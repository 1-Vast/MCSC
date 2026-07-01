"""Dataset loading and cold-split construction for PRISM affinity protocols."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model.staticfeatures import (
    build_chemberta_features,
    build_morgan_features,
    build_protein_plm_features,
    build_smiles_cnn_features,
)
from scripts.runtime import RunSettings, load_davis, select_split, validation_indices


@dataclass
class AffinityBundle:
    dataset: str
    drug_raw: np.ndarray
    target_raw: np.ndarray
    drug_ids: list[str]
    target_ids: list[str]
    target_seqs: list[str]
    feature_meta: dict
    davis_data: dict | None = None
    kiba_y: np.ndarray | None = None


def load_affinity_bundle(dataset: str, args: argparse.Namespace, device: torch.device) -> AffinityBundle:
    cache_dir = REPO / "dataset" / "cache"
    if dataset == "DAVIS":
        settings = RunSettings(device=str(device), data_dir="dataset/davis")
        data = load_davis(settings.repo_path(settings.data_dir))
        drug_ids = data["drugIds"]
        smiles = data["drugSmiles"]
        target_ids = data["targetIds"]
        seqs = data["targetSeqs"]
        morgan = build_morgan_features(smiles, device, bits=args.morgan_bits)
        davis_data = data
        y = None
    elif dataset == "KIBA":
        root = REPO / "dataset" / "kiba"
        drug_dict = json.loads((root / "ligands_can.txt").read_text(encoding="utf-8"))
        protein_dict = json.loads((root / "proteins.txt").read_text(encoding="utf-8"))
        drug_ids = list(drug_dict.keys())
        smiles = [drug_dict[key] for key in drug_ids]
        target_ids = list(protein_dict.keys())
        seqs = [protein_dict[key] for key in target_ids]
        if args.morgan_bits == 1024 and (root / "morgan_cache_1024.npy").exists():
            morgan = np.load(root / "morgan_cache_1024.npy").astype(np.float32)
        else:
            morgan = build_morgan_features(smiles, device, bits=args.morgan_bits)
        davis_data = None
        y = np.asarray(pickle.load(open(root / "Y", "rb"), encoding="latin1"), dtype=float)
    else:
        raise ValueError(dataset)

    drug_encoder = getattr(args, "drug_encoder", "morgan")
    drug_parts: list[np.ndarray] = []
    rep_parts: list[str] = []
    drug_meta = {
        "morganBits": int(args.morgan_bits),
        "morganRadius": 2,
        "externalPretrainedDrug": False,
        "drugEncoderMode": drug_encoder,
    }
    if drug_encoder in ("morgan", "morgan-chemberta"):
        drug_parts.append(morgan.astype(np.float32))
        rep_parts.append("morgan")
        if args.smiles_cnn:
            smiles_feat, smiles_meta = build_smiles_cnn_features(
                smiles,
                device=device,
                batch_size=args.feature_batch_size,
                max_len=args.smiles_max_len,
                seed=args.feature_seed,
            )
            drug_parts.append(smiles_feat)
            drug_meta.update(smiles_meta)
            rep_parts.append("smiles_cnn")
    if drug_encoder in ("chemberta", "morgan-chemberta"):
        cb_cache = Path(args.drug_cache_path) if getattr(args, "drug_cache_path", None) else cache_dir
        cb_feat, cb_meta = build_chemberta_features(
            smiles,
            model_id=args.chemberta_model,
            cache_dir=cb_cache,
            device=device,
            batch_size=args.feature_batch_size,
            max_len=args.smiles_max_len,
        )
        drug_parts.append(cb_feat)
        drug_meta.update({
            f"chemberta_{key}" if key in {"cache", "digest", "pooling", "maxLen"} else key: value
            for key, value in cb_meta.items()
        })
        drug_meta["externalPretrainedDrug"] = True
        rep_parts.append("chemberta")
    if not drug_parts:
        raise SystemExit(f"unknown --drug-encoder {drug_encoder!r}")
    drug_raw = np.concatenate(drug_parts, axis=1).astype(np.float32)

    target_raw, target_meta = build_protein_plm_features(
        target_ids,
        seqs,
        source=args.plm_source,
        cache_dir=cache_dir,
        device=device,
        batch_size=args.esm_batch_size,
        max_len=args.esm_max_len,
    )
    feature_meta = {
        "drugRepresentation": "+".join(rep_parts),
        "drugDim": int(drug_raw.shape[1]),
        "targetDim": int(target_raw.shape[1]),
        **drug_meta,
        **target_meta,
    }
    return AffinityBundle(
        dataset=dataset,
        drug_raw=drug_raw,
        target_raw=target_raw.astype(np.float32),
        drug_ids=drug_ids,
        target_ids=target_ids,
        target_seqs=seqs,
        feature_meta=feature_meta,
        davis_data=davis_data,
        kiba_y=y,
    )


def davis_split(bundle: AffinityBundle, split: str, seed: int) -> dict:
    sp = select_split(bundle.davis_data, split, seed, target_feat=bundle.target_raw)
    train_idx, val_idx, basis = validation_indices(sp, split, seed, target_feat=bundle.target_raw)
    return {**sp, "tr_idx": train_idx, "val_idx": val_idx, "seed": seed, "validationBasis": basis}


def kiba_split(bundle: AffinityBundle, split: str, seed: int) -> dict:
    y = bundle.kiba_y
    drug_idx, target_idx = np.where(np.isfinite(y))
    n_targets = bundle.target_raw.shape[0]
    rng = np.random.RandomState(seed + 5)
    if split == "target-cold":
        held = set(rng.choice(n_targets, max(1, int(n_targets * 0.2)), replace=False).tolist())
    elif split == "cluster-cold":
        labels = KMeans(n_clusters=8, random_state=seed + 5, n_init=10).fit(bundle.target_raw).labels_
        held_clusters = set(rng.choice(8, max(1, int(8 * 0.3)), replace=False).tolist())
        held = set(np.where(np.isin(labels, list(held_clusters)))[0].tolist())
    else:
        raise ValueError(split)
    test_mask = np.isin(target_idx, list(held))
    train_d, train_t = drug_idx[~test_mask], target_idx[~test_mask]
    test_d, test_t = drug_idx[test_mask], target_idx[test_mask]
    train_y = y[train_d, train_t]
    test_y = y[test_d, test_t]
    rng_val = np.random.RandomState(seed + 99)
    train_targets = np.unique(train_t)
    val_targets = set(rng_val.choice(train_targets, max(1, int(len(train_targets) * 0.2)), replace=False).tolist())
    val_mask = np.isin(train_t, list(val_targets))
    return {
        "name": split,
        "trainD": train_d,
        "trainT": train_t,
        "trainY": train_y,
        "testD": test_d,
        "testT": test_t,
        "testY": test_y,
        "tr_idx": np.where(~val_mask)[0],
        "val_idx": np.where(val_mask)[0],
        "seed": seed,
        "validationBasis": "cold_target",
        "heldTargets": sorted(int(x) for x in held),
    }


def make_split(bundle: AffinityBundle, split: str, seed: int) -> dict:
    if bundle.dataset == "DAVIS":
        return davis_split(bundle, split, seed)
    return kiba_split(bundle, split, seed)


def parse_cells(values: list[str] | None) -> list[tuple[str, str]]:
    allowed = {
        "DAVIS/target-cold": ("DAVIS", "target-cold"),
        "DAVIS/family-cold": ("DAVIS", "family-cold"),
        "KIBA/target-cold": ("KIBA", "target-cold"),
        "KIBA/cluster-cold": ("KIBA", "cluster-cold"),
    }
    if not values:
        return list(allowed.values())
    cells = []
    for value in values:
        if value not in allowed:
            raise SystemExit(f"unknown split {value!r}; choices: {', '.join(sorted(allowed))}")
        cells.append(allowed[value])
    return cells

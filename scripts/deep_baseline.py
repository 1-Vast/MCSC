"""DeepDTA baseline gate under this repo's exact cold-split protocol.

This is a compact, dependency-light DeepDTA reproduction: character CNN over
SMILES and protein sequence, validation-only early stopping, no mainline model
changes, and resume-safe per split/seed results.

Run examples:
  D:/anaconda/envs/drug/python.exe scripts/deep_baseline.py
  D:/anaconda/envs/drug/python.exe scripts/deep_baseline.py --splits KIBA/target-cold --seeds 1
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model.metrics import compute_metrics
from scripts.seq_descriptors import ctriad
from scripts.runtime import RunSettings, load_data, load_features, select_split, validation_indices

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEV.type == "cuda":
    torch.backends.cudnn.benchmark = True

OUT = REPO / "doc" / "deep-baseline-results.json"
SPLITS = ["DAVIS/target-cold", "DAVIS/family-cold", "KIBA/target-cold", "KIBA/cluster-cold"]
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]

SMI_LEN = 100
SEQ_LEN = 1000
SMI_VOCAB = "#%()+-.0123456789=@ABCDEFGHIKLMNOPRSTVXYZ[\\]abcdefgilmnoprstuy/"
SEQ_VOCAB = "ACDEFGHIKLMNPQRSTVWXYBZUO"
SMI_I = {c: i + 1 for i, c in enumerate(SMI_VOCAB)}
SEQ_I = {c: i + 1 for i, c in enumerate(SEQ_VOCAB)}


def encode_chars(value: str, table: dict[str, int], length: int) -> np.ndarray:
    encoded = np.zeros(length, np.int64)
    for i, char in enumerate(value[:length]):
        encoded[i] = table.get(char, 0)
    return encoded


class DeepDTA(nn.Module):
    def __init__(self, n_smi: int, n_seq: int, emb: int = 128) -> None:
        super().__init__()
        self.drug_embedding = nn.Embedding(n_smi + 1, emb, padding_idx=0)
        self.target_embedding = nn.Embedding(n_seq + 1, emb, padding_idx=0)
        self.drug_conv = nn.Sequential(
            nn.Conv1d(emb, 32, 4),
            nn.ReLU(),
            nn.Conv1d(32, 64, 6),
            nn.ReLU(),
            nn.Conv1d(64, 96, 8),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.target_conv = nn.Sequential(
            nn.Conv1d(emb, 32, 4),
            nn.ReLU(),
            nn.Conv1d(32, 64, 8),
            nn.ReLU(),
            nn.Conv1d(64, 96, 12),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(192, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
        )

    def encode_drug(self, drug: torch.Tensor) -> torch.Tensor:
        return self.drug_conv(self.drug_embedding(drug).transpose(1, 2)).squeeze(-1)

    def encode_target(self, target: torch.Tensor) -> torch.Tensor:
        return self.target_conv(self.target_embedding(target).transpose(1, 2)).squeeze(-1)

    def score_features(self, drug: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([drug, target], 1)).squeeze(-1)

    def forward(self, drug: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.score_features(self.encode_drug(drug), self.encode_target(target))

    def forward_ids(self, drug_table: torch.Tensor, target_table: torch.Tensor, drug_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        unique_drug, drug_inverse = torch.unique(drug_ids, sorted=False, return_inverse=True)
        unique_target, target_inverse = torch.unique(target_ids, sorted=False, return_inverse=True)
        drug_features = self.encode_drug(drug_table[unique_drug])[drug_inverse]
        target_features = self.encode_target(target_table[unique_target])[target_inverse]
        return self.score_features(drug_features, target_features)


def parameter_count() -> int:
    return int(sum(p.numel() for p in DeepDTA(len(SMI_VOCAB), len(SEQ_VOCAB)).parameters()))


def autocast_context(enabled: bool):
    return torch.amp.autocast("cuda", enabled=enabled)


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def predict(
    model: nn.Module,
    drug_table: torch.Tensor,
    target_table: torch.Tensor,
    drug_idx: np.ndarray,
    target_idx: np.ndarray,
    eval_batch_size: int,
    amp_enabled: bool,
) -> np.ndarray:
    drug_ids = torch.from_numpy(drug_idx).long().to(DEV)
    target_ids = torch.from_numpy(target_idx).long().to(DEV)
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(drug_idx), eval_batch_size):
            end = start + eval_batch_size
            with autocast_context(amp_enabled):
                pred = model.forward_ids(drug_table, target_table, drug_ids[start:end], target_ids[start:end])
            chunks.append(pred.float().cpu().numpy())
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)


def worst_target_group_r2(target_feat: np.ndarray, test_targets: np.ndarray, y: np.ndarray, pred: np.ndarray) -> Optional[float]:
    unique_targets = np.unique(test_targets)
    n_clusters = min(4, len(unique_targets))
    if n_clusters < 2:
        return None
    labels = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(target_feat[unique_targets]).labels_
    target_to_group = dict(zip(unique_targets.tolist(), labels.tolist()))
    groups = np.array([target_to_group[int(t)] for t in test_targets])
    scores = []
    for group in range(n_clusters):
        mask = groups == group
        if int(mask.sum()) >= 5:
            scores.append(compute_metrics(y[mask], pred[mask])["r2"])
    return round(float(min(scores)), 4) if scores else None


def evaluate_predictions(
    y_true: np.ndarray,
    pred: np.ndarray,
    target_feat: np.ndarray,
    test_targets: np.ndarray,
) -> dict:
    metrics = compute_metrics(y_true, pred)
    metrics["worstgrp_R2"] = worst_target_group_r2(target_feat, test_targets, y_true, pred)
    return metrics


def train_eval(
    drug_table: torch.Tensor,
    target_table: torch.Tensor,
    train_drug: np.ndarray,
    train_target: np.ndarray,
    train_y: np.ndarray,
    val_drug: np.ndarray,
    val_target: np.ndarray,
    val_y: np.ndarray,
    test_drug: np.ndarray,
    test_target: np.ndarray,
    test_y: np.ndarray,
    target_feat: np.ndarray,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    eval_batch_size: int,
    amp_enabled: bool,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if DEV.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    started = time.time()
    model = DeepDTA(len(SMI_VOCAB), len(SEQ_VOCAB)).to(DEV)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = make_scaler(amp_enabled)

    train_drug_t = torch.from_numpy(train_drug).long().to(DEV)
    train_target_t = torch.from_numpy(train_target).long().to(DEV)
    train_y_t = torch.from_numpy(train_y).float().to(DEV)

    best_mse = float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    epochs_run = 0

    for epoch in range(1, epochs + 1):
        epochs_run = epoch
        model.train()
        order = torch.randperm(len(train_drug), device=DEV)
        for start in range(0, len(train_drug), batch_size):
            idx = order[start:start + batch_size]
            if len(idx) < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(amp_enabled):
                pred = model.forward_ids(drug_table, target_table, train_drug_t[idx], train_target_t[idx])
                loss = ((pred - train_y_t[idx]) ** 2).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        val_pred = predict(model, drug_table, target_table, val_drug, val_target, eval_batch_size, amp_enabled)
        val_mse = float(np.mean((val_y - val_pred) ** 2))
        if val_mse < best_mse - 1e-4:
            best_mse = val_mse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("DeepDTA did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    test_pred = predict(model, drug_table, target_table, test_drug, test_target, eval_batch_size, amp_enabled)
    metrics = evaluate_predictions(test_y, test_pred, target_feat, test_target)
    return {
        "metrics": metrics,
        "bestValMSE": round(best_mse, 6),
        "bestEpoch": int(best_epoch),
        "epochsRun": int(epochs_run),
        "runtimeSec": round(time.time() - started, 2),
    }


def bootstrap_ci(values: list[float]) -> Optional[list[float]]:
    finite = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if finite.size == 0:
        return None
    boot = np.random.RandomState(0).choice(finite, (10000, finite.size), replace=True).mean(1)
    return [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)]


def summarize_cell(cell: dict, expected_seeds: list[int]) -> None:
    runs = cell.setdefault("runs", {})
    present = sorted(int(seed) for seed in runs)
    summary = {
        "status": "complete" if present == sorted(expected_seeds) else "partial",
        "nSeeds": len(present),
        "seeds": present,
    }
    metric_names = ["r2", "mse", "rmse", "mae", "pearson", "spearman", "worstgrp_R2"]
    for name in metric_names:
        values = []
        for seed in present:
            value = runs[str(seed)]["metrics"].get(name)
            if value is not None and np.isfinite(value):
                values.append(float(value))
        summary[name] = round(float(np.mean(values)), 4) if values else None
        if name == "r2":
            summary["r2_ci95"] = bootstrap_ci(values)
    runtime = [float(runs[str(seed)].get("runtimeSec", 0.0)) for seed in present]
    summary["runtimeSecTotal"] = round(float(np.sum(runtime)), 2) if runtime else 0.0
    summary["runtimeSecMean"] = round(float(np.mean(runtime)), 2) if runtime else None
    cell["summary"] = summary


def load_results() -> dict:
    if not OUT.exists():
        return {}
    raw = json.loads(OUT.read_text())
    if raw.get("_schema") == "deepdta-v2":
        return raw

    migrated = {"_schema": "deepdta-v2", "_migratedFrom": "r2-only legacy"}
    for key, value in raw.items():
        if key.startswith("_"):
            continue
        migrated[key] = {"legacyR2Only": value.get("R2", []), "runs": {}}
    return migrated


def save_results(out: dict, expected_seeds: list[int]) -> None:
    out["_schema"] = "deepdta-v2"
    out["_metadata"] = {
        "baseline": "DeepDTA compact char-CNN",
        "device": str(DEV),
        "seeds": SEEDS,
        "smilesLength": SMI_LEN,
        "sequenceLength": SEQ_LEN,
        "parameterCount": parameter_count(),
        "note": "Per-seed metrics are validation-selected; no test tuning.",
    }
    for key in SPLITS:
        if key in out:
            summarize_cell(out[key], expected_seeds)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    tmp.replace(OUT)


def load_esm150_target_features(n_targets: int) -> np.ndarray:
    hits = sorted((REPO / "dataset" / "cache").glob(f"esm_esm2_t30_150M_UR50D_n{n_targets}_*.npy"))
    if not hits:
        raise FileNotFoundError("Missing KIBA ESM-2 150M cache; run scripts/representation_plm.py first")
    values = np.load(hits[-1]).astype(np.float32)
    return values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-9)


def build_davis_context() -> dict:
    settings = RunSettings(device=str(DEV))
    settings.encoder.drug = "morgan"
    settings.encoder.target = "kb"
    data = load_data(settings)
    features = load_features(settings)
    drug_table = torch.from_numpy(np.stack([encode_chars(s, SMI_I, SMI_LEN) for s in data["drugSmiles"]])).long().to(DEV)
    target_table = torch.from_numpy(np.stack([encode_chars(s, SEQ_I, SEQ_LEN) for s in data["targetSeqs"]])).long().to(DEV)
    return {
        "data": data,
        "splitTargetFeat": features["targetFeat"],
        "groupTargetFeat": ctriad(data["targetSeqs"]),
        "drugTable": drug_table,
        "targetTable": target_table,
    }


def build_kiba_context() -> dict:
    data_dir = REPO / "dataset" / "kiba"
    with open(data_dir / "Y", "rb") as handle:
        y_matrix = np.asarray(pickle.load(handle, encoding="latin1"), dtype=float)
    smiles = list(json.loads((data_dir / "ligands_can.txt").read_text()).values())
    seqs = list(json.loads((data_dir / "proteins.txt").read_text()).values())
    drug_table = torch.from_numpy(np.stack([encode_chars(s, SMI_I, SMI_LEN) for s in smiles])).long().to(DEV)
    target_table = torch.from_numpy(np.stack([encode_chars(s, SEQ_I, SEQ_LEN) for s in seqs])).long().to(DEV)
    return {
        "yMatrix": y_matrix,
        "groupTargetFeat": load_esm150_target_features(len(seqs)),
        "drugTable": drug_table,
        "targetTable": target_table,
    }


def davis_seed_split(ctx: dict, split: str, seed: int) -> dict:
    split_data = select_split(ctx["data"], split, seed, target_feat=ctx["splitTargetFeat"])
    train_idx, val_idx, _ = validation_indices(split_data, split, seed, target_feat=ctx["splitTargetFeat"])
    return {
        "trainD": split_data["trainD"][train_idx],
        "trainT": split_data["trainT"][train_idx],
        "trainY": split_data["trainY"][train_idx],
        "valD": split_data["trainD"][val_idx],
        "valT": split_data["trainT"][val_idx],
        "valY": split_data["trainY"][val_idx],
        "testD": split_data["testD"],
        "testT": split_data["testT"],
        "testY": split_data["testY"],
        "seed": seed,
    }


def kiba_seed_splits(split: str) -> dict[int, dict]:
    from scripts.mcsc import make_split

    return {int(sp["seed"]): {
        "trainD": sp["trainD"][sp["tr_idx"]],
        "trainT": sp["trainT"][sp["tr_idx"]],
        "trainY": sp["trainY"][sp["tr_idx"]],
        "valD": sp["trainD"][sp["val_idx"]],
        "valT": sp["trainT"][sp["val_idx"]],
        "valY": sp["trainY"][sp["val_idx"]],
        "testD": sp["testD"],
        "testT": sp["testT"],
        "testY": sp["testY"],
        "seed": int(sp["seed"]),
    } for _, sp in (make_split("KIBA", split, seed) for seed in SEEDS)}


def run_one_split(key: str, seeds: list[int], out: dict, args: argparse.Namespace, contexts: dict) -> None:
    dataset, split = key.split("/", 1)
    cell = out.setdefault(key, {"runs": {}})
    runs = cell.setdefault("runs", {})

    if dataset == "DAVIS":
        ctx = contexts.setdefault("DAVIS", build_davis_context())
        seed_splits = {seed: davis_seed_split(ctx, split, seed) for seed in seeds}
    elif dataset == "KIBA":
        ctx = contexts.setdefault("KIBA", build_kiba_context())
        seed_splits = kiba_seed_splits(split)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    for seed in seeds:
        existing = runs.get(str(seed))
        if (
            existing
            and not args.force
            and int(existing.get("epochsMax", -1)) == args.epochs
            and int(existing.get("patience", -1)) == args.patience
            and bool(existing.get("amp", False)) == bool(args.amp and DEV.type == "cuda")
            and int(existing.get("batchSize", -1)) == args.batch_size
        ):
            print(f"[skip] {key} seed{seed} already has matching metrics", flush=True)
            continue
        split_data = seed_splits[seed]
        print(f"[run] {key} seed{seed} on {DEV}", flush=True)
        result = train_eval(
            ctx["drugTable"],
            ctx["targetTable"],
            split_data["trainD"],
            split_data["trainT"],
            split_data["trainY"],
            split_data["valD"],
            split_data["valT"],
            split_data["valY"],
            split_data["testD"],
            split_data["testT"],
            split_data["testY"],
            ctx["groupTargetFeat"],
            seed,
            args.epochs,
            args.patience,
            args.batch_size,
            args.eval_batch_size,
            args.amp and DEV.type == "cuda",
        )
        result.update({
            "seed": seed,
            "split": key,
            "epochsMax": args.epochs,
            "patience": args.patience,
            "batchSize": args.batch_size,
            "evalBatchSize": args.eval_batch_size,
            "amp": bool(args.amp and DEV.type == "cuda"),
        })
        runs[str(seed)] = result
        save_results(out, seeds)
        print(
            f"[done] {key} seed{seed} R2={result['metrics']['r2']:.4f} "
            f"RMSE={result['metrics']['rmse']:.4f} runtime={result['runtimeSec']:.1f}s",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="*", default=SPLITS, choices=SPLITS)
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--amp", dest="amp", action="store_true", default=False)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    invalid_seeds = [seed for seed in args.seeds if seed not in SEEDS]
    if invalid_seeds:
        raise SystemExit(f"Unsupported seeds: {invalid_seeds}; expected subset of {SEEDS}")
    print(f"device={DEV} splits={args.splits} seeds={args.seeds}", flush=True)
    out = load_results()
    save_results(out, args.seeds)
    contexts: dict = {}
    for key in args.splits:
        run_one_split(key, args.seeds, out, args, contexts)
    save_results(out, args.seeds)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

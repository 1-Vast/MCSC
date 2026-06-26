"""Train and evaluate the current MCSC mainline.

Current mainline:

    prior = validation-global blend memory prior
    refiner = trained ResidualRefiner
    final = prior + alpha * (refiner - prior)

Only this path is promotable. Older selectors and safety layers are summarized
as failed directions under experiments/analysis.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import pickle
import subprocess
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("OMP_NUM_THREADS", "1")

from sklearn.cluster import KMeans


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model.memory import InteractionMemory
from model.metrics import compute_metrics
from model.refiners import ResidualRefiner
from scripts.runtime import RunSettings, load_data, load_features, select_split, validation_indices
from scripts.seq_descriptors import aac_dip, ctriad, split_norm


OUT_DIR = REPO / "outputs" / "mcsc"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
MANIFEST = OUT_DIR / "manifest.json"
RESULTS = REPO / "doc" / "mcsc-mainline-results.json"
REPORT = REPO / "doc" / "mcsc-mainline-report.md"

CELLS = (
    ("DAVIS", "target-cold"),
    ("DAVIS", "family-cold"),
    ("KIBA", "target-cold"),
    ("KIBA", "cluster-cold"),
)
SEEDS = tuple(range(1, 9))
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class DatasetBundle:
    dataset: str
    drug_raw: np.ndarray
    target_raw: np.ndarray
    split_target_raw: np.ndarray | None = None
    davis_data: dict | None = None
    kiba_y: np.ndarray | None = None


_BUNDLES: dict[str, DatasetBundle] = {}


def json_load(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def repo_rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


class GpuMonitor:
    """Optional lightweight nvidia-smi sampler for real GPU utilization audits."""

    def __init__(self, path: Path | None, interval: float = 0.5) -> None:
        self.path = path
        self.interval = max(float(interval), 0.2)
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        if self.path is None:
            return self
        self._thread = threading.Thread(target=self._run, name="mcsc-gpu-monitor", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.path is None:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.write()

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = self._sample()
            if sample:
                self.samples.append(sample)
            self._stop.wait(self.interval)

    @staticmethod
    def _sample() -> dict | None:
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        first = proc.stdout.strip().splitlines()[0]
        parts = [part.strip() for part in first.split(",")]
        if len(parts) < 4:
            return None
        return {
            "timestamp": parts[0],
            "utilizationGpuPct": float(parts[1]),
            "memoryUsedMiB": float(parts[2]),
            "memoryTotalMiB": float(parts[3]),
        }

    def write(self) -> None:
        if self.path is None:
            return
        util = np.asarray([row["utilizationGpuPct"] for row in self.samples], dtype=float)
        mem = np.asarray([row["memoryUsedMiB"] for row in self.samples], dtype=float)
        summary = {
            "schema": "drugtarget-gpu-monitor-v1",
            "note": "Samples include CPU-side split/data phases; short DAVIS/KIBA cells can be bursty.",
            "intervalSec": self.interval,
            "nSamples": int(len(self.samples)),
            "utilizationGpuPct": {
                "mean": round(float(util.mean()), 2) if util.size else None,
                "p50": round(float(np.percentile(util, 50)), 2) if util.size else None,
                "p90": round(float(np.percentile(util, 90)), 2) if util.size else None,
                "max": round(float(util.max()), 2) if util.size else None,
                "shareAtLeast80Pct": round(float((util >= 80).mean()), 4) if util.size else None,
                "shareAtLeast95Pct": round(float((util >= 95).mean()), 4) if util.size else None,
            },
            "memoryMiB": {
                "meanUsed": round(float(mem.mean()), 2) if mem.size else None,
                "maxUsed": round(float(mem.max()), 2) if mem.size else None,
            },
            "samples": self.samples,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def checkpoint_path(dataset: str, split: str, seed: int) -> Path:
    name = f"{dataset.lower()}_{split.replace('-', '_')}_seed{seed}.pt"
    return CHECKPOINT_DIR / name


def load_esm150(n_targets: int) -> np.ndarray:
    hits = sorted((REPO / "dataset" / "cache").glob(f"esm_esm2_t30_150M_UR50D_n{n_targets}_*.npy"))
    if not hits:
        raise SystemExit("missing KIBA ESM-2 150M cache; run the PLM embedding preparation first")
    values = np.load(hits[-1]).astype(np.float32)
    return (values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-9)).astype(np.float32)


def load_bundle(dataset: str) -> DatasetBundle:
    if dataset in _BUNDLES:
        return _BUNDLES[dataset]
    if dataset == "DAVIS":
        settings = RunSettings(device="cpu")
        settings.encoder.drug = "morgan"
        settings.encoder.target = "kb"
        data = load_data(settings)
        features = load_features(settings)
        bundle = DatasetBundle(
            dataset="DAVIS",
            drug_raw=features["drugFeat"].cpu().numpy().astype(np.float32),
            target_raw=ctriad(data["targetSeqs"]).astype(np.float32),
            split_target_raw=features["targetFeat"].cpu().numpy().astype(np.float32),
            davis_data=data,
        )
    elif dataset == "KIBA":
        root = REPO / "dataset" / "kiba"
        y = np.asarray(pickle.load(open(root / "Y", "rb"), encoding="latin1"), dtype=float)
        seqs = list(json.loads((root / "proteins.txt").read_text(encoding="utf-8")).values())
        bundle = DatasetBundle(
            dataset="KIBA",
            drug_raw=np.load(root / "morgan_cache_1024.npy").astype(np.float32),
            target_raw=load_esm150(len(seqs)),
            split_target_raw=aac_dip(seqs).astype(np.float32),
            kiba_y=y,
        )
    else:
        raise ValueError(dataset)
    _BUNDLES[dataset] = bundle
    return bundle


def davis_split(bundle: DatasetBundle, split: str, seed: int) -> dict:
    sp = select_split(bundle.davis_data, split, seed, target_feat=bundle.split_target_raw)
    train_idx, val_idx, basis = validation_indices(sp, split, seed, target_feat=bundle.split_target_raw)
    return {**sp, "tr_idx": train_idx, "val_idx": val_idx, "seed": seed, "validationBasis": basis}


def kiba_split(bundle: DatasetBundle, split: str, seed: int) -> dict:
    y = bundle.kiba_y
    drug_idx, target_idx = np.where(np.isfinite(y))
    n_targets = bundle.target_raw.shape[0]
    rng = np.random.RandomState(seed + 5)
    if split == "target-cold":
        held = set(rng.choice(n_targets, max(1, int(n_targets * 0.2)), replace=False).tolist())
    elif split == "cluster-cold":
        labels = KMeans(n_clusters=8, random_state=seed + 5, n_init=10).fit(bundle.split_target_raw).labels_
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


def make_split(dataset: str, split: str, seed: int) -> tuple[DatasetBundle, dict]:
    bundle = load_bundle(dataset)
    if dataset == "DAVIS":
        return bundle, davis_split(bundle, split, seed)
    return bundle, kiba_split(bundle, split, seed)


def alpha_config() -> dict[tuple[str, str], float]:
    data = json_load(REPO / "config" / "residual-alpha-calibration.json", {})
    entries = {}
    for item in data.get("entries", []):
        entries[(item.get("dataset"), item.get("split"))] = float(item["alpha"])
    return entries


def frozen_alpha_for(dataset: str, split: str) -> float:
    cfg = alpha_config()
    key = (dataset, split)
    if key not in cfg:
        raise SystemExit(f"missing frozen residual alpha for {dataset}/{split}")
    return cfg[key]


def drug_stats(drugs: np.ndarray, labels: np.ndarray, n_drugs: int) -> tuple[np.ndarray, np.ndarray, float]:
    values: dict[int, list[float]] = defaultdict(list)
    for drug, label in zip(drugs, labels):
        values[int(drug)].append(float(label))
    counts = np.zeros(n_drugs, dtype=np.float32)
    means = np.zeros(n_drugs, dtype=np.float32)
    for drug, vals in values.items():
        counts[drug] = len(vals)
        means[drug] = float(np.mean(vals))
    return counts, means, float(np.mean(labels))


def marginal_query(means: np.ndarray, counts: np.ndarray, global_mean: float, query_drugs: np.ndarray) -> np.ndarray:
    return np.asarray([
        means[int(drug)] if counts[int(drug)] > 0 else global_mean
        for drug in query_drugs
    ], dtype=np.float32)


def marginal_loo(drugs: np.ndarray, labels: np.ndarray, global_mean: float) -> np.ndarray:
    sums: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)
    for drug, label in zip(drugs, labels):
        sums[int(drug)] += float(label)
        counts[int(drug)] += 1
    out = np.empty(len(drugs), dtype=np.float32)
    for i, (drug, label) in enumerate(zip(drugs, labels)):
        drug = int(drug)
        out[i] = (sums[drug] - float(label)) / (counts[drug] - 1) if counts[drug] > 1 else global_mean
    return out


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return compute_metrics(y_true, y_pred)["r2"]


def select_global_blend_weight(val_fine: np.ndarray, val_marginal: np.ndarray, val_y: np.ndarray) -> float:
    best_r2 = -1e9
    best_weight = 1.0
    for weight in np.linspace(0.0, 1.0, 21):
        score = r2(val_y, weight * val_fine + (1.0 - weight) * val_marginal)
        if score > best_r2:
            best_r2 = score
            best_weight = float(weight)
    return best_weight


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = DEFAULT_DEVICE
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    return torch.device(name)


def autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=enabled)
    return nullcontext()


def train_residual_refiner(
    drug_feat: torch.Tensor,
    target_feat: torch.Tensor,
    train_d: np.ndarray,
    train_t: np.ndarray,
    train_prior: np.ndarray,
    train_y: np.ndarray,
    val_d: np.ndarray,
    val_t: np.ndarray,
    val_prior: np.ndarray,
    val_y: np.ndarray,
    seed: int,
    device: torch.device,
    batch_size: int,
    amp: bool,
    epochs: int = 12,
) -> tuple[ResidualRefiner, float]:
    torch.manual_seed(seed)
    model = ResidualRefiner(
        drug_feat.shape[1],
        target_feat.shape[1],
        hidden=(256, 128),
        dropout=0.2,
        norm="batch",
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    drug_feat = drug_feat.to(device)
    target_feat = target_feat.to(device)
    train_d_t = torch.from_numpy(train_d).long().to(device)
    train_t_t = torch.from_numpy(train_t).long().to(device)
    train_prior_t = torch.from_numpy(train_prior).float().to(device)
    train_y_t = torch.from_numpy(train_y).float().to(device)
    val_d_t = torch.from_numpy(val_d).long().to(device)
    val_t_t = torch.from_numpy(val_t).long().to(device)
    val_prior_t = torch.from_numpy(val_prior).float().to(device)
    val_y_t = torch.from_numpy(val_y).float().to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    best_val = float("inf")
    best_state = None
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(train_d), device=device)
        for start in range(0, len(train_d), batch_size):
            idx = order[start:start + batch_size]
            if idx.numel() < 2:
                continue
            opt.zero_grad(set_to_none=True)
            with autocast_context(device, amp):
                pred = model(drug_feat[train_d_t[idx]], target_feat[train_t_t[idx]], train_prior_t[idx])
                loss = ((pred - train_y_t[idx]) ** 2).mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        model.eval()
        with torch.no_grad():
            with autocast_context(device, amp):
                val_pred = model(drug_feat[val_d_t], target_feat[val_t_t], val_prior_t)
                val_loss = float(((val_pred - val_y_t) ** 2).mean().item())
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("MCSC residual refiner did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, best_val


@torch.no_grad()
def predict_refiner(
    model: ResidualRefiner,
    drug_feat: torch.Tensor,
    target_feat: torch.Tensor,
    drug_idx: np.ndarray,
    target_idx: np.ndarray,
    prior: np.ndarray,
    device: torch.device,
    batch_size: int,
    amp: bool,
) -> np.ndarray:
    model = model.to(device)
    drug_feat = drug_feat.to(device)
    target_feat = target_feat.to(device)
    drug_all = torch.as_tensor(np.ascontiguousarray(drug_idx, dtype=np.int64), dtype=torch.long, device=device)
    target_all = torch.as_tensor(np.ascontiguousarray(target_idx, dtype=np.int64), dtype=torch.long, device=device)
    prior_all = torch.as_tensor(np.ascontiguousarray(prior, dtype=np.float32), dtype=torch.float32, device=device)
    out = []
    with torch.no_grad():
        for start in range(0, len(drug_idx), batch_size):
            sl = slice(start, start + batch_size)
            with autocast_context(device, amp):
                pred = model(drug_feat[drug_all[sl]], target_feat[target_all[sl]], prior_all[sl])
            out.append(pred.detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def train_one(
    dataset: str,
    split: str,
    seed: int,
    force: bool,
    device: torch.device,
    batch_size: int,
    eval_batch_size: int,
    amp: bool,
) -> dict:
    out = checkpoint_path(dataset, split, seed)
    meta_out = out.with_suffix(".json")
    if out.exists() and meta_out.exists() and not force:
        return json_load(meta_out, {})

    bundle, sp = make_split(dataset, split, seed)
    train_d, train_t, train_y = sp["trainD"], sp["trainT"], sp["trainY"]
    fit_idx, val_idx = sp["tr_idx"], sp["val_idx"]
    fit_d, fit_t, fit_y = train_d[fit_idx], train_t[fit_idx], train_y[fit_idx]
    val_d, val_t, val_y = train_d[val_idx], train_t[val_idx], train_y[val_idx]

    drug_feat_np = split_norm(bundle.drug_raw, np.unique(fit_d))
    target_feat_np = split_norm(bundle.target_raw, np.unique(fit_t))
    drug_feat = torch.from_numpy(drug_feat_np).float().to(device)
    target_feat = torch.from_numpy(target_feat_np).float().to(device)

    n_drugs = bundle.drug_raw.shape[0]
    full_counts, full_means, full_global = drug_stats(train_d, train_y, n_drugs)
    fit_counts, fit_means, fit_global = drug_stats(fit_d, fit_y, n_drugs)

    train_marginal = marginal_loo(train_d, train_y, full_global)
    val_marginal = marginal_query(fit_means, fit_counts, fit_global, val_d)

    mem_fit = InteractionMemory(drug_feat, target_feat, fit_d, fit_t, fit_y, normalize=False)
    mem_full = InteractionMemory(drug_feat, target_feat, train_d, train_t, train_y, normalize=False)

    val_fine = mem_fit.predict(val_d, val_t)
    train_fine = mem_full.predict(train_d, train_t, exclude_self=True)

    blend_weight = select_global_blend_weight(val_fine, val_marginal, val_y)
    train_prior = blend_weight * train_fine + (1.0 - blend_weight) * train_marginal
    val_prior = blend_weight * val_fine + (1.0 - blend_weight) * val_marginal

    model, best_val = train_residual_refiner(
        drug_feat,
        target_feat,
        train_d,
        train_t,
        train_prior,
        train_y,
        val_d,
        val_t,
        val_prior,
        val_y,
        seed,
        device,
        batch_size,
        amp,
    )
    alpha = frozen_alpha_for(dataset, split)
    metadata = {
        "schema": "drugtarget-mcsc-checkpoint-v1",
        "dataset": dataset,
        "split": split,
        "seed": int(seed),
        "checkpoint": repo_rel(out),
        "modelType": "ResidualRefiner",
        "drugDim": int(drug_feat.shape[1]),
        "targetDim": int(target_feat.shape[1]),
        "targetRepresentation": "ctriad" if dataset == "DAVIS" else "esm2_t30_150M_UR50D",
        "prior": "global_blend",
        "device": str(device),
        "amp": bool(amp and device.type == "cuda"),
        "batchSize": int(batch_size),
        "evalBatchSize": int(eval_batch_size),
        "blendWeight": float(blend_weight),
        "frozenAlpha": float(alpha),
        "bestValLoss": float(best_val),
        "validationBasis": sp.get("validationBasis"),
        "trainedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **metadata,
            "stateDict": model.state_dict(),
        },
        out,
    )
    json_dump(meta_out, metadata)
    return metadata


def load_checkpoint(dataset: str, split: str, seed: int, device: torch.device) -> tuple[dict, ResidualRefiner]:
    path = checkpoint_path(dataset, split, seed)
    if not path.exists():
        raise SystemExit(f"missing checkpoint: {repo_rel(path)}; run `python main.py mcsc --stage train` first")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = ResidualRefiner(
        int(payload["drugDim"]),
        int(payload["targetDim"]),
        hidden=(256, 128),
        dropout=0.2,
        norm="batch",
    ).to(device)
    model.load_state_dict(payload["stateDict"])
    model.eval()
    return payload, model


def metrics(y_true: np.ndarray, pred: np.ndarray) -> dict:
    values = compute_metrics(y_true, pred)
    return {
        "R2": float(values["r2"]),
        "MSE": float(values["mse"]),
        "RMSE": float(values["rmse"]),
        "Pearson": float(values["pearson"]),
        "Spearman": float(values["spearman"]),
    }


def harmful_rate(prior: np.ndarray, pred: np.ndarray, y_true: np.ndarray) -> float:
    moved = np.abs(pred - prior) > 1e-9
    if not moved.any():
        return 0.0
    return float((np.abs(pred[moved] - y_true[moved]) > np.abs(prior[moved] - y_true[moved])).mean())


def worstgrp(target_feat: np.ndarray, target_idx: np.ndarray, y_true: np.ndarray, pred: np.ndarray) -> float:
    targets = np.unique(target_idx)
    k = min(4, len(targets))
    if k < 2:
        return float("nan")
    labels = KMeans(n_clusters=k, random_state=0, n_init=10).fit(target_feat[targets]).labels_
    target_to_group = dict(zip(targets.tolist(), labels.tolist()))
    groups = np.asarray([target_to_group[int(t)] for t in target_idx])
    values = [r2(y_true[groups == group], pred[groups == group]) for group in range(k) if (groups == group).sum() >= 5]
    return float(min(values)) if values else float("nan")


def infer_one(dataset: str, split: str, seed: int, device: torch.device, eval_batch_size: int, amp: bool) -> dict:
    payload, model = load_checkpoint(dataset, split, seed, device)
    bundle, sp = make_split(dataset, split, seed)
    train_d, train_t, train_y = sp["trainD"], sp["trainT"], sp["trainY"]
    fit_idx = sp["tr_idx"]
    fit_d, fit_t = train_d[fit_idx], train_t[fit_idx]
    test_d, test_t, test_y = sp["testD"], sp["testT"], sp["testY"]

    drug_feat_np = split_norm(bundle.drug_raw, np.unique(fit_d))
    target_feat_np = split_norm(bundle.target_raw, np.unique(fit_t))
    drug_feat = torch.from_numpy(drug_feat_np).float().to(device)
    target_feat = torch.from_numpy(target_feat_np).float().to(device)

    n_drugs = bundle.drug_raw.shape[0]
    full_counts, full_means, full_global = drug_stats(train_d, train_y, n_drugs)
    test_marginal = marginal_query(full_means, full_counts, full_global, test_d)
    memory = InteractionMemory(drug_feat, target_feat, train_d, train_t, train_y, normalize=False)
    test_fine = memory.predict(test_d, test_t)
    blend_weight = float(payload["blendWeight"])
    prior = blend_weight * test_fine + (1.0 - blend_weight) * test_marginal
    refiner = predict_refiner(model, drug_feat, target_feat, test_d, test_t, prior, device, eval_batch_size, amp)
    alpha = float(payload["frozenAlpha"])
    final = prior + alpha * (refiner - prior)

    target_group_feat = split_norm(bundle.target_raw, np.unique(train_t))
    methods = {
        "prior_only": prior,
        "full_refiner": refiner,
        "mcsc_frozen_alpha": final,
    }
    metric_block = {}
    for name, pred in methods.items():
        block = metrics(test_y, pred)
        block["harm_worse"] = harmful_rate(prior, pred, test_y)
        block["worstgrp_R2"] = worstgrp(target_group_feat, test_t.astype(int), test_y, pred)
        metric_block[name] = {
            key: round(float(value), 6) if isinstance(value, (float, np.floating)) else value
            for key, value in block.items()
        }

    return {
        "dataset": dataset,
        "split": split,
        "seed": int(seed),
        "checkpoint": payload["checkpoint"],
        "targetRepresentation": payload["targetRepresentation"],
        "prior": "global_blend",
        "device": str(device),
        "amp": bool(amp and device.type == "cuda"),
        "blendWeight": blend_weight,
        "frozenAlpha": alpha,
        "validationBasis": payload.get("validationBasis"),
        "metrics": metric_block,
    }


def boot(delta: np.ndarray) -> list[float]:
    values = np.asarray(delta, dtype=float)
    samples = np.random.RandomState(0).choice(values, (10000, len(values))).mean(axis=1)
    return [
        round(float(np.percentile(samples, 2.5)), 4),
        round(float(np.percentile(samples, 97.5)), 4),
    ]


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return {
        "mean": round(float(delta.mean()), 4),
        "ci95": boot(delta),
        "wins": f"{int((delta > 0).sum())}/{len(delta)}",
    }


def aggregate(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(f"{row['dataset']}/{row['split']}", []).append(row)
    cells = {}
    for cell, items in sorted(grouped.items()):
        items = sorted(items, key=lambda item: int(item["seed"]))
        summary = {}
        for method in ("prior_only", "full_refiner", "mcsc_frozen_alpha"):
            vals = {
                metric: np.asarray([item["metrics"][method][metric] for item in items], dtype=float)
                for metric in ("R2", "MSE", "RMSE", "Pearson", "Spearman", "harm_worse", "worstgrp_R2")
            }
            summary[method] = {
                "R2": round(float(vals["R2"].mean()), 4),
                "MSE": round(float(vals["MSE"].mean()), 4),
                "RMSE": round(float(vals["RMSE"].mean()), 4),
                "Pearson": round(float(vals["Pearson"].mean()), 4),
                "Spearman": round(float(vals["Spearman"].mean()), 4),
                "harm_worse": round(float(vals["harm_worse"].mean()), 4),
                "worstgrp_R2": round(float(np.nanmean(vals["worstgrp_R2"])), 4),
            }
        final = np.asarray([item["metrics"]["mcsc_frozen_alpha"]["R2"] for item in items], dtype=float)
        prior = np.asarray([item["metrics"]["prior_only"]["R2"] for item in items], dtype=float)
        refiner = np.asarray([item["metrics"]["full_refiner"]["R2"] for item in items], dtype=float)
        cells[cell] = {
            "dataset": cell.split("/", 1)[0],
            "split": cell.split("/", 1)[1],
            "nSeeds": len(items),
            "targetRepresentation": items[0]["targetRepresentation"],
            "alpha": items[0]["frozenAlpha"],
            "summary": summary,
            "deltaVsPrior": paired(final, prior),
            "deltaVsFullRefiner": paired(final, refiner),
            "perSeed": items,
        }
    return {
        "schema": "drugtarget-mcsc-mainline-results-v1",
        "status": "trainable_mainline",
        "model": "MCSC",
        "variant": "MCSC-FrozenAlpha",
        "formula": "prior + alpha * (refiner - prior)",
        "training": "python main.py mcsc --stage train",
        "inference": "python main.py mcsc --stage infer",
        "claimScope": "reproduced-frontier SOTA-level only under this repository's same split/seed/metric protocol",
        "cells": cells,
    }


def write_report(result: dict) -> None:
    lines = [
        "# MCSC Mainline Report",
        "",
        "Current trainable mainline: **MCSC-FrozenAlpha**.",
        "",
        "`final = prior + alpha * (refiner - prior)`",
        "",
        "The residual refiner is trained by `python main.py mcsc --stage train`; alpha is loaded",
        "from `config/residual-alpha-calibration.json` and frozen before final evaluation.",
        "",
        "## Results",
        "",
        "| split | target rep | alpha | prior R2 | full refiner R2 | MCSC R2 | delta vs prior | delta vs refiner | RMSE | Pearson | Spearman | worst-group |",
        "|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for cell, item in result["cells"].items():
        summary = item["summary"]
        final = summary["mcsc_frozen_alpha"]
        lines.append(
            f"| {cell} | {item['targetRepresentation']} | {float(item['alpha']):.2f} | "
            f"{summary['prior_only']['R2']:.4f} | {summary['full_refiner']['R2']:.4f} | **{final['R2']:.4f}** | "
            f"{item['deltaVsPrior']['mean']:+.4f} {item['deltaVsPrior']['ci95']}, {item['deltaVsPrior']['wins']} | "
            f"{item['deltaVsFullRefiner']['mean']:+.4f} {item['deltaVsFullRefiner']['ci95']}, {item['deltaVsFullRefiner']['wins']} | "
            f"{final['RMSE']:.4f} | {final['Pearson']:.4f} | {final['Spearman']:.4f} | {final['worstgrp_R2']:.4f} |"
        )
    lines.extend([
        "",
        "## Boundary",
        "",
        "- This is the only current MCSC mainline.",
        "- Older selectors, RCSC, dispersion, and full-refiner-only paths are retained only as failure/analysis history.",
        "- SOTA-level wording is limited to reproduced frontier comparisons under this repository's identical protocol.",
    ])
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(metas: list[dict]) -> None:
    merged = {}
    for path in CHECKPOINT_DIR.glob("*.json"):
        row = json_load(path, {})
        if row:
            merged[(row.get("dataset"), row.get("split"), int(row.get("seed", -1)))] = row
    for row in metas:
        merged[(row.get("dataset"), row.get("split"), int(row.get("seed", -1)))] = row
    payload = {
        "schema": "drugtarget-mcsc-manifest-v1",
        "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "checkpoints": sorted(
            merged.values(),
            key=lambda row: (str(row.get("dataset")), str(row.get("split")), int(row.get("seed", -1))),
        ),
    }
    json_dump(MANIFEST, payload)


def parse_cells(values: list[str] | None) -> list[tuple[str, str]]:
    if not values:
        return list(CELLS)
    allowed = {f"{dataset}/{split}": (dataset, split) for dataset, split in CELLS}
    cells = []
    for value in values:
        if value not in allowed:
            raise SystemExit(f"unknown split {value!r}; choices: {', '.join(sorted(allowed))}")
        cells.append(allowed[value])
    return cells


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate the current MCSC mainline")
    parser.add_argument("--stage", choices=["train", "infer", "full", "report", "cleanup"], default="full")
    parser.add_argument("--splits", nargs="*", help="Subset like DAVIS/target-cold KIBA/cluster-cold")
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    parser.add_argument("--force", action="store_true", help="retrain selected cells even if cached")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=65536)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument(
        "--gpu-monitor",
        nargs="?",
        const=str(OUT_DIR / "gpu-monitor.json"),
        default="",
        help="Optionally sample nvidia-smi utilization into a JSON file.",
    )
    parser.add_argument("--gpu-monitor-interval", type=float, default=0.5)
    return parser.parse_args()


def run_train(
    cells: list[tuple[str, str]],
    seeds: list[int],
    force: bool,
    device: torch.device,
    batch_size: int,
    eval_batch_size: int,
    amp: bool,
) -> list[dict]:
    metas = []
    for dataset, split in cells:
        for seed in seeds:
            print(f"[mcsc train] {dataset}/{split}/seed{seed} device={device} batch={batch_size} amp={amp and device.type == 'cuda'}")
            metas.append(train_one(dataset, split, int(seed), force, device, batch_size, eval_batch_size, amp))
    write_manifest(metas)
    print(f"wrote {MANIFEST.relative_to(REPO)}")
    return metas


def run_infer(cells: list[tuple[str, str]], seeds: list[int], device: torch.device, eval_batch_size: int, amp: bool) -> dict:
    rows = []
    for dataset, split in cells:
        for seed in seeds:
            print(f"[mcsc infer] {dataset}/{split}/seed{seed} device={device} amp={amp and device.type == 'cuda'}")
            rows.append(infer_one(dataset, split, int(seed), device, eval_batch_size, amp))
    result = aggregate(rows)
    RESULTS.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(result)
    print(f"wrote {RESULTS.relative_to(REPO)}")
    print(f"wrote {REPORT.relative_to(REPO)}")
    return result


def cleanup() -> None:
    print("No destructive cleanup is attached to the mainline command; use explicit repo cleanup steps.")


def main() -> None:
    args = parse_args()
    cells = parse_cells(args.splits)
    seeds = [int(seed) for seed in args.seeds]
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    monitor_path = Path(args.gpu_monitor) if args.gpu_monitor else None
    if monitor_path is not None and not monitor_path.is_absolute():
        monitor_path = REPO / monitor_path
    with GpuMonitor(monitor_path, args.gpu_monitor_interval):
        if args.stage == "train":
            run_train(cells, seeds, args.force, device, args.batch_size, args.eval_batch_size, args.amp)
        elif args.stage in {"infer", "report"}:
            run_infer(cells, seeds, device, args.eval_batch_size, args.amp)
        elif args.stage == "full":
            run_train(cells, seeds, args.force, device, args.batch_size, args.eval_batch_size, args.amp)
            run_infer(cells, seeds, device, args.eval_batch_size, args.amp)
        elif args.stage == "cleanup":
            cleanup()


if __name__ == "__main__":
    main()

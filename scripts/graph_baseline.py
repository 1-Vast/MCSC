"""GraphDTA-style baseline gate under this repo's exact cold-split protocol.

This is a compact GraphDTA reproduction: RDKit molecular graph -> GCN encoder,
protein sequence -> character CNN encoder, validation-only early stopping, and
the same DAVIS/KIBA split/seed protocol used by the MCSC and DeepDTA gates.

It is deliberately a baseline script, not a mainline model change.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from sklearn.cluster import KMeans
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, global_max_pool

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover
    def tqdm(x, **_kwargs):
        return x


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model.metrics import compute_metrics
from scripts.deep_baseline import (
    SEQ_I,
    SEQ_LEN,
    SEQ_VOCAB,
    build_kiba_context as _deep_kiba_context,
    davis_seed_split,
    encode_chars,
    kiba_seed_splits,
    load_esm150_target_features,
)
from scripts.seq_descriptors import ctriad
from scripts.runtime import RunSettings, load_data, load_features


DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEV.type == "cuda":
    torch.backends.cudnn.benchmark = True

OUT = REPO / "doc" / "graph-baseline-results.json"
REPORT = REPO / "doc" / "graph-baseline-report.md"
SPLITS = ["DAVIS/target-cold", "DAVIS/family-cold", "KIBA/target-cold", "KIBA/cluster-cold"]
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]

ATOM_NUMS = [1, 5, 6, 7, 8, 9, 11, 12, 15, 16, 17, 19, 20, 26, 30, 35, 53]
DEGREES = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]


def one_hot_unknown(value, choices: list) -> list[float]:
    return [float(value == item) for item in choices] + [float(value not in choices)]


def atom_features(atom: Chem.Atom) -> list[float]:
    return (
        one_hot_unknown(atom.GetAtomicNum(), ATOM_NUMS)
        + one_hot_unknown(atom.GetDegree(), DEGREES)
        + one_hot_unknown(atom.GetFormalCharge(), FORMAL_CHARGES)
        + one_hot_unknown(atom.GetHybridization(), HYBRIDIZATIONS)
        + [
            float(atom.GetIsAromatic()),
            float(atom.GetTotalNumHs()),
            float(atom.IsInRing()),
        ]
    )


def smiles_to_graph(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        x = torch.zeros((1, atom_dim()), dtype=torch.float32)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        return Data(x=x, edge_index=edge_index)

    x = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float32)
    edges: list[tuple[int, int]] = []
    for bond in mol.GetBonds():
        a = int(bond.GetBeginAtomIdx())
        b = int(bond.GetEndAtomIdx())
        edges.append((a, b))
        edges.append((b, a))
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.zeros((2, 0), dtype=torch.long)
    )
    return Data(x=x, edge_index=edge_index)


def atom_dim() -> int:
    return len(ATOM_NUMS) + 1 + len(DEGREES) + 1 + len(FORMAL_CHARGES) + 1 + len(HYBRIDIZATIONS) + 1 + 3


class GraphDTACompact(nn.Module):
    def __init__(self, n_seq: int, atom_feature_dim: int, emb: int = 128) -> None:
        super().__init__()
        self.gcn1 = GCNConv(atom_feature_dim, 128)
        self.gcn2 = GCNConv(128, 128)
        self.gcn3 = GCNConv(128, 128)
        self.drug_proj = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1))

        self.target_embedding = nn.Embedding(n_seq + 1, emb, padding_idx=0)
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
            nn.Linear(224, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def encode_drug_batch(self, graph_batch: Batch) -> torch.Tensor:
        x = torch.relu(self.gcn1(graph_batch.x, graph_batch.edge_index))
        x = torch.relu(self.gcn2(x, graph_batch.edge_index))
        x = torch.relu(self.gcn3(x, graph_batch.edge_index))
        return self.drug_proj(global_max_pool(x, graph_batch.batch))

    def encode_target(self, target: torch.Tensor) -> torch.Tensor:
        return self.target_conv(self.target_embedding(target).transpose(1, 2)).squeeze(-1)

    def score_features(self, drug_feat: torch.Tensor, target_feat: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([drug_feat, target_feat], dim=1)).squeeze(-1)

    def forward_ids(
        self,
        drug_graphs: list[Data],
        target_table: torch.Tensor,
        drug_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        unique_drug, drug_inverse = torch.unique(drug_ids, sorted=False, return_inverse=True)
        unique_target, target_inverse = torch.unique(target_ids, sorted=False, return_inverse=True)
        graph_batch = Batch.from_data_list([drug_graphs[int(i)] for i in unique_drug.detach().cpu().tolist()]).to(DEV)
        drug_features = self.encode_drug_batch(graph_batch)[drug_inverse]
        target_features = self.encode_target(target_table[unique_target])[target_inverse]
        return self.score_features(drug_features, target_features)


def parameter_count() -> int:
    return int(sum(p.numel() for p in GraphDTACompact(len(SEQ_VOCAB), atom_dim()).parameters()))


def set_threads() -> int:
    workers = max(1, os.cpu_count() or 1)
    torch.set_num_threads(max(1, min(workers, 8)))
    return workers


def predict(
    model: nn.Module,
    drug_graphs: list[Data],
    target_table: torch.Tensor,
    drug_idx: np.ndarray,
    target_idx: np.ndarray,
    eval_batch_size: int,
) -> np.ndarray:
    drug_ids = torch.from_numpy(drug_idx).long().to(DEV)
    target_ids = torch.from_numpy(target_idx).long().to(DEV)
    chunks = []
    model.eval()
    with torch.no_grad():
        iterator = range(0, len(drug_idx), eval_batch_size)
        for start in iterator:
            end = start + eval_batch_size
            pred = model.forward_ids(drug_graphs, target_table, drug_ids[start:end], target_ids[start:end])
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


def evaluate_predictions(y_true: np.ndarray, pred: np.ndarray, target_feat: np.ndarray, test_targets: np.ndarray) -> dict:
    metrics = compute_metrics(y_true, pred)
    metrics["worstgrp_R2"] = worst_target_group_r2(target_feat, test_targets, y_true, pred)
    return metrics


def train_eval(
    drug_graphs: list[Data],
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
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if DEV.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    started = time.time()
    model = GraphDTACompact(len(SEQ_VOCAB), atom_dim()).to(DEV)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_drug_t = torch.from_numpy(train_drug).long().to(DEV)
    train_target_t = torch.from_numpy(train_target).long().to(DEV)
    train_y_t = torch.from_numpy(train_y).float().to(DEV)

    best_mse = float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    epochs_run = 0

    progress = tqdm(range(1, epochs + 1), desc=f"GraphDTA seed{seed}", leave=False)
    for epoch in progress:
        epochs_run = epoch
        model.train()
        order = torch.randperm(len(train_drug), device=DEV)
        losses = []
        for start in range(0, len(train_drug), batch_size):
            idx = order[start:start + batch_size]
            if int(idx.numel()) < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            pred = model.forward_ids(drug_graphs, target_table, train_drug_t[idx], train_target_t[idx])
            loss = ((pred - train_y_t[idx]) ** 2).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        val_pred = predict(model, drug_graphs, target_table, val_drug, val_target, eval_batch_size)
        val_mse = float(np.mean((val_y - val_pred) ** 2))
        progress.set_postfix({"val_mse": f"{val_mse:.4f}", "loss": f"{np.mean(losses):.4f}" if losses else "nan"})
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
        raise RuntimeError("GraphDTA did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    test_pred = predict(model, drug_graphs, target_table, test_drug, test_target, eval_batch_size)
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


def summarize_cell(cell: dict) -> None:
    runs = cell.setdefault("runs", {})
    present = sorted(int(seed) for seed in runs)
    summary = {
        "status": "complete" if present == SEEDS else "partial",
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
    return json.loads(OUT.read_text(encoding="utf-8"))


def save_results(out: dict) -> None:
    out["_schema"] = "graphdta-compact-v1"
    out["_metadata"] = {
        "baseline": "GraphDTA compact GCN-CNN",
        "device": str(DEV),
        "seeds": SEEDS,
        "sequenceLength": SEQ_LEN,
        "atomFeatureDim": atom_dim(),
        "parameterCount": parameter_count(),
        "note": "Per-seed metrics are validation-selected; no test tuning. This is a compact reproduction, not paper-number comparison.",
    }
    for key in SPLITS:
        if key in out:
            summarize_cell(out[key])
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    tmp.replace(OUT)


def build_davis_context() -> dict:
    settings = RunSettings(device=str(DEV))
    settings.encoder.drug = "morgan"
    settings.encoder.target = "kb"
    data = load_data(settings)
    features = load_features(settings)
    drug_graphs = [smiles_to_graph(smi) for smi in tqdm(data["drugSmiles"], desc="DAVIS graphs")]
    target_table = torch.from_numpy(np.stack([encode_chars(seq, SEQ_I, SEQ_LEN) for seq in data["targetSeqs"]])).long().to(DEV)
    return {
        "data": data,
        "splitTargetFeat": features["targetFeat"],
        "groupTargetFeat": ctriad(data["targetSeqs"]),
        "drugGraphs": drug_graphs,
        "targetTable": target_table,
    }


def build_kiba_context() -> dict:
    data_dir = REPO / "dataset" / "kiba"
    with open(data_dir / "Y", "rb") as handle:
        y_matrix = np.asarray(pickle.load(handle, encoding="latin1"), dtype=float)
    smiles = list(json.loads((data_dir / "ligands_can.txt").read_text(encoding="utf-8")).values())
    seqs = list(json.loads((data_dir / "proteins.txt").read_text(encoding="utf-8")).values())
    drug_graphs = [smiles_to_graph(smi) for smi in tqdm(smiles, desc="KIBA graphs")]
    target_table = torch.from_numpy(np.stack([encode_chars(seq, SEQ_I, SEQ_LEN) for seq in seqs])).long().to(DEV)
    return {
        "yMatrix": y_matrix,
        "groupTargetFeat": load_esm150_target_features(len(seqs)),
        "drugGraphs": drug_graphs,
        "targetTable": target_table,
    }


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
            and int(existing.get("batchSize", -1)) == args.batch_size
        ):
            print(f"[skip] {key} seed{seed} already has matching metrics", flush=True)
            continue
        split_data = seed_splits[seed]
        print(f"[run] {key} seed{seed} on {DEV}", flush=True)
        result = train_eval(
            ctx["drugGraphs"],
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
        )
        result.update({
            "seed": seed,
            "split": key,
            "epochsMax": args.epochs,
            "patience": args.patience,
            "batchSize": args.batch_size,
            "evalBatchSize": args.eval_batch_size,
        })
        runs[str(seed)] = result
        save_results(out)
        print(
            f"[done] {key} seed{seed} R2={result['metrics']['r2']:.4f} "
            f"RMSE={result['metrics']['rmse']:.4f} runtime={result['runtimeSec']:.1f}s",
            flush=True,
        )


def frontier_refs() -> dict:
    refs: dict[str, dict] = {}
    mcsc_path = REPO / "doc" / "mcsc-mainline-results.json"
    if mcsc_path.exists():
        mcsc = json.loads(mcsc_path.read_text(encoding="utf-8"))
        for cell, payload in mcsc.get("cells", {}).items():
            refs.setdefault(cell, {})["frozen_alpha"] = payload["summary"]["mcsc_frozen_alpha"]["R2"]
    deep_path = REPO / "doc" / "deep-baseline-results.json"
    if deep_path.exists():
        deep = json.loads(deep_path.read_text(encoding="utf-8"))
        for cell in SPLITS:
            if cell in deep and "summary" in deep[cell]:
                refs.setdefault(cell, {})["deepdta"] = deep[cell]["summary"].get("r2")
            comp = deep.get("_comparison", {}).get("cells", {}).get(cell, {})
            if comp.get("xgb_gpu_R2_mean_ref") is not None:
                refs.setdefault(cell, {})["xgb"] = comp.get("xgb_gpu_R2_mean_ref")
    return refs


def write_report(out: dict) -> None:
    refs = frontier_refs()
    lines = [
        "# GraphDTA Compact Baseline Report",
        "",
        "Purpose: reproduce a graph-based deep DTA baseline under the same local split, seed,",
        "validation, and metric protocol. This does not change the MCSC mainline.",
        "",
        "## Method",
        "",
        "- Drug encoder: RDKit molecular graph with a 3-layer GCN and global max pooling.",
        "- Target encoder: protein character CNN with sequence length 1000.",
        "- Selection: validation MSE only; test labels are evaluation-only.",
        "- Boundary: compact GraphDTA-style reproduction, not paper-table comparison.",
        "",
        "## Results",
        "",
        "| split | status | seeds | GraphDTA R2 | CI95 | RMSE | Pearson | Spearman | worst-group | frozen alpha | DeepDTA | XGBoost |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in SPLITS:
        if cell not in out:
            continue
        summary = out[cell]["summary"]
        ref = refs.get(cell, {})
        lines.append(
            f"| {cell} | {summary['status']} | {summary['nSeeds']} | "
            f"{summary.get('r2')} | {summary.get('r2_ci95')} | {summary.get('rmse')} | "
            f"{summary.get('pearson')} | {summary.get('spearman')} | {summary.get('worstgrp_R2')} | "
            f"{ref.get('frozen_alpha')} | {ref.get('deepdta')} | {ref.get('xgb')} |"
        )
    lines.extend([
        "",
        "## Claim Boundary",
        "",
        "- A complete baseline gate requires all four splits with seeds 1-8.",
        "- Partial smoke results are implementation evidence only and cannot support superiority claims.",
        "- SOTA remains forbidden unless the complete reproduced deep frontier is beaten under this protocol.",
    ])
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="*", default=SPLITS, choices=SPLITS)
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    invalid_seeds = [seed for seed in args.seeds if seed not in SEEDS]
    if invalid_seeds:
        raise SystemExit(f"Unsupported seeds: {invalid_seeds}; expected subset of {SEEDS}")
    workers = set_threads()
    print(f"device={DEV} cpu_threads={torch.get_num_threads()} os_workers={workers} splits={args.splits} seeds={args.seeds}", flush=True)
    out = load_results()
    save_results(out)
    contexts: dict = {}
    for key in args.splits:
        run_one_split(key, args.seeds, out, args, contexts)
    save_results(out)
    write_report(out)
    print(f"wrote {OUT}", flush=True)
    print(f"wrote {REPORT}", flush=True)


if __name__ == "__main__":
    main()

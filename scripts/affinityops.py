"""Shared CUDA DTA utilities used by the static and GKN experiment lines."""
from __future__ import annotations

from contextlib import nullcontext
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from model.memory import InteractionMemory
from model.metrics import compute_metrics


REPO = Path(__file__).resolve().parents[1]
MEM_DIM = 5


def repo_rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def json_load(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("DTA experiment workflows are CUDA-only; run inside the drug CUDA environment")
    return device


def autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=enabled)
    return nullcontext()


def long_tensor(values: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(values):
        return values.to(device=device, dtype=torch.long)
    return torch.as_tensor(np.ascontiguousarray(values, dtype=np.int64), dtype=torch.long, device=device)


def float_tensor(values: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(values):
        return values.to(device=device, dtype=torch.float32)
    return torch.as_tensor(np.ascontiguousarray(values, dtype=np.float32), dtype=torch.float32, device=device)


def split_norm_tensor(values: np.ndarray, fit_rows: np.ndarray, device: torch.device) -> torch.Tensor:
    x = float_tensor(values, device)
    rows = long_tensor(np.unique(fit_rows), device)
    mean = x[rows].mean(dim=0, keepdim=True)
    centered = x - mean
    return centered / centered.norm(dim=1, keepdim=True).clamp_min(1e-9)


def split_norm_numpy(values: np.ndarray, fit_rows: np.ndarray) -> np.ndarray:
    mean = values[np.unique(fit_rows)].mean(axis=0, keepdims=True)
    centered = values - mean
    return (centered / (np.linalg.norm(centered, axis=1, keepdims=True) + 1e-9)).astype(np.float32)


def memory_standardizer_tensor(train_mem: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_mem.mean(dim=0, keepdim=True)
    std = train_mem.std(dim=0, keepdim=True)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean, std


def apply_standardizer_tensor(mem: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (mem - mean) / std


@torch.no_grad()
def memory_fine_prior_tensor(
    memory: InteractionMemory,
    drug_idx: torch.Tensor,
    target_idx: torch.Tensor,
    exclude_self: bool = False,
) -> torch.Tensor:
    return memory.predict_tensor(drug_idx, target_idx, exclude_self=exclude_self)


def drug_stats_tensor(drugs: torch.Tensor, labels: torch.Tensor, n_drugs: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    counts = torch.bincount(drugs, minlength=n_drugs).float()
    sums = torch.zeros(n_drugs, dtype=torch.float32, device=labels.device)
    sums.scatter_add_(0, drugs, labels)
    global_mean = labels.mean()
    means = torch.where(counts > 0, sums / counts.clamp_min(1.0), global_mean.expand_as(counts))
    return counts, means, global_mean


def marginal_query_tensor(means: torch.Tensor, counts: torch.Tensor, global_mean: torch.Tensor, query_drugs: torch.Tensor) -> torch.Tensor:
    return torch.where(counts[query_drugs] > 0, means[query_drugs], global_mean.expand_as(query_drugs.float()))


def marginal_loo_tensor(drugs: torch.Tensor, labels: torch.Tensor, n_drugs: int) -> tuple[torch.Tensor, torch.Tensor]:
    counts, means, global_mean = drug_stats_tensor(drugs, labels, n_drugs)
    sums = means * counts
    loo_counts = counts[drugs] - 1.0
    loo_sums = sums[drugs] - labels
    loo = torch.where(loo_counts > 0, loo_sums / loo_counts.clamp_min(1.0), global_mean.expand_as(labels))
    return loo, global_mean


def r2_tensor(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum().clamp_min(1e-8)
    return 1.0 - ss_res / ss_tot


def select_global_blend_weight_tensor(val_fine: torch.Tensor, val_marginal: torch.Tensor, val_y: torch.Tensor) -> torch.Tensor:
    best_w = val_fine.new_tensor(1.0)
    best = val_fine.new_tensor(-1e9)
    for value in np.linspace(0.0, 1.0, 21):
        w = val_fine.new_tensor(float(value))
        pred = w * val_fine + (1.0 - w) * val_marginal
        score = r2_tensor(val_y, pred)
        if bool((score > best).detach().cpu().item()):
            best = score
            best_w = w
    return best_w


def _rank(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    ranks = torch.empty_like(values)
    ranks[order] = torch.arange(values.numel(), dtype=values.dtype, device=values.device)
    return ranks


def spearman_tensor(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if pred.numel() < 3:
        return pred.new_zeros(())
    rp = _rank(pred.float())
    rl = _rank(labels.float())
    rp = rp - rp.mean()
    rl = rl - rl.mean()
    denom = (rp.norm() * rl.norm()).clamp_min(1e-9)
    return (rp * rl).sum() / denom


def harm_worse_tensor(prior: torch.Tensor, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    moved = (pred - prior).abs() > 1e-9
    if not bool(moved.any().detach().cpu().item()):
        return prior.new_zeros(())
    old_err = (prior[moved] - labels[moved]).abs()
    new_err = (pred[moved] - labels[moved]).abs()
    return (new_err > old_err).float().mean()


def select_alpha_tensor(
    val_prior: torch.Tensor,
    val_refiner: torch.Tensor,
    val_y: torch.Tensor,
    max_harm: float = 0.40,
    rank_tol: float = 0.005,
) -> dict:
    residual = val_refiner - val_prior
    abs_res = residual.abs()
    grid = torch.linspace(0.0, 1.0, 41, dtype=torch.float32, device=val_y.device)
    band_qs = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    if abs_res.numel() > 0:
        band_values = [0.0] + [float(torch.quantile(abs_res, q).detach().cpu().item()) for q in band_qs[1:]]
    else:
        band_values = [0.0]
    prior_spearman = float(spearman_tensor(val_prior, val_y).detach().cpu().item())
    candidates: list[dict] = []
    for band in band_values:
        active = abs_res >= band
        gated = torch.where(active, residual, torch.zeros_like(residual))
        preds = val_prior[None, :] + grid[:, None] * gated[None, :]
        mse = ((preds - val_y[None, :]) ** 2).mean(dim=1)
        idx = int(torch.argmin(mse).detach().cpu().item())
        alpha = float(grid[idx].detach().cpu().item())
        candidates.append({
            "alpha": alpha,
            "residualBand": float(band),
            "moveShare": 0.0 if alpha <= 1e-9 else float(active.float().mean().detach().cpu().item()),
            "validMSE": float(mse[idx].detach().cpu().item()),
            "validHarmWorse": float(harm_worse_tensor(val_prior, preds[idx], val_y).detach().cpu().item()),
            "validSpearman": float(spearman_tensor(preds[idx], val_y).detach().cpu().item()),
            "harmGuard": float(max_harm),
            "rankGuard": float(rank_tol),
            "priorSpearman": prior_spearman,
        })
    feasible = [
        c for c in candidates
        if c["validHarmWorse"] <= float(max_harm) + 1e-9
        and c["validSpearman"] >= prior_spearman - float(rank_tol) - 1e-9
    ]
    if not feasible:
        return {
            "alpha": 0.0, "residualBand": 0.0, "moveShare": 0.0,
            "validMSE": float(F.mse_loss(val_prior, val_y).detach().cpu().item()),
            "validHarmWorse": 0.0, "validSpearman": prior_spearman,
            "priorSpearman": prior_spearman, "harmGuard": float(max_harm),
            "rankGuard": float(rank_tol), "acceptedByGuards": False,
        }
    best_mse = min(c["validMSE"] for c in feasible)
    tol = max(1e-4, 0.0025 * best_mse)
    near = [c for c in feasible if c["validMSE"] <= best_mse + tol]
    near.sort(key=lambda c: (c["moveShare"], c["validMSE"]))
    best = dict(near[0])
    best["acceptedByGuards"] = True
    return best


def apply_routed_alpha(prior: torch.Tensor, refiner: torch.Tensor, alpha: float, band: float) -> torch.Tensor:
    residual = refiner - prior
    gated = torch.where(residual.abs() >= float(band), residual, torch.zeros_like(residual))
    return prior + float(alpha) * gated


def harmful_rate(prior: np.ndarray, pred: np.ndarray, y_true: np.ndarray) -> float:
    moved = np.abs(pred - prior) > 1e-9
    if not moved.any():
        return 0.0
    return float((np.abs(pred[moved] - y_true[moved]) > np.abs(prior[moved] - y_true[moved])).mean())


def r2_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = ((y_true - y_true.mean()) ** 2).sum()
    if denom <= 1e-12:
        return float("nan")
    return float(1.0 - ((y_true - y_pred) ** 2).sum() / denom)


def worstgrp_r2(target_feat: np.ndarray, target_idx: np.ndarray, y_true: np.ndarray, pred: np.ndarray) -> float:
    targets = np.unique(target_idx)
    k = min(4, len(targets))
    if k < 2:
        return float("nan")
    labels = KMeans(n_clusters=k, random_state=0, n_init=10).fit(target_feat[targets]).labels_
    target_to_group = dict(zip(targets.tolist(), labels.tolist()))
    groups = np.asarray([target_to_group[int(t)] for t in target_idx])
    values = [r2_np(y_true[groups == group], pred[groups == group]) for group in range(k) if (groups == group).sum() >= 5]
    return float(min(values)) if values else float("nan")


def core_metrics(y_true: np.ndarray, pred: np.ndarray, prior: np.ndarray, target_feat: np.ndarray, target_idx: np.ndarray) -> dict:
    values = compute_metrics(y_true, pred)
    return {
        "RMSE": float(values["rmse"]),
        "MSE": float(values["mse"]),
        "Spearman": float(values["spearman"]),
        "CI": float(values["ci"]),
        "worstgrp_R2": worstgrp_r2(target_feat, target_idx.astype(int), y_true, pred),
        "harm_worse": harmful_rate(prior, pred, y_true),
    }


def subset_rows(values: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if limit <= 0 or limit >= len(values):
        return values
    rng = np.random.RandomState(seed)
    return rng.choice(values, size=limit, replace=False)


def apply_limits(sp: dict, args, seed: int) -> dict:
    fit_idx = subset_rows(np.asarray(sp["tr_idx"], dtype=np.int64), args.limit_train, seed + 101)
    val_idx = subset_rows(np.asarray(sp["val_idx"], dtype=np.int64), args.limit_val, seed + 202)
    test_idx = subset_rows(np.arange(len(sp["testD"]), dtype=np.int64), args.limit_test, seed + 303)
    return {
        "fitD": sp["trainD"][fit_idx],
        "fitT": sp["trainT"][fit_idx],
        "fitY": sp["trainY"][fit_idx],
        "valD": sp["trainD"][val_idx],
        "valT": sp["trainT"][val_idx],
        "valY": sp["trainY"][val_idx],
        "testD": sp["testD"][test_idx],
        "testT": sp["testT"][test_idx],
        "testY": sp["testY"][test_idx],
        "trainD": np.concatenate([sp["trainD"][fit_idx], sp["trainD"][val_idx]]),
        "trainT": np.concatenate([sp["trainT"][fit_idx], sp["trainT"][val_idx]]),
        "trainY": np.concatenate([sp["trainY"][fit_idx], sp["trainY"][val_idx]]),
        "limits": {
            "fitRows": int(len(fit_idx)),
            "valRows": int(len(val_idx)),
            "testRows": int(len(test_idx)),
        },
    }


def prepare_priors(
    drug_feat: torch.Tensor,
    target_feat: torch.Tensor,
    rows: dict,
    device: torch.device,
) -> dict:
    n_drugs = drug_feat.shape[0]
    fit_d_t = long_tensor(rows["fitD"], device)
    fit_t_t = long_tensor(rows["fitT"], device)
    fit_y_t = float_tensor(rows["fitY"], device)
    val_d_t = long_tensor(rows["valD"], device)
    val_t_t = long_tensor(rows["valT"], device)
    val_y_t = float_tensor(rows["valY"], device)

    fit_marginal, _ = marginal_loo_tensor(fit_d_t, fit_y_t, n_drugs)
    fit_counts_t, fit_means_t, fit_global_t = drug_stats_tensor(fit_d_t, fit_y_t, n_drugs)
    val_marginal = marginal_query_tensor(fit_means_t, fit_counts_t, fit_global_t, val_d_t)

    mem_fit = InteractionMemory(drug_feat, target_feat, rows["fitD"], rows["fitT"], rows["fitY"], normalize=False)
    fit_fine = memory_fine_prior_tensor(mem_fit, fit_d_t, fit_t_t, exclude_self=True)
    val_fine = memory_fine_prior_tensor(mem_fit, val_d_t, val_t_t)
    blend_weight_t = select_global_blend_weight_tensor(val_fine, val_marginal, val_y_t)
    fit_prior = blend_weight_t * fit_fine + (1.0 - blend_weight_t) * fit_marginal
    val_prior = blend_weight_t * val_fine + (1.0 - blend_weight_t) * val_marginal

    fit_mem_raw = mem_fit.memory_features_tensor(fit_d_t, fit_t_t, exclude_self=True)
    val_mem_raw = mem_fit.memory_features_tensor(val_d_t, val_t_t)
    mem_mean, mem_std = memory_standardizer_tensor(fit_mem_raw)
    return {
        "fitD": fit_d_t,
        "fitT": fit_t_t,
        "fitY": fit_y_t,
        "valD": val_d_t,
        "valT": val_t_t,
        "valY": val_y_t,
        "fitPrior": fit_prior,
        "valPrior": val_prior,
        "fitMem": apply_standardizer_tensor(fit_mem_raw, mem_mean, mem_std),
        "valMem": apply_standardizer_tensor(val_mem_raw, mem_mean, mem_std),
        "memMean": mem_mean,
        "memStd": mem_std,
        "blendWeight": blend_weight_t,
    }


def prepare_test_prior(
    drug_feat: torch.Tensor,
    target_feat: torch.Tensor,
    rows: dict,
    blend_weight: float,
    mem_mean: torch.Tensor,
    mem_std: torch.Tensor,
    device: torch.device,
) -> dict:
    n_drugs = drug_feat.shape[0]
    train_d_t = long_tensor(rows["trainD"], device)
    train_y_t = float_tensor(rows["trainY"], device)
    test_d_t = long_tensor(rows["testD"], device)
    test_t_t = long_tensor(rows["testT"], device)
    full_counts_t, full_means_t, full_global_t = drug_stats_tensor(train_d_t, train_y_t, n_drugs)
    test_marginal = marginal_query_tensor(full_means_t, full_counts_t, full_global_t, test_d_t)
    memory = InteractionMemory(drug_feat, target_feat, rows["trainD"], rows["trainT"], rows["trainY"], normalize=False)
    test_fine = memory_fine_prior_tensor(memory, test_d_t, test_t_t)
    w = torch.tensor(float(blend_weight), dtype=torch.float32, device=device)
    prior = w * test_fine + (1.0 - w) * test_marginal
    test_mem_raw = memory.memory_features_tensor(test_d_t, test_t_t)
    return {
        "testD": test_d_t,
        "testT": test_t_t,
        "testPrior": prior,
        "testMem": apply_standardizer_tensor(test_mem_raw, mem_mean, mem_std),
    }


def contrastive_loss(drug_pool: torch.Tensor, target_pool: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    if labels.numel() < 4:
        return labels.new_zeros(())
    drug_z = F.normalize(drug_pool.float(), dim=1)
    target_z = F.normalize(target_pool.float(), dim=1)
    logits = drug_z @ target_z.t() / max(float(temperature), 1e-6)
    target = torch.arange(labels.numel(), device=labels.device)
    lf = labels.detach().float()
    high_cut = torch.quantile(lf, 0.60)
    low_cut = torch.quantile(lf, 0.40)
    high = labels >= high_cut
    low = labels <= low_cut
    h_w = high.float()
    h_n = h_w.sum().clamp_min(1.0)
    ce_rows = F.cross_entropy(logits, target, reduction="none")
    ce_cols = F.cross_entropy(logits.t(), target, reduction="none")
    pull = 0.5 * ((ce_rows * h_w).sum() + (ce_cols * h_w).sum()) / h_n
    l_w = low.float()
    l_n = l_w.sum().clamp_min(1.0)
    push = (F.softplus(logits.diag()) * l_w).sum() / l_n
    return pull + 0.10 * push


def pairwise_rank_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if labels.numel() < 2:
        return labels.new_zeros(())
    diff_y = labels[:, None] - labels[None, :]
    pos = diff_y > 1e-6
    diff_pred = pred[:, None] - pred[None, :]
    weight = diff_y.abs() * pos.float()
    loss = F.softplus(-diff_pred)
    return (weight * loss).sum() / weight.sum().clamp_min(1e-6)

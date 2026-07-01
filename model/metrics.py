"""Regression metrics for affinity evaluation."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def centered_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yc = y_true - y_true.mean()
    pc = y_pred - y_pred.mean()
    denom = np.linalg.norm(yc) * np.linalg.norm(pc)
    return float((yc * pc).sum() / max(denom, 1e-8))


def concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pairwise ranking agreement used by DAVIS/KIBA DTA affinity papers."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    n = y_true.shape[0]
    concordant = 0.0
    comparable = 0.0
    for i in range(n - 1):
        true_diff = y_true[i] - y_true[i + 1:]
        pred_diff = y_pred[i] - y_pred[i + 1:]
        mask = true_diff != 0.0
        if not np.any(mask):
            continue
        prod = true_diff[mask] * pred_diff[mask]
        concordant += float(np.sum(prod > 0.0))
        concordant += 0.5 * float(np.sum(pred_diff[mask] == 0.0))
        comparable += float(np.sum(mask))
    return float(concordant / comparable) if comparable > 0.0 else float("nan")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    pearson = centered_pearson(y_true, y_pred)
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
    try:
        spearman = float(spearmanr(y_true, y_pred)[0])
    except Exception:
        spearman = float("nan")
    ci = concordance_index(y_true, y_pred)
    return {
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "pearson": round(pearson, 4),
        "spearman": round(spearman, 4),
        "ci": round(ci, 4),
        "r2": round(float(r2), 4),
    }

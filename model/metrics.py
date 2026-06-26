"""Regression metrics for affinity evaluation."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def centered_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yc = y_true - y_true.mean()
    pc = y_pred - y_pred.mean()
    denom = np.linalg.norm(yc) * np.linalg.norm(pc)
    return float((yc * pc).sum() / max(denom, 1e-8))


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
    return {
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "pearson": round(pearson, 4),
        "spearman": round(spearman, 4),
        "r2": round(float(r2), 4),
    }

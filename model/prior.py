"""Validation-selected global blend prior utilities for MCSC."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


BLEND_WEIGHTS = tuple(round(x, 2) for x in np.linspace(0.0, 1.0, 21))
SIGNAL_BLEND = "global_blend_weight"


def global_blend(fine_pred, marginal_pred, weight):
    """Return weight * fine + (1 - weight) * per-drug marginal."""
    return float(weight) * np.asarray(fine_pred) + (1.0 - float(weight)) * np.asarray(marginal_pred)


def select_blend_weight_on_validation(val_fine, val_marginal, val_y, r2_fn, weights=BLEND_WEIGHTS):
    """Pick the single blend weight maximizing validation R2."""
    best_weight = 1.0
    best_r2 = float(r2_fn(val_y, val_fine))
    for weight in weights:
        score = float(r2_fn(val_y, global_blend(val_fine, val_marginal, weight)))
        if score > best_r2:
            best_r2 = score
            best_weight = float(weight)
    return float(best_weight), float(best_r2), float(r2_fn(val_y, val_fine))


def per_drug_marginal(train_drug, train_label, query_drug, global_mean):
    """Mean train affinity per drug, with a global fallback for unseen drugs."""
    sums: defaultdict[int, float] = defaultdict(float)
    counts: defaultdict[int, int] = defaultdict(int)
    for drug, label in zip(train_drug, train_label):
        sums[int(drug)] += float(label)
        counts[int(drug)] += 1
    return np.asarray([
        sums[int(drug)] / counts[int(drug)] if counts[int(drug)] else global_mean
        for drug in query_drug
    ], dtype=np.float32)

"""Small sequence descriptor helpers used by active baselines and MCSC."""
from __future__ import annotations

import numpy as np


AA = "ACDEFGHIKLMNPQRSTVWY"
AA_I = {aa: i for i, aa in enumerate(AA)}

CT_GROUPS = {
    "A": 0, "G": 0, "V": 0,
    "I": 1, "L": 1, "F": 1, "P": 1,
    "Y": 2, "M": 2, "T": 2, "S": 2,
    "H": 3, "N": 3, "Q": 3, "W": 3,
    "R": 4, "K": 4,
    "D": 5, "E": 5,
    "C": 6,
}


def aac_dip(seqs: list[str]) -> np.ndarray:
    aac = np.zeros((len(seqs), 20), np.float32)
    dip = np.zeros((len(seqs), 400), np.float32)
    for i, seq in enumerate(seqs):
        chars = [char for char in seq if char in AA_I]
        for char in chars:
            aac[i, AA_I[char]] += 1
        for left, right in zip(chars, chars[1:]):
            dip[i, AA_I[left] * 20 + AA_I[right]] += 1
        if chars:
            aac[i] /= len(chars)
        if len(chars) > 1:
            dip[i] /= len(chars) - 1
    return np.concatenate([aac, dip], axis=1).astype(np.float32)


def ctriad(seqs: list[str]) -> np.ndarray:
    out = np.zeros((len(seqs), 343), np.float32)
    for i, seq in enumerate(seqs):
        groups = [CT_GROUPS[char] for char in seq if char in CT_GROUPS]
        for a, b, c in zip(groups, groups[1:], groups[2:]):
            out[i, a * 49 + b * 7 + c] += 1
        total = out[i].sum()
        if total > 0:
            out[i] /= total
    return out


def split_norm(values: np.ndarray, train_rows: np.ndarray) -> np.ndarray:
    mean = values[train_rows].mean(axis=0, keepdims=True)
    centered = values - mean
    return (centered / (np.linalg.norm(centered, axis=1, keepdims=True) + 1e-9)).astype(np.float32)

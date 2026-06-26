"""Frozen drug and target encoders for affinity prediction.

No text model is downloaded or loaded. Target mechanism text is represented by a
deterministic hashing encoder, so KB and API text can be compared without an
external sentence-embedding dependency.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch


def normalize_descriptors(x: torch.Tensor) -> torch.Tensor:
    """Center columns over ALL rows and L2-normalize rows. Transductive: only safe when
    every row is train-visible (e.g. warm). Cold splits must use split_normalize."""
    x = x - x.mean(dim=0, keepdim=True)
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-6)


def split_normalize(x: torch.Tensor, fit_rows: torch.Tensor) -> torch.Tensor:
    """Center columns using ONLY the train-visible rows, then L2-normalize every row.
    The row L2 norm is per-row (no cross-row fit), so the only fitted statistic is the
    column mean, which is computed from fit_rows alone -> no held-out leakage."""
    mean = x[fit_rows].mean(dim=0, keepdim=True)
    x = x - mean
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-6)


class MorganEncoder:
    """Offline Morgan fingerprint encoder for compounds."""

    def __init__(self, radius: int = 2, bits: int = 1024, device: str = "cpu"):
        self.radius = radius
        self.bits = bits
        self.device = device

    def build(self, smiles: list[str]) -> torch.Tensor:
        try:
            from rdkit import Chem
            from rdkit.Chem import rdFingerprintGenerator
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "RDKit is required for Morgan fingerprints. Install it with "
                "`python -m pip install rdkit` or run smoke tests with `--drug hash`."
            ) from exc

        rows: list[np.ndarray] = []
        valid = 0
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius,
            fpSize=self.bits,
        )
        for value in smiles:
            mol = Chem.MolFromSmiles(value)
            if mol is None:
                rows.append(np.zeros(self.bits, dtype=np.float32))
                continue
            fp = generator.GetFingerprint(mol)
            arr = np.zeros(self.bits, dtype=np.float32)
            Chem.DataStructs.ConvertToNumpyArray(fp, arr)
            rows.append(arr)
            valid += 1
        print(f"[drug] Morgan fingerprints: {valid}/{len(smiles)} valid, dim={self.bits}")
        return torch.from_numpy(np.stack(rows)).to(self.device)


class HashSmilesEncoder:
    """Deterministic hashed SMILES n-gram encoder for dependency-light smoke tests."""

    def __init__(self, bits: int = 1024, device: str = "cpu"):
        self.bits = bits
        self.device = device

    def build(self, smiles: list[str]) -> torch.Tensor:
        rows = []
        for value in smiles:
            row = np.zeros(self.bits, dtype=np.float32)
            padded = f"^{value}$"
            for width in (2, 3, 4):
                for start in range(max(len(padded) - width + 1, 0)):
                    idx = stable_hash(padded[start : start + width], self.bits)
                    row[idx] += 1.0
            rows.append(row)
        print(f"[drug] hashed SMILES descriptors: {len(smiles)} rows, dim={self.bits}")
        return torch.from_numpy(np.stack(rows)).to(self.device)


class MechanismTextEncoder:
    """Deterministic hashed encoder for target mechanism text."""

    def __init__(self, dim: int = 256, device: str = "cpu"):
        self.dim = dim
        self.device = device

    def build(self, texts: list[str], center_mask: list[bool] | None = None) -> torch.Tensor:
        rows = []
        for text in texts:
            row = np.zeros(self.dim, dtype=np.float32)
            tokens = tokenize(text)
            for token in tokens:
                row[stable_hash(f"tok:{token}", self.dim)] += 1.0
            for left, right in zip(tokens, tokens[1:]):
                row[stable_hash(f"bi:{left}_{right}", self.dim)] += 0.5
            rows.append(row)

        x = torch.from_numpy(np.stack(rows)).float()
        x = torch.log1p(x)
        # Raw descriptors only. Centering/L2 is applied SPLIT-AWARE downstream (fit on
        # train-visible targets), so held-out cold targets never influence the statistics.
        # center_mask is accepted for backward compatibility but no longer used here.
        _ = center_mask
        print(f"[target] hashed mechanism descriptors (raw): {tuple(x.shape)}")
        return x.to(self.device)


def tokenize(text: str) -> list[str]:
    cleaned = []
    for ch in text.lower():
        cleaned.append(ch if ch.isalnum() else " ")
    return [token for token in "".join(cleaned).split() if token]


def stable_hash(value: str, modulo: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little") % modulo

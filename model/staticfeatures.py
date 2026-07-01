"""Static sequence feature builders for isolated DTA upgrade experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model.encode import MorganEncoder
from scripts.seqdescriptors import signed_kmer


SMILES_CHARS = "#%()+-./0123456789=@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]\\abcdefghijklmnopqrstuvwxyz"
SMILES_INDEX = {ch: i + 2 for i, ch in enumerate(SMILES_CHARS)}
PAD_INDEX = 0
UNK_INDEX = 1

ESM_MODELS = {
    "esm2_t6_8M_UR50D": "facebook/esm2_t6_8M_UR50D",
    "esm2_t30_150M_UR50D": "facebook/esm2_t30_150M_UR50D",
}


def safe_name(value: str) -> str:
    return value.split("/")[-1].replace("-", "_")


def sequence_digest(model_id: str, seqs: list[str], max_len: int) -> str:
    joined = model_id + "|" + "|".join(seq[:max_len] for seq in seqs)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]


def _encode_smiles(smiles: list[str], max_len: int) -> np.ndarray:
    arr = np.full((len(smiles), max_len), PAD_INDEX, dtype=np.int64)
    for i, value in enumerate(smiles):
        for j, ch in enumerate(value[:max_len]):
            arr[i, j] = SMILES_INDEX.get(ch, UNK_INDEX)
    return arr


class FrozenSmilesCNNEncoder(nn.Module):
    """Deterministic frozen 1D-CNN over SMILES characters."""

    def __init__(
        self,
        max_len: int = 192,
        embed_dim: int = 32,
        channels: int = 64,
        kernels: tuple[int, ...] = (3, 5, 7),
        seed: int = 19,
    ) -> None:
        super().__init__()
        self.max_len = int(max_len)
        self.output_dim = int(channels) * len(kernels) * 2
        generator_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        self.embed = nn.Embedding(len(SMILES_INDEX) + 2, embed_dim, padding_idx=PAD_INDEX)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, channels, kernel_size=k, padding=k // 2)
            for k in kernels
        ])
        torch.random.set_rng_state(generator_state)
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens).transpose(1, 2)
        parts = []
        for conv in self.convs:
            h = torch.relu(conv(x))
            parts.append(h.mean(dim=-1))
            parts.append(h.amax(dim=-1))
        return torch.cat(parts, dim=-1)


@torch.no_grad()
def build_smiles_cnn_features(
    smiles: list[str],
    device: str | torch.device,
    batch_size: int = 1024,
    max_len: int = 192,
    seed: int = 19,
) -> tuple[np.ndarray, dict]:
    """Build deterministic frozen 1D-CNN SMILES descriptors without label access."""
    dev = torch.device(device)
    encoder = FrozenSmilesCNNEncoder(max_len=max_len, seed=seed).to(dev).eval()
    tokens = _encode_smiles(smiles, max_len)
    rows = []
    for start in range(0, len(smiles), batch_size):
        batch = torch.as_tensor(tokens[start:start + batch_size], dtype=torch.long, device=dev)
        rows.append(encoder(batch).detach().cpu().numpy())
    feat = np.concatenate(rows, axis=0).astype(np.float32)
    meta = {
        "smilesFeature": "frozen_1d_cnn",
        "smilesFeatureTrainable": False,
        "smilesMaxLen": int(max_len),
        "smilesDim": int(feat.shape[1]),
        "smilesSeed": int(seed),
    }
    return feat, meta


def build_morgan_features(
    smiles: list[str],
    device: str | torch.device,
    bits: int = 1024,
    radius: int = 2,
) -> np.ndarray:
    encoder = MorganEncoder(radius=radius, bits=bits, device=str(device))
    return encoder.build(smiles).detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def build_chemberta_features(
    smiles: list[str],
    model_id: str,
    cache_dir: Path,
    device: str | torch.device,
    batch_size: int = 64,
    max_len: int = 256,
    pooling: str = "mean",
) -> tuple[np.ndarray, dict]:
    """Build or load frozen ChemBERTa SMILES embeddings (external pretrained drug encoder).

    One embedding per molecule; cached by a digest of (model_id, smiles, max_len, pooling) so
    repeated runs reuse the cache. SMILES strings are static inputs; no labels/test stats used.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(
        (model_id + f"|{max_len}|{pooling}|" + "|".join(s[:max_len] for s in smiles)).encode("utf-8")
    ).hexdigest()[:12]
    cache = cache_dir / f"chemberta_{safe_name(model_id)}_n{len(smiles)}_{pooling}_{digest}.npy"
    meta_path = cache.with_suffix(".json")
    base_meta = {
        "drugRepresentationModel": model_id,
        "drugEncoder": "chemberta",
        "pooling": pooling,
        "maxLen": int(max_len),
        "moleculeCount": len(smiles),
        "digest": digest,
        "cache": cache.name,
        "externalPretrainedDrug": True,
    }
    if cache.exists():
        feat = np.load(cache).astype(np.float32)
        return feat, {**base_meta, "drugDim": int(feat.shape[1]), "cacheHit": True}

    from transformers import AutoModel, AutoTokenizer

    dev = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(dev).eval()
    rows = []
    for start in range(0, len(smiles), batch_size):
        batch = [s[:max_len] for s in smiles[start:start + batch_size]]
        encoded = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len
        ).to(dev)
        hidden = model(**encoded).last_hidden_state
        if pooling == "cls":
            pooled = hidden[:, 0, :]
        else:
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        rows.append(pooled.detach().cpu().numpy())
    feat = np.concatenate(rows, axis=0).astype(np.float32)
    np.save(cache, feat)
    meta = {**base_meta, "drugDim": int(feat.shape[1]), "cacheHit": False,
            "tokenizer": "smiles_bpe", "source": "frozen ChemBERTa last_hidden_state"}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return feat, meta


@torch.no_grad()
def build_protein_plm_features(
    target_ids: list[str],
    seqs: list[str],
    source: str,
    cache_dir: Path,
    device: str | torch.device,
    batch_size: int = 8,
    max_len: int = 1022,
    hash_dim: int = 640,
) -> tuple[np.ndarray, dict]:
    """Build or load frozen protein sequence embeddings.

    `source=hash` is a dependency-light smoke fallback and is marked as non-ESM in
    metadata. ESM sources are sequence-only frozen pLM embeddings.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    if source == "hash":
        feat = signed_kmer(seqs, bits=hash_dim, min_k=3, max_k=5).astype(np.float32)
        return feat, {
            "targetRepresentation": "hash_signed_kmer_smoke",
            "targetDim": int(feat.shape[1]),
            "externalPretrainedTarget": False,
            "plmSource": "hash",
            "cache": None,
        }

    model_id = ESM_MODELS.get(source, source)
    key = sequence_digest(model_id, seqs, max_len)
    cache = cache_dir / f"esm_{safe_name(model_id)}_n{len(seqs)}_{key}.npy"
    meta_path = cache.with_suffix(".json")
    if cache.exists():
        feat = np.load(cache).astype(np.float32)
        return feat, {
            "targetRepresentation": safe_name(model_id),
            "targetDim": int(feat.shape[1]),
            "externalPretrainedTarget": True,
            "plmSource": model_id,
            "cache": cache.name,
            "cacheHit": True,
            "maxLen": int(max_len),
        }

    from transformers import AutoModel, AutoTokenizer

    dev = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(dev).eval()
    rows = []
    for start in range(0, len(seqs), batch_size):
        batch = [seq[:max_len] for seq in seqs[start:start + batch_size]]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(dev)
        hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        rows.append(pooled.detach().cpu().numpy())
    feat = np.concatenate(rows, axis=0).astype(np.float32)
    np.save(cache, feat)
    meta = {
        "model": model_id,
        "targetCount": len(target_ids),
        "targetIdsHash": hashlib.sha1("|".join(target_ids).encode("utf-8")).hexdigest()[:16],
        "dim": int(feat.shape[1]),
        "pooling": "attention_mask_mean",
        "maxLen": int(max_len),
        "source": "sequence-only frozen ESM-2",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return feat, {
        "targetRepresentation": safe_name(model_id),
        "targetDim": int(feat.shape[1]),
        "externalPretrainedTarget": True,
        "plmSource": model_id,
        "cache": cache.name,
        "cacheHit": False,
        "maxLen": int(max_len),
    }

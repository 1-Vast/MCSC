"""Train-only kNN interaction memory used as the PRISM prior."""
from __future__ import annotations

import numpy as np
import torch

from model.encode import normalize_descriptors


def _long_index(values, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(values):
        return values.to(device=device, dtype=torch.long)
    return torch.as_tensor(np.asarray(values, dtype=np.int64), dtype=torch.long, device=device)


class InteractionMemory:
    """kNN affinity retrieval in frozen drug and target descriptor spaces."""

    def __init__(
        self,
        drug_feat: torch.Tensor,
        target_feat: torch.Tensor,
        train_drug: np.ndarray,
        train_target: np.ndarray,
        train_label: np.ndarray,
        k: int = 5,
        temperature: float = 0.1,
        mode: str = "both",
        normalize: bool = True,
    ):
        # normalize=False when callers pass features already normalized SPLIT-AWARE; the
        # internal normalize_descriptors centers over ALL rows (transductive) and would
        # reintroduce held-out leakage in cold splits.
        self.drug_feat = normalize_descriptors(drug_feat) if normalize else drug_feat
        self.target_feat = normalize_descriptors(target_feat) if normalize else target_feat
        self.k = k
        self.temperature = temperature
        self.mode = mode
        self.drug_count = drug_feat.shape[0]
        self.target_count = target_feat.shape[0]
        self.device = drug_feat.device

        self.y_train = torch.full(
            (self.drug_count, self.target_count),
            float("nan"),
            device=self.device,
        )
        drug_t = torch.as_tensor(np.asarray(train_drug, dtype=np.int64), dtype=torch.long, device=self.device)
        target_t = torch.as_tensor(np.asarray(train_target, dtype=np.int64), dtype=torch.long, device=self.device)
        label_t = torch.as_tensor(np.asarray(train_label, dtype=np.float32), dtype=torch.float32, device=self.device)
        self.y_train[drug_t, target_t] = label_t
        self.global_mean = self.y_train[~self.y_train.isnan()].mean()

    @torch.no_grad()
    def _knn_weights(
        self,
        query_feat: torch.Tensor,
        key_feat: torch.Tensor,
        known_mask: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        sim = query_feat @ key_feat.t()
        sim = sim.masked_fill(~known_mask, -float("inf"))
        known_count = known_mask.sum(dim=1)
        k = min(self.k, key_feat.shape[0])
        if k == 0:
            return None, None
        empty = known_count == 0
        sim = torch.where(empty.unsqueeze(1), torch.zeros_like(sim), sim)
        top_vals, top_idx = sim.topk(k, dim=1)
        weights = torch.softmax(top_vals / max(self.temperature, 1e-6), dim=1)
        weights = torch.where(
            empty.unsqueeze(1),
            torch.full_like(weights, float("nan")),
            weights,
        )
        return weights, top_idx

    @torch.no_grad()
    def _predict_drug_side(
        self, drug_idx: torch.Tensor, target_idx: torch.Tensor, exclude_self: bool = False
    ) -> torch.Tensor:
        known = ~self.y_train[:, target_idx].t().isnan()
        if exclude_self:
            # Leave-one-out: hide the query pair's own cell so a train pair cannot retrieve
            # its own label (otherwise the residual target y - prior collapses to ~0).
            known[torch.arange(len(drug_idx), device=self.device), drug_idx] = False
        weights, top_idx = self._knn_weights(self.drug_feat[drug_idx], self.drug_feat, known)
        if weights is None or top_idx is None:
            return torch.full((len(drug_idx),), float("nan"), device=self.device)
        gathered = self.y_train[top_idx, target_idx.unsqueeze(1).expand(-1, top_idx.shape[1])]
        gathered = torch.nan_to_num(gathered, nan=0.0)
        return (weights * gathered).sum(dim=1)

    @torch.no_grad()
    def _predict_target_side(
        self, drug_idx: torch.Tensor, target_idx: torch.Tensor, exclude_self: bool = False
    ) -> torch.Tensor:
        known = ~self.y_train[drug_idx, :].isnan()
        if exclude_self:
            known[torch.arange(len(target_idx), device=self.device), target_idx] = False
        weights, top_idx = self._knn_weights(
            self.target_feat[target_idx],
            self.target_feat,
            known,
        )
        if weights is None or top_idx is None:
            return torch.full((len(drug_idx),), float("nan"), device=self.device)
        gathered = self.y_train[drug_idx.unsqueeze(1).expand(-1, top_idx.shape[1]), top_idx]
        gathered = torch.nan_to_num(gathered, nan=0.0)
        return (weights * gathered).sum(dim=1)

    @torch.no_grad()
    def predict(
        self,
        drug_idx: np.ndarray,
        target_idx: np.ndarray,
        exclude_self: bool = False,
        chunk_size: int | None = 32768,
    ) -> np.ndarray:
        """Predict pKd for the given pairs. With exclude_self=True the query pair's own
        cell is masked from retrieval (leave-one-out), used to build a clean residual
        target on TRAIN pairs for the Memory-Calibrated Residual Refiner."""
        drug_np = np.ascontiguousarray(np.asarray(drug_idx, dtype=np.int64))
        target_np = np.ascontiguousarray(np.asarray(target_idx, dtype=np.int64))
        if chunk_size and len(drug_np) > chunk_size:
            return np.concatenate([
                self.predict(drug_np[i:i + chunk_size], target_np[i:i + chunk_size],
                             exclude_self=exclude_self, chunk_size=None)
                for i in range(0, len(drug_np), chunk_size)
            ])
        drug = torch.as_tensor(drug_np, dtype=torch.long, device=self.device)
        target = torch.as_tensor(target_np, dtype=torch.long, device=self.device)

        if self.mode == "drug":
            pred = self._predict_drug_side(drug, target, exclude_self)
        elif self.mode == "target":
            pred = self._predict_target_side(drug, target, exclude_self)
        else:
            drug_pred = self._fill_nan(self._predict_drug_side(drug, target, exclude_self))
            target_pred = self._fill_nan(self._predict_target_side(drug, target, exclude_self))
            pred = (drug_pred + target_pred) / 2.0

        pred = self._fill_nan(pred)
        return pred.cpu().numpy()

    def _fill_nan(self, values: torch.Tensor) -> torch.Tensor:
        return torch.where(torch.isnan(values), self.global_mean.expand_as(values), values)

    @torch.no_grad()
    def predict_tensor(
        self,
        drug_idx,
        target_idx,
        exclude_self: bool = False,
        chunk_size: int | None = 32768,
    ) -> torch.Tensor:
        """GPU-resident prediction path used by the PRISM mainline."""
        drug = _long_index(drug_idx, self.device)
        target = _long_index(target_idx, self.device)
        if chunk_size and len(drug) > chunk_size:
            return torch.cat([
                self.predict_tensor(drug[i:i + chunk_size], target[i:i + chunk_size],
                                    exclude_self=exclude_self, chunk_size=None)
                for i in range(0, len(drug), chunk_size)
            ], dim=0)

        if self.mode == "drug":
            pred = self._predict_drug_side(drug, target, exclude_self)
        elif self.mode == "target":
            pred = self._predict_target_side(drug, target, exclude_self)
        else:
            drug_pred = self._fill_nan(self._predict_drug_side(drug, target, exclude_self))
            target_pred = self._fill_nan(self._predict_target_side(drug, target, exclude_self))
            pred = (drug_pred + target_pred) / 2.0
        return self._fill_nan(pred)

    @torch.no_grad()
    def memory_features(
        self,
        drug_idx: np.ndarray,
        target_idx: np.ndarray,
        exclude_self: bool = False,
        chunk_size: int | None = 32768,
    ) -> torch.Tensor:
        """Per-pair memory coverage/uncertainty diagnostics, shape [N, 5]:
        [drug_side_prior, target_side_prior, |disagreement|, drug_coverage, target_coverage].
        Coverage is 1.0 when that side retrieved at least one train neighbor, else 0.0 (and
        the prior falls back to the global train mean). exclude_self mirrors predict()."""
        return self.memory_features_tensor(drug_idx, target_idx, exclude_self, chunk_size)

    @torch.no_grad()
    def memory_features_tensor(
        self,
        drug_idx,
        target_idx,
        exclude_self: bool = False,
        chunk_size: int | None = 32768,
    ) -> torch.Tensor:
        """GPU-resident memory diagnostics used by the PRISM mainline."""
        drug = _long_index(drug_idx, self.device)
        target = _long_index(target_idx, self.device)
        if chunk_size and len(drug) > chunk_size:
            return torch.cat([
                self.memory_features_tensor(drug[i:i + chunk_size], target[i:i + chunk_size],
                                            exclude_self=exclude_self, chunk_size=None)
                for i in range(0, len(drug), chunk_size)
            ], dim=0)
        d = self._predict_drug_side(drug, target, exclude_self)
        t = self._predict_target_side(drug, target, exclude_self)
        d_cov = (~torch.isnan(d)).float()
        t_cov = (~torch.isnan(t)).float()
        d_fill = self._fill_nan(d)
        t_fill = self._fill_nan(t)
        disagree = (d_fill - t_fill).abs()
        return torch.stack([d_fill, t_fill, disagree, d_cov, t_cov], dim=1)

    def predict_shuffled_targets(
        self,
        drug_idx: np.ndarray,
        target_idx: np.ndarray,
        seed: int = 999,
    ) -> np.ndarray:
        """Negative control: shuffle target descriptors before retrieval."""
        gen = torch.Generator(device=self.device).manual_seed(seed)
        perm = torch.randperm(self.target_feat.shape[0], generator=gen, device=self.device)
        original = self.target_feat.clone()
        self.target_feat = self.target_feat[perm]
        result = self.predict(drug_idx, target_idx)
        self.target_feat = original
        return result

"""PRISM selective affinity protocol with GKN prototypes and DeepSeek-QC defer."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model.encode import MechanismTextEncoder, tokenize
from model.enhanced import PrismSelectiveRefiner
from model.graph import GraphKnowledgeNetwork, dense_normalized_adjacency
from scripts.affinitydata import load_affinity_bundle, make_split, parse_cells
from scripts.affinityops import (
    MEM_DIM,
    apply_limits,
    autocast_context,
    contrastive_loss,
    core_metrics,
    float_tensor,
    harm_worse_tensor,
    harmful_rate,
    pairwise_rank_loss,
    json_dump,
    long_tensor,
    prepare_priors,
    prepare_test_prior,
    repo_rel,
    resolve_device,
    seed_everything,
    spearman_tensor,
    split_norm_numpy,
    split_norm_tensor,
)
from scripts.promptprofiles import build_family_profiles, PROMPT_VERSION_DEFAULT
from scripts.runtime import build_knowledge_texts


OUT_DIR = REPO / "outputs" / "prism"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
LEGACY_CHECKPOINT_DIR = REPO / "outputs" / "dta_gkn" / "checkpoints"
GRAPH_DIR = OUT_DIR / "graphs"
RESULTS = REPO / "doc" / "prism-results.json"
REPORT = REPO / "doc" / "prism-report.md"

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "acts", "also",
    "protein", "target", "human", "primary", "function", "family", "pathway",
    "pathways", "member", "contains", "domain", "domains", "regulates", "cell",
}


def require_cuda_device(name: str) -> torch.device:
    device = resolve_device(name)
    if device.type != "cuda":
        raise SystemExit("PRISM enhanced workflow is GPU-only; pass --device cuda and do not use CPU fallback")
    if not torch.cuda.is_available():
        raise SystemExit("PRISM enhanced workflow requires CUDA, but torch.cuda.is_available() is false")
    try:
        _ = torch.empty(1, device=device)
    except Exception as exc:
        raise SystemExit(f"PRISM enhanced workflow could not allocate on {device}: {exc}") from exc
    return device


def load_llm_cache(dataset: str) -> dict[str, str]:
    candidates = [
        REPO / "dataset" / "cache" / f"llm-mechanism-{dataset.lower()}-sanitized.json",
        REPO / "dataset" / "cache" / f"llm-mechanism-{dataset.lower()}.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("entries"), dict):
            return {
                str(key): str(value.get("summary", value.get("text", "")))
                for key, value in data["entries"].items()
                if isinstance(value, dict)
            }
        if isinstance(data, dict) and isinstance(data.get("descriptions"), dict):
            return {str(key): str(value) for key, value in data["descriptions"].items()}
        if isinstance(data, dict):
            return {
                str(key): str(value)
                for key, value in data.items()
                if isinstance(value, str) and key not in {"schema", "dataset", "method"}
            }
    return {}


def mechanism_texts(dataset: str, target_ids: list[str], source: str) -> tuple[dict[str, str], dict]:
    if source == "llm-cache":
        texts = load_llm_cache(dataset)
        if texts:
            return texts, {
                "mechanismTextSource": "llm-cache",
                "availableTexts": len(texts),
                "apiCalledInRun": False,
            }
    kb_texts, covered = build_knowledge_texts(target_ids)
    return dict(zip(target_ids, kb_texts)), {
        "mechanismTextSource": "public-kb-fallback" if source == "llm-cache" else "public-kb",
        "availableTexts": int(sum(covered)),
        "apiCalledInRun": False,
    }


def select_entities(text_by_id: dict[str, str], train_target_ids: list[str], max_entities: int, min_df: int) -> list[str]:
    df: Counter[str] = Counter()
    for tid in train_target_ids:
        seen = {
            token for token in tokenize(text_by_id.get(tid, ""))
            if len(token) >= 4 and token not in STOPWORDS and not token.isdigit()
        }
        df.update(seen)
    kept = [token for token, count in df.items() if count >= min_df]
    kept.sort(key=lambda token: (-df[token], token))
    return kept[:max_entities]


def text_feature_matrix(
    dataset: str,
    target_ids: list[str],
    fit_targets: np.ndarray,
    source: str,
    dim: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, str], dict]:
    text_by_id, meta = mechanism_texts(dataset, target_ids, source)
    fit_id_set = {target_ids[int(i)] for i in np.unique(fit_targets)}
    texts = [text_by_id.get(tid, "") if tid in fit_id_set else "" for tid in target_ids]
    feat = MechanismTextEncoder(dim, str(device)).build(texts).detach().cpu().numpy().astype(np.float32)
    meta.update({
        "textDim": int(dim),
        "textPolicy": "inner-train targets keep mechanism text; non-inner-train targets are zeroed before and after normalization",
        "fitTextTargets": int(len(fit_id_set)),
    })
    return feat, text_by_id, meta


def train_only_text_tensor(text_raw: np.ndarray, fit_targets: np.ndarray, device: torch.device) -> torch.Tensor:
    text_feat = split_norm_tensor(text_raw, fit_targets, device)
    keep = torch.zeros(text_feat.shape[0], dtype=torch.bool, device=device)
    keep[long_tensor(np.unique(fit_targets), device)] = True
    return text_feat * keep.unsqueeze(1)


def domain_distance_score(domain_dist: torch.Tensor) -> torch.Tensor:
    """Scalar OOD score: lower means closer to at least one train-derived prototype."""
    return domain_dist.float().min(dim=1).values


def select_domain_alpha_tensor(
    val_prior: torch.Tensor,
    val_refiner: torch.Tensor,
    val_y: torch.Tensor,
    val_domain_dist: torch.Tensor,
    max_harm: float,
    rank_tol: float = 0.0,
) -> dict:
    """Select alpha, residual deadband, and prototype-distance cutoff on validation only.

    `domainThreshold=None` reproduces the previous residual-deadband selector. Numeric
    thresholds monotonically defer far-from-prototype targets to the prior.

    Feasibility is gated on two validation-only guards: ``validHarmWorse <= max_harm``
    (a fixed harm budget) and ``validSpearman >= prior_spearman - rank_tol`` (ranking must
    be held within tolerance of the prior). The prior (alpha=0) is always feasible.
    """
    residual = val_refiner - val_prior
    abs_res = residual.abs()
    score = domain_distance_score(val_domain_dist)
    prior_spearman = float(spearman_tensor(val_prior, val_y).detach().cpu().item())
    grid = torch.linspace(0.0, 1.0, 41, dtype=torch.float32, device=val_y.device)
    band_qs = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    if abs_res.numel() > 0:
        band_values = [0.0] + [float(torch.quantile(abs_res, q).detach().cpu().item()) for q in band_qs[1:]]
        threshold_values: list[float | None] = [None] + [
            float(torch.quantile(score, q).detach().cpu().item())
            for q in (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
        ]
    else:
        band_values = [0.0]
        threshold_values = [None]
    candidates: list[dict] = []
    for band in band_values:
        band_active = abs_res >= band
        for threshold in threshold_values:
            domain_active = torch.ones_like(band_active) if threshold is None else score <= float(threshold)
            active = band_active & domain_active
            gated = torch.where(active, residual, torch.zeros_like(residual))
            preds = val_prior[None, :] + grid[:, None] * gated[None, :]
            mse = ((preds - val_y[None, :]) ** 2).mean(dim=1)
            idx = int(torch.argmin(mse).detach().cpu().item())
            alpha = float(grid[idx].detach().cpu().item())
            route_share = 0.0 if alpha <= 1e-9 else float(active.float().mean().detach().cpu().item())
            candidates.append({
                "alpha": alpha,
                "residualBand": float(band),
                "domainThreshold": None if threshold is None else float(threshold),
                "domainRouteShare": route_share,
                "validMSE": float(mse[idx].detach().cpu().item()),
                "validHarmWorse": float(harm_worse_tensor(val_prior, preds[idx], val_y).detach().cpu().item()),
                "validSpearman": float(spearman_tensor(preds[idx], val_y).detach().cpu().item()),
                "harmGuard": float(max_harm),
                "rankGuard": float(rank_tol),
                "priorSpearman": prior_spearman,
                "domainScore": "min_standardized_distance_to_GKN_prototypes",
            })
    if not candidates:
        return {
            "alpha": 0.0, "residualBand": 0.0, "domainThreshold": None,
            "domainRouteShare": 0.0, "validMSE": float("inf"),
            "validHarmWorse": 0.0, "harmGuard": float(max_harm),
            "acceptedByHarmGuard": False,
            "domainScore": "min_standardized_distance_to_GKN_prototypes",
        }
    feasible = [
        c for c in candidates
        if c["validHarmWorse"] <= float(max_harm) + 1e-9
        and c["validSpearman"] >= prior_spearman - float(rank_tol) - 1e-9
    ]
    if not feasible:
        prior_mse = float(F.mse_loss(val_prior, val_y).detach().cpu().item())
        return {
            "alpha": 0.0,
            "residualBand": 0.0,
            "domainThreshold": None,
            "domainRouteShare": 0.0,
            "validMSE": prior_mse,
            "validHarmWorse": 0.0,
            "validSpearman": prior_spearman,
            "priorSpearman": prior_spearman,
            "harmGuard": float(max_harm),
            "rankGuard": float(rank_tol),
            "acceptedByHarmGuard": False,
            "domainScore": "min_standardized_distance_to_GKN_prototypes",
        }
    harm_ok = feasible
    # Conservative tie-break: among configs whose validation MSE is within `tol` of the
    # best, prefer the one that moves the fewest points (largest deferral). This defends
    # `harm_worse` and worst-group robustness without giving up measured validation MSE,
    # since the alpha=0 / no-route config (prior) is always within tol of itself.
    best_mse = min(c["validMSE"] for c in harm_ok)
    tol = max(1e-4, 0.0025 * best_mse)
    near = [c for c in harm_ok if c["validMSE"] <= best_mse + tol]
    near.sort(key=lambda c: (c["domainRouteShare"], c["validMSE"]))
    best = dict(near[0])
    best["validMSEbest"] = float(best_mse)
    best["tieTolerance"] = float(tol)
    best["acceptedByHarmGuard"] = True
    return best


def apply_domain_routed_alpha(
    prior: torch.Tensor,
    refiner: torch.Tensor,
    alpha: float,
    band: float,
    domain_dist: torch.Tensor,
    threshold: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual = refiner - prior
    active = residual.abs() >= float(band)
    if threshold is not None:
        active = active & (domain_distance_score(domain_dist) <= float(threshold))
    if float(alpha) <= 1e-9:
        active = torch.zeros_like(active)
    gated = torch.where(active, residual, torch.zeros_like(residual))
    return prior + float(alpha) * gated, active


def build_mechanism_graph(
    target_ids: list[str],
    fit_targets: np.ndarray,
    text_by_id: dict[str, str],
    text_dim: int,
    max_entities: int,
    min_entity_df: int,
    device: torch.device,
    hierarchical: bool = False,
    higcn_tiers: int = 2,
) -> tuple[torch.Tensor, list[tuple[int, int]], int, dict]:
    fit_unique = np.unique(fit_targets).astype(int)
    train_target_ids = [target_ids[int(i)] for i in fit_unique]
    entities = select_entities(text_by_id, train_target_ids, max_entities=max_entities, min_df=min_entity_df)
    entity_index = {entity: i for i, entity in enumerate(entities)}
    target_pos = {int(t): i for i, t in enumerate(fit_unique)}
    node_texts = [text_by_id.get(target_ids[int(t)], "") for t in fit_unique]
    node_texts.extend(entities)
    if not node_texts:
        node_texts = [""]
    node_feat = MechanismTextEncoder(text_dim, str(device)).build(node_texts).float()
    # Inner-train document frequency per entity (used for HiGCN-style hierarchical masking).
    entity_df = {e: 0 for e in entities}
    for tid in train_target_ids:
        for token in {tok for tok in tokenize(text_by_id.get(tid, "")) if tok in entity_index}:
            entity_df[token] += 1
    tier_cut = 0
    if hierarchical and entity_df:
        # K=2 tiers: keep edges only to high-frequency (reliable) entities, blocking noisy
        # low-frequency entities from corrupting prototypes; flow stays high->low at projection.
        dfs = np.asarray(sorted(entity_df.values()))
        q = max(0.0, 1.0 - 1.0 / max(1, int(higcn_tiers)))  # tiers=2 -> median cut
        tier_cut = int(np.quantile(dfs, q)) if dfs.size else 0
    edges: list[tuple[int, int]] = []
    dropped = 0
    for t in fit_unique:
        tid = target_ids[int(t)]
        tokens = {
            token for token in tokenize(text_by_id.get(tid, ""))
            if token in entity_index
        }
        for token in tokens:
            if hierarchical and entity_df.get(token, 0) < tier_cut:
                dropped += 1
                continue
            edges.append((target_pos[int(t)], len(fit_unique) + entity_index[token]))
    meta = {
        "nTrainTargets": int(len(fit_unique)),
        "nEntities": int(len(entities)),
        "nEdges": int(len(edges)),
        "minEntityDf": int(min_entity_df),
        "maxEntities": int(max_entities),
        "entityExamples": entities[:20],
        "hierarchical": bool(hierarchical),
        "higcnTiers": int(higcn_tiers),
        "higcnTierCutDf": int(tier_cut),
        "higcnEdgesDropped": int(dropped),
    }
    return node_feat, edges, len(fit_unique), meta


def train_gkn_prototypes(
    all_target_feat: np.ndarray,
    fit_targets: np.ndarray,
    target_ids: list[str],
    text_by_id: dict[str, str],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    node_feat, edges, n_train_targets, graph_meta = build_mechanism_graph(
        target_ids,
        fit_targets,
        text_by_id,
        args.text_dim,
        args.max_entities,
        args.min_entity_df,
        device,
        hierarchical=bool(getattr(args, "hierarchical_gkn", False)),
        higcn_tiers=int(getattr(args, "higcn_tiers", 2)),
    )
    adj = dense_normalized_adjacency(node_feat.shape[0], edges, device)
    model = GraphKnowledgeNetwork(args.text_dim, args.gkn_hidden, args.domain_dim, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.gkn_lr, weight_decay=args.weight_decay)
    target_adj = (adj > 0).float()
    best_state = None
    best_loss = float("inf")
    for _ in range(args.gkn_epochs):
        emb = model(node_feat, adj)
        logits = emb @ emb.t() / max(1.0, args.domain_dim ** 0.5)
        loss = F.binary_cross_entropy_with_logits(logits, target_adj)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        value = float(loss.detach().cpu().item())
        if value < best_loss:
            best_loss = value
            best_state = {key: val.detach().clone() for key, val in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        gkn_nodes = model(node_feat, adj)
        train_domain = gkn_nodes[:n_train_targets].detach()

    fit_unique = np.unique(fit_targets).astype(int)
    x_train = float_tensor(all_target_feat[fit_unique], device)
    y_train = train_domain.float()
    projector = torch.nn.Linear(x_train.shape[1], args.domain_dim).to(device)
    proj_opt = torch.optim.AdamW(projector.parameters(), lr=args.gkn_lr, weight_decay=args.weight_decay)
    for _ in range(args.projector_epochs):
        pred = projector(x_train)
        loss = F.mse_loss(pred, y_train)
        proj_opt.zero_grad(set_to_none=True)
        loss.backward()
        proj_opt.step()
    with torch.no_grad():
        all_domain = projector(float_tensor(all_target_feat, device)).float()
        train_all = all_domain[long_tensor(fit_unique, device)]
        mean = train_all.mean(dim=0, keepdim=True)
        std = train_all.std(dim=0, keepdim=True).clamp_min(1e-6)
        all_z = (all_domain - mean) / std
        train_z = all_z[long_tensor(fit_unique, device)].detach().cpu().numpy()

    k = int(min(args.prototypes, max(1, train_z.shape[0])))
    if k == 1:
        proto_np = train_z.mean(axis=0, keepdims=True)
    else:
        proto_np = KMeans(n_clusters=k, random_state=args.seed + 77, n_init=10).fit(train_z).cluster_centers_
    prototypes = float_tensor(proto_np.astype(np.float32), device)
    dist = torch.cdist(all_z, prototypes).float()
    family_id = dist.argmin(dim=1).detach().cpu().numpy().astype(np.int64)
    train_dist = dist[long_tensor(fit_unique, device)]
    dist_mean = train_dist.mean(dim=0, keepdim=True)
    dist_std = train_dist.std(dim=0, keepdim=True).clamp_min(1e-6)
    dist_z = (dist - dist_mean) / dist_std
    meta = {
        "graph": graph_meta,
        "gkn": {
            "type": "dense_gcn_link_reconstruction",
            "domainDim": int(args.domain_dim),
            "hidden": int(args.gkn_hidden),
            "epochs": int(args.gkn_epochs),
            "bestLoss": float(best_loss),
            "prototypeCount": int(k),
            "hierarchicalGkn": bool(getattr(args, "hierarchical_gkn", False)),
            "prototypeSource": "GKN embeddings from inner-train target mechanism graph only",
            "targetProjector": "linear pLM-to-GKN-domain map fit on inner-train targets only",
        },
    }
    return dist_z.detach(), prototypes.detach(), int(k), family_id, meta


def checkpoint_path(dataset: str, split: str, seed: int, args: argparse.Namespace) -> Path:
    plm = str(args.plm_source).replace("/", "_")
    src = str(args.mechanism_source).replace("-", "_")
    mode = "smoke" if args.smoke else "full"
    rw = int(round(float(getattr(args, "rank_weight", 0.0)) * 1000))
    cw = int(round(float(getattr(args, "contrast_weight", 0.0)) * 1000))
    extra = ""
    if rw:
        extra += f"_rw{rw}"
    if cw:
        extra += f"_cw{cw}"
    if int(args.epochs) != 4:
        extra += f"_e{args.epochs}"
    if float(getattr(args, "ema_decay", 0.0)) > 0:
        extra += f"_ema{int(round(float(args.ema_decay) * 1000))}"
    if float(getattr(args, "pair_affinity_weight", 0.0)) > 0:
        extra += f"_pa{int(round(float(args.pair_affinity_weight) * 1000))}"
    if bool(getattr(args, "conformal_defer", False)):
        extra += "_conf"
    if bool(getattr(args, "deterministic", False)):
        extra += "_det"
    drug_mode = getattr(args, "drug_encoder", "morgan")
    if drug_mode != "morgan":
        extra += f"_{drug_mode.replace('-', '')}"
    pmode = getattr(args, "prompt_profile_mode", "off")
    if bool(getattr(args, "family_calibration", False)):
        extra += "_fcal"
    if pmode != "off":
        fus = getattr(args, "prompt_profile_fusion", "off").replace("-", "")
        extra += f"_pp{pmode.replace('-', '')}_{fus}"
        if getattr(args, "prompt_profile_source", "cached") == "deepseek":
            extra += "_ds"
        if getattr(args, "hierarchical_gkn", False):
            extra += "_hi"
        if float(getattr(args, "mechanism_align_weight", 0.0)) > 0:
            extra += f"_ma{int(round(float(args.mechanism_align_weight) * 1000))}"
        ctrl = getattr(args, "prompt_control", "none")
        if ctrl != "none":
            extra += f"_{ctrl.replace('-', '')}"
    name = (
        f"{dataset.lower()}_{split.replace('-', '_')}_{plm}_{src}_{mode}_"
        f"d{args.d_model}_l{args.layers}_dom{args.domain_dim}_p{args.prototypes}{extra}_seed{seed}.pt"
    )
    return CHECKPOINT_DIR / name


def resolve_checkpoint_path(dataset: str, split: str, seed: int, args: argparse.Namespace) -> Path:
    path = checkpoint_path(dataset, split, seed, args)
    if path.exists():
        return path
    legacy = LEGACY_CHECKPOINT_DIR / path.name
    if legacy.exists():
        return legacy
    return path


class WeightEMA:
    """Exponential moving average of floating-point model weights (surgical, no new module)."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if value.dtype.is_floating_point
        }

    def update(self, model: torch.nn.Module) -> None:
        sd = model.state_dict()
        for key, shadow in self.shadow.items():
            shadow.mul_(self.decay).add_(sd[key].detach(), alpha=1.0 - self.decay)

    def state_dict_over(self, model: torch.nn.Module) -> dict:
        sd = {key: value.detach().clone() for key, value in model.state_dict().items()}
        for key, shadow in self.shadow.items():
            sd[key] = shadow.detach().clone()
        return sd


def configure_determinism(args: argparse.Namespace, device: torch.device) -> dict:
    """Apply deterministic CUDA/cuDNN settings when requested; record what was set."""
    deterministic = bool(getattr(args, "deterministic", False))
    disable_amp = bool(getattr(args, "disable_amp", False)) or deterministic
    if disable_amp:
        args.amp = False
    if not deterministic:
        torch.backends.cudnn.benchmark = device.type == "cuda"
        return {"deterministic": False, "ampDisabled": bool(disable_amp)}
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    warn_only = not bool(getattr(args, "strict_deterministic", False))
    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
    except Exception:
        warn_only = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    # Force the math SDPA path when strict mode is on: Memory-Efficient attention has a
    # nondeterministic backward on Ada/Ampere and is the source of the warn-only fall-back
    # in the round-5 variance report. The math path is deterministic and slightly slower.
    sdpa_forced_math = False
    if bool(getattr(args, "strict_deterministic", False)) and device.type == "cuda":
        try:
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            sdpa_forced_math = True
        except Exception:
            sdpa_forced_math = False
    return {
        "deterministic": True,
        "cudnnDeterministic": True,
        "cudnnBenchmark": False,
        "ampDisabled": True,
        "cublasWorkspaceConfig": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "useDeterministicAlgorithmsWarnOnly": bool(warn_only),
        "sdpaForcedMath": bool(sdpa_forced_math),
    }


def train_gkn_model(
    drug_feat: torch.Tensor,
    target_feat: torch.Tensor,
    text_feat: torch.Tensor,
    domain_dist: torch.Tensor,
    prep: dict,
    args: argparse.Namespace,
    device: torch.device,
    prompt_cats: torch.Tensor | None = None,
    prompt_cov: torch.Tensor | None = None,
) -> tuple[PrismSelectiveRefiner, dict]:
    seed_everything(args.seed)
    prompt_dim = int(prompt_cats.shape[-1]) if prompt_cats is not None else 0
    model = PrismSelectiveRefiner(
        drug_feat.shape[1],
        target_feat.shape[1],
        text_feat.shape[1],
        domain_dist.shape[1],
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        mem_dim=MEM_DIM,
        min_gamma=args.min_gamma,
        prompt_profile_dim=prompt_dim,
        prompt_fusion=getattr(args, "prompt_profile_fusion", "off"),
    ).to(device)

    def _pp(t_idx: torch.Tensor):
        if prompt_cats is None:
            return None, None
        return prompt_cats[t_idx], prompt_cov[t_idx]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    ema = WeightEMA(model, args.ema_decay) if float(getattr(args, "ema_decay", 0.0)) > 0 else None
    best_val = float("inf")
    best_state = None

    def _val_loss() -> float:
        model.eval()
        loss_sum = prep["valY"].new_zeros(())
        count = 0
        with torch.no_grad():
            for start in range(0, prep["valD"].numel(), args.eval_batch_size):
                sl = slice(start, start + args.eval_batch_size)
                t_idx = prep["valT"][sl]
                vc, vcov = _pp(t_idx)
                with autocast_context(device, args.amp):
                    val_pred = model(
                        drug_feat[prep["valD"][sl]],
                        target_feat[t_idx],
                        prep["valPrior"][sl],
                        prep["valMem"][sl],
                        text_feat[t_idx],
                        domain_dist[t_idx],
                        vc,
                        vcov,
                    )
                loss_sum = loss_sum + ((val_pred - prep["valY"][sl]) ** 2).sum()
                count += int(prep["valY"][sl].numel())
        return float((loss_sum / max(1, count)).detach().cpu().item())

    fit_n = prep["fitD"].numel()
    val_every = max(1, int(getattr(args, "val_every", 1)))
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(fit_n, device=device)
        for start in range(0, fit_n, args.batch_size):
            idx = order[start:start + args.batch_size]
            if idx.numel() < 2:
                continue
            d_idx = prep["fitD"][idx]
            t_idx = prep["fitT"][idx]
            pc, pcov = _pp(t_idx)
            use_pa = float(getattr(args, "pair_affinity_weight", 0.0)) > 0
            with autocast_context(device, args.amp):
                out = model.residual_gate(
                    drug_feat[d_idx],
                    target_feat[t_idx],
                    prep["fitMem"][idx],
                    text_feat[t_idx],
                    domain_dist[t_idx],
                    pc,
                    pcov,
                    return_pair=use_pa,
                )
                if use_pa:
                    residual, gamma, pair_repr = out
                else:
                    residual, gamma = out
                pred = prep["fitPrior"][idx] + gamma * residual
                loss = F.mse_loss(pred, prep["fitY"][idx])
                if use_pa:
                    # Train-only direct pair-affinity regression: forces the cross-attn fused
                    # representation to carry pair-affinity signal explicitly. NOT used at infer.
                    pa_pred = model.pair_affinity(pair_repr)
                    loss = loss + float(args.pair_affinity_weight) * F.mse_loss(pa_pred, prep["fitY"][idx])
                if args.defer_l1 > 0:
                    loss = loss + float(args.defer_l1) * (1.0 - gamma).mean()
                if getattr(args, "rank_weight", 0.0) > 0:
                    loss = loss + float(args.rank_weight) * pairwise_rank_loss(pred, prep["fitY"][idx])
                if getattr(args, "contrast_weight", 0.0) > 0:
                    drug_pool, target_pool, _ = model.base.modality_pools(drug_feat[d_idx], target_feat[t_idx])
                    loss = loss + float(args.contrast_weight) * contrastive_loss(
                        drug_pool, target_pool, prep["fitY"][idx], args.contrast_temperature
                    )
                if getattr(args, "mechanism_align_weight", 0.0) > 0 and pc is not None and model.align_proj is not None:
                    denom = pcov.sum(dim=1, keepdim=True).clamp_min(1.0)
                    avg_profile = (pc * pcov.unsqueeze(-1)).sum(dim=1) / denom
                    drug_pool, _, _ = model.base.modality_pools(drug_feat[d_idx], target_feat[t_idx])
                    loss = loss + float(args.mechanism_align_weight) * contrastive_loss(
                        drug_pool, model.profile_align(avg_profile), prep["fitY"][idx], args.contrast_temperature
                    )
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update(model)
        if epoch % val_every != 0 and epoch != int(args.epochs):
            continue
        val_loss = _val_loss()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("GKN DTA refiner did not produce a validation checkpoint")
    # Choose raw-best vs EMA weights by validation MSE only (no test access).
    weights_used = "raw"
    raw_val = best_val
    ema_val = None
    if ema is not None:
        ema_state = ema.state_dict_over(model)
        model.load_state_dict(ema_state)
        ema_val = _val_loss()
        if ema_val < raw_val:
            weights_used = "ema"
            best_state = ema_state
            best_val = ema_val
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "bestValLoss": best_val,
        "weightsUsed": weights_used,
        "rawValLoss": float(raw_val),
        "emaValLoss": None if ema_val is None else float(ema_val),
        "emaDecay": float(getattr(args, "ema_decay", 0.0)),
    }


@torch.no_grad()
def predict_gkn(
    model: PrismSelectiveRefiner,
    drug_feat: torch.Tensor,
    target_feat: torch.Tensor,
    text_feat: torch.Tensor,
    domain_dist: torch.Tensor,
    drug_idx: torch.Tensor,
    target_idx: torch.Tensor,
    prior: torch.Tensor,
    mem: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    prompt_cats: torch.Tensor | None = None,
    prompt_cov: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    preds = []
    gammas = []
    for start in range(0, drug_idx.numel(), args.eval_batch_size):
        sl = slice(start, start + args.eval_batch_size)
        t_idx = target_idx[sl]
        pc = None if prompt_cats is None else prompt_cats[t_idx]
        pcov = None if prompt_cov is None else prompt_cov[t_idx]
        with autocast_context(device, args.amp):
            residual, gamma = model.residual_gate(
                drug_feat[drug_idx[sl]],
                target_feat[t_idx],
                mem[sl],
                text_feat[t_idx],
                domain_dist[t_idx],
                pc,
                pcov,
            )
            pred = prior[sl] + gamma * residual
        preds.append(pred.detach())
        gammas.append(gamma.detach())
    return torch.cat(preds, dim=0), torch.cat(gammas, dim=0)


def build_prompt_pack(
    args: argparse.Namespace,
    bundle,
    rows: dict,
    text_by_id: dict,
    domain_dist: torch.Tensor,
    n_families: int,
    family_id: np.ndarray,
    device: torch.device,
    split: str = "target-cold",
) -> dict:
    """Build per-target PromptSE-style profile tensors and the augmented gate context.

    When prompt mode is off, returns the unchanged domain_dist as the gate context and no
    profiles, so default behaviour is preserved exactly.
    """
    mode = getattr(args, "prompt_profile_mode", "off")
    if mode == "off":
        return {
            "cats": None,
            "cov": None,
            "gateDist": domain_dist,
            "profile": None,
            "familyId": family_id,
            "nFamilies": n_families,
        }
    source = getattr(args, "prompt_profile_source", "cached")
    fit_unique = np.unique(rows["fitT"]).astype(int)
    quality = uncertainty = None
    if source == "deepseek":
        from scripts.promptprofiles import load_deepseek_family_profiles
        from scripts.mechanismcache import cache_path as ds_cache_path
        members = {f: [int(t) for t in fit_unique if int(family_id[int(t)]) == f] for f in range(n_families)}
        profile = load_deepseek_family_profiles(
            ds_cache_path(bundle.dataset, split, int(args.seed)),
            n_families, members, args.text_dim, device,
            control=getattr(args, "prompt_control", "none"), seed=int(args.seed),
        )
        quality = profile["quality"]
        uncertainty = profile["uncertainty"]
    else:
        fit_pairs = {"D": rows["fitD"], "T": rows["fitT"], "Y": rows["fitY"]}
        profile = build_family_profiles(
            bundle.target_ids, fit_unique, family_id, text_by_id,
            n_families=n_families, text_dim=args.text_dim, device=device, fit_pairs=fit_pairs,
            sample_cap=int(getattr(args, "llm_sample_cap", 20)),
            prompt_version=getattr(args, "llm_prompt_version", PROMPT_VERSION_DEFAULT),
            control=getattr(args, "prompt_control", "none"), seed=int(args.seed),
        )
    fam = long_tensor(family_id, device)                       # [n_targets]
    profile_fam = fam
    fmap = profile.get("familyMap")
    if fmap is not None:
        fmap = fmap.to(device=device, dtype=torch.long)
        profile_fam = fmap[fam.clamp(max=fmap.numel() - 1)]
    cats = profile["profileTensor"][profile_fam]               # [n_targets,4,text_dim]
    cov = profile["coverage"][profile_fam]                     # [n_targets,4]
    conf = profile["confidence"][profile_fam].unsqueeze(1)     # [n_targets,1]
    cov_frac = cov.mean(dim=1, keepdim=True)                   # [n_targets,1]
    coldness = domain_distance_score(domain_dist).unsqueeze(1) # [n_targets,1]
    extra = [domain_dist, conf, cov_frac, coldness]
    if quality is not None:
        extra.append(quality[profile_fam].unsqueeze(1))
        extra.append(uncertainty[profile_fam].unsqueeze(1))
    gate_dist = torch.cat(extra, dim=1)
    return {"cats": cats, "cov": cov, "gateDist": gate_dist, "profile": profile,
            "familyId": family_id, "nFamilies": n_families}


def run_train_one(dataset: str, split: str, seed: int, args: argparse.Namespace, device: torch.device) -> dict:
    out = checkpoint_path(dataset, split, seed, args)
    meta_out = out.with_suffix(".json")
    if out.exists() and meta_out.exists() and not args.force:
        return json.loads(meta_out.read_text(encoding="utf-8"))
    t0 = time.time()
    args.seed = int(seed)
    # Seed the whole pipeline up front: the GKN graph network and projector in
    # train_gkn_prototypes consume the global RNG before train_gkn_model reseeds, so without
    # this the prototypes (and thus domain routing) were drawn from process entropy and varied
    # run-to-run. This is the dominant nondeterminism source identified in the variance round.
    seed_everything(args.seed)
    bundle = load_affinity_bundle(dataset, args, device)
    sp = make_split(bundle, split, seed)
    rows = apply_limits(sp, args, seed)
    drug_feat = split_norm_tensor(bundle.drug_raw, rows["fitD"], device)
    target_feat = split_norm_tensor(bundle.target_raw, rows["fitT"], device)
    prep = prepare_priors(drug_feat, target_feat, rows, device)
    text_raw, text_by_id, text_meta = text_feature_matrix(
        dataset,
        bundle.target_ids,
        rows["fitT"],
        args.mechanism_source,
        args.text_dim,
        device,
    )
    domain_dist, prototypes, n_families, family_id, graph_meta = train_gkn_prototypes(
        bundle.target_raw,
        rows["fitT"],
        bundle.target_ids,
        text_by_id,
        args,
        device,
    )
    pack = build_prompt_pack(args, bundle, rows, text_by_id, domain_dist, n_families, family_id, device, split)
    gate_dist = pack["gateDist"]
    text_feat = train_only_text_tensor(text_raw, rows["fitT"], device)
    model, train_meta = train_gkn_model(
        drug_feat, target_feat, text_feat, gate_dist, prep, args, device, pack["cats"], pack["cov"]
    )
    val_refiner, val_gamma = predict_gkn(
        model,
        drug_feat,
        target_feat,
        text_feat,
        gate_dist,
        prep["valD"],
        prep["valT"],
        prep["valPrior"],
        prep["valMem"],
        args,
        device,
        pack["cats"],
        pack["cov"],
    )
    alpha_meta = select_domain_alpha_tensor(
        prep["valPrior"],
        val_refiner,
        prep["valY"],
        domain_dist[prep["valT"]],
        args.alpha_harm_guard,
        args.alpha_rank_tol,
    )
    # Part D.3: per-family risk-calibrated deferral (validation-only). For each train target
    # family, estimate the routed correction's harm on its validation pairs; families whose
    # validation harm exceeds the budget are deferred (residual->prior) at test time. The prior
    # remains the guaranteed fallback. Uses train/val only; no test labels/statistics.
    defer_families: list[int] = []
    family_val_harm: dict[str, float] = {}
    if getattr(args, "family_calibration", False) and pack.get("familyId") is not None:
        val_final_t = apply_domain_routed_alpha(
            prep["valPrior"], val_refiner, float(alpha_meta["alpha"]),
            float(alpha_meta.get("residualBand", 0.0)), domain_dist[prep["valT"]],
            alpha_meta.get("domainThreshold"),
        )[0]
        vfam = np.asarray(pack["familyId"])[prep["valT"].detach().cpu().numpy()]
        vp = prep["valPrior"].detach().cpu().numpy()
        vf = val_final_t.detach().cpu().numpy()
        vy = prep["valY"].detach().cpu().numpy()
        for f in range(int(pack["nFamilies"])):
            m = vfam == f
            if int(m.sum()) >= 5:
                h = harmful_rate(vp[m], vf[m], vy[m])
                family_val_harm[str(f)] = round(float(h), 4)
                if h > float(args.alpha_harm_guard) + 1e-9:
                    defer_families.append(int(f))
    val_final_for_reliability = apply_domain_routed_alpha(
        prep["valPrior"], val_refiner, float(alpha_meta["alpha"]),
        float(alpha_meta.get("residualBand", 0.0)), domain_dist[prep["valT"]],
        alpha_meta.get("domainThreshold"),
    )[0]
    if defer_families and pack.get("familyId") is not None:
        vfam_all = np.asarray(pack["familyId"])[prep["valT"].detach().cpu().numpy()]
        defer_mask = torch.from_numpy(np.isin(vfam_all, defer_families)).to(device)
        val_final_for_reliability = torch.where(defer_mask, prep["valPrior"], val_final_for_reliability)
    val_rel, val_rel_meta = pack_deepseek_reliability(pack, prep["valT"].detach().cpu().numpy())
    reliability_meta = select_reliability_beta(
        prep["valPrior"].detach().cpu().numpy(),
        val_final_for_reliability.detach().cpu().numpy(),
        prep["valY"].detach().cpu().numpy(),
        val_rel,
        args.alpha_harm_guard,
        args.alpha_rank_tol,
    )
    reliability_meta["validationReliability"] = val_rel_meta
    harmfirst_meta = select_harmfirst_beta(
        prep["valPrior"].detach().cpu().numpy(),
        val_final_for_reliability.detach().cpu().numpy(),
        prep["valY"].detach().cpu().numpy(),
        val_rel,
        args.alpha_harm_guard,
        args.alpha_rank_tol,
    )
    harmfirst_meta["validationReliability"] = val_rel_meta
    conformal_meta = {"enabled": False}
    if getattr(args, "conformal_defer", False) and pack.get("familyId") is not None:
        val_fam_for_conf = np.asarray(pack["familyId"])[prep["valT"].detach().cpu().numpy()]
        conformal_meta = build_conformal_defer_meta(
            val_fam_for_conf,
            prep["valPrior"].detach().cpu().numpy(),
            val_final_for_reliability.detach().cpu().numpy(),
            prep["valY"].detach().cpu().numpy(),
            val_rel,
            int(pack["nFamilies"]),
            args.alpha_harm_guard,
        )
    # Selective prediction risk score (validation-only). For each train family, record the
    # validation harm rate so test pairs can be ranked by per-family expected harm. At infer
    # time the top-K riskiest fraction is deferred to the prior. Test labels are NOT used.
    selective_meta: dict = {"enabled": False}
    if getattr(args, "selective", False) and pack.get("familyId") is not None:
        # Risk is measured against the FULL refiner (no family-cal deferral), so it has signal
        # even when family-calibration has already zeroed routed validation harm. Combines:
        # (1) per-family harm rate of the un-routed refiner vs prior (coarse, family-level), and
        # (2) per-pair |refiner - prior| residual magnitude (fine, pair-level tie-break).
        vfam = np.asarray(pack["familyId"])[prep["valT"].detach().cpu().numpy()]
        vp = prep["valPrior"].detach().cpu().numpy()
        vr = val_refiner.detach().cpu().numpy()
        vy = prep["valY"].detach().cpu().numpy()
        # Per-family harm of the *un-routed* refiner (validation only)
        global_h = harmful_rate(vp, vr, vy)
        per_fam_risk: dict[str, float] = {}
        for f in range(int(pack["nFamilies"])):
            m = vfam == f
            per_fam_risk[str(f)] = round(
                float(harmful_rate(vp[m], vr[m], vy[m]) if int(m.sum()) >= 5 else global_h), 4)
        # Per-family residual-magnitude scale (mean |refiner - prior|), used to normalize
        # pair-level residual magnitude so families with intrinsically large corrections
        # aren't all flagged risky.
        per_fam_res_scale: dict[str, float] = {}
        for f in range(int(pack["nFamilies"])):
            m = vfam == f
            per_fam_res_scale[str(f)] = round(
                float(np.mean(np.abs(vr[m] - vp[m])) if int(m.sum()) >= 5 else
                      np.mean(np.abs(vr - vp))), 6)
        selective_meta = {
            "enabled": True,
            "globalValHarm": round(float(global_h), 4),
            "perFamilyValHarm": per_fam_risk,
            "perFamilyResidualScale": per_fam_res_scale,
            "score": "per_family_val_harm_plus_per_pair_residual_magnitude",
        }
        val_family_for_selective = vfam
        val_rel_for_selective = val_rel
        hbeta_for_selective = float(harmfirst_meta.get("beta", reliability_meta.get("beta", 1.0)))
        if val_rel_for_selective is not None:
            hscale = np.clip(1.0 - hbeta_for_selective * (1.0 - val_rel_for_selective), 0.0, 1.0)
            val_harmfirst = (
                prep["valPrior"].detach().cpu().numpy()
                + hscale * (
                    val_final_for_reliability.detach().cpu().numpy()
                    - prep["valPrior"].detach().cpu().numpy()
                )
            )
            selected = select_selective_coverage(
                prep["valPrior"].detach().cpu().numpy(),
                val_harmfirst,
                val_refiner.detach().cpu().numpy(),
                prep["valY"].detach().cpu().numpy(),
                val_family_for_selective,
                val_rel_for_selective,
                selective_meta,
                list(getattr(args, "selective_coverages", [1.0, 0.95, 0.90, 0.80])),
                args.alpha_harm_guard,
                args.alpha_rank_tol,
            )
            selective_meta["selectedCoverage"] = selected
    metadata = {
        "schema": "drugtarget-PRISM-checkpoint-v1",
        "dataset": dataset,
        "split": split,
        "seed": int(seed),
        "checkpoint": repo_rel(out),
        "modelType": "PrismSelectiveRefiner",
        "architectureName": "PRISM",
        "physicalLine": "outputs/prism",
        "drugDim": int(drug_feat.shape[1]),
        "targetDim": int(target_feat.shape[1]),
        "textDim": int(text_feat.shape[1]),
        "domainDim": int(gate_dist.shape[1]),
        "baseDomainDim": int(domain_dist.shape[1]),
        "promptProfileMode": getattr(args, "prompt_profile_mode", "off"),
        "promptProfileFusion": getattr(args, "prompt_profile_fusion", "off"),
        "promptControl": getattr(args, "prompt_control", "none"),
        "promptProfileSource": getattr(args, "prompt_profile_source", "cached"),
        "familyCalibration": bool(getattr(args, "family_calibration", False)),
        "deferFamilies": defer_families,
        "familyValHarm": family_val_harm,
        "deepseekReliability": reliability_meta,
        "deepseekHarmFirst": harmfirst_meta,
        "conformalDefer": conformal_meta,
        "selective": selective_meta,
        "selectiveCoverages": list(getattr(args, "selective_coverages", [1.0, 0.95, 0.90, 0.80])),
        "hierarchicalGkn": bool(getattr(args, "hierarchical_gkn", False)),
        "mechanismAlignWeight": float(getattr(args, "mechanism_align_weight", 0.0)),
        "pairAffinityWeight": float(getattr(args, "pair_affinity_weight", 0.0)),
        "promptMeta": (pack["profile"]["meta"] if pack.get("profile") else None),
        "featureMeta": bundle.feature_meta,
        "textMeta": text_meta,
        "graphMeta": graph_meta,
        "dModel": int(args.d_model),
        "heads": int(args.heads),
        "layers": int(args.layers),
        "ffDim": int(args.ff_dim),
        "dropout": float(args.dropout),
        "epochs": int(args.epochs),
        "batchSize": int(args.batch_size),
        "evalBatchSize": int(args.eval_batch_size),
        "valEvery": int(getattr(args, "val_every", 1)),
        "learningRate": float(args.lr),
        "weightDecay": float(args.weight_decay),
        "deferL1": float(args.defer_l1),
        "minGamma": float(args.min_gamma),
        "rankWeight": float(getattr(args, "rank_weight", 0.0)),
        "contrastWeight": float(getattr(args, "contrast_weight", 0.0)),
        "emaDecay": float(getattr(args, "ema_decay", 0.0)),
        "weightsUsed": train_meta.get("weightsUsed", "raw"),
        "rawValLoss": train_meta.get("rawValLoss"),
        "emaValLoss": train_meta.get("emaValLoss"),
        "determinism": getattr(args, "determinismMeta", {"deterministic": False}),
        "amp": bool(args.amp),
        "device": str(device),
        "devicePolicy": "cuda_required_no_cpu_fallback",
        "validationBasis": sp.get("validationBasis"),
        "limits": rows["limits"],
        "blendWeight": float(prep["blendWeight"].detach().cpu().item()),
        "alpha": float(alpha_meta["alpha"]),
        "residualBand": float(alpha_meta.get("residualBand", 0.0)),
        "domainThreshold": alpha_meta.get("domainThreshold"),
        "domainRouteShare": float(alpha_meta.get("domainRouteShare", 0.0)),
        "alphaSelection": alpha_meta,
        "valMeanGamma": float(val_gamma.float().mean().detach().cpu().item()),
        "memMean": prep["memMean"].detach().cpu().reshape(-1).tolist(),
        "memStd": prep["memStd"].detach().cpu().reshape(-1).tolist(),
        "trainableParams": int(sum(param.numel() for param in model.parameters() if param.requires_grad)),
        "bestValLoss": float(train_meta["bestValLoss"]),
        "leakagePolicy": "GKN graph, prototypes, text availability, normalizers, and projector are fit from inner-train targets only; test labels/statistics are not used",
        "trainSeconds": round(float(time.time() - t0), 3),
        "trainedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    save_blob = {
        **metadata,
        "stateDict": model.state_dict(),
        "domainDist": domain_dist.detach().cpu(),
        "gateDist": gate_dist.detach().cpu(),
        "familyId": torch.from_numpy(np.asarray(family_id, dtype=np.int64)),
        "textFeatRaw": torch.from_numpy(text_raw.astype(np.float32)),
        "prototypes": prototypes.detach().cpu(),
    }
    if pack.get("profile") is not None:
        save_blob["promptProfileTensor"] = pack["profile"]["profileTensor"].detach().cpu()
        save_blob["promptCoverage"] = pack["profile"]["coverage"].detach().cpu()
        for key, save_key in (
            ("confidence", "promptConfidence"),
            ("quality", "promptQuality"),
            ("uncertainty", "promptUncertainty"),
            ("familyMap", "promptFamilyMap"),
        ):
            if key in pack["profile"]:
                save_blob[save_key] = pack["profile"][key].detach().cpu()
    torch.save(save_blob, out)
    json_dump(meta_out, {key: value for key, value in metadata.items() if key not in {"stateDict"}})
    return metadata


def load_checkpoint(dataset: str, split: str, seed: int, args: argparse.Namespace, device: torch.device) -> tuple[dict, PrismSelectiveRefiner]:
    path = resolve_checkpoint_path(dataset, split, seed, args)
    if not path.exists():
        raise SystemExit(f"missing checkpoint: {repo_rel(path)}; run train first")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    prompt_dim = int(payload["promptProfileTensor"].shape[-1]) if "promptProfileTensor" in payload else 0
    model = PrismSelectiveRefiner(
        int(payload["drugDim"]),
        int(payload["targetDim"]),
        int(payload["textDim"]),
        int(payload["domainDim"]),
        d_model=int(payload["dModel"]),
        n_heads=int(payload["heads"]),
        n_layers=int(payload["layers"]),
        ff_dim=int(payload["ffDim"]),
        dropout=float(payload["dropout"]),
        mem_dim=MEM_DIM,
        min_gamma=float(payload["minGamma"]),
        prompt_profile_dim=prompt_dim,
        prompt_fusion=payload.get("promptProfileFusion", "off"),
    ).to(device)
    model.load_state_dict(payload["stateDict"])
    model.eval()
    return payload, model


def args_selective_coverages_payload(payload: dict) -> list[float]:
    return payload.get("selectiveCoverages") or [1.0, 0.95, 0.90, 0.80]


def deepseek_qc_from_cache(payload: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    meta = payload.get("promptMeta") or {}
    cache_name = meta.get("cache")
    if not cache_name:
        return None
    path = REPO / "dataset" / "cache" / "deepseek_promptdta" / str(cache_name)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    n = int(data.get("nFamilies", 0) or 0)
    if n <= 0:
        return None
    coverage = np.zeros((n, 4), dtype=np.float32)
    confidence = np.full((n,), 0.5, dtype=np.float32)
    quality = np.full((n,), 0.5, dtype=np.float32)
    uncertainty = np.full((n,), 0.5, dtype=np.float32)
    for key, rec in (data.get("families") or {}).items():
        try:
            f = int(key)
        except ValueError:
            continue
        if not (0 <= f < n):
            continue
        qc = rec.get("qc") or {}
        accepted = qc.get("acceptedProfiles") or rec.get("profiles") or {}
        names = ["binding_domain_compatibility", "target_family_selectivity",
                 "pathway_function_context", "ligand_scaffold_physicochemical"]
        for c, name in enumerate(names):
            summary = str((accepted.get(name) or {}).get("summary", "None"))
            coverage[f, c] = 0.0 if summary.strip().lower() in ("none", "") else 1.0
        confidence[f] = float(qc.get("confidence", confidence[f]))
        quality[f] = float(qc.get("familyQuality", quality[f]))
        uncertainty[f] = float(qc.get("uncertainty", uncertainty[f]))
    if (payload.get("promptControl") == "name-only") or ((payload.get("promptMeta") or {}).get("control") == "name-only"):
        coverage.fill(1.0)
        confidence.fill(0.5)
        quality.fill(0.5)
        uncertainty.fill(0.5)
    return coverage, confidence, quality, uncertainty


def profile_family_ids(payload: dict, test_target_idx: np.ndarray) -> np.ndarray | None:
    if "familyId" not in payload:
        return None
    fam = payload["familyId"].numpy().astype(np.int64)
    profile_fam = fam[test_target_idx]
    fmap = payload.get("promptFamilyMap")
    if fmap is not None:
        fmap_np = fmap.numpy().astype(np.int64)
        profile_fam = fmap_np[np.clip(profile_fam, 0, fmap_np.size - 1)]
    return profile_fam


def deepseek_reliability(payload: dict, test_target_idx: np.ndarray) -> tuple[np.ndarray | None, dict]:
    if payload.get("promptProfileSource") != "deepseek":
        return None, {"enabled": False, "reason": "not_deepseek_profile_source"}
    profile_fam = profile_family_ids(payload, test_target_idx)
    if profile_fam is None:
        return None, {"enabled": False, "reason": "missing_family_ids"}
    if all(key in payload for key in ("promptCoverage", "promptConfidence", "promptQuality", "promptUncertainty")):
        coverage = payload["promptCoverage"].numpy().astype(np.float32)
        confidence = payload["promptConfidence"].numpy().astype(np.float32)
        quality = payload["promptQuality"].numpy().astype(np.float32)
        uncertainty = payload["promptUncertainty"].numpy().astype(np.float32)
    else:
        qc = deepseek_qc_from_cache(payload)
        if qc is None:
            return None, {"enabled": False, "reason": "missing_deepseek_qc"}
        coverage, confidence, quality, uncertainty = qc
    profile_fam = np.clip(profile_fam, 0, quality.shape[0] - 1)
    cov_frac = coverage[profile_fam].mean(axis=1)
    rel = 0.40 * quality[profile_fam] + 0.25 * confidence[profile_fam] + 0.20 * cov_frac + 0.15 * (1.0 - uncertainty[profile_fam])
    rel = np.clip(rel.astype(np.float64), 0.05, 1.0)
    return rel, {
        "enabled": True,
        "source": "deepseek_qc_quality_confidence_coverage_uncertainty",
        "meanReliability": round(float(np.mean(rel)), 4),
        "minReliability": round(float(np.min(rel)), 4),
        "maxReliability": round(float(np.max(rel)), 4),
    }


def pack_deepseek_reliability(pack: dict, target_idx: np.ndarray) -> tuple[np.ndarray | None, dict]:
    profile = pack.get("profile")
    if not profile:
        return None, {"enabled": False, "reason": "missing_profile_pack"}
    meta = profile.get("meta") or {}
    if meta.get("profileSource") != "deepseek-offline-cache":
        return None, {"enabled": False, "reason": "not_deepseek_profile_source"}
    family_id = np.asarray(pack.get("familyId"), dtype=np.int64)
    profile_fam = family_id[target_idx]
    fmap = profile.get("familyMap")
    if fmap is not None:
        fmap_np = fmap.detach().cpu().numpy().astype(np.int64)
        profile_fam = fmap_np[np.clip(profile_fam, 0, fmap_np.size - 1)]
    coverage = profile["coverage"].detach().cpu().numpy().astype(np.float32)
    confidence = profile["confidence"].detach().cpu().numpy().astype(np.float32)
    quality = profile["quality"].detach().cpu().numpy().astype(np.float32)
    uncertainty = profile["uncertainty"].detach().cpu().numpy().astype(np.float32)
    profile_fam = np.clip(profile_fam, 0, quality.shape[0] - 1)
    cov_frac = coverage[profile_fam].mean(axis=1)
    rel = 0.40 * quality[profile_fam] + 0.25 * confidence[profile_fam] + 0.20 * cov_frac + 0.15 * (1.0 - uncertainty[profile_fam])
    rel = np.clip(rel.astype(np.float64), 0.05, 1.0)
    return rel, {
        "enabled": True,
        "source": "deepseek_qc_quality_confidence_coverage_uncertainty",
        "meanReliability": round(float(np.mean(rel)), 4),
        "minReliability": round(float(np.min(rel)), 4),
        "maxReliability": round(float(np.max(rel)), 4),
    }


def spearman_np(pred: np.ndarray, labels: np.ndarray) -> float:
    if pred.size < 3:
        return 0.0
    rp = np.empty_like(pred, dtype=np.float64)
    rl = np.empty_like(labels, dtype=np.float64)
    rp[np.argsort(pred)] = np.arange(pred.size, dtype=np.float64)
    rl[np.argsort(labels)] = np.arange(labels.size, dtype=np.float64)
    rp = rp - rp.mean()
    rl = rl - rl.mean()
    denom = max(float(np.linalg.norm(rp) * np.linalg.norm(rl)), 1e-9)
    return float(np.dot(rp, rl) / denom)


def selective_risk_score(
    family: np.ndarray,
    refiner: np.ndarray,
    prior: np.ndarray,
    meta: dict,
    rel: np.ndarray | None,
) -> np.ndarray:
    """Validation-derived pair risk used only to decide how much residual to defer."""
    per_fam = meta.get("perFamilyValHarm", {}) or {}
    per_fam_scale = meta.get("perFamilyResidualScale", {}) or {}
    global_h = float(meta.get("globalValHarm", 0.0))
    coarse = np.asarray([per_fam.get(str(int(f)), global_h) for f in family], dtype=np.float64)
    pair_res = np.abs(refiner - prior).astype(np.float64)
    fam_scale = np.asarray(
        [max(per_fam_scale.get(str(int(f)), 1e-6), 1e-6) for f in family],
        dtype=np.float64,
    )
    fine = pair_res / fam_scale
    fine_rank = np.argsort(np.argsort(fine)).astype(np.float64) / max(fine.size - 1, 1)
    reliability_penalty = 0.0 if rel is None else (1.0 - rel)
    return coarse + 0.10 * reliability_penalty + 1e-3 * fine_rank


def apply_selective_defer(pred: np.ndarray, prior: np.ndarray, risk: np.ndarray, coverage: float) -> tuple[np.ndarray, int]:
    cov = float(np.clip(coverage, 0.0, 1.0))
    k_defer = int(round((1.0 - cov) * risk.size))
    out = pred.copy()
    if k_defer > 0:
        order = np.lexsort((np.arange(risk.size), risk))[::-1]
        out[order[:k_defer]] = prior[order[:k_defer]]
    return out, k_defer


def select_selective_coverage(
    val_prior: np.ndarray,
    val_pred: np.ndarray,
    val_refiner: np.ndarray,
    val_y: np.ndarray,
    val_family: np.ndarray,
    val_rel: np.ndarray | None,
    selective_meta: dict,
    coverages: list[float],
    max_harm: float,
    rank_tol: float,
) -> dict:
    if not selective_meta.get("enabled"):
        return {"enabled": False, "reason": "selective_meta_disabled"}
    prior_s = spearman_np(val_prior, val_y)
    risk = selective_risk_score(val_family, val_refiner, val_prior, selective_meta, val_rel)
    candidates: list[dict] = []
    for cov in sorted({float(c) for c in coverages}, reverse=True):
        pred, k_defer = apply_selective_defer(val_pred, val_prior, risk, cov)
        candidates.append({
            "coverage": float(cov),
            "kDeferred": int(k_defer),
            "validMSE": float(np.mean((pred - val_y) ** 2)),
            "validHarmWorse": harmful_rate(val_prior, pred, val_y),
            "validSpearman": spearman_np(pred, val_y),
        })
    feasible = [
        c for c in candidates
        if c["validHarmWorse"] <= float(max_harm) + 1e-9
        and c["validSpearman"] >= prior_s - float(rank_tol) - 1e-9
    ]
    pool = feasible or candidates
    best_harm = min(c["validHarmWorse"] for c in pool)
    near = [c for c in pool if c["validHarmWorse"] <= best_harm + 0.02]
    # Promotion policy is harm-first: when validation cannot meaningfully distinguish
    # coverages, prefer stronger deferral. This protects hard seeds where the validation
    # risk ordering is flat but the test residual can still be harmful.
    near.sort(key=lambda c: (c["coverage"], c["validMSE"]))
    best = dict(near[0])
    best["enabled"] = True
    best["selector"] = "validation_harm_first_conservative_coverage"
    best["acceptedByGuards"] = bool(feasible)
    best["priorSpearman"] = prior_s
    best["harmGuard"] = float(max_harm)
    best["rankGuard"] = float(rank_tol)
    best["candidates"] = [
        {k: round(float(v), 6) if isinstance(v, (float, np.floating)) else v for k, v in c.items()}
        for c in candidates
    ]
    return best


def select_reliability_beta(
    val_prior: np.ndarray,
    val_final: np.ndarray,
    val_y: np.ndarray,
    val_rel: np.ndarray | None,
    max_harm: float,
    rank_tol: float,
) -> dict:
    if val_rel is None:
        return {"beta": 0.0, "enabled": False, "reason": "missing_validation_reliability"}
    prior_s = spearman_np(val_prior, val_y)
    betas = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.50], dtype=np.float64)
    residual = val_final - val_prior
    candidates: list[dict] = []
    for beta in betas:
        scale = np.clip(1.0 - float(beta) * (1.0 - val_rel), 0.0, 1.0)
        pred = val_prior + scale * residual
        candidates.append({
            "beta": float(beta),
            "validMSE": float(np.mean((pred - val_y) ** 2)),
            "validHarmWorse": harmful_rate(val_prior, pred, val_y),
            "validSpearman": spearman_np(pred, val_y),
            "meanScale": float(np.mean(scale)),
        })
    feasible = [
        c for c in candidates
        if c["validHarmWorse"] <= float(max_harm) + 1e-9
        and c["validSpearman"] >= prior_s - float(rank_tol) - 1e-9
    ]
    if feasible:
        best_mse = min(c["validMSE"] for c in feasible)
        tol = max(1e-4, 0.0025 * best_mse)
        near = [c for c in feasible if c["validMSE"] <= best_mse + tol]
        near.sort(key=lambda c: (c["validHarmWorse"], c["validMSE"], c["beta"]))
        best = dict(near[0])
        best["acceptedByGuards"] = True
    else:
        best = min(candidates, key=lambda c: (c["validHarmWorse"], c["validMSE"]))
        best = dict(best)
        best["acceptedByGuards"] = False
    best["enabled"] = True
    best["priorSpearman"] = prior_s
    best["harmGuard"] = float(max_harm)
    best["rankGuard"] = float(rank_tol)
    best["candidates"] = [
        {k: round(float(v), 6) if isinstance(v, (float, np.floating)) else v for k, v in c.items()}
        for c in candidates
    ]
    return best


def select_harmfirst_beta(
    val_prior: np.ndarray,
    val_final: np.ndarray,
    val_y: np.ndarray,
    val_rel: np.ndarray | None,
    max_harm: float,
    rank_tol: float,
) -> dict:
    if val_rel is None:
        return {"beta": 0.0, "enabled": False, "reason": "missing_validation_reliability"}
    prior_s = spearman_np(val_prior, val_y)
    residual = val_final - val_prior
    betas = np.asarray(
        [0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0, 2.25, 2.50],
        dtype=np.float64,
    )
    candidates: list[dict] = []
    for beta in betas:
        scale = np.clip(1.0 - float(beta) * (1.0 - val_rel), 0.0, 1.0)
        pred = val_prior + scale * residual
        candidates.append({
            "beta": float(beta),
            "validMSE": float(np.mean((pred - val_y) ** 2)),
            "validHarmWorse": harmful_rate(val_prior, pred, val_y),
            "validSpearman": spearman_np(pred, val_y),
            "meanScale": float(np.mean(scale)),
        })
    rank_ok = [c for c in candidates if c["validSpearman"] >= prior_s - float(rank_tol) - 1e-9]
    pool = rank_ok or candidates
    # Harm-first selector: use validation to find a near-minimal harm band, then choose
    # the strongest bounded DeepSeek-QC shrinkage in that band. This avoids beta=3-style
    # over-shrink, which can game the moved-sample denominator of harm_worse.
    best_harm = min(c["validHarmWorse"] for c in pool)
    harm_tol = 0.02
    near = [c for c in pool if c["validHarmWorse"] <= best_harm + harm_tol]
    prior_mse = float(np.mean((val_prior - val_y) ** 2))
    mse_safe = [c for c in near if c["validMSE"] <= prior_mse + 1e-9]
    if mse_safe:
        near = mse_safe
    near.sort(key=lambda c: (-c["beta"], c["validHarmWorse"], c["validMSE"]))
    best = dict(near[0])
    best["enabled"] = True
    best["acceptedByHarmGuard"] = best["validHarmWorse"] <= float(max_harm) + 1e-9
    best["priorSpearman"] = prior_s
    best["harmGuard"] = float(max_harm)
    best["rankGuard"] = float(rank_tol)
    best["selector"] = "harm_first_validation_beta"
    best["harmNearTolerance"] = float(harm_tol)
    best["betaMax"] = float(betas.max())
    best["candidates"] = [
        {k: round(float(v), 6) if isinstance(v, (float, np.floating)) else v for k, v in c.items()}
        for c in candidates
    ]
    return best


def build_conformal_defer_meta(
    val_family: np.ndarray,
    val_prior: np.ndarray,
    val_final: np.ndarray,
    val_y: np.ndarray,
    rel: np.ndarray | None,
    n_families: int,
    harm_guard: float,
) -> dict:
    residual = np.abs(val_final - val_prior)
    moved = residual > 1e-9
    harmful = np.zeros_like(residual, dtype=bool)
    harmful[moved] = np.abs(val_final[moved] - val_y[moved]) > np.abs(val_prior[moved] - val_y[moved])
    global_threshold = float(np.quantile(residual[moved] if moved.any() else residual, 0.70))
    global_harm = harmful_rate(val_prior, val_final, val_y)
    thresholds: dict[str, float] = {}
    rates: dict[str, float] = {}
    counts: dict[str, int] = {}
    rel_means: dict[str, float] = {}
    defer_families: list[int] = []
    for f in range(int(n_families)):
        mask = val_family == f
        counts[str(f)] = int(mask.sum())
        if int(mask.sum()) < 5:
            thresholds[str(f)] = global_threshold
            rates[str(f)] = round(float(global_harm), 4)
            rel_means[str(f)] = 0.5 if rel is None else round(float(np.mean(rel)), 4)
            continue
        fam_res = residual[mask]
        thresholds[str(f)] = float(np.quantile(fam_res, 0.65))
        rates[str(f)] = round(float(harmful[mask & moved].mean() if (mask & moved).any() else 0.0), 4)
        rel_means[str(f)] = 0.5 if rel is None else round(float(np.mean(rel[mask])), 4)
        if rates[str(f)] > float(harm_guard) or rel_means[str(f)] < 0.66:
            defer_families.append(int(f))
    return {
        "enabled": True,
        "score": "validation_family_harm_plus_residual_quantile_plus_deepseek_reliability",
        "globalValHarm": round(float(global_harm), 4),
        "globalResidualThreshold": round(global_threshold, 6),
        "perFamilyResidualThreshold": {k: round(float(v), 6) for k, v in thresholds.items()},
        "perFamilyValHarm": rates,
        "perFamilyValCount": counts,
        "perFamilyMeanReliability": rel_means,
        "deferFamilies": defer_families,
        "harmGuard": float(harm_guard),
    }


def calibrate_harmfirst_at_infer(
    payload: dict,
    model: PrismSelectiveRefiner,
    drug_feat: torch.Tensor,
    target_feat: torch.Tensor,
    text_feat: torch.Tensor,
    domain_dist: torch.Tensor,
    gate_dist: torch.Tensor,
    prep: dict,
    args: argparse.Namespace,
    device: torch.device,
    prompt_cats: torch.Tensor | None,
    prompt_cov: torch.Tensor | None,
) -> dict:
    val_refiner, _ = predict_gkn(
        model,
        drug_feat,
        target_feat,
        text_feat,
        gate_dist,
        prep["valD"],
        prep["valT"],
        prep["valPrior"],
        prep["valMem"],
        args,
        device,
        prompt_cats,
        prompt_cov,
    )
    val_final = apply_domain_routed_alpha(
        prep["valPrior"],
        val_refiner,
        float(payload["alpha"]),
        float(payload.get("residualBand", 0.0)),
        domain_dist[prep["valT"]],
        payload.get("domainThreshold"),
    )[0]
    defer_families = set(int(f) for f in payload.get("deferFamilies", []) or [])
    if defer_families and "familyId" in payload:
        fam_np = payload["familyId"].numpy()
        val_fam = fam_np[prep["valT"].detach().cpu().numpy()]
        defer_mask = torch.from_numpy(np.isin(val_fam, list(defer_families))).to(device)
        val_final = torch.where(defer_mask, prep["valPrior"], val_final)
    val_rel, val_rel_meta = deepseek_reliability(payload, prep["valT"].detach().cpu().numpy())
    meta = select_harmfirst_beta(
        prep["valPrior"].detach().cpu().numpy(),
        val_final.detach().cpu().numpy(),
        prep["valY"].detach().cpu().numpy(),
        val_rel,
        args.alpha_harm_guard,
        args.alpha_rank_tol,
    )
    meta["validationReliability"] = val_rel_meta
    meta["calibrationAtInfer"] = True
    return meta


def calibrate_selective_at_infer(
    payload: dict,
    model: PrismSelectiveRefiner,
    drug_feat: torch.Tensor,
    target_feat: torch.Tensor,
    text_feat: torch.Tensor,
    domain_dist: torch.Tensor,
    gate_dist: torch.Tensor,
    prep: dict,
    args: argparse.Namespace,
    device: torch.device,
    prompt_cats: torch.Tensor | None,
    prompt_cov: torch.Tensor | None,
    harm_meta: dict,
) -> dict:
    sel_meta = dict(payload.get("selective", {}) or {})
    if not sel_meta.get("enabled") or "familyId" not in payload:
        return {"enabled": False, "reason": "selective_meta_disabled"}
    val_refiner, _ = predict_gkn(
        model,
        drug_feat,
        target_feat,
        text_feat,
        gate_dist,
        prep["valD"],
        prep["valT"],
        prep["valPrior"],
        prep["valMem"],
        args,
        device,
        prompt_cats,
        prompt_cov,
    )
    val_final = apply_domain_routed_alpha(
        prep["valPrior"],
        val_refiner,
        float(payload["alpha"]),
        float(payload.get("residualBand", 0.0)),
        domain_dist[prep["valT"]],
        payload.get("domainThreshold"),
    )[0]
    defer_families = set(int(f) for f in payload.get("deferFamilies", []) or [])
    if defer_families:
        fam_np = payload["familyId"].numpy()
        val_fam_np = fam_np[prep["valT"].detach().cpu().numpy()]
        defer_mask = torch.from_numpy(np.isin(val_fam_np, list(defer_families))).to(device)
        val_final = torch.where(defer_mask, prep["valPrior"], val_final)
    val_rel, rel_meta = deepseek_reliability(payload, prep["valT"].detach().cpu().numpy())
    if val_rel is None:
        return {"enabled": False, "reason": "missing_deepseek_reliability", "validationReliability": rel_meta}
    beta = float(harm_meta.get("beta", (payload.get("deepseekHarmFirst") or {}).get("beta", 1.0)))
    scale = np.clip(1.0 - beta * (1.0 - val_rel), 0.0, 1.0)
    val_prior_np = prep["valPrior"].detach().cpu().numpy()
    val_harmfirst = val_prior_np + scale * (val_final.detach().cpu().numpy() - val_prior_np)
    fam_np = payload["familyId"].numpy()
    selected = select_selective_coverage(
        val_prior_np,
        val_harmfirst,
        val_refiner.detach().cpu().numpy(),
        prep["valY"].detach().cpu().numpy(),
        fam_np[prep["valT"].detach().cpu().numpy()],
        val_rel,
        sel_meta,
        args_selective_coverages_payload(payload),
        args.alpha_harm_guard,
        args.alpha_rank_tol,
    )
    selected["calibrationAtInfer"] = True
    selected["validationReliability"] = rel_meta
    return selected


def run_infer_one(
    dataset: str,
    split: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    bundle_cache: dict[str, object] | None = None,
) -> dict:
    payload, model = load_checkpoint(dataset, split, seed, args, device)
    args.seed = int(seed)
    if bundle_cache is None:
        bundle = load_affinity_bundle(dataset, args, device)
    else:
        if dataset not in bundle_cache:
            bundle_cache[dataset] = load_affinity_bundle(dataset, args, device)
        bundle = bundle_cache[dataset]
    sp = make_split(bundle, split, seed)
    rows = apply_limits(sp, args, seed)
    drug_feat = split_norm_tensor(bundle.drug_raw, rows["fitD"], device)
    target_feat = split_norm_tensor(bundle.target_raw, rows["fitT"], device)
    prep = prepare_priors(drug_feat, target_feat, rows, device)
    text_feat = train_only_text_tensor(payload["textFeatRaw"].numpy().astype(np.float32), rows["fitT"], device)
    domain_dist = payload["domainDist"].to(device=device, dtype=torch.float32)
    gate_dist = payload.get("gateDist")
    gate_dist = domain_dist if gate_dist is None else gate_dist.to(device=device, dtype=torch.float32)
    prompt_cats = prompt_cov = None
    if "promptProfileTensor" in payload:
        fam = payload["familyId"].to(device=device, dtype=torch.long)
        profile_tensor = payload["promptProfileTensor"].to(device=device, dtype=torch.float32)
        coverage = payload["promptCoverage"].to(device=device, dtype=torch.float32)
        prompt_cats = profile_tensor[fam]
        prompt_cov = coverage[fam]
    mem_mean = float_tensor(np.asarray(payload["memMean"], dtype=np.float32).reshape(1, -1), device)
    mem_std = float_tensor(np.asarray(payload["memStd"], dtype=np.float32).reshape(1, -1), device)
    test = prepare_test_prior(
        drug_feat,
        target_feat,
        rows,
        float(payload["blendWeight"]),
        mem_mean,
        mem_std,
        device,
    )
    refiner, gamma = predict_gkn(
        model,
        drug_feat,
        target_feat,
        text_feat,
        gate_dist,
        test["testD"],
        test["testT"],
        test["testPrior"],
        test["testMem"],
        args,
        device,
        prompt_cats,
        prompt_cov,
    )
    final, route_active = apply_domain_routed_alpha(
        test["testPrior"],
        refiner,
        float(payload["alpha"]),
        float(payload.get("residualBand", 0.0)),
        domain_dist[test["testT"]],
        payload.get("domainThreshold"),
    )
    # Part D.3 apply: defer (final->prior) for test targets in validation-flagged high-risk families.
    defer_families = set(int(f) for f in payload.get("deferFamilies", []) or [])
    if defer_families and "familyId" in payload:
        fam_np = payload["familyId"].numpy()
        test_fam = fam_np[test["testT"].detach().cpu().numpy()]
        defer_mask = torch.from_numpy(np.isin(test_fam, list(defer_families))).to(device)
        final = torch.where(defer_mask, test["testPrior"], final)
    target_group_feat = split_norm_numpy(bundle.target_raw, rows["fitT"])
    y = rows["testY"].astype(np.float32)
    prior_np = test["testPrior"].detach().cpu().numpy()
    refiner_np = refiner.detach().cpu().numpy()
    final_np = final.detach().cpu().numpy()
    methods = {
        "prior_only": prior_np,
        "gkn_refiner": refiner_np,
        "domain_alpha": final_np,
    }
    test_target_idx = test["testT"].detach().cpu().numpy()
    ds_rel, ds_rel_meta = deepseek_reliability(payload, test_target_idx)
    harmfirst_np: np.ndarray | None = None
    if ds_rel is not None:
        rel_meta = payload.get("deepseekReliability") or {}
        beta = float(rel_meta.get("beta", 1.0))
        ds_scale = np.clip(1.0 - beta * (1.0 - ds_rel), 0.0, 1.0)
        methods["deepseek_reliable_alpha"] = prior_np + ds_scale * (final_np - prior_np)
        ds_rel_meta = {**ds_rel_meta, "beta": round(float(beta), 6),
                       "meanScale": round(float(np.mean(ds_scale)), 4),
                       "validationSelected": bool(rel_meta)}
        harm_meta = calibrate_harmfirst_at_infer(
            payload,
            model,
            drug_feat,
            target_feat,
            text_feat,
            domain_dist,
            gate_dist,
            prep,
            args,
            device,
            prompt_cats,
            prompt_cov,
        )
        hbeta = float(getattr(args, "harmfirst_beta_override", -1.0))
        if hbeta < 0:
            hbeta = float(harm_meta.get("beta", beta))
        hscale = np.clip(1.0 - hbeta * (1.0 - ds_rel), 0.0, 1.0)
        harmfirst_np = prior_np + hscale * (final_np - prior_np)
        methods["deepseek_harmfirst_alpha"] = harmfirst_np
        harm_meta_out = {
            **harm_meta,
            **ds_rel_meta,
            "beta": round(float(hbeta), 6),
            "meanScale": round(float(np.mean(hscale)), 4),
            "validationSelected": bool(harm_meta),
            "selector": harm_meta.get("selector", "harm_first_validation_beta"),
        }
    else:
        harm_meta_out = {"enabled": False, "reason": "missing_deepseek_reliability"}
    conf_meta = payload.get("conformalDefer") or {}
    conformal_table: dict = {"enabled": False}
    if conf_meta.get("enabled") and "familyId" in payload:
        fam_np = payload["familyId"].numpy()
        test_fam = fam_np[test_target_idx]
        thresholds = conf_meta.get("perFamilyResidualThreshold", {}) or {}
        risky_fams = set(int(f) for f in conf_meta.get("deferFamilies", []) or [])
        pair_res = np.abs(final_np - prior_np)
        defer_mask_np = np.zeros_like(pair_res, dtype=bool)
        for i, fam_i in enumerate(test_fam):
            fam_int = int(fam_i)
            threshold = float(thresholds.get(str(fam_int), conf_meta.get("globalResidualThreshold", float("inf"))))
            if fam_int in risky_fams and pair_res[i] >= threshold:
                defer_mask_np[i] = True
        conf_pred = final_np.copy()
        conf_pred[defer_mask_np] = prior_np[defer_mask_np]
        methods["deepseek_conformal_defer"] = conf_pred
        conformal_table = {
            "enabled": True,
            "deferred": int(defer_mask_np.sum()),
            "coverage": round(float(1.0 - defer_mask_np.mean()), 6),
            "deferFamilies": sorted(risky_fams),
            "score": conf_meta.get("score"),
        }
    # Selective prediction: at K coverage levels, defer the per-family-riskiest (1-K) fraction
    # of test pairs to the prior. Risk score is the validation-only per-family harm rate
    # (no test labels). Tie-broken by random within family for determinism.
    selective_table: list[dict] = []
    sel_meta = payload.get("selective", {}) or {}
    selected_coverage_meta: dict = {"enabled": False}
    if sel_meta.get("enabled") and "familyId" in payload:
        fam_np = payload["familyId"].numpy()
        test_fam = fam_np[test_target_idx]
        risk = selective_risk_score(test_fam, refiner_np, prior_np, sel_meta, ds_rel)
        selected_coverage_meta = sel_meta.get("selectedCoverage") or calibrate_selective_at_infer(
            payload,
            model,
            drug_feat,
            target_feat,
            text_feat,
            domain_dist,
            gate_dist,
            prep,
            args,
            device,
            prompt_cats,
            prompt_cov,
            harm_meta_out,
        )
        for cov in list(args_selective_coverages_payload(payload)):
            sel_pred, k_defer = apply_selective_defer(final_np, prior_np, risk, float(cov))
            name = f"selective_alpha_cov{int(round(cov * 100))}"
            methods[name] = sel_pred
            selective_table.append({"coverage": float(cov), "k_deferred": k_defer, "method": name})
            if harmfirst_np is not None:
                hsel_pred, k_defer = apply_selective_defer(harmfirst_np, prior_np, risk, float(cov))
                hname = f"deepseek_harmfirst_selective_cov{int(round(cov * 100))}"
                methods[hname] = hsel_pred
                selective_table.append({"coverage": float(cov), "k_deferred": k_defer, "method": hname})
        selected_cov = float(selected_coverage_meta.get("coverage", 0.80))
        if harmfirst_np is not None:
            auto_pred, auto_defer = apply_selective_defer(harmfirst_np, prior_np, risk, selected_cov)
            methods["deepseek_qc_selective"] = auto_pred
            selective_table.append({
                "coverage": selected_cov,
                "k_deferred": auto_defer,
                "method": "deepseek_qc_selective",
                "selector": selected_coverage_meta.get("selector", "validation_harm_first_coverage"),
            })
    metric_block = {
        name: {
            key: round(float(value), 6) if isinstance(value, (float, np.floating)) else value
            for key, value in core_metrics(y, pred, prior_np, target_group_feat, rows["testT"]).items()
        }
        for name, pred in methods.items()
    }
    return {
        "dataset": dataset,
        "split": split,
        "seed": int(seed),
        "checkpoint": payload["checkpoint"],
        "modelType": payload["modelType"],
        "architectureName": payload["architectureName"],
        "featureMeta": payload["featureMeta"],
        "textMeta": payload["textMeta"],
        "graphMeta": payload["graphMeta"],
        "validationBasis": payload.get("validationBasis"),
        "blendWeight": payload["blendWeight"],
        "alpha": payload["alpha"],
        "residualBand": payload.get("residualBand", 0.0),
        "domainThreshold": payload.get("domainThreshold"),
        "domainRouteShare": payload.get("domainRouteShare", 0.0),
        "testRouteShare": round(float(route_active.float().mean().detach().cpu().item()), 6),
        "meanGamma": round(float(gamma.float().mean().detach().cpu().item()), 6),
        "minGammaObserved": round(float(gamma.float().min().detach().cpu().item()), 6),
        "maxGammaObserved": round(float(gamma.float().max().detach().cpu().item()), 6),
        "limits": payload["limits"],
        "selectiveTable": selective_table,
        "selectiveMeta": sel_meta,
        "selectedCoverageMeta": selected_coverage_meta,
        "deepseekReliabilityMeta": ds_rel_meta,
        "deepseekHarmFirstMeta": harm_meta_out,
        "conformalDeferMeta": conformal_table,
        "metrics": metric_block,
    }


def aggregate(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(f"{row['dataset']}/{row['split']}", []).append(row)

    metric_names = ("RMSE", "MSE", "Spearman", "CI", "worstgrp_R2", "harm_worse")
    cells: dict[str, dict] = {}
    for cell, items in sorted(grouped.items()):
        items = sorted(items, key=lambda item: int(item["seed"]))
        methods = sorted({method for item in items for method in item["metrics"]})
        summary: dict[str, dict] = {}
        for method in methods:
            vals = {
                metric: np.asarray(
                    [item["metrics"].get(method, {}).get(metric, np.nan) for item in items],
                    dtype=float,
                )
                for metric in metric_names
            }
            summary[method] = {
                metric: round(float(np.nanmean(vals[metric])), 4)
                for metric in metric_names
            }
            summary[method].update({
                "RMSE_std": round(float(np.nanstd(vals["RMSE"])), 4),
                "Spearman_std": round(float(np.nanstd(vals["Spearman"])), 4),
                "CI_std": round(float(np.nanstd(vals["CI"])), 4),
                "worstgrp_R2_std": round(float(np.nanstd(vals["worstgrp_R2"])), 4),
                "harm_worse_std": round(float(np.nanstd(vals["harm_worse"])), 4),
            })
        dataset, split = cell.split("/", 1)
        cells[cell] = {
            "dataset": dataset,
            "split": split,
            "nSeeds": len(items),
            "featureMeta": items[0]["featureMeta"],
            "textMeta": items[0]["textMeta"],
            "graphMeta": items[0]["graphMeta"],
            "summary": summary,
            "perSeed": items,
        }
    return {
        "schema": "drugtarget-prism-results-v1",
        "status": "selected_mainline",
        "model": "PRISM",
        "selectedMethod": PRIMARY_METHOD,
        "selectedMethodLabel": METHOD_LABELS[PRIMARY_METHOD],
        "selectedPolicy": "DeepSeek-QC harm-first residual shrink plus validation-selected selective defer",
        "claimScope": "memory-calibrated affinity refiner with train-only GKN prototypes and domain-aware defer",
        "cells": cells,
    }


PRIMARY_METHOD = "deepseek_qc_selective"
LEGACY_PRIMARY_METHOD = "deepseek_harmfirst_selective_cov80"
METHOD_LABELS = {
    "prior_only": "Prior",
    "gkn_refiner": "GKNRefiner",
    "domain_alpha": "DomainAlpha",
    "deepseek_reliable_alpha": "DeepSeekQC",
    "deepseek_harmfirst_alpha": "DeepSeekQCHarmFirst",
    "deepseek_qc_selective": "DeepSeekQCSelective",
    PRIMARY_METHOD: "DeepSeekQCSelective",
    LEGACY_PRIMARY_METHOD: "DeepSeekQCSelectiveCov80",
}
REPORT_METHODS = ["prior_only", "domain_alpha", PRIMARY_METHOD]


def write_report(result: dict, report_path: Path = REPORT) -> None:
    lines = [
        "# PRISM Report",
        "",
        "PRISM mainline: train-only GKN domain prototypes plus a DeepSeek-QC residual "
        "trust audit. DeepSeek is used as an offline quality signal, not as a direct affinity feature.",
        "",
        f"## Selected primary method: `{METHOD_LABELS[PRIMARY_METHOD]}`",
        "",
        "The public method is intentionally compact: residual prediction remains neural, while "
        "DeepSeek-QC calibrates residual trust and validation-selected selective defer. It does "
        "NOT claim that generated mechanism text predicts affinity. The name-only control "
        "collapses to prior-only, so the promoted claim is reliability calibration, not text "
        "enhancement.",
        "",
        "## Results",
        "",
        "| split | text source | prototypes | method | RMSE | MSE | Spearman | CI | worstgrp_R2 | harm_worse | gamma | route |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell, item in result["cells"].items():
        text_source = item["textMeta"].get("mechanismTextSource")
        prototypes = item["graphMeta"]["gkn"]["prototypeCount"]
        gamma = np.mean([row.get("meanGamma", float("nan")) for row in item["perSeed"]])
        route = np.mean([row.get("testRouteShare", float("nan")) for row in item["perSeed"]])
        methods = [m for m in REPORT_METHODS if m in item["summary"]]
        if PRIMARY_METHOD not in item["summary"] and LEGACY_PRIMARY_METHOD in item["summary"]:
            methods.append(LEGACY_PRIMARY_METHOD)
        for method in methods:
            metrics = item["summary"][method]
            marker = " **[SELECTED]**" if method in {PRIMARY_METHOD, LEGACY_PRIMARY_METHOD} else ""
            label = METHOD_LABELS.get(method, method)
            lines.append(
                f"| {cell} | {text_source} | {int(prototypes)} | {label}{marker} | "
                f"{metrics['RMSE']:.4f} | {metrics['MSE']:.4f} | {metrics['Spearman']:.4f} | "
                f"{metrics['CI']:.4f} | {metrics['worstgrp_R2']:.4f} | {metrics['harm_worse']:.4f} | {gamma:.4f} | {route:.4f} |"
            )
        # Per-seed harm for the primary method, so budget compliance is visible without opening JSON.
        primary_key = PRIMARY_METHOD if PRIMARY_METHOD in item["summary"] else LEGACY_PRIMARY_METHOD
        primary_row = item["summary"].get(primary_key)
        if primary_row is not None:
            per_seed_harm = [
                round(float(row["metrics"].get(primary_key, {}).get("harm_worse", float("nan"))), 4)
                for row in item["perSeed"]
            ]
            lines.append("")
            lines.append(f"Per-seed `harm_worse` for `{METHOD_LABELS[primary_key]}` on `{cell}`: {per_seed_harm}")
            lines.append(f"All seeds <= 0.40: **{all(h <= 0.40 for h in per_seed_harm)}**.")
            selected_cov = [
                row.get("selectedCoverageMeta", {}).get("coverage")
                for row in item["perSeed"]
                if row.get("selectedCoverageMeta", {}).get("enabled")
            ]
            if selected_cov:
                lines.append(f"Validation-selected coverages: {[round(float(c), 3) for c in selected_cov]}.")
    lines.extend([
        "",
        "## Rejected claims (kept explicit)",
        "",
        "- Direct DeepSeek mechanism text/profile fusion is NOT supported by controls (name-only "
        "  family identity matched or beat the full mechanism profiles in round 7).",
        "- Name-only family identity alone is NOT a valid mechanism-content claim.",
        "- Conformal hard defer worsened both ranking and harm.",
        "- Global shrink alone (fixed beta) improved mean harm but failed seed 3.",
        "- beta >= 3 over-shrink was unstable because harm_worse conditions on moved samples; "
        "  the current harm-first beta selector caps beta at 2.5.",
        "",
        "## Boundary",
        "",
        "- Outputs are isolated under `outputs/prism/` and `doc/prism-*`.",
        "- This enhanced workflow is CUDA-only; CPU fallback is rejected before train/infer.",
        "- Mechanism graph/prototypes use inner-train targets only; test labels/statistics are final-evaluation only.",
        "- LLM/API calls are not made during train or inference; cached text or public KB fallback is used.",
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate the PRISM selective affinity model")
    parser.add_argument("--stage", choices=["train", "infer", "full"], default="full")
    parser.add_argument("--splits", nargs="*", help="Subset like KIBA/target-cold DAVIS/target-cold")
    parser.add_argument("--seeds", nargs="*", type=int, default=[1])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--plm-source", default="esm2_t6_8M_UR50D")
    parser.add_argument("--morgan-bits", type=int, default=1024)
    parser.add_argument("--smiles-cnn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drug-encoder", choices=["morgan", "chemberta", "morgan-chemberta"],
                        default="morgan", help="drug feature source (default morgan = current baseline)")
    parser.add_argument("--chemberta-model", default="DeepChem/ChemBERTa-77M-MLM")
    parser.add_argument("--drug-cache-path", default=None, help="override dir for cached drug embeddings")
    parser.add_argument("--smiles-max-len", type=int, default=192)
    parser.add_argument("--feature-batch-size", type=int, default=1024)
    parser.add_argument("--feature-seed", type=int, default=19)
    parser.add_argument("--esm-batch-size", type=int, default=8)
    parser.add_argument("--esm-max-len", type=int, default=1022)
    parser.add_argument("--mechanism-source", choices=["llm-cache", "kb"], default="llm-cache")
    parser.add_argument("--text-dim", type=int, default=256)
    parser.add_argument("--max-entities", type=int, default=256)
    parser.add_argument("--min-entity-df", type=int, default=2)
    parser.add_argument("--gkn-hidden", type=int, default=128)
    parser.add_argument("--domain-dim", type=int, default=64)
    parser.add_argument("--prototypes", type=int, default=8)
    parser.add_argument("--gkn-epochs", type=int, default=50)
    parser.add_argument("--projector-epochs", type=int, default=80)
    parser.add_argument("--gkn-lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--val-every", type=int, default=1,
                        help="run validation every N epochs and always on the final epoch (default 1)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--defer-l1", type=float, default=1e-3)
    parser.add_argument("--min-gamma", type=float, default=0.0)
    parser.add_argument("--rank-weight", type=float, default=0.0,
                        help="weight of the pairwise rank-aware auxiliary loss (0 disables)")
    parser.add_argument("--contrast-weight", type=float, default=0.0,
                        help="weight of the cross-modal contrastive auxiliary loss (0 disables)")
    parser.add_argument("--contrast-temperature", type=float, default=0.2)
    parser.add_argument("--ema-decay", type=float, default=0.0,
                        help="EMA decay for refiner weights (0 disables); EMA vs raw chosen by validation MSE")
    parser.add_argument("--deterministic", action="store_true",
                        help="enable cuDNN/CUDA deterministic mode and disable AMP for reproducible runs")
    parser.add_argument("--disable-amp", action="store_true",
                        help="disable AMP without forcing full determinism")
    parser.add_argument("--strict-deterministic", action="store_true",
                        help="error (instead of warn) on nondeterministic ops to diagnose the source")
    # PromptSE-inspired LLM mechanism profile conditioning (all default off = baseline).
    parser.add_argument("--prompt-profile-mode", choices=["off", "target", "family", "target-family"],
                        default="off", help="condition on per-target-family LLM mechanism profiles")
    parser.add_argument("--prompt-profile-cache", default=None, help="optional profile cache dir (recorded)")
    parser.add_argument("--prompt-profile-source", choices=["cached", "deepseek"], default="cached",
                        help="cached = derived-from-summary profiles; deepseek = offline DeepSeek v2 cache")
    parser.add_argument("--prompt-profile-encoder", choices=["hash", "biobert"], default="hash")
    parser.add_argument("--prompt-profile-fusion", choices=["off", "mlp", "mlp-conv", "mlp-conv-attn"],
                        default="off", help="PromptSE+-style fusion of the profile into the refiner")
    parser.add_argument("--prompt-control", choices=["none", "shuffle", "name-only"], default="none",
                        help="ablation control: shuffle permutes family->profile; name-only drops mechanism text")
    parser.add_argument("--hierarchical-gkn", action="store_true",
                        help="HiGCN-style hierarchical edge masking that blocks low-frequency entities")
    parser.add_argument("--higcn-tiers", type=int, default=2)
    parser.add_argument("--pair-affinity-weight", type=float, default=0.0,
                        help="train-only auxiliary direct affinity regression from the cross-attn "
                             "pair representation (forces the fused token state to carry pair-affinity "
                             "signal; 0 disables; never used at inference)")
    parser.add_argument("--mechanism-align-weight", type=float, default=0.0,
                        help="weight of the drug<->family-profile mechanism alignment loss (0 disables)")
    parser.add_argument("--family-calibration", action="store_true",
                        help="record family-aware calibration intent (served by existing domain routing)")
    parser.add_argument("--selective", action="store_true",
                        help="enable selective prediction: validation-only per-family harm risk score; "
                             "at infer, top-K riskiest test pairs are deferred to the prior and "
                             "selective metrics are reported at multiple coverage levels")
    parser.add_argument("--selective-coverages", nargs="*", type=float,
                        default=[1.0, 0.95, 0.90, 0.80],
                        help="coverage levels for selective evaluation (1.0 = no abstention)")
    parser.add_argument("--conformal-defer", action="store_true",
                        help="validation-only per-family residual-quantile defer using DeepSeek-QC reliability")
    parser.add_argument("--harmfirst-beta-override", type=float, default=-1.0,
                        help="analysis-only override for deepseek_harmfirst_alpha beta; negative uses checkpoint")
    parser.add_argument("--llm-prompt-version", default=PROMPT_VERSION_DEFAULT)
    parser.add_argument("--llm-sample-cap", type=int, default=20)
    parser.add_argument("--alpha-harm-guard", type=float, default=0.40,
                        help="max validation harm_worse a routed config may incur (fixed budget)")
    parser.add_argument("--alpha-rank-tol", type=float, default=0.005,
                        help="routed config must hold validation Spearman within this of the prior")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke", action="store_true")
    raw_args = sys.argv[1:]
    args = parser.parse_args()
    supplied = set()
    for token in raw_args:
        if token.startswith("--"):
            supplied.add(token.split("=", 1)[0])

    def has_opt(*names: str) -> bool:
        return any(name in supplied for name in names)

    if os.environ.get("DRUGTARGET_ENTRYPOINT") == "prism":
        # Public selected-line preset. `prism` defaults to the audited
        # DeepSeekQCSelective configuration so users do not have to remember a long
        # ablation command.
        if not has_opt("--plm-source"):
            args.plm_source = "esm2_t30_150M_UR50D"
        if not has_opt("--d-model"):
            args.d_model = 192
        if not has_opt("--layers"):
            args.layers = 3
        if not has_opt("--ff-dim"):
            args.ff_dim = 768
        if not has_opt("--epochs"):
            args.epochs = max(int(args.epochs), 8)
        if not has_opt("--rank-weight") and float(args.rank_weight) == 0.0:
            args.rank_weight = 0.1
        if not has_opt("--contrast-weight") and float(args.contrast_weight) == 0.0:
            args.contrast_weight = 0.05
        if not has_opt("--pair-affinity-weight") and float(args.pair_affinity_weight) == 0.0:
            args.pair_affinity_weight = 0.05
        if not has_opt("--prompt-profile-mode"):
            args.prompt_profile_mode = "family"
        if not has_opt("--prompt-profile-source"):
            args.prompt_profile_source = "deepseek"
        if not has_opt("--prompt-profile-fusion"):
            args.prompt_profile_fusion = "mlp-conv-attn"
        if not has_opt("--prompt-control"):
            args.prompt_control = "none"
        args.family_calibration = True
        args.selective = True
        if not has_opt("--prototypes"):
            args.prototypes = 8
        if not has_opt("--batch-size"):
            args.batch_size = max(int(args.batch_size), 2048)
        if not has_opt("--eval-batch-size"):
            args.eval_batch_size = max(int(args.eval_batch_size), 8192)
        if not has_opt("--val-every"):
            args.val_every = max(int(args.val_every), 2)
    if args.smoke:
        args.d_model = 64
        args.heads = 4
        args.layers = 1
        args.ff_dim = 128
        args.epochs = 1
        args.gkn_epochs = 5
        args.projector_epochs = 10
        args.gkn_hidden = 64
        args.domain_dim = 32
        if args.prompt_profile_source != "deepseek":
            args.prototypes = 4
        args.max_entities = 64
        args.batch_size = min(args.batch_size, 256)
        args.eval_batch_size = min(args.eval_batch_size, 2048)
        args.limit_train = args.limit_train or 1024
        args.limit_val = args.limit_val or 512
        args.limit_test = args.limit_test or 512
    if args.d_model % args.heads != 0:
        raise SystemExit("--d-model must be divisible by --heads")
    return args


def main() -> None:
    args = parse_args()
    cells = parse_cells(args.splits)
    seeds = [int(seed) for seed in args.seeds]
    device = require_cuda_device(args.device)
    args.determinismMeta = configure_determinism(args, device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    if args.determinismMeta.get("deterministic"):
        print(f"[PRISM] deterministic mode: {args.determinismMeta}")

    rows = []
    if args.stage in {"train", "full"}:
        for dataset, split in cells:
            for seed in seeds:
                print(f"[PRISM train] {dataset}/{split}/seed{seed} device={device}")
                run_train_one(dataset, split, seed, args, device)
    if args.stage in {"infer", "full"}:
        bundle_cache: dict[str, object] = {}
        for dataset, split in cells:
            for seed in seeds:
                print(f"[PRISM infer] {dataset}/{split}/seed{seed} device={device}")
                rows.append(run_infer_one(dataset, split, seed, args, device, bundle_cache))
        result = aggregate(rows)
        json_dump(RESULTS, result)
        write_report(result, REPORT)
        print(f"wrote {repo_rel(RESULTS)}")
        print(f"wrote {repo_rel(REPORT)}")


if __name__ == "__main__":
    main()

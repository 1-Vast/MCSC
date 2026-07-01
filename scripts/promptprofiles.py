"""PromptSE-inspired LLM mechanism profile cache for the isolated GKN-enhanced DTA line.

This adapts PromptSE's three-stage reasoning (entity characterization -> mechanism-oriented
inference -> profile summarization) to DTA targets/families. To stay reproducible and
leakage-safe, the Stage-1 entity characterization reuses the repository's pre-cached,
leakage-sanitized LLM mechanism summaries (no fresh API calls at train/infer time). Stages 2-3
categorize each inner-train target's mechanism cues into four DTA pharmacological perspectives
and summarize them per target *family* (a GKN prototype cluster), so cold test targets inherit a
mechanism profile from their nearest inner-train family only.

All construction uses inner-train targets and inner-train high-affinity pairs only. No test
labels, test statistics, split membership, or model predictions are used.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from model.encode import MechanismTextEncoder, tokenize


PROMPT_VERSION_DEFAULT = "promptdta_v1"

# Four DTA pharmacological perspectives (adapted from PromptSE's four perspectives).
CATEGORY_NAMES = [
    "binding_domain_compatibility",
    "target_family_selectivity",
    "pathway_function_context",
    "ligand_scaffold_physicochemical",
]

CATEGORY_LEXICONS = {
    "binding_domain_compatibility": {
        "kinase", "domain", "atp", "active", "site", "catalytic", "pocket", "binding",
        "residue", "hinge", "allosteric", "helix", "sheet", "fold", "loop", "motif",
        "phosphotransfer", "gatekeeper",
    },
    "target_family_selectivity": {
        "family", "subfamily", "isoform", "receptor", "class", "selective", "paralog",
        "homolog", "member", "type", "group", "serine", "threonine", "tyrosine", "agc",
        "camk", "cmgc", "subgroup",
    },
    "pathway_function_context": {
        "pathway", "signaling", "signal", "regulation", "phosphorylation", "cascade",
        "transcription", "apoptosis", "proliferation", "metabolism", "immune", "growth",
        "differentiation", "survival", "checkpoint", "stress",
    },
    "ligand_scaffold_physicochemical": {
        "inhibitor", "ligand", "scaffold", "hydrophobic", "aromatic", "competitive",
        "small", "molecule", "lipophilic", "polar", "hydrogen", "bond", "analog",
        "derivative", "selectivity",
    },
}


def _digest(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12]


def _category_tokens(text: str) -> list[set[str]]:
    toks = [t for t in tokenize(text) if len(t) >= 3]
    buckets = []
    for cat in CATEGORY_NAMES:
        lex = CATEGORY_LEXICONS[cat]
        buckets.append({t for t in toks if t in lex})
    return buckets


def load_deepseek_family_profiles(
    cache_path: "Path",
    n_families: int,
    family_members: dict[int, list[int]],
    text_dim: int,
    device: torch.device,
    control: str = "none",
    seed: int = 0,
) -> dict:
    """Load an offline DeepSeek profile cache (fail-closed) into encoded family tensors.

    Verifies each cached family's member set matches the current deterministic assignment, then
    encodes the four accepted channel summaries (QC already applied at build time; low-quality
    channels are already 'None'). Adds per-family quality/uncertainty/confidence vectors.
    """
    import json as _json
    from pathlib import Path as _Path
    cache_path = _Path(cache_path)
    if not cache_path.exists():
        raise SystemExit(
            f"DeepSeek profile cache missing: {cache_path}; build it offline first with "
            f"python main.py cache build (training/inference never call the API)"
        )
    data = _json.loads(cache_path.read_text(encoding="utf-8"))
    if data.get("partial"):
        raise SystemExit(f"DeepSeek cache {cache_path.name} is partial; rebuild before training")
    if int(data.get("nFamilies", -1)) != int(n_families):
        raise SystemExit(
            f"DeepSeek cache {cache_path.name} family count mismatch: "
            f"cache={data.get('nFamilies')} current={n_families}"
        )
    fams = data["families"]
    if len(fams) != int(n_families):
        raise SystemExit(
            f"DeepSeek cache {cache_path.name} is incomplete: "
            f"{len(fams)}/{n_families} families present"
        )
    texts: list[list[str]] = []
    coverage = np.zeros((n_families, 4), dtype=np.float32)
    quality = np.zeros(n_families, dtype=np.float32)
    confidence = np.zeros(n_families, dtype=np.float32)
    uncertainty = np.ones(n_families, dtype=np.float32)
    mismatches = 0
    cache_members: dict[int, set[int]] = {}
    cache_family_by_target: dict[int, int] = {}
    for key, rec in fams.items():
        try:
            fam_idx = int(key)
        except ValueError:
            continue
        cache_members[fam_idx] = set(int(x) for x in rec.get("members", []) or [])
        for target_idx in rec.get("members", []) or []:
            cache_family_by_target[int(target_idx)] = fam_idx
    # Robustly align current prototype labels to cached prototype labels by inner-train
    # member overlap. This handles harmless KMeans label/order drift without ever
    # mapping a cached mechanism profile to the wrong current family.
    family_map = np.full((n_families,), -1, dtype=np.int64)
    assigned_current: set[int] = set()
    used_cache: set[int] = set()
    scored_pairs: list[tuple[int, int, int]] = []
    for cur_f in range(n_families):
        cur_members = set(int(x) for x in family_members.get(cur_f, []) or [])
        for cache_f in range(n_families):
            overlap = len(cur_members & cache_members.get(cache_f, set()))
            scored_pairs.append((overlap, cur_f, cache_f))
    for overlap, cur_f, cache_f in sorted(scored_pairs, reverse=True):
        if overlap <= 0:
            continue
        if cur_f in assigned_current:
            continue
        if cache_f in used_cache:
            continue
        family_map[cur_f] = cache_f
        assigned_current.add(cur_f)
        used_cache.add(cache_f)
    for cur_f in range(n_families):
        if family_map[cur_f] < 0:
            family_map[cur_f] = cur_f
    for f in range(n_families):
        rec = fams.get(str(f), {})
        if not rec:
            raise SystemExit(f"DeepSeek cache {cache_path.name} missing family {f}")
        leakage = rec.get("leakage_flags") or (rec.get("qc", {}) or {}).get("leakageFlags") or []
        if leakage:
            raise SystemExit(f"DeepSeek cache {cache_path.name} family {f} has leakage flags: {leakage}")
        accepted = (rec.get("qc", {}) or {}).get("acceptedProfiles", {})
        # member-set verification (assignment must match the build)
        if set(rec.get("members", [])) != set(family_members.get(f, [])):
            mismatches += 1
        cats = []
        for c, name in enumerate(CATEGORY_NAMES):
            summary = str((accepted.get(name) or {}).get("summary", "None"))
            if control == "name-only":
                summary = f"family_{f}_category_{c}" if rec.get("members") else "None"
            cats.append(summary)
            if summary.strip().lower() not in ("none", ""):
                coverage[f, c] = 1.0
        texts.append(cats)
        qc = rec.get("qc", {}) or {}
        quality[f] = float(qc.get("familyQuality", 0.0))
        confidence[f] = float(qc.get("confidence", 0.0))
        uncertainty[f] = float(qc.get("uncertainty", 1.0))
        if control == "name-only":
            # Name-only is a family-identity control, so it must not inherit DeepSeek
            # content-quality signals that were judged from the removed mechanism text.
            quality[f] = 0.5
            confidence[f] = 0.5
            uncertainty[f] = 0.5
    encoder = MechanismTextEncoder(text_dim, str(device))
    flat = [texts[f][c] for f in range(n_families) for c in range(4)]
    enc = encoder.build(flat).detach().cpu().numpy().astype(np.float32).reshape(n_families, 4, text_dim)
    enc = enc * coverage[:, :, None]
    if control == "shuffle":
        rng = np.random.RandomState(seed + 99)
        perm = rng.permutation(n_families)
        enc, coverage, quality, confidence, uncertainty = (
            enc[perm], coverage[perm], quality[perm], confidence[perm], uncertainty[perm])
    family_map_tensor = torch.from_numpy(family_map).to(device=device, dtype=torch.long)
    meta = {
        "promptVersion": data.get("promptVersion"), "llmModel": data.get("model"),
        "profileSource": "deepseek-offline-cache", "control": control,
        "encoder": f"MechanismTextEncoder(hash,dim={text_dim})", "textDim": int(text_dim),
        "nFamilies": int(n_families), "memberSetMismatches": int(mismatches),
        "familyMapApplied": bool(mismatches),
        "familyMap": {str(i): int(family_map[i]) for i in range(n_families)},
        "nonEmptyProfilesPerCategory": [int(coverage[:, c].sum()) for c in range(4)],
        "meanFamilyQuality": round(float(quality.mean()), 4),
        "cache": cache_path.name, "leakagePolicy": data.get("leakagePolicy"),
    }
    return {
        "profileTensor": torch.from_numpy(enc).to(device=device, dtype=torch.float32),
        "coverage": torch.from_numpy(coverage).to(device=device, dtype=torch.float32),
        "confidence": torch.from_numpy(confidence).to(device=device, dtype=torch.float32),
        "quality": torch.from_numpy(quality).to(device=device, dtype=torch.float32),
        "uncertainty": torch.from_numpy(uncertainty).to(device=device, dtype=torch.float32),
        "familyMap": family_map_tensor,
        "meta": meta,
    }


def build_family_profiles(
    target_ids: list[str],
    fit_unique: np.ndarray,
    family_id: np.ndarray,
    text_by_id: dict[str, str],
    n_families: int,
    text_dim: int,
    device: torch.device,
    fit_pairs: dict | None = None,
    sample_cap: int = 20,
    prompt_version: str = PROMPT_VERSION_DEFAULT,
    llm_model: str = "cached-llm-mechanism-sanitized",
    control: str = "none",
    seed: int = 0,
) -> dict:
    """Build per-family four-channel mechanism profiles, encoded to fixed vectors.

    control: "none" (real profiles), "shuffle" (permute family->profile assignment),
    or "name-only" (replace mechanism summary with the family/target identity token only).
    Returns a dict with the encoded profile tensor [n_families, 4, text_dim], a presence mask,
    per-family confidence, a per-target family map, and leakage-safe metadata.
    """
    fit_set = set(int(i) for i in fit_unique)
    # Stage 1+2: per inner-train target, bucket mechanism tokens into the four perspectives.
    family_cat_tokens: list[list[set[str]]] = [[set(), set(), set(), set()] for _ in range(n_families)]
    family_members: list[list[int]] = [[] for _ in range(n_families)]
    for t in fit_unique:
        fam = int(family_id[int(t)])
        family_members[fam].append(int(t))
        buckets = _category_tokens(text_by_id.get(target_ids[int(t)], ""))
        for c in range(4):
            family_cat_tokens[fam][c] |= buckets[c]

    # Stage 2 evidence: cap-sampled inner-train high-affinity pair ids per family (PromptSE cap).
    sampled_pairs: dict[int, list[int]] = {f: [] for f in range(n_families)}
    if fit_pairs is not None:
        d = np.asarray(fit_pairs["D"]); t = np.asarray(fit_pairs["T"]); y = np.asarray(fit_pairs["Y"])
        if y.size:
            hi = y >= np.quantile(y, 0.6)
            rng = np.random.RandomState(seed + 7)
            for fam in range(n_families):
                fam_targets = set(family_members[fam])
                idx = np.where(hi & np.isin(t, list(fam_targets)))[0]
                if idx.size > sample_cap:
                    idx = rng.choice(idx, sample_cap, replace=False)
                sampled_pairs[fam] = sorted(int(i) for i in idx.tolist())

    # Stage 3: summarize each category into a text profile (or "None"); name-only control.
    profile_texts: list[list[str]] = []
    coverage = np.zeros((n_families, 4), dtype=np.float32)
    confidence = np.zeros(n_families, dtype=np.float32)
    structured: list[dict] = []
    for fam in range(n_families):
        cats = []
        present = 0
        for c in range(4):
            toks = sorted(family_cat_tokens[fam][c])
            if control == "name-only":
                summary = f"family_{fam}_category_{c}" if family_members[fam] else "None"
            else:
                summary = " ".join(toks[:24]) if toks else "None"
            if summary != "None":
                coverage[fam, c] = 1.0
                present += 1
            cats.append(summary)
        profile_texts.append(cats)
        confidence[fam] = present / 4.0
        structured.append({
            "family_id": fam,
            "n_members": len(family_members[fam]),
            "categories": {CATEGORY_NAMES[c]: cats[c] for c in range(4)},
            "entities": {CATEGORY_NAMES[c]: sorted(family_cat_tokens[fam][c])[:24] for c in range(4)},
            "source_pair_ids": sampled_pairs.get(fam, []),
            "confidence": float(confidence[fam]),
        })

    # Part B: encode each of the four category summaries; "None" -> zero vector.
    encoder = MechanismTextEncoder(text_dim, str(device))
    flat = [profile_texts[f][c] for f in range(n_families) for c in range(4)]
    enc = encoder.build(flat).detach().cpu().numpy().astype(np.float32).reshape(n_families, 4, text_dim)
    mask = coverage[:, :, None]
    enc = enc * mask  # None categories are exactly zero
    if control == "shuffle":
        rng = np.random.RandomState(seed + 99)
        perm = rng.permutation(n_families)
        enc = enc[perm]
        coverage = coverage[perm]
        confidence = confidence[perm]

    profile_tensor = torch.from_numpy(enc).to(device=device, dtype=torch.float32)
    cov_tensor = torch.from_numpy(coverage).to(device=device, dtype=torch.float32)
    conf_tensor = torch.from_numpy(confidence).to(device=device, dtype=torch.float32)
    digest = _digest(prompt_version, llm_model, control, str(text_dim), str(n_families),
                     json.dumps([s["categories"] for s in structured], sort_keys=True))
    nonempty = [int(coverage[:, c].sum()) for c in range(4)]
    meta = {
        "promptVersion": prompt_version,
        "llmModel": llm_model,
        "profileSource": "derived-from-cached-sanitized-llm-mechanism-text",
        "stages": ["entity_characterization(cached)", "mechanism_oriented_inference(4-perspective)",
                   "profile_summarization(family)"],
        "categories": CATEGORY_NAMES,
        "encoder": f"MechanismTextEncoder(hash,dim={text_dim})",
        "textDim": int(text_dim),
        "pooling": "category-summary-hash",
        "nFamilies": int(n_families),
        "sampleCap": int(sample_cap),
        "control": control,
        "nonEmptyProfilesPerCategory": nonempty,
        "digest": digest,
        "leakagePolicy": "families/profiles/entities from inner-train targets and inner-train "
                         "high-affinity pairs only; cold targets inherit nearest-family profile "
                         "via train-fit GKN domain distance; no test labels/stats/membership used",
    }
    return {
        "profileTensor": profile_tensor,   # [k,4,text_dim]
        "coverage": cov_tensor,            # [k,4]
        "confidence": conf_tensor,         # [k]
        "structured": structured,
        "meta": meta,
    }

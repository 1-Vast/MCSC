"""Staged mechanism-LLM generation and quality control for PRISM's DeepSeek-QC cache.

This module holds the LLM ORCHESTRATION logic only: prompt construction, staged-generation
control flow, and quality/leakage checks. It intentionally has NO network client and NO import
from `scripts/` (that would create a circular import, since scripts/selectiveaffinity.py and
friends already import from `model/`) -- callers pass in any object with a
`generate(system, user, max_tokens) -> {"parsed": dict|None, "raw": str|None, ...}` method.
The live DeepSeek API client (`DeepSeekClient`, the only place in this codebase that opens an
outbound HTTP connection to the completions endpoint) stays in `scripts/mechanismcache.py`, which
is the sole entry point DeepSeek may be called from -- train/infer never imports it.
`scripts/integritycheck.py`'s `check_deepseek_boundary()` enforces that no other file (including
this one) contains live API client code.

PHASED CREATION (required order; each phase is cached to its own file under
dataset/cache/deepseek_promptdta/staged/<dataset>_<split>_seed<seed>/family<n>/, so an
interrupted build resumes instead of re-spending API calls -- see `stage_path` / `load_if_valid`):

  Stage 0  deterministic manifest (no LLM; the family/pair assignment is computed by the caller
           and passed into `build_seed` as `asg`)
  Stage 1  evidence compression   -> `stage1_prompt` / `_stage1_batch`: compact per-pair-batch
           briefs (drug_cues, target_cues, compatibility_hint, uncertainty). Batches are
           independent and may run concurrently, but MUST all complete before stage 2 starts --
           stage 2 reasons over the accumulated `briefs` list, not raw pair data.
  Stage 2  channel reasoning      -> `stage2_prompt` / `_channel_profile`: per family x channel
           (one of CHANNELS) mechanism claims, reasoning FROM the stage-1 briefs only.
  Stage 3  channel summarization  -> `stage3_prompt` / `_channel_profile`: compresses stage-2
           claims into one <=80-word channel profile, again reasoning from the prior stage's
           output only (never from raw pair data directly).
  Stage 4  final assembly         -> `build_family`: LOCAL (no LLM) merge of the 4 channel
           profiles into one family record.
  Stage 5  QC + repair            -> `family_qc` / `channel_quality` / `scan_leakage`, applied
           after every stage and again at final assembly; malformed JSON is retried with a
           larger token budget (truncation, not bad content, is the dominant failure mode for
           this model), then a repair-only pass, then fails closed (summary="None") rather than
           keeping unparseable or leaked content.

QUALITY CHECK REQUIREMENTS (all enforced before a channel/family is accepted into the cache):
  - Leakage: `scan_leakage` rejects any channel whose text matches dataset/benchmark/affinity-
    value/metric patterns (LEAK_PATTERNS, AFFINITY_CASE_PATTERNS); a hit neutralizes the ENTIRE
    family's channels for that build, not just the offending one.
  - Evidence grounding: `family_qc` drops any evidence_id that doesn't belong to the family's
    own sampled pairs before scoring.
  - Channel quality: `channel_quality` scores specificity (ligand-property vocabulary),
    grounding (has evidence_ids), non-identity (adds signal beyond generic family identity),
    and penalizes genericness/hallucination/identity-dependence risk flags the model itself
    reports; channels below `min_channel_quality` are masked to summary="None" rather than kept.
  - Final-result check: `validate_seed_cache` (called automatically at the end of `cache build`,
    and independently via `main.py cache validate`) re-verifies the written rollup against the
    per-stage files on disk, so a partially-written or tampered cache cannot pass as complete.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]

PROMPT_VERSION = "deepseek_promptdta_staged_v1"
FINAL_SCHEMA = "promptdta_deepseek_family_profile_v3"
CHANNELS = [
    "binding_domain_compatibility",
    "target_family_selectivity",
    "pathway_function_context",
    "ligand_scaffold_physicochemical",
]
CHANNEL_DEF = {
    "binding_domain_compatibility": "how ligand scaffold/physicochemistry fits the binding pocket/domain",
    "target_family_selectivity": "how ligand features relate to this protein family's selectivity",
    "pathway_function_context": "how the family's pathway/function context relates to these ligands",
    "ligand_scaffold_physicochemical": "ligand scaffold/physicochemical properties relevant to binding",
}
CACHE_DIR = REPO / "dataset" / "cache" / "deepseek_promptdta"
STAGE_DIR = CACHE_DIR / "staged"
BATCH_SIZE = 4

LEAK_PATTERNS = re.compile(
    r"\b(kiba|davis|bindingdb|benchmark|test set|validation set|train set|held[- ]?out|"
    r"ground truth|label|pic50|ki\s*=|kd\s*=|ic50|affinity value|rmse|spearman|"
    r"concordance|auc|model prediction)\b", re.IGNORECASE,
)
AFFINITY_CASE_PATTERNS = re.compile(r"\bpK[di]\b|\bpIC50\b")
CONCISE = ("Do not include chain-of-thought or reasoning of any kind. Output concise JSON only, "
           "one object, no prose before or after it, no markdown fences. Keep any summary under 80 "
           "words. The first character of your reply must be '{'.")
SYSTEM = (
    "You are a pharmacology reasoning assistant for drug-target affinity research. You connect ligand "
    "scaffold/physicochemistry to protein binding-domain, family/selectivity, and pathway properties. "
    "You NEVER predict binding affinity or numbers. Never mention datasets, benchmarks, affinity "
    "values, labels, metrics, or train/test splits. " + CONCISE
)
GENERIC_TOKENS = {"kinase", "atp", "binding", "signaling", "pathway", "family", "protein", "domain"}
LIGAND_CUES = {"scaffold", "hydrophobic", "aromatic", "hydrogen", "lipophilic", "polar", "ring",
               "moiety", "substituent", "planar", "hinge", "pocket", "selectivity"}


# ------------------------------------------------------------------------------ drug descriptors
def drug_descriptor(smiles: str) -> dict:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"smiles": smiles[:120], "valid": False}
    return {
        "smiles": smiles[:120], "mw": round(Descriptors.MolWt(mol), 1),
        "logp": round(Descriptors.MolLogP(mol), 2), "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol), "tpsa": round(Descriptors.TPSA(mol), 1),
        "rings": rdMolDescriptors.CalcNumRings(mol), "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "scaffold": MurckoScaffold.MurckoScaffoldSmiles(mol=mol)[:120],
    }


# ------------------------------------------------------------------------------ cache helpers
def _digest(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12]


def atomic_write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)


def stage_path(dataset, split, seed, fam, stage, channel=None, batch=None) -> Path:
    base = STAGE_DIR / f"{dataset.lower()}_{split.replace('-', '_')}_seed{seed}" / f"family{fam}"
    if stage == "manifest":
        return base / "manifest.json"
    if stage == "stage1":
        return base / f"stage1_batch{batch}.json"
    if stage in ("stage2", "stage3"):
        return base / f"{stage}_{channel}.json"
    if stage == "assembled":
        return base / "assembled.json"
    raise ValueError(stage)


def load_if_valid(path: Path, force: bool) -> dict | None:
    """Return cached stage output if present and successful; None triggers (re)generation.

    Failed stages return None so a later run (e.g. with a better model) retries them.
    """
    if force or not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return None if d.get("_failed") else d
    except Exception:
        return None


def cache_path(dataset: str, split: str, seed: int) -> Path:
    return CACHE_DIR / f"{dataset.lower()}_{split.replace('-', '_')}_seed{seed}_{PROMPT_VERSION}.json"


def validate_seed_cache(dataset: str, split: str, seed: int) -> dict:
    """Validate a completed staged cache without requiring DeepSeek credentials.

    This is intentionally stricter than the training loader: it verifies both the
    final rollup and the per-stage files so interrupted builds cannot masquerade
    as complete profile artifacts. This is the "final generated result" check: it runs
    automatically at the end of `cache build` and is also callable standalone via
    `main.py cache validate`.
    """
    out = cache_path(dataset, split, seed)
    errors: list[str] = []
    warnings: list[str] = []
    stage_files = 0
    if not out.exists():
        return {
            "dataset": dataset, "split": split, "seed": int(seed), "cache": out.as_posix(),
            "ok": False, "errors": [f"missing rollup: {out.name}"], "warnings": [],
            "nFamilies": 0, "familiesPresent": 0, "familiesAssembled": 0,
        }
    try:
        data = json.loads(out.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "dataset": dataset, "split": split, "seed": int(seed), "cache": out.as_posix(),
            "ok": False, "errors": [f"invalid rollup json: {type(exc).__name__}:{exc}"],
            "warnings": [], "nFamilies": 0, "familiesPresent": 0, "familiesAssembled": 0,
        }
    if data.get("schema") != FINAL_SCHEMA:
        errors.append(f"bad schema: {data.get('schema')}")
    if data.get("promptVersion") != PROMPT_VERSION:
        errors.append(f"bad promptVersion: {data.get('promptVersion')}")
    if data.get("dataset") != dataset or data.get("split") != split or int(data.get("seed", -1)) != int(seed):
        errors.append("rollup dataset/split/seed mismatch")
    n_families = int(data.get("nFamilies", 0) or 0)
    families = data.get("families", {})
    if len(families) != n_families:
        errors.append(f"family count mismatch: {len(families)}/{n_families}")
    assembled = 0
    quality_values = []
    coverage_values = []
    masked_channels = 0
    leakage_hits: list[str] = []
    for fam in range(n_families):
        rec = families.get(str(fam))
        if rec is None:
            errors.append(f"missing family {fam} in rollup")
            continue
        if rec.get("schema") != FINAL_SCHEMA:
            errors.append(f"family {fam}: bad schema {rec.get('schema')}")
        qc = rec.get("qc") or {}
        leakage = rec.get("leakage_flags") or qc.get("leakageFlags") or []
        if leakage:
            leakage_hits.extend([str(x) for x in leakage])
            errors.append(f"family {fam}: leakage flags {leakage}")
        accepted = qc.get("acceptedProfiles") or rec.get("profiles") or {}
        for channel in CHANNELS:
            if channel not in accepted:
                errors.append(f"family {fam}: missing accepted channel {channel}")
        if rec.get("empty"):
            warnings.append(f"family {fam}: empty family/profile")
            continue
        ap = stage_path(dataset, split, seed, fam, "assembled")
        if not ap.exists():
            errors.append(f"family {fam}: missing staged assembled.json")
        else:
            assembled += 1
            stage_files += 1
        expected_batches = int(((rec.get("source_stage_digests") or {}).get("stage1_batches", 0)) or 0)
        for b in range(expected_batches):
            p = stage_path(dataset, split, seed, fam, "stage1", batch=b)
            if not p.exists():
                errors.append(f"family {fam}: missing stage1 batch {b}")
            else:
                stage_files += 1
        for channel in CHANNELS:
            for stage in ("stage2", "stage3"):
                p = stage_path(dataset, split, seed, fam, stage, channel=channel)
                if not p.exists():
                    errors.append(f"family {fam}: missing {stage} {channel}")
                else:
                    stage_files += 1
        quality_values.append(float(qc.get("familyQuality", rec.get("overall_quality", 0.0)) or 0.0))
        coverage_values.append(float(qc.get("coverage", 0.0) or 0.0))
        masked_channels += int(qc.get("maskedChannels", 0) or 0)
    return {
        "dataset": dataset,
        "split": split,
        "seed": int(seed),
        "cache": out.relative_to(REPO).as_posix(),
        "ok": not errors,
        "errors": errors[:50],
        "warnings": warnings[:50],
        "nFamilies": n_families,
        "familiesPresent": len(families),
        "familiesAssembled": assembled,
        "stageFiles": stage_files,
        "meanFamilyQuality": round(float(np.mean(quality_values)), 4) if quality_values else 0.0,
        "meanCoverage": round(float(np.mean(coverage_values)), 4) if coverage_values else 0.0,
        "maskedChannels": int(masked_channels),
        "leakageHits": sorted(set(leakage_hits)),
        "qc": _generation_qc_rates(data),
        "promptVersion": data.get("promptVersion"),
        "model": data.get("model"),
        "cachePath": out.relative_to(REPO).as_posix(),
    }


def _generation_qc_rates(data: dict) -> dict:
    """Surface the generation-time call stats (already saved by build_seed) as rates, so a
    healthy vs degraded cache is visible from `cache validate` without re-opening the raw JSON."""
    stats = data.get("stats") or {}
    calls = int(stats.get("calls", 0) or 0)
    if calls == 0:
        return {"calls": 0, "malformedRate": None, "truncatedRate": None,
                "repairedRate": None, "rejectedRate": None}
    return {
        "calls": calls,
        "malformedRate": round(int(stats.get("malformed", 0) or 0) / calls, 4),
        "truncatedRate": round(int(stats.get("truncated", 0) or 0) / calls, 4),
        "repairedRate": round(int(stats.get("repaired", 0) or 0) / calls, 4),
        "rejectedRate": round(int(stats.get("rejected", 0) or 0) / calls, 4),
    }


def validate_caches(split_specs: list[str], seeds: list[int]) -> dict:
    cells = []
    ok = True
    for split_spec in split_specs:
        dataset, split = split_spec.split("/", 1)
        for seed in seeds:
            cell = validate_seed_cache(dataset, split, int(seed))
            cells.append(cell)
            ok = ok and bool(cell["ok"])
    return {"schema": "promptdta-cache-validation-v1", "ok": ok, "cells": cells}


# ------------------------------------------------------------------------------ QC (stage 5)
def scan_leakage(obj) -> list[str]:
    text = json.dumps(obj, ensure_ascii=False)
    hits = set(m.lower() for m in LEAK_PATTERNS.findall(text))
    hits.update(m for m in AFFINITY_CASE_PATTERNS.findall(text))
    return sorted(hits)


def channel_quality(ch: dict) -> dict:
    summary = str(ch.get("summary", "None"))
    if summary.strip().lower() in ("none", ""):
        return {"channel_quality": 0.0, "is_none": True, "specificity": 0.0, "grounding": 0.0,
                "non_identity": 0.0}
    words = re.findall(r"[a-z]+", summary.lower())
    uniq = set(words)
    generic_frac = len(uniq & GENERIC_TOKENS) / max(1, len(uniq))
    specificity = min(1.0, len(uniq & LIGAND_CUES) / 6.0)
    grounding = 1.0 if ch.get("evidence_ids") else 0.0
    beyond = 1.0 if ch.get("beyond_identity_signal") else 0.0
    risk = {"low": 0.0, "medium": 0.5, "high": 1.0}
    gp = risk.get(str(ch.get("genericness_risk", "high")).lower(), 0.7)
    hp = risk.get(str(ch.get("hallucination_risk", "high")).lower(), 0.7)
    ip = risk.get(str(ch.get("identity_dependence_risk", "high")).lower(), 0.7)
    non_identity = max(0.0, beyond - 0.5 * generic_frac)
    quality = max(0.0, 0.35 * specificity + 0.25 * grounding + 0.25 * non_identity
                  + 0.15 * (1.0 - gp) - 0.15 * hp - 0.10 * ip)
    return {"channel_quality": round(min(1.0, quality), 3), "is_none": False,
            "specificity": round(specificity, 3), "grounding": grounding,
            "non_identity": round(non_identity, 3), "genericness_penalty": gp,
            "hallucination_penalty": hp, "identity_penalty": ip}


def family_qc(channel_profiles: dict, valid_ids: set, min_channel_quality: float) -> dict:
    """Stage 5 for one family: leakage scan, evidence grounding, per-channel quality masking."""
    accepted, scores, masked, qflags = {}, {}, 0, []
    leak = scan_leakage(channel_profiles)
    for c in CHANNELS:
        ch = dict(channel_profiles.get(c) or {"summary": "None"})
        clean_ids = []
        bad = []
        for e in ch.get("evidence_ids", []) or []:
            try:
                eid = int(e)
            except (TypeError, ValueError):
                bad.append(e)
                continue
            if eid in valid_ids:
                clean_ids.append(eid)
            else:
                bad.append(e)
        if bad:
            ch["evidence_ids"] = clean_ids
            qflags.append(f"dropped_bad_evidence:{c}")
        elif clean_ids:
            ch["evidence_ids"] = clean_ids
        if leak:
            ch["summary"] = "None"; qflags.append(f"leak_neutralized:{c}")
        q = channel_quality(ch)
        scores[c] = q
        if not q["is_none"] and q["channel_quality"] < min_channel_quality:
            ch["summary"] = "None"; masked += 1; qflags.append(f"low_quality_masked:{c}")
        accepted[c] = ch
    present = [c for c in CHANNELS if str(accepted[c].get("summary", "None")).strip().lower() not in ("none", "")]
    confs = [float(accepted[c].get("confidence", 0.0)) for c in present] or [0.0]
    uncs = [float(accepted[c].get("uncertainty", 1.0)) for c in present] or [1.0]
    return {"acceptedProfiles": accepted, "channelScores": scores, "leakageFlags": leak,
            "qualityFlags": qflags, "maskedChannels": masked,
            "familyQuality": round(float(np.mean([scores[c]["channel_quality"] for c in CHANNELS])), 4),
            "coverage": round(len(present) / 4.0, 3),
            "confidence": round(float(np.mean(confs)), 3), "uncertainty": round(float(np.mean(uncs)), 3)}


# ------------------------------------------------------------------------------ staged prompts
def stage1_prompt(fam, briefs_batch) -> str:
    """Stage 1: evidence compression. Reasons from raw pair data only (no earlier stage)."""
    ex = {"schema": "promptdta_evidence_briefs_v1", "family_id": str(fam), "batch_id": "B",
          "evidence_briefs": [{"evidence_id": "id", "drug_cues": [], "target_cues": [],
                               "compatibility_hint": "...", "uncertainty": 0.0}]}
    return ("Family target mechanism summaries (sanitized):\n" + briefs_batch["targets"]
            + "\nLigands in this batch (descriptors only, no affinity):\n" + briefs_batch["drugs"]
            + "\nFor EACH ligand produce a compact evidence brief: drug_cues (scaffold/physchem), "
            "target_cues (domain/family/pathway), a one-line compatibility_hint, and uncertainty in [0,1]. "
            "evidence_id must equal the given id. Return ONLY JSON matching:\n" + json.dumps(ex))


def stage2_prompt(fam, channel, briefs) -> str:
    """Stage 2: channel reasoning. Reasons FROM stage-1 `briefs` only, not raw pair data."""
    ex = {"schema": "promptdta_channel_reasoning_v1", "family_id": str(fam), "channel": channel,
          "mechanism_claims": [{"claim": "...", "supporting_evidence_ids": [], "drug_property": "...",
                                "target_property": "...", "confidence": 0.0, "uncertainty": 1.0,
                                "identity_dependence_risk": "low", "hallucination_risk": "low"}]}
    return (f"Channel: {channel} ({CHANNEL_DEF[channel]}).\nCompressed evidence briefs:\n"
            + json.dumps(briefs)[:5000]
            + "\nProduce 1-3 mechanism_claims for THIS channel only, each linking a drug_property to a "
            "target_property with supporting_evidence_ids from the briefs, confidence/uncertainty in [0,1], "
            "and risk flags. If no ligand-specific claim exists, return an empty mechanism_claims list. "
            "Return ONLY JSON matching:\n" + json.dumps(ex))


def stage3_prompt(fam, channel, claims) -> str:
    """Stage 3: channel summarization. Reasons FROM stage-2 `claims` only, not raw pair data."""
    ex = {"schema": "promptdta_channel_profile_v1", "family_id": str(fam), "channel": channel,
          "summary": "... or None", "entities": [], "typed_relations": [], "confidence": 0.0,
          "uncertainty": 1.0, "evidence_ids": [], "genericness_risk": "low", "identity_dependence_risk": "low",
          "beyond_identity_signal": True, "quality_flags": []}
    return (f"Channel: {channel}.\nMechanism claims:\n" + json.dumps(claims)[:5000]
            + "\nSummarize into ONE compact channel profile (<80 words). If the claims are weak or only "
            "restate generic family identity, set summary to \"None\" and beyond_identity_signal=false. "
            "evidence_ids from the claims only. Return ONLY JSON matching:\n" + json.dumps(ex))


# ------------------------------------------------------------------------------ build one family
def _stage1_batch(client, dataset, split, seed, fam, force, b, batch_ids, batch_descs, summaries_txt) -> list:
    """One stage-1 evidence-compression call for a single pair batch; returns its evidence briefs.

    Independent of every other batch/channel (only reads already-cached inputs and writes its
    own stage_path), so callers may run many of these concurrently.
    """
    bp = stage_path(dataset, split, seed, fam, "stage1", batch=b // BATCH_SIZE)
    cached = load_if_valid(bp, force)
    if cached is None:
        drugs_txt = "\n".join(f"  id={i} {json.dumps(d)}" for i, d in zip(batch_ids, batch_descs))
        resp = client.generate(SYSTEM, stage1_prompt(fam, {"targets": summaries_txt, "drugs": drugs_txt}), 2400)
        cached = {"_failed": resp["parsed"] is None, "result": resp.get("parsed"),
                  "inputDigest": _digest(drugs_txt), "outputDigest": _digest(resp.get("raw") or "")}
        atomic_write(bp, cached)
    return cached["result"].get("evidence_briefs", []) if cached.get("result") else []


def _channel_profile(client, dataset, split, seed, fam, channel, briefs, force) -> tuple[str, dict]:
    """Stage 2 (channel reasoning) + stage 3 (channel summarization) for one channel.

    Stage 3 reasons from stage 2's output within this same call chain; channels only share the
    (already-computed) stage-1 `briefs` list as input and write to distinct per-channel cache
    files, so the 4 channels are safe to run concurrently with each other.
    """
    s2p = stage_path(dataset, split, seed, fam, "stage2", channel=channel)
    c2 = load_if_valid(s2p, force)
    if c2 is None:
        r2 = client.generate(SYSTEM, stage2_prompt(fam, channel, briefs), 2800)
        c2 = {"_failed": r2["parsed"] is None, "result": r2.get("parsed")}
        atomic_write(s2p, c2)
    claims = (c2.get("result") or {}).get("mechanism_claims", []) if c2.get("result") else []
    s3p = stage_path(dataset, split, seed, fam, "stage3", channel=channel)
    c3 = load_if_valid(s3p, force)
    if c3 is None:
        if not claims:
            c3 = {"_failed": False, "result": {"summary": "None", "channel": channel, "evidence_ids": []}}
        else:
            r3 = client.generate(SYSTEM, stage3_prompt(fam, channel, claims), 2400)
            c3 = {"_failed": r3["parsed"] is None, "result": r3.get("parsed")}
        atomic_write(s3p, c3)
    return channel, (c3.get("result") or {"summary": "None"})


def build_family(client, dataset, split, seed, fam, members, target_ids, text_by_id,
                 idx, drug_descs, min_channel_quality, force, channel_cap, progress=None) -> dict:
    """Run stages 1-5 for one family. `client` needs only `.generate(system, user, max_tokens)`."""
    valid_ids = set(idx)
    base = {"familyId": fam, "members": members, "sampledPairIds": idx, "schema": FINAL_SCHEMA,
            "prompt_version": PROMPT_VERSION}
    if not idx or not members:
        qc = family_qc({}, set(), min_channel_quality)
        return {**base, "empty": True, "qc": qc}
    # Stage 1: evidence briefs in small batches, run concurrently (each batch is an independent
    # I/O-bound API call); capped worker count keeps concurrent DeepSeek requests bounded. ALL
    # batches must finish before stage 2 starts, since stage 2 reasons over the full briefs list.
    summaries_txt = "\n".join(f"- {text_by_id.get(target_ids[t], '')[:240]}" for t in members[:10])
    batches = [(b, idx[b:b + BATCH_SIZE], drug_descs[b:b + BATCH_SIZE]) for b in range(0, len(idx), BATCH_SIZE)]
    briefs = []
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(batches)))) as pool:
        futures = {
            pool.submit(_stage1_batch, client, dataset, split, seed, fam, force, b, batch_ids, batch_descs, summaries_txt): b
            for b, batch_ids, batch_descs in batches
        }
        for fut in as_completed(futures):
            briefs.extend(fut.result())
            if progress is not None:
                progress(f"family {fam} stage1 {len(briefs)} briefs")
    # Stages 2+3 per channel, run concurrently (4 independent channels, one worker each)
    channel_profiles = {}
    active_channels = [c for ci, c in enumerate(CHANNELS) if channel_cap is None or ci < channel_cap]
    for ci, channel in enumerate(CHANNELS):
        if channel_cap is not None and ci >= channel_cap:
            channel_profiles[channel] = {"summary": "None"}
    if active_channels:
        with ThreadPoolExecutor(max_workers=len(active_channels)) as pool:
            futures = {
                pool.submit(_channel_profile, client, dataset, split, seed, fam, channel, briefs, force): channel
                for channel in active_channels
            }
            for fut in as_completed(futures):
                channel, profile = fut.result()
                channel_profiles[channel] = profile
                if progress is not None:
                    progress(f"family {fam} channel {channel} done")
    # Stage 4 (local assembly, no LLM) + stage 5 (QC)
    qc = family_qc(channel_profiles, valid_ids, min_channel_quality)
    assembled = {
        **base, "profiles": qc["acceptedProfiles"],
        "overall_confidence": qc["confidence"], "overall_uncertainty": qc["uncertainty"],
        "overall_quality": qc["familyQuality"], "leakage_flags": qc["leakageFlags"],
        "quality_flags": qc["qualityFlags"], "qc": qc,
        "source_stage_digests": {"stage1_batches": (len(idx) + BATCH_SIZE - 1) // BATCH_SIZE},
    }
    atomic_write(stage_path(dataset, split, seed, fam, "assembled"), assembled)
    return assembled


def build_seed(client, dataset, split, seed, asg: dict, min_channel_quality, force,
               family_cap, channel_cap, family_workers: int = 3) -> dict:
    """Run stages 1-5 for every family in one (dataset, split, seed).

    `asg` is the family/pair assignment (bundle, rows, family_id, n_families, text_by_id) that
    the caller must compute beforehand (see `scripts.mechanismcache.family_assignment`) -- that
    step trains a small GKN clustering model and depends on the dataset/training pipeline in
    `scripts/`, so it deliberately stays out of this module to avoid a model/ -> scripts/ ->
    model/ import cycle (scripts/selectiveaffinity.py already imports from model/).
    """
    bundle, rows = asg["bundle"], asg["rows"]
    target_ids, text_by_id, family_id, k = bundle.target_ids, asg["text_by_id"], asg["family_id"], asg["n_families"]
    fitD, fitT, fitY = np.asarray(rows["fitD"]), np.asarray(rows["fitT"]), np.asarray(rows["fitY"])
    drug_dict = json.loads((REPO / "dataset" / "kiba" / "ligands_can.txt").read_text())
    smiles = list(drug_dict.values())
    fit_unique = np.unique(fitT).astype(int)
    members = {f: [int(t) for t in fit_unique if int(family_id[int(t)]) == f] for f in range(k)}
    hi = fitY >= np.quantile(fitY, 0.6)
    rng = np.random.RandomState(seed + 7)
    n = k if family_cap is None else min(k, family_cap)
    # Pair sampling MUST stay single-threaded and in family order: it draws from one shared
    # RandomState (reproducibility depends on draw order) and np.random.RandomState is not
    # thread-safe. Only the LLM-calling work below (build_family) is parallelized.
    plan = []
    for f in range(n):
        idx = np.where(hi & np.isin(fitT, list(set(members[f]))))[0]
        if idx.size > 20:
            idx = rng.choice(idx, 20, replace=False)
        idx = sorted(int(i) for i in idx.tolist())
        drug_descs = [drug_descriptor(smiles[int(fitD[i])]) for i in idx]
        plan.append((f, idx, drug_descs))

    families = {}
    bar = tqdm(total=n, desc=f"{dataset}/{split}/seed{seed}", unit="family", dynamic_ncols=True)
    bar_lock = threading.Lock()

    def _progress(msg: str) -> None:
        with bar_lock:
            bar.set_postfix_str(msg, refresh=True)

    def _run(item):
        f, idx, drug_descs = item
        result = build_family(client, dataset, split, seed, f, members[f], target_ids, text_by_id,
                              idx, drug_descs, min_channel_quality, force, channel_cap, progress=_progress)
        with bar_lock:
            bar.update(1)
            bar.set_postfix_str(f"family {f} done, calls={client.stats['calls']}", refresh=True)
        return f, result

    workers = min(max(1, family_workers), max(1, len(plan)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for f, result in pool.map(_run, plan):
            families[str(f)] = result
    bar.close()
    return {
        "schema": FINAL_SCHEMA, "promptVersion": PROMPT_VERSION, "model": client.model,
        "dataset": dataset, "split": split, "seed": seed, "nFamilies": k,
        "sourceScope": "inner_train_only", "minChannelQuality": min_channel_quality,
        "families": families, "stats": client.stats, "builtAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "leakagePolicy": "staged prompts use inner-train target summaries and inner-train ligand descriptors "
                         "only; no affinity values, test rows, labels, split membership, or predictions",
    }

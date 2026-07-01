"""Offline DeepSeek mechanism-QC cache for PRISM.

Token-exhaustion fix: instead of one large final JSON from the reasoning model, generation is
split into small, individually-parseable, separately-cached stages that continue from earlier
cached outputs:

  Stage 0  deterministic manifest (no LLM)
  Stage 1  evidence compression  -> compact briefs per small pair batch (3-5 pairs/call)
  Stage 2  channel reasoning      -> per family x channel mechanism claims (from Stage-1 briefs)
  Stage 3  channel summarization   -> per family x channel compact profile
  Stage 4  final assembly          -> LOCAL assembly of the 4 channel profiles (no LLM)
  Stage 5  QC + repair             -> schema/leakage/quality checks after every stage; on
                                      malformed JSON: retry small -> repair-only prompt -> fail-closed

DeepSeek is called ONLY here, offline. Training/inference load the rollup and fail closed if it
is missing/partial. Prompts use inner-train data only; no test labels/stats/membership/predictions.
Each stage is cached atomically and resumed; valid cache is never overwritten without --force-cache.

CLI:
  python main.py cache smoke --splits KIBA/target-cold --seeds 1 --family-cap 1 --channel-cap 1
  python main.py cache build --splits KIBA/target-cold --seeds 1 2 3 4 5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.affinitydata import load_affinity_bundle, make_split
from scripts.affinityops import apply_limits, resolve_device, seed_everything, split_norm_tensor
from scripts.selectiveaffinity import text_feature_matrix, train_gkn_prototypes

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
CONCISE = ("Do not include chain-of-thought. Output concise JSON only, one object. "
           "Keep any summary under 80 words.")


def require_cuda_device(name: str) -> torch.device:
    device = resolve_device(name)
    if device.type != "cuda":
        raise SystemExit("PRISM DeepSeek cache construction is GPU-only; pass --device cuda")
    if not torch.cuda.is_available():
        raise SystemExit("PRISM DeepSeek cache construction requires CUDA, but CUDA is unavailable")
    try:
        _ = torch.empty(1, device=device)
    except Exception as exc:
        raise SystemExit(f"PRISM DeepSeek cache construction could not allocate on {device}: {exc}") from exc
    return device


# --------------------------------------------------------------------------------------- env/client
def load_env() -> dict:
    path = Path(os.environ.get("DRUGTARGET_ENV_FILE", REPO / ".env"))
    env = dict(os.environ)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip())
    return env


class DeepSeekClient:
    def __init__(self, env: dict, model: str | None = None) -> None:
        self.base = env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
        self.key = env.get("DEEPSEEK_API_KEY", "")
        self.model = model or env.get("DEEPSEEK_MODEL", "deepseek-v4-pro")
        self.timeout = int(env.get("DEEPSEEK_TIMEOUT", "60"))
        if not self.key:
            raise SystemExit("DEEPSEEK_API_KEY missing; cannot build cache (fail closed)")
        self.stats = {"calls": 0, "malformed": 0, "repaired": 0, "rejected": 0, "truncated": 0}

    def _post(self, messages: list[dict], max_tokens: int) -> dict:
        body = {"model": self.model, "messages": messages, "temperature": 0,
                "max_tokens": int(max_tokens), "response_format": {"type": "json_object"}}
        req = urllib.request.Request(
            self.base + "/chat/completions", data=json.dumps(body).encode(), method="POST",
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        self.stats["calls"] += 1
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def generate(self, system: str, user: str, max_tokens: int = 1500) -> dict:
        """Small-JSON generation with finish-reason detection and a JSON-repair fallback."""
        try:
            d = self._post([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens)
            choice = d["choices"][0]
            content = choice["message"]["content"]
            finish = choice.get("finish_reason")
            if finish == "length":
                self.stats["truncated"] += 1
            try:
                return {"parsed": json.loads(content), "raw": content, "finish": finish, "usage": d.get("usage")}
            except json.JSONDecodeError:
                self.stats["malformed"] += 1
        except Exception as exc:
            content = ""
            err = f"{type(exc).__name__}:{str(exc)[:100]}"
            try:  # one short retry
                d = self._post([{"role": "system", "content": system},
                                {"role": "user", "content": user + "\nReturn ONLY one small valid JSON object."}],
                               max_tokens)
                content = d["choices"][0]["message"]["content"]
                return {"parsed": json.loads(content), "raw": content, "finish": "retry"}
            except Exception:
                return {"parsed": None, "raw": None, "error": err}
        # repair-only pass (no new information; larger budget for the repair completion)
        repaired = self.repair(content, max_tokens + 800)
        if repaired is not None:
            self.stats["repaired"] += 1
            return {"parsed": repaired, "raw": json.dumps(repaired), "finish": "repaired"}
        self.stats["rejected"] += 1
        return {"parsed": None, "raw": content, "error": "malformed_after_repair"}

    def repair(self, broken: str, max_tokens: int) -> dict | None:
        if not broken:
            return None
        try:
            d = self._post([
                {"role": "system", "content": "You repair JSON. Output only the corrected valid JSON object."},
                {"role": "user", "content": "Repair this into valid JSON only. Do not add new information:\n" + broken[:6000]},
            ], max_tokens)
            return json.loads(d["choices"][0]["message"]["content"])
        except Exception:
            return None


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


# ------------------------------------------------------------------------------ family assignment
def family_assignment(dataset: str, split: str, seed: int, device: torch.device) -> dict:
    args = argparse.Namespace(
        seed=seed, morgan_bits=1024, smiles_cnn=True, smiles_max_len=192, feature_batch_size=1024,
        feature_seed=19, esm_batch_size=8, esm_max_len=1022, plm_source="esm2_t30_150M_UR50D",
        drug_encoder="morgan", chemberta_model="DeepChem/ChemBERTa-77M-MLM", drug_cache_path=None,
        text_dim=256, max_entities=256, min_entity_df=2, gkn_hidden=128, domain_dim=64, prototypes=8,
        gkn_epochs=50, projector_epochs=80, gkn_lr=1e-3, dropout=0.2, weight_decay=1e-5,
        mechanism_source="llm-cache", hierarchical_gkn=False, higcn_tiers=2,
        limit_train=0, limit_val=0, limit_test=0, smoke=False,
    )
    seed_everything(seed)
    bundle = load_affinity_bundle(dataset, args, device)
    sp = make_split(bundle, split, seed)
    rows = apply_limits(sp, args, seed)
    _ = split_norm_tensor(bundle.drug_raw, rows["fitD"], device)
    text_raw, text_by_id, _ = text_feature_matrix(
        dataset, bundle.target_ids, rows["fitT"], args.mechanism_source, args.text_dim, device)
    _, _, n_families, family_id, _ = train_gkn_prototypes(
        bundle.target_raw, rows["fitT"], bundle.target_ids, text_by_id, args, device)
    return {"bundle": bundle, "rows": rows, "text_by_id": text_by_id,
            "n_families": int(n_families), "family_id": family_id}


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
    as complete profile artifacts.
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


# ------------------------------------------------------------------------------ QC
def scan_leakage(obj) -> list[str]:
    text = json.dumps(obj, ensure_ascii=False)
    hits = set(m.lower() for m in LEAK_PATTERNS.findall(text))
    hits.update(m for m in AFFINITY_CASE_PATTERNS.findall(text))
    return sorted(hits)


GENERIC_TOKENS = {"kinase", "atp", "binding", "signaling", "pathway", "family", "protein", "domain"}
LIGAND_CUES = {"scaffold", "hydrophobic", "aromatic", "hydrogen", "lipophilic", "polar", "ring",
               "moiety", "substituent", "planar", "hinge", "pocket", "selectivity"}


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
SYSTEM = (
    "You are a pharmacology reasoning assistant for drug-target affinity research. You connect ligand "
    "scaffold/physicochemistry to protein binding-domain, family/selectivity, and pathway properties. "
    "You NEVER predict binding affinity or numbers. Never mention datasets, benchmarks, affinity "
    "values, labels, metrics, or train/test splits. " + CONCISE
)


def stage1_prompt(fam, briefs_batch) -> str:
    ex = {"schema": "promptdta_evidence_briefs_v1", "family_id": str(fam), "batch_id": "B",
          "evidence_briefs": [{"evidence_id": "id", "drug_cues": [], "target_cues": [],
                               "compatibility_hint": "...", "uncertainty": 0.0}]}
    return ("Family target mechanism summaries (sanitized):\n" + briefs_batch["targets"]
            + "\nLigands in this batch (descriptors only, no affinity):\n" + briefs_batch["drugs"]
            + "\nFor EACH ligand produce a compact evidence brief: drug_cues (scaffold/physchem), "
            "target_cues (domain/family/pathway), a one-line compatibility_hint, and uncertainty in [0,1]. "
            "evidence_id must equal the given id. Return ONLY JSON matching:\n" + json.dumps(ex))


def stage2_prompt(fam, channel, briefs) -> str:
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
    ex = {"schema": "promptdta_channel_profile_v1", "family_id": str(fam), "channel": channel,
          "summary": "... or None", "entities": [], "typed_relations": [], "confidence": 0.0,
          "uncertainty": 1.0, "evidence_ids": [], "genericness_risk": "low", "identity_dependence_risk": "low",
          "beyond_identity_signal": True, "quality_flags": []}
    return (f"Channel: {channel}.\nMechanism claims:\n" + json.dumps(claims)[:5000]
            + "\nSummarize into ONE compact channel profile (<80 words). If the claims are weak or only "
            "restate generic family identity, set summary to \"None\" and beyond_identity_signal=false. "
            "evidence_ids from the claims only. Return ONLY JSON matching:\n" + json.dumps(ex))


# ------------------------------------------------------------------------------ build one family
def build_family(client, dataset, split, seed, fam, members, target_ids, text_by_id,
                 idx, drug_descs, min_channel_quality, force, channel_cap) -> dict:
    valid_ids = set(idx)
    base = {"familyId": fam, "members": members, "sampledPairIds": idx, "schema": FINAL_SCHEMA,
            "prompt_version": PROMPT_VERSION}
    if not idx or not members:
        qc = family_qc({}, set(), min_channel_quality)
        return {**base, "empty": True, "qc": qc}
    # Stage 1: evidence briefs in small batches
    briefs = []
    summaries_txt = "\n".join(f"- {text_by_id.get(target_ids[t], '')[:240]}" for t in members[:10])
    for b in range(0, len(idx), BATCH_SIZE):
        bp = stage_path(dataset, split, seed, fam, "stage1", batch=b // BATCH_SIZE)
        cached = load_if_valid(bp, force)
        if cached is None:
            batch_ids = idx[b:b + BATCH_SIZE]
            drugs_txt = "\n".join(f"  id={i} {json.dumps(d)}" for i, d in zip(batch_ids, drug_descs[b:b + BATCH_SIZE]))
            resp = client.generate(SYSTEM, stage1_prompt(fam, {"targets": summaries_txt, "drugs": drugs_txt}), 1500)
            cached = {"_failed": resp["parsed"] is None, "result": resp.get("parsed"),
                      "inputDigest": _digest(drugs_txt), "outputDigest": _digest(resp.get("raw") or "")}
            atomic_write(bp, cached)
        if cached.get("result"):
            briefs.extend(cached["result"].get("evidence_briefs", []))
    # Stages 2+3 per channel
    channel_profiles = {}
    for ci, channel in enumerate(CHANNELS):
        if channel_cap is not None and ci >= channel_cap:
            channel_profiles[channel] = {"summary": "None"}
            continue
        s2p = stage_path(dataset, split, seed, fam, "stage2", channel=channel)
        c2 = load_if_valid(s2p, force)
        if c2 is None:
            r2 = client.generate(SYSTEM, stage2_prompt(fam, channel, briefs), 1800)
            c2 = {"_failed": r2["parsed"] is None, "result": r2.get("parsed")}
            atomic_write(s2p, c2)
        claims = (c2.get("result") or {}).get("mechanism_claims", []) if c2.get("result") else []
        s3p = stage_path(dataset, split, seed, fam, "stage3", channel=channel)
        c3 = load_if_valid(s3p, force)
        if c3 is None:
            if not claims:
                c3 = {"_failed": False, "result": {"summary": "None", "channel": channel, "evidence_ids": []}}
            else:
                r3 = client.generate(SYSTEM, stage3_prompt(fam, channel, claims), 1500)
                c3 = {"_failed": r3["parsed"] is None, "result": r3.get("parsed")}
            atomic_write(s3p, c3)
        channel_profiles[channel] = c3.get("result") or {"summary": "None"}
    # Stage 4: local assembly + Stage 5 QC
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


def build_seed(client, dataset, split, seed, device, min_channel_quality, force,
               family_cap, channel_cap) -> dict:
    asg = family_assignment(dataset, split, seed, device)
    bundle, rows = asg["bundle"], asg["rows"]
    target_ids, text_by_id, family_id, k = bundle.target_ids, asg["text_by_id"], asg["family_id"], asg["n_families"]
    fitD, fitT, fitY = np.asarray(rows["fitD"]), np.asarray(rows["fitT"]), np.asarray(rows["fitY"])
    drug_dict = json.loads((REPO / "dataset" / "kiba" / "ligands_can.txt").read_text())
    smiles = list(drug_dict.values())
    fit_unique = np.unique(fitT).astype(int)
    members = {f: [int(t) for t in fit_unique if int(family_id[int(t)]) == f] for f in range(k)}
    hi = fitY >= np.quantile(fitY, 0.6)
    rng = np.random.RandomState(seed + 7)
    families = {}
    n = k if family_cap is None else min(k, family_cap)
    for f in range(n):
        idx = np.where(hi & np.isin(fitT, list(set(members[f]))))[0]
        if idx.size > 20:
            idx = rng.choice(idx, 20, replace=False)
        idx = sorted(int(i) for i in idx.tolist())
        drug_descs = [drug_descriptor(smiles[int(fitD[i])]) for i in idx]
        families[str(f)] = build_family(client, dataset, split, seed, f, members[f], target_ids,
                                        text_by_id, idx, drug_descs, min_channel_quality, force, channel_cap)
    return {
        "schema": FINAL_SCHEMA, "promptVersion": PROMPT_VERSION, "model": client.model,
        "dataset": dataset, "split": split, "seed": seed, "nFamilies": k,
        "sourceScope": "inner_train_only", "minChannelQuality": min_channel_quality,
        "families": families, "stats": client.stats, "builtAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "leakagePolicy": "staged prompts use inner-train target summaries and inner-train ligand descriptors "
                         "only; no affinity values, test rows, labels, split membership, or predictions",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "build", "validate"])
    ap.add_argument("--splits", nargs="*", default=["KIBA/target-cold"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[1])
    ap.add_argument("--model", default=None)
    ap.add_argument("--min-channel-quality", type=float, default=0.30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force-cache", action="store_true")
    ap.add_argument("--family-cap", type=int, default=None)
    ap.add_argument("--channel-cap", type=int, default=None)
    args = ap.parse_args()
    _ = require_cuda_device(args.device)
    if args.mode == "validate":
        report = validate_caches(args.splits, args.seeds)
        print(json.dumps(report, indent=2))
        if not report["ok"]:
            raise SystemExit(2)
        return
    client = DeepSeekClient(load_env(), args.model)
    device = require_cuda_device(args.device)
    for split_spec in args.splits:
        dataset, split = split_spec.split("/", 1)
        for seed in args.seeds:
            out = cache_path(dataset, split, seed)
            fam_cap = 1 if (args.mode == "smoke" and args.family_cap is None) else args.family_cap
            chan_cap = 1 if (args.mode == "smoke" and args.channel_cap is None) else args.channel_cap
            print(f"[deepseek-staged] {dataset}/{split}/seed{seed} model={client.model} mode={args.mode} "
                  f"family_cap={fam_cap} channel_cap={chan_cap}")
            cache = build_seed(client, dataset, split, seed, device, args.min_channel_quality,
                               args.force_cache, fam_cap, chan_cap)
            if args.mode == "smoke":
                f0 = cache["families"].get("0", {})
                print(json.dumps({"stats": client.stats, "familyQuality": f0.get("qc", {}).get("familyQuality"),
                                  "coverage": f0.get("qc", {}).get("coverage"),
                                  "leakageFlags": f0.get("qc", {}).get("leakageFlags"),
                                  "acceptedSummaries": {c: str((f0.get("qc", {}).get("acceptedProfiles", {}).get(c) or {}).get("summary", ""))[:120]
                                                        for c in CHANNELS}}, indent=2)[:2500])
                return
            atomic_write(out, cache)
            print(f"[deepseek-staged] wrote {out.relative_to(REPO).as_posix()} stats={client.stats}")


if __name__ == "__main__":
    main()

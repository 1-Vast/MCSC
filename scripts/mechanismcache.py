"""Offline DeepSeek mechanism-QC cache for PRISM.

This script is the ONLY place the live DeepSeek API may be called from (`DeepSeekClient` below,
the sole user of `urllib.request` / `chat/completions` in this codebase outside this file).
Training/inference load the rollup this script writes and fail closed if it is missing/partial;
they never make a network call. `scripts/integritycheck.py::check_deepseek_boundary()` enforces
this split automatically.

The staged prompt/generation/quality-control pipeline itself (phase order, per-stage caching,
leakage/quality checks) lives in `model/mechanismllm.py` -- see that module's docstring for the
full phase-by-phase and quality-check breakdown. It has no import from `scripts/`, so it can be
imported from the model package without creating a circular dependency; this script supplies the
dataset/family-assignment context (`family_assignment`, which needs the training pipeline in
`scripts/`) and the live network client, and wires them together in `main()`.

CLI:
  python main.py cache smoke --splits KIBA/target-cold --seeds 1 --family-cap 1 --channel-cap 1
  python main.py cache build --splits KIBA/target-cold --seeds 1 2 3 4 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import urllib.request
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model.mechanismllm import (  # noqa: E402
    CHANNELS,
    atomic_write,
    build_seed,
    cache_path,
    validate_caches,
    validate_seed_cache,
)
from scripts.affinitydata import load_affinity_bundle, make_split
from scripts.affinityops import apply_limits, resolve_device, seed_everything, split_norm_tensor
from scripts.selectiveaffinity import text_feature_matrix, train_gkn_prototypes


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
        # Channels/batches within a family are called concurrently (see model.mechanismllm.build_family),
        # so stats mutations and the shared call counter need a lock; urllib requests themselves are
        # each independent socket calls and are safe to run from multiple threads.
        self._lock = threading.Lock()

    def _bump(self, key: str) -> None:
        with self._lock:
            self.stats[key] += 1

    def _post(self, messages: list[dict], max_tokens: int) -> dict:
        body = {"model": self.model, "messages": messages, "temperature": 0,
                "max_tokens": int(max_tokens), "response_format": {"type": "json_object"}}
        req = urllib.request.Request(
            self.base + "/chat/completions", data=json.dumps(body).encode(), method="POST",
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        self._bump("calls")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def _attempt(self, system: str, user: str, max_tokens: int) -> dict:
        """One raw generation call. Returns parsed=None (not an exception) on any failure so
        callers can inspect `finish`/`raw` to decide how to recover."""
        try:
            d = self._post([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens)
            choice = d["choices"][0]
            content = choice["message"]["content"]
            finish = choice.get("finish_reason")
            if finish == "length":
                self._bump("truncated")
            try:
                return {"parsed": json.loads(content), "raw": content, "finish": finish, "usage": d.get("usage")}
            except json.JSONDecodeError:
                self._bump("malformed")
                return {"parsed": None, "raw": content, "finish": finish, "usage": d.get("usage")}
        except Exception as exc:
            return {"parsed": None, "raw": None, "finish": None, "error": f"{type(exc).__name__}:{str(exc)[:100]}"}

    def generate(self, system: str, user: str, max_tokens: int = 1500) -> dict:
        """Small-JSON generation with finish-reason detection and a JSON-repair fallback.

        The dominant failure mode observed for this model is truncation (finish_reason=="length"
        with the response cut off mid-JSON): malformed-parse count matches truncated count almost
        exactly across every build so far. Repairing already-truncated JSON rarely recovers it
        (the model has no more information than what was cut off), so a truncated first attempt
        gets ONE full regeneration with a substantially larger budget before falling back to
        repair-from-broken-text.
        """
        first = self._attempt(system, user, max_tokens)
        if first["parsed"] is not None:
            return first
        if first.get("finish") == "length":
            bigger = self._attempt(
                system,
                user + "\nBe extremely concise. Output JSON only: no prose, no reasoning, "
                       "no markdown fences. The first character of your reply must be '{'.",
                max_tokens * 2,
            )
            if bigger["parsed"] is not None:
                return bigger
            first = bigger if bigger.get("raw") else first
        elif first.get("raw") is None:
            # network/transport error, not a truncation -> one short plain retry
            retry = self._attempt(system, user + "\nReturn ONLY one small valid JSON object.", max_tokens)
            if retry["parsed"] is not None:
                return retry
            if retry.get("raw") is not None:
                first = retry
        if not first.get("raw"):
            return {"parsed": None, "raw": None, "error": first.get("error", "no_content")}
        # repair-only pass (no new information; larger budget for the repair completion)
        repaired = self.repair(first["raw"], max_tokens + 1500)
        if repaired is not None:
            self._bump("repaired")
            return {"parsed": repaired, "raw": json.dumps(repaired), "finish": "repaired"}
        self._bump("rejected")
        return {"parsed": None, "raw": first["raw"], "error": "malformed_after_repair"}

    def repair(self, broken: str, max_tokens: int) -> dict | None:
        if not broken:
            return None
        try:
            d = self._post([
                {"role": "system", "content": "You repair JSON. Output only the corrected valid JSON object. "
                                               "No prose, no reasoning, no markdown fences."},
                {"role": "user", "content": "Repair this into valid JSON only. Do not add new information:\n" + broken[:6000]},
            ], max_tokens)
            return json.loads(d["choices"][0]["message"]["content"])
        except Exception:
            return None


# ------------------------------------------------------------------------------ family assignment
def family_assignment(dataset: str, split: str, seed: int, device: torch.device) -> dict:
    """Compute the (dataset, split, seed) train-only target-family clustering that mechanism
    prompts are grouped by. Needs the training pipeline (dataset loading, GKN prototype training)
    in `scripts/`, so it stays here rather than in model/mechanismllm.py -- see that module's
    docstring for why. Not LLM logic itself: no `generate()` calls happen in this function."""
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
    ap.add_argument("--family-workers", type=int, default=3,
                     help="max families processed concurrently within a seed (channels/batches "
                          "within each family are already concurrent); higher = faster but more "
                          "simultaneous DeepSeek requests")
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
            asg = family_assignment(dataset, split, seed, device)
            cache = build_seed(client, dataset, split, seed, asg, args.min_channel_quality,
                               args.force_cache, fam_cap, chan_cap, args.family_workers)
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
            # Check the final generated result immediately (structural completeness, leakage,
            # quality/coverage), not only on a later manual `cache validate` call.
            check = validate_seed_cache(dataset, split, seed)
            status = "OK" if check["ok"] else "FAILED"
            print(f"[deepseek-staged] check {dataset}/{split}/seed{seed}: {status} "
                  f"families={check['familiesAssembled']}/{check['nFamilies']} "
                  f"meanQuality={check['meanFamilyQuality']} meanCoverage={check['meanCoverage']}")
            if not check["ok"]:
                print(f"[deepseek-staged] check errors: {check['errors'][:10]}")
            if check["meanFamilyQuality"] < 0.15 or check["meanCoverage"] < 0.15:
                print(f"[deepseek-staged] WARNING: low mean quality/coverage for {dataset}/{split}/seed{seed} "
                      "- inspect before using this cache for training")


if __name__ == "__main__":
    main()

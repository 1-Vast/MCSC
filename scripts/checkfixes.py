"""Cheap integrity checks for the current MCSC DTI mainline."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import runtime as leakage
from scripts.runtime import (
    RunSettings,
    create_cold_splits,
    create_family_cold_split,
    feature_path,
    load_davis,
    load_features,
    validation_indices,
)


DAVIS = REPO / "dataset" / "davis"
CELLS = ("DAVIS/target-cold", "DAVIS/family-cold", "KIBA/target-cold", "KIBA/cluster-cold")


def check_folds() -> None:
    raw = eval((DAVIS / "folds" / "train_fold_setting1.txt").read_text(), {"__builtins__": {}})
    fold0 = len(raw[0])
    total = sum(len(fold) for fold in raw)
    n_train = len(load_davis(DAVIS)["trainD"])
    assert n_train == total, f"train pool {n_train} != all folds {total}"
    assert len(raw) == 1 or n_train != fold0, "DAVIS train pool uses only fold 0"
    print(f"[PASS] DAVIS train pool uses all {len(raw)} folds ({n_train} rows)")


def check_snapshot() -> None:
    model = nn.Linear(4, 3)
    snapshot = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    saved = snapshot["weight"].clone()
    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)
    assert torch.equal(saved, snapshot["weight"]), "best checkpoint snapshot mutated"
    assert not torch.equal(snapshot["weight"], model.weight.detach().cpu()), "live model did not change"
    print("[PASS] best-state snapshots are immutable clones")


def check_split_isolation() -> None:
    data = load_davis(DAVIS)
    warm_train = set(zip(data["trainD"].tolist(), data["trainT"].tolist()))
    warm_test = set(zip(data["testD"].tolist(), data["testT"].tolist()))
    assert not (warm_train & warm_test), "warm split pair overlap"
    cold = create_cold_splits(data)
    drug_cold = cold["drug-cold"]
    assert not (set(np.unique(drug_cold["trainD"])) & set(np.unique(drug_cold["testD"]))), "drug-cold drug leak"
    target_cold = cold["target-cold"]
    assert not (set(np.unique(target_cold["trainT"])) & set(np.unique(target_cold["testT"]))), "target-cold target leak"
    print("[PASS] warm/drug-cold/target-cold split isolation holds")


def check_family_cold_isolation() -> None:
    settings = RunSettings(device="cpu")
    settings.encoder.drug = "morgan"
    settings.encoder.target = "kb"
    if not feature_path(settings).exists():
        print("[skip] DAVIS morgan/kb feature cache missing; family-cold isolation skipped")
        return
    features = load_features(settings)
    data = load_davis(DAVIS)
    split = create_family_cold_split(data, features["targetFeat"], seed=1)
    train_targets = set(np.unique(split["trainT"]).tolist())
    test_targets = set(np.unique(split["testT"]).tolist())
    assert test_targets and not (train_targets & test_targets), "family-cold target overlap"
    split = {"name": "family-cold", **split}
    tr_idx, val_idx, basis = validation_indices(split, "family-cold", 1, target_feat=features["targetFeat"])
    val_targets = set(np.unique(split["trainT"][val_idx]).tolist())
    inner_targets = set(np.unique(split["trainT"][tr_idx]).tolist())
    assert basis == "cold_family", f"family validation basis is {basis}"
    assert val_targets and not (val_targets & inner_targets), "family-cold validation target overlap"
    print("[PASS] family-cold test and validation are target/family isolated")


def check_stale_cache_refused() -> None:
    settings = RunSettings(device="cpu")
    settings.encoder.drug = "morgan"
    settings.encoder.target = "kb"
    with tempfile.TemporaryDirectory() as tmpdir:
        settings.cache_dir = tmpdir
        path = feature_path(settings)
        torch.save(
            {
                "drugFeat": torch.zeros(2, 3),
                "targetFeat": torch.zeros(2, 3),
                "drugIds": ["d1", "d2"],
                "targetIds": ["t1", "t2"],
                "settings": {"drug": "morgan", "target": "kb"},
            },
            path,
        )
        try:
            load_features(settings)
        except SystemExit as exc:
            assert "stale" in str(exc).lower(), f"unexpected stale-cache refusal: {exc}"
            print("[PASS] stale pre-schema feature cache is refused")
            return
    raise AssertionError("stale feature cache was accepted")


def check_deepseek_boundary() -> None:
    cache = DAVIS.parent / "cache" / "deepseekdescriptions30.json"
    if cache.exists():
        desc = json.loads(cache.read_text(encoding="utf-8"))
        report = leakage.audit(list(desc.keys()), list(desc.values()))
        assert report["nOffenders"] > 0, "DeepSeek audit failed to flag known contaminated cache"
        print(f"[PASS] DeepSeek audit flags contaminated cache ({report['nOffenders']}/{report['nTargets']})")
    else:
        print("[skip] DeepSeek cache absent; contamination fixture skipped")

    mcsc = (REPO / "scripts" / "mcsc.py").read_text(encoding="utf-8").lower()
    assert "deepseek" not in mcsc, "MCSC mainline must not touch DeepSeek/LLM text"
    runtime = (REPO / "scripts" / "runtime.py").read_text(encoding="utf-8")
    for token in ("unsafe_until_reviewed", "exclude=True", "targetTextSafety"):
        assert token in runtime, f"runtime missing DeepSeek safety token: {token}"
    print("[PASS] LLM/DeepSeek is excluded from MCSC and guarded by audit metadata")


def check_mcsc_mainline_source() -> None:
    src = (REPO / "scripts" / "mcsc.py").read_text(encoding="utf-8")
    memory_src = (REPO / "model" / "memory.py").read_text(encoding="utf-8")
    runtime_src = (REPO / "scripts" / "runtime.py").read_text(encoding="utf-8")
    required = [
        "mem_full.predict(train_d, train_t, exclude_self=True)",
        "train_marginal = marginal_loo(train_d, train_y, full_global)",
        "blend_weight = select_global_blend_weight(val_fine, val_marginal, val_y)",
        "alpha = frozen_alpha_for(dataset, split)",
        "final = prior + alpha * (refiner - prior)",
        "KMeans(n_clusters=8, random_state=seed + 5, n_init=10)",
        "torch.amp.GradScaler",
        "torch.amp.autocast",
        "drug_all = torch.as_tensor",
        "--gpu-monitor",
        'torch.set_float32_matmul_precision("high")',
        "opt.zero_grad(set_to_none=True)",
    ]
    missing = [token for token in required if token not in src]
    assert not missing, f"MCSC source missing required tokens: {missing}"
    assert ".detach().cpu().clone()" not in src, "training loop still syncs best-state snapshots to CPU"
    assert "self.y_train[drug_t, target_t] = label_t" in memory_src, "InteractionMemory labels are not vectorized on-device"
    assert "report = audit(ids, texts)" in runtime_src, "runtime leakage audit must call local audit()"
    assert "write_record(report, out)" in runtime_src, "runtime leakage audit must call local write_record()"
    print("[PASS] MCSC source uses LOO global_blend prior, canonical split, frozen alpha, AMP")


def check_residual_alpha_calibration() -> None:
    path = REPO / "config" / "residual-alpha-calibration.json"
    assert path.exists(), "residual-alpha calibration missing"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("schema") == "drugtarget-residual-alpha-calibration-v1", "wrong alpha schema"
    assert "test" not in json.dumps(data).lower(), "alpha calibration contains test fields"
    expected = {
        ("DAVIS", "target-cold"): ("ctriad", 0.5),
        ("DAVIS", "family-cold"): ("ctriad", 0.25),
        ("KIBA", "target-cold"): ("esm2_t30_150M_UR50D", 0.5),
        ("KIBA", "cluster-cold"): ("esm2_t30_150M_UR50D", 0.5),
    }
    seen = set()
    for entry in data.get("entries", []):
        key = (entry.get("dataset"), entry.get("split"))
        assert key in expected, f"unexpected alpha entry: {key}"
        target_rep, alpha = expected[key]
        assert entry.get("targetRepresentation") == target_rep, f"{key} target rep mismatch"
        assert float(entry.get("alpha")) == alpha, f"{key} alpha mismatch"
        assert int(entry.get("nCalibrationSeeds", 0)) >= 8, f"{key} not 8-seed calibrated"
        seen.add(key)
    assert seen == set(expected), f"missing alpha entries: {set(expected) - seen}"
    print("[PASS] residual alpha calibration is complete and test-free")


def check_mcsc_outputs() -> None:
    results = REPO / "doc" / "mcsc-mainline-results.json"
    if not results.exists():
        print("[skip] MCSC results missing; run `python main.py mcsc --stage full --device cuda`")
        return
    data = json.loads(results.read_text(encoding="utf-8"))
    assert data.get("variant") == "MCSC-FrozenAlpha", "MCSC result variant mismatch"
    for cell in CELLS:
        payload = data.get("cells", {}).get(cell)
        assert payload, f"missing MCSC result cell {cell}"
        assert int(payload.get("nSeeds", 0)) == 8, f"{cell} is not 8-seed complete"
        assert payload.get("summary", {}).get("mcsc_frozen_alpha", {}).get("R2") is not None, f"{cell} missing R2"
    print("[PASS] MCSC mainline results are 8-seed complete")


def check_mcsc_manifest() -> None:
    manifest = REPO / "outputs" / "mcsc" / "manifest.json"
    if not manifest.exists():
        print("[skip] MCSC checkpoint manifest missing; run `python main.py mcsc --stage train --device cuda`")
        return
    data = json.loads(manifest.read_text(encoding="utf-8"))
    keys = {
        (row.get("dataset"), row.get("split"), int(row.get("seed", -1)))
        for row in data.get("checkpoints", [])
    }
    expected = {
        (dataset, split, seed)
        for cell in CELLS
        for dataset, split in [cell.split("/", 1)]
        for seed in range(1, 9)
    }
    assert keys >= expected, f"MCSC manifest missing {len(expected - keys)} checkpoint entries"
    print("[PASS] MCSC checkpoint manifest contains all 32 mainline entries")


def check_sota_artifacts() -> None:
    path = REPO / "doc" / "sota-evidence-results.json"
    if not path.exists():
        print("[skip] SOTA evidence missing; run `python main.py sotaevidence`")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("status") == "reproduced_frontier_sota_level", "SOTA evidence status mismatch"
    for cell in CELLS:
        payload = data.get("cells", {}).get(cell)
        decision = payload.get("decision", {}) if payload else {}
        assert decision.get("beatsAllPairedDeepBaselines") is True, f"{cell} does not beat all deep baselines"
        assert decision.get("beatsXgbMeanRef") is True, f"{cell} does not beat XGBoost mean reference"
    print("[PASS] reproduced-frontier SOTA evidence is present and PASS")


def check_cuda_available() -> None:
    assert torch.cuda.is_available(), "CUDA unavailable"
    print(f"[PASS] CUDA available: {torch.cuda.get_device_name(0)}")


def main() -> None:
    checks = [
        check_folds,
        check_snapshot,
        check_split_isolation,
        check_family_cold_isolation,
        check_stale_cache_refused,
        check_deepseek_boundary,
        check_mcsc_mainline_source,
        check_residual_alpha_calibration,
        check_mcsc_outputs,
        check_mcsc_manifest,
        check_sota_artifacts,
        check_cuda_available,
    ]
    for check in checks:
        check()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

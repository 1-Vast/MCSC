"""Fast verification gate for the current MCSC-FrozenAlpha mainline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CELLS = {
    ("DAVIS", "target-cold"): ("ctriad", 0.5),
    ("DAVIS", "family-cold"): ("ctriad", 0.25),
    ("KIBA", "target-cold"): ("esm2_t30_150M_UR50D", 0.5),
    ("KIBA", "cluster-cold"): ("esm2_t30_150M_UR50D", 0.5),
}
SEEDS = set(range(1, 9))


def pass_(message: str) -> None:
    print(f"[PASS] {message}")


def fail(message: str) -> None:
    raise AssertionError(message)


def read(path: str) -> str:
    return (REPO / path).read_text(encoding="utf-8")


def load(path: str) -> dict:
    return json.loads((REPO / path).read_text(encoding="utf-8"))


def check_dispatcher_is_current() -> None:
    text = read("main.py")
    required = [
        '"mcsc": "scripts/mcsc.py"',
        '"train": "scripts/train.py"',
        '"infer": "scripts/infer.py"',
        '"evidence": "scripts/evidence.py"',
        '"sotaevidence": "scripts/sota_evidence.py"',
        '"plmcache": "scripts/representation_plm.py"',
    ]
    missing = [token for token in required if token not in text]
    if missing:
        fail(f"main.py missing current command routes: {missing}")
    retired = ['"evidencefast"', '"experiment"', '"rcsc"', '"alphafrontier"']
    found = [token for token in retired if token in text.lower()]
    if found:
        fail(f"main.py still exposes retired routes: {found}")
    pass_("public dispatcher exposes current MCSC/baseline/evidence commands only")


def check_mcsc_source_contract() -> None:
    src = read("scripts/mcsc.py")
    required = [
        "mem_full.predict(train_d, train_t, exclude_self=True)",
        "train_marginal = marginal_loo(train_d, train_y, full_global)",
        "blend_weight = select_global_blend_weight(val_fine, val_marginal, val_y)",
        "alpha = frozen_alpha_for(dataset, split)",
        "final = prior + alpha * (refiner - prior)",
        "KMeans(n_clusters=8, random_state=seed + 5, n_init=10)",
        "torch.amp.autocast",
        "torch.amp.GradScaler",
        'parser.add_argument("--device"',
        'parser.add_argument("--batch-size"',
        'parser.add_argument("--eval-batch-size"',
        "torch.backends.cudnn.benchmark = True",
        "drug_all = torch.as_tensor",
        "--gpu-monitor",
        'torch.set_float32_matmul_precision("high")',
        "opt.zero_grad(set_to_none=True)",
    ]
    missing = [token for token in required if token not in src]
    if missing:
        fail(f"scripts/mcsc.py missing required mainline tokens: {missing}")
    if ".detach().cpu().clone()" in src:
        fail("MCSC training loop still syncs best checkpoint snapshots to CPU")
    forbidden = ["verify_kiba_loo", "selector_search", "residual_shrinkage", "scripts.rcsc", "deepseek"]
    found = [token for token in forbidden if token.lower() in src.lower()]
    if found:
        fail(f"scripts/mcsc.py contains retired/LLM path tokens: {found}")
    pass_("MCSC source uses LOO global_blend prior, frozen alpha, canonical KMeans, GPU/AMP path")


def check_gpu_and_leakage_bugfixes() -> None:
    memory = read("model/memory.py")
    runtime = read("scripts/runtime.py")
    if "self.y_train[drug_t, target_t] = label_t" not in memory:
        fail("InteractionMemory train labels are not vectorized on the active device")
    if "report = audit(ids, texts)" not in runtime or "write_record(report, out)" not in runtime:
        fail("runtime target-text leakage audit still uses a broken self-reference")
    pass_("GPU data path and target-text leakage audit bugfixes are present")


def check_baseline_split_contract() -> None:
    deep = read("scripts/deep_baseline.py")
    if "configure_builder" in deep:
        fail("DeepDTA baseline still imports removed configure_builder")
    if 'make_split("KIBA", split, seed)' not in deep:
        fail("DeepDTA baseline does not reuse current MCSC KIBA split builder")
    for path in ("scripts/graph_baseline.py", "scripts/moltrans_baseline.py"):
        src = read(path)
        if "kiba_seed_splits" not in src or "davis_seed_split" not in src:
            fail(f"{path} does not reuse shared same-protocol split helpers")
    pass_("deep baselines reuse the current same-protocol split helpers")


def check_alpha_calibration() -> None:
    data = load("config/residual-alpha-calibration.json")
    if data.get("schema") != "drugtarget-residual-alpha-calibration-v1":
        fail("residual-alpha calibration schema mismatch")
    text = json.dumps(data).lower()
    if "test" in text:
        fail("residual-alpha calibration contains test-derived fields")
    seen = {}
    grid = {0.0, 0.25, 0.5, 0.75, 1.0}
    for entry in data.get("entries", []):
        key = (entry.get("dataset"), entry.get("split"))
        if key not in CELLS:
            fail(f"unexpected residual-alpha entry: {key}")
        target_rep, alpha = CELLS[key]
        if entry.get("targetRepresentation") != target_rep:
            fail(f"{key} target representation mismatch")
        if float(entry.get("alpha")) != alpha:
            fail(f"{key} alpha mismatch: {entry.get('alpha')} != {alpha}")
        if set(float(x) for x in entry.get("alphaGrid", [])) != grid:
            fail(f"{key} alpha grid mismatch")
        if int(entry.get("nCalibrationSeeds", 0)) < 8:
            fail(f"{key} residual alpha was not calibrated across 8 seeds")
        source = entry.get("source")
        if source and not (REPO / source).exists():
            fail(f"{key} calibration source is missing: {source}")
        seen[key] = True
    missing = [key for key in CELLS if key not in seen]
    if missing:
        fail(f"missing residual-alpha entries: {missing}")
    pass_("frozen residual-alpha calibration is complete, grid-bounded, and calibration-only")


def check_checkpoint_metadata() -> None:
    ckpt_dir = REPO / "outputs" / "mcsc" / "checkpoints"
    rows = []
    for path in ckpt_dir.glob("*.json"):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    seen = {(row.get("dataset"), row.get("split"), int(row.get("seed", -1))): row for row in rows}
    missing = [
        (dataset, split, seed)
        for dataset, split in CELLS
        for seed in SEEDS
        if (dataset, split, seed) not in seen
    ]
    if missing:
        fail(f"missing MCSC checkpoint metadata for {len(missing)} cells/seeds; run `python main.py mcsc --stage full --device cuda`")
    for (dataset, split, seed), row in seen.items():
        if (dataset, split) not in CELLS:
            continue
        target_rep, alpha = CELLS[(dataset, split)]
        if row.get("schema") != "drugtarget-mcsc-checkpoint-v1":
            fail(f"{dataset}/{split}/seed{seed} checkpoint schema mismatch")
        if row.get("prior") != "global_blend":
            fail(f"{dataset}/{split}/seed{seed} prior is not global_blend")
        if row.get("targetRepresentation") != target_rep:
            fail(f"{dataset}/{split}/seed{seed} target representation mismatch")
        if float(row.get("frozenAlpha")) != alpha:
            fail(f"{dataset}/{split}/seed{seed} frozen alpha mismatch")
        if row.get("blendWeight") is None:
            fail(f"{dataset}/{split}/seed{seed} missing blendWeight")
        if row.get("device") != "cuda":
            fail(f"{dataset}/{split}/seed{seed} was not trained on CUDA metadata")
        if row.get("amp") is not True:
            fail(f"{dataset}/{split}/seed{seed} does not record AMP=True")
    pass_("all 32 MCSC checkpoints record CUDA, AMP, global_blend, target rep, and frozen alpha")


def check_result_artifacts() -> None:
    result = load("doc/mcsc-mainline-results.json")
    if result.get("variant") != "MCSC-FrozenAlpha":
        fail("mcsc-mainline results are not the FrozenAlpha variant")
    for dataset, split in CELLS:
        key = f"{dataset}/{split}"
        cell = result.get("cells", {}).get(key)
        if not cell:
            fail(f"missing MCSC result cell: {key}")
        if int(cell.get("nSeeds", 0)) != 8:
            fail(f"{key} result is not 8-seed complete")
        if cell.get("summary", {}).get("mcsc_frozen_alpha", {}).get("R2") is None:
            fail(f"{key} missing FrozenAlpha R2")

    sota = load("doc/sota-evidence-results.json")
    if sota.get("status") != "reproduced_frontier_sota_level":
        fail("SOTA evidence is not in reproduced-frontier SOTA-level status")
    for dataset, split in CELLS:
        key = f"{dataset}/{split}"
        cell = sota.get("cells", {}).get(key)
        decision = cell.get("decision", {}) if cell else {}
        if decision.get("beatsAllPairedDeepBaselines") is not True:
            fail(f"{key} does not beat all paired deep baselines")
        if decision.get("beatsXgbMeanRef") is not True:
            fail(f"{key} does not beat XGBoost mean reference")
        for name, baseline in cell.get("baselines", {}).items():
            if baseline.get("status") != "complete" or int(baseline.get("nSeeds", 0)) != 8:
                fail(f"{key} baseline {name} is not 8-seed complete")
    pass_("MCSC mainline and reproduced-frontier evidence artifacts are 8-seed complete")


def check_gpu_available() -> None:
    if not torch.cuda.is_available():
        fail("CUDA is unavailable; current mainline is expected to run GPU-first")
    name = torch.cuda.get_device_name(0)
    pass_(f"CUDA available for GPU-first flow: {name}")


def check_llm_boundary() -> None:
    mcsc = read("scripts/mcsc.py").lower()
    if "deepseek" in mcsc or "target=deepseek" in mcsc:
        fail("MCSC mainline touches DeepSeek/LLM target text")
    runtime = read("scripts/runtime.py")
    required = ["targetTextSafety", "unsafe_until_reviewed", "exclude=True"]
    missing = [token for token in required if token not in runtime]
    if missing:
        fail(f"DeepSeek leakage boundary missing runtime tokens: {missing}")
    pass_("LLM/DeepSeek remains outside MCSC and fail-closed behind audit/unsafe metadata")


def check_retired_code_removed() -> None:
    retired = [
        "scripts/alpha_frontier.py",
        "scripts/baseline_comparison.py",
        "scripts/cold_residual_selector.py",
        "scripts/rcsc.py",
        "scripts/residual_shrinkage.py",
        "scripts/selector_search.py",
        "scripts/verify_kiba_loo.py",
    ]
    present = [path for path in retired if (REPO / path).exists()]
    if present:
        fail(f"retired executable scripts still present: {present}")
    legacy = REPO / "scripts" / "legacy"
    if legacy.exists() and any(legacy.glob("*.py")):
        fail("scripts/legacy still contains executable retired Python files")
    pass_("retired executable branches are absent from scripts/")


def main() -> None:
    for check in (
        check_dispatcher_is_current,
        check_mcsc_source_contract,
        check_gpu_and_leakage_bugfixes,
        check_baseline_split_contract,
        check_alpha_calibration,
        check_checkpoint_metadata,
        check_result_artifacts,
        check_gpu_available,
        check_llm_boundary,
        check_retired_code_removed,
    ):
        check()
    print("VERIFICATION GATE COMPLETE")


if __name__ == "__main__":
    main()

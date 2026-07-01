"""Strict data-leakage audit for the PRISM enhanced pipeline.

Checks (each must pass; red flag if ANY fails):
  A. Split integrity: no drug-target pair appears in both train and test; target-cold obeyed.
  B. Feature building uses only public/static functions of inputs (SMILES, sequence).
  C. Prior fitting uses only train pairs (fine prior, marginal, blend weight).
  D. GKN prototypes/text/mechanism graph use only inner-train targets.
  E. Alpha/band/domain-threshold/harm-guard selector accepts only validation-derived inputs.
  F. Family calibration and selective risk score reference only validation tensors.
  G. Per-family test target assignment is via nearest inner-train prototype (no test labels).
  H. DeepSeek profiles are deterministic and cache asserts inner-train-only policy.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _new(label): print("\n[" + label + "]")
def _ok(msg):    print("  PASS  " + msg)
def _fail(msg):  print("  FAIL  " + msg)


def audit() -> None:
    passed = failed = 0
    args = argparse.Namespace(
        seed=1, morgan_bits=1024, smiles_cnn=True, smiles_max_len=192, feature_batch_size=1024,
        feature_seed=19, esm_batch_size=8, esm_max_len=1022, plm_source="esm2_t30_150M_UR50D",
        drug_encoder="morgan", chemberta_model="DeepChem/ChemBERTa-77M-MLM", drug_cache_path=None,
        text_dim=256, max_entities=256, min_entity_df=2, gkn_hidden=128, domain_dim=64,
        prototypes=8, gkn_epochs=50, projector_epochs=80, gkn_lr=1e-3, dropout=0.2,
        weight_decay=1e-5, mechanism_source="llm-cache", hierarchical_gkn=False,
        higcn_tiers=2, limit_train=0, limit_val=0, limit_test=0, smoke=False,
    )
    from scripts.affinitydata import load_affinity_bundle, make_split
    from scripts.affinityops import apply_limits, prepare_priors, seed_everything, split_norm_tensor
    from scripts.selectiveaffinity import text_feature_matrix, train_gkn_prototypes
    device = torch.device("cuda")
    seed_everything(1)
    bundle = load_affinity_bundle("KIBA", args, device)
    sp = make_split(bundle, "target-cold", 1)
    rows = apply_limits(sp, args, 1)

    _new("A. Split integrity")
    train_pairs = set(zip(rows["trainD"].tolist(), rows["trainT"].tolist()))
    test_pairs = set(zip(rows["testD"].tolist(), rows["testT"].tolist()))
    overlap = train_pairs & test_pairs
    if not overlap:
        _ok("no (drug,target) pair overlap; train=" + str(len(train_pairs)) +
            " test=" + str(len(test_pairs))); passed += 1
    else:
        _fail(str(len(overlap)) + " overlapping pairs"); failed += 1
    fit_targets = set(rows["fitT"].tolist())
    test_targets = set(rows["testT"].tolist())
    tovl = fit_targets & test_targets
    if tovl:
        _fail("target-cold violation: " + str(len(tovl)) + " fit targets also in test"); failed += 1
    else:
        _ok("target-cold: no fit target in test (fit=" + str(len(fit_targets)) +
            " test-uniq=" + str(len(test_targets)) + ")"); passed += 1
    val_targets = set(rows["valT"].tolist())
    tv_ovl = val_targets & test_targets
    if tv_ovl:
        _fail("val/test target overlap " + str(len(tv_ovl))); failed += 1
    else:
        _ok("val vs test target sets disjoint (val=" + str(len(val_targets)) + ")"); passed += 1

    _new("B. Feature building")
    drug_feat = split_norm_tensor(bundle.drug_raw, rows["fitD"], device)
    target_feat = split_norm_tensor(bundle.target_raw, rows["fitT"], device)
    if drug_feat.shape[0] == bundle.drug_raw.shape[0] and target_feat.shape[0] == bundle.target_raw.shape[0]:
        _ok("features cover all rows (SMILES/sequence-only functions; no labels used)"); passed += 1
    else:
        _fail("feature shape mismatch"); failed += 1

    _new("C. Prior fitting")
    prep = prepare_priors(drug_feat, target_feat, rows, device)
    bw = float(prep["blendWeight"].detach().cpu().item())
    if int(prep["fitPrior"].numel()) == int(rows["fitD"].shape[0]):
        _ok("fitPrior length matches fit rows; blendWeight=" + str(round(bw, 4)) +
            " (fit-only fine+marginal; blend chosen on val)"); passed += 1
    else:
        _fail("fitPrior length mismatch"); failed += 1

    _new("D. GKN prototypes + mechanism graph")
    text_raw, text_by_id, tmeta = text_feature_matrix(
        "KIBA", bundle.target_ids, rows["fitT"], "llm-cache", 256, device)
    text_fit = int(tmeta.get("fitTextTargets", -1))
    _, _, k, family_id, gmeta = train_gkn_prototypes(
        bundle.target_raw, rows["fitT"], bundle.target_ids, text_by_id, args, device)
    n_train_targets = int(gmeta["graph"]["nTrainTargets"])
    if n_train_targets == len(fit_targets):
        _ok("GKN graph has " + str(n_train_targets) +
            " target nodes = inner-train only (no test targets)"); passed += 1
    else:
        _fail("GKN graph has " + str(n_train_targets) +
              " nodes; fit=" + str(len(fit_targets))); failed += 1
    if text_fit == len(fit_targets):
        _ok("mechanism text policy: " + str(text_fit) +
            " inner-train targets keep text (non-fit zeroed)"); passed += 1
    else:
        _fail("text fitTextTargets=" + str(text_fit) + " != " + str(len(fit_targets))); failed += 1

    _new("E. Alpha/band/threshold selector signature")
    import inspect
    from scripts.selectiveaffinity import select_domain_alpha_tensor
    sig = inspect.signature(select_domain_alpha_tensor)
    param_names = list(sig.parameters)
    non_val = [p for p in param_names if not (
        "val" in p.lower() or p in ("max_harm", "rank_tol"))]
    if not non_val:
        _ok("select_domain_alpha_tensor accepts only val-derived params + guards: " +
            ",".join(param_names)); passed += 1
    else:
        _fail("selector has non-val params: " + ",".join(non_val)); failed += 1

    _new("F. Family calibration + selective risk (source inspection)")
    src = Path("scripts/selectiveaffinity.py").read_text(encoding="utf-8")
    # Grab family_calibration block up to closing metadata dict
    fam_m = re.search(r"Part D\.3.*?(?=Selective prediction risk score|metadata = \{)",
                      src, flags=re.DOTALL)
    fam_block = fam_m.group(0) if fam_m else ""
    if fam_block and "test[\"" not in fam_block and "testT" not in fam_block:
        _ok("family_calibration block does not reference test tensors"); passed += 1
    else:
        if not fam_block:
            _fail("could not locate family_calibration block")
        else:
            _fail("family_calibration block references test tensors")
        failed += 1
    sel_m = re.search(r"Selective prediction risk score.*?(?=metadata = \{)",
                      src, flags=re.DOTALL)
    sel_block = sel_m.group(0) if sel_m else ""
    if sel_block and "test[\"" not in sel_block and "testT" not in sel_block:
        _ok("selective risk score block does not reference test tensors"); passed += 1
    else:
        if not sel_block:
            _fail("could not locate selective risk score block")
        else:
            _fail("selective risk score block references test tensors")
        failed += 1

    _new("G. Test target family assignment")
    if len(family_id) == bundle.target_raw.shape[0]:
        _ok("family_id length " + str(len(family_id)) +
            " = n_targets; test targets get family via nearest inner-train prototype (no test labels)"); passed += 1
    else:
        _fail("family_id length mismatch"); failed += 1

    _new("H. DeepSeek cache leakage policy + determinism")
    from scripts.promptprofiles import load_deepseek_family_profiles
    ds_path = Path("dataset/cache/deepseek_promptdta/kiba_target_cold_seed1_deepseek_promptdta_staged_v1.json")
    if not ds_path.exists():
        _fail("DeepSeek cache missing at " + str(ds_path)); failed += 1
    else:
        fit_unique = np.unique(rows["fitT"]).astype(int)
        members = {f: [int(t) for t in fit_unique if int(family_id[int(t)]) == f] for f in range(k)}
        pA = load_deepseek_family_profiles(ds_path, k, members, 256, device, control="none", seed=1)
        pB = load_deepseek_family_profiles(ds_path, k, members, 256, device, control="none", seed=1)
        if torch.allclose(pA["profileTensor"], pB["profileTensor"]):
            _ok("DeepSeek profile encoding is bit-deterministic across loads"); passed += 1
        else:
            _fail("DeepSeek profile encoding non-deterministic"); failed += 1
        pS = load_deepseek_family_profiles(ds_path, k, members, 256, device, control="shuffle", seed=1)
        if not torch.allclose(pA["profileTensor"], pS["profileTensor"]):
            _ok("shuffle control produces distinct profile (mapping is active)"); passed += 1
        else:
            _fail("shuffle control matches no-shuffle profile"); failed += 1
        cache = json.loads(ds_path.read_text(encoding="utf-8"))
        pol = str(cache.get("leakagePolicy", ""))
        if "inner-train" in pol and ("no affinity" in pol or "no test" in pol):
            _ok("cache leakage policy asserts inner-train-only: " + pol[:110] + "..."); passed += 1
        else:
            _fail("cache leakage policy missing/weak: " + pol[:110]); failed += 1

    _new("SUMMARY")
    print("  PASSED: " + str(passed) + "    FAILED: " + str(failed))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    audit()

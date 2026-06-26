# Representation-next round — PLM target embeddings (KIBA ceiling)

Goal: push the KIBA representation ceiling (sequence-composition descriptors saturate e2e) while
keeping the verified DAVIS ctriad gain. No selector/prior-logic changes. Frozen ESM-2 embeddings,
sequence-only (leakage-safe), recomputed from `dataset/kiba/proteins.txt` for guaranteed alignment,
cached with sha256 in `config/representation-manifest.json`.

## Part A — inventory/audit
ESM-2 mean-pooled embeddings (8M→320d, 150M→640d) are deterministic functions of the protein
sequence; no labels/affinity/benchmark/split → **safe public descriptor**. Drug = Morgan (fixed).
DeepSeek text remains unsafe-until-reviewed (unused). Full audit: `doc/representation-audit.md`.

## Part B — prior-level probe (KIBA, 8 seeds; `doc/representation-next-results.json`)
| rep | target-cold Δ vs current | cluster-cold Δ vs current |
|---|---|---|
| ctriad | +0.006 (7/8) | +0.004 (5/8) |
| **esm-8M** | +0.014 (8/8) | +0.016 (7/8) |
| **esm-150M** | +0.013 (8/8) | +0.015 (8/8) |
Both ESM sizes beat current AND ctriad at the prior level (8/8, CI excl 0). 8M ≈ 150M here.

## Part C — end-to-end (`doc/representation-next-e2e-results.json`, full refiner, 8 seeds)
| KIBA split | current e2e | esm-8M e2e | **esm-150M e2e** | xgb_gpu |
|---|---|---|---|---|
| target-cold | 0.474 | 0.477 (+0.003, 5/8, ns) | **0.503 (+0.030, 8/8)** | 0.467 |
| cluster-cold | 0.285 | 0.267 (−0.018, 2/8) | **0.330 (+0.045, 7/8)** | 0.346 |
**Capacity matters: only esm-150M survives e2e** (8M's prior gain washes out, matching the earlier
ctriad saturation). esm-150M improves the KIBA mainline e2e on **both** cold splits; on target-cold
it beats XGBoost (0.503 vs 0.467); on cluster-cold it beats the current mainline (+0.045) but
XGBoost still edges that single split (0.346).

## Part D — deep baseline gate
**Incomplete.** DeepDTA/GraphDTA/MolTrans/DrugBAN not reproduced this round (dependency + GPU-time
cost). See `doc/baseline-gate-report.md`. Therefore **no SOTA / no deep-baseline-superiority claim**;
results scoped to internal improvement over the global_blend mainline + the reproduced XGBoost(GPU)
shallow baseline.

## Decision: **A (promote, internal) + D (baseline gate incomplete)**
- **Promote ESM-2 150M as the KIBA target representation.** It lifts the KIBA ceiling end-to-end on
  both cold splits (target +0.030 8/8 beating XGBoost; cluster +0.045 7/8), is leakage-safe, and is
  a genuine representation lever (not selector decoration) — the e2e gain that ctriad/8M could not
  deliver. **DAVIS keeps conjoint triad** (unchanged, not harmed).
- The baseline-relative / SOTA claim remains **gated** until deep baselines are reproduced.

## Recommendation / next
- KIBA mainline target rep → **ESM-2 150M** (frozen, cached); DAVIS → ctriad. The model is now
  dataset-adaptive in its target representation, both sequence-derived and leakage-safe.
- Reproduce ≥1 deep DTA baseline (gates SOTA). Optional: test ESM-2 on DAVIS (442 targets) to see if
  it beats ctriad there too; 650M is staged locally if 150M plateaus.

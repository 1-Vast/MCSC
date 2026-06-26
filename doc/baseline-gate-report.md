# Deep baseline gate — status: INCOMPLETE (Decision D)

## What is required
A fair, same-protocol reproduction of ≥1 strong deep DTA baseline (DeepDTA / GraphDTA / MolTrans /
DrugBAN) under identical DAVIS/KIBA cold splits, 8 seeds, sklearn-canonical cluster/family-cold, LOO
priors, no test tuning — to enable any baseline-relative or SOTA claim.

## Status: not reproduced this round
- **Blocker:** each baseline is a separate training framework (own deps, data adapters, training
  loops) and full reproduction under our exact split/seed protocol is a multi-day, GPU-heavy effort
  on a single unstable RTX 4060. Not completed in this round.
- `external_models/` (manifest + scoped .gitignore) is staged for this, but the baselines are **not**
  run; paper-reported numbers are NOT used as a fair comparison.

## What IS reproduced (fair, same protocol)
- **memory / fine / drug_marginal / global_blend** priors.
- **XGBoost(GPU)** shallow baseline on [Morgan ‖ target_rep] — same splits/seeds/features.
- MCSC global_blend mainline + the new ESM-2 representation.

## Consequence for claims
- **Allowed:** internal improvements over the global_blend mainline and over the reproduced
  XGBoost(GPU) baseline (e.g. KIBA target-cold ESM-150M e2e 0.503 > XGBoost 0.467).
- **Forbidden until this gate closes:** SOTA; superiority over DeepDTA/GraphDTA/MolTrans/DrugBAN.
- KIBA cluster-cold: XGBoost(GPU) (0.346) still edges the ESM-150M e2e refiner (0.330) — reported
  honestly; the representation gain is over the mainline, not over every shallow baseline on every
  split.

## To close the gate
Reproduce one baseline (DeepDTA is the simplest: CNN over SMILES + sequence) under
`scripts/`-driven splits/seeds, emit `doc/baseline-gate-results.json`, and only then make any
baseline-relative claim.

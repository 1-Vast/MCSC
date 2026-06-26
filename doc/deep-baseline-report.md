# Deep Baseline Gate Report

Purpose: position the fixed MCSC mainline against a reproduced deep DTA baseline under this repo's
exact cold-split protocol. Mainline was not changed.

## Baseline

**DeepDTA compact reproduction** (`scripts/deep_baseline.py`): character CNN over canonical SMILES
and protein sequence, validation-only early stopping, no test tuning.

- Splits: DAVIS target-cold, DAVIS family-cold, KIBA target-cold, KIBA cluster-cold.
- Seeds: 1-8 for every split.
- Model scale: 919,169 parameters.
- Fixed lengths: SMILES 100, protein sequence 1000.
- Training: Adam, max 60 epochs, patience 10, batch 512, FP32 on RTX 4060.
- Speed-only implementation detail: repeated drug/target ids inside a batch are encoded once and
  expanded back before scoring. This is mathematically equivalent to the naive DeepDTA forward pass.

## Completeness

DeepDTA is now complete for all required cells: **4 splits x 8 seeds**. Results are stored in
`doc/deep-baseline-results.json` with per-seed R2, MSE, RMSE, Pearson, Spearman, worst-group R2,
best validation MSE, epoch count, and runtime.

One limitation remains: the promoted MCSC result artifacts expose 8-seed mean references, but not
per-seed prediction arrays for the promoted DAVIS-ctriad and KIBA-ESM150M mainlines. Therefore the
DeepDTA-vs-MCSC deltas below are against the MCSC 8-seed mean references, not true paired bootstrap
deltas. I do not label them as paired CIs.

## Main Result

| split | DeepDTA R2 | MCSC mainline R2 | XGBoost R2 | delta vs MCSC ref | wins vs MCSC ref | verdict |
|---|---:|---:|---:|---:|---:|---|
| DAVIS target-cold | 0.467 [0.447, 0.485] | 0.444 | 0.484 | +0.023 [0.003, 0.041] | 6/8 | DeepDTA stronger than MCSC; XGB still edges mean |
| DAVIS family-cold | 0.254 [0.231, 0.279] | 0.234 | 0.284 | +0.020 [-0.004, 0.045] | 5/8 | close; DeepDTA slightly above MCSC, below XGB |
| KIBA target-cold | 0.415 [0.391, 0.440] | 0.503 | 0.467 | -0.089 [-0.113, -0.063] | 0/8 | MCSC clearly stronger |
| KIBA cluster-cold | 0.301 [0.242, 0.356] | 0.330 | 0.347 | -0.029 [-0.088, 0.027] | 4/8 | close/unstable; XGB remains strongest mean |

## Additional Metrics

| split | RMSE | MSE | Pearson | Spearman | worst-group R2 | mean runtime/seed |
|---|---:|---:|---:|---:|---:|---:|
| DAVIS target-cold | 0.654 | 0.428 | 0.687 | 0.564 | 0.210 | 69.3s |
| DAVIS family-cold | 0.731 | 0.547 | 0.521 | 0.485 | 0.157 | 60.2s |
| KIBA target-cold | 0.634 | 0.402 | 0.648 | 0.572 | 0.241 | 153.4s |
| KIBA cluster-cold | 0.668 | 0.449 | 0.558 | 0.504 | -0.045 | 106.5s |

## Decision

**Decision A+B, split-dependent.**

- **A on KIBA target-cold:** MCSC with frozen ESM-2 150M target representation is clearly stronger
  than reproduced DeepDTA and XGBoost under the same split protocol.
- **Competitive/close on KIBA cluster-cold:** MCSC mean is above DeepDTA, but the DeepDTA-vs-MCSC
  reference interval includes zero; XGBoost remains the strongest mean on this split.
- **B on DAVIS target-cold:** reproduced DeepDTA beats MCSC; MCSC must not claim superiority over the
  deep frontier on DAVIS target-cold.
- **DAVIS family-cold is close:** DeepDTA is slightly above MCSC by mean, but uncertainty overlaps;
  XGBoost has the best reproduced mean.

## Claim Impact

Allowed:

- MCSC is competitive with a reproduced deep baseline, and **stronger on KIBA target-cold**.
- The KIBA ESM-2 150M promotion survives a fair DeepDTA check on target-cold.
- MCSC remains a strong internal/shallow-frontier method, with split-specific deep-baseline evidence.

Forbidden:

- No SOTA claim.
- No blanket "MCSC beats deep baselines" claim.
- No claim of deep-frontier superiority on DAVIS target-cold or KIBA cluster-cold.

Next evidence needed for a stronger claim: save promoted MCSC per-seed predictions or per-seed R2
arrays and compute true paired deltas; optionally reproduce GraphDTA/MolTrans/DrugBAN under the same
splits.

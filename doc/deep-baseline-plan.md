# Deep Baseline Gate Plan

Purpose: position the fixed MCSC mainline against at least one reproduced deep DTA baseline under
this repo's exact split/seed protocol. This round is not an architecture search.

## Baseline

DeepDTA compact reproduction (`scripts/deep_baseline.py`): label-encoded SMILES (max length 100) and
protein sequence (max length 1000), three Conv1d layers per modality, global max pooling, concat,
FC[1024, 512, 1], MSE, Adam, validation-only early stopping.

Why DeepDTA first: it is the simplest strong deep DTA baseline to reproduce without external
framework dependencies. GraphDTA/MolTrans/DrugBAN remain follow-up baselines.

## Protocol

- Splits: DAVIS target-cold, DAVIS family-cold, KIBA target-cold, KIBA cluster-cold.
- Seeds: 1-8.
- Same split generators already used by the repo, including sklearn KMeans(n_init=10) for canonical
  family/cluster cold splits.
- No test tuning; model selection uses validation MSE only.
- MCSC comparison target is the fixed mainline: DAVIS ctriad + global_blend + current refiner; KIBA
  ESM-2 150M + global_blend + current refiner.

## Documented Simplifications

- Fixed maximum lengths: SMILES 100, protein 1000.
- Compact filter widths and three-layer per-modality CNN.
- Training cap: 60 epochs, patience 10, batch 512, FP32 on RTX 4060.
- Batch-level duplicate drug/target encodings are computed once and expanded back before scoring;
  this is a speed-only equivalent computation, not a protocol or model change.

## Outputs

- `doc/deep-baseline-results.json`
- `doc/deep-baseline-report.md`
- updated `doc/claim-boundary.md`

Metrics: R2, MSE, RMSE, Pearson, Spearman, worst-group R2, seed wins, CI95, runtime, parameter scale.

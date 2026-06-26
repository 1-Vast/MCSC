# MCSC Mainline

**MCSC = Memory-Calibrated Selective Correction.** The active trainable variant
is **MCSC-FrozenAlpha**.

```text
prior = global_blend(fine_memory, drug_marginal; validation-selected w)
refiner = ResidualRefiner(drug_descriptor, target_descriptor, prior)
final = prior + alpha * (refiner - prior)
```

## Fixed Configuration

| component | DAVIS | KIBA |
|---|---|---|
| drug representation | Morgan | Morgan |
| target representation | ctriad | frozen ESM-2 150M |
| prior | validation-global blend | validation-global blend |
| residual policy | frozen split-level alpha | frozen split-level alpha |

Alpha is loaded from `config/residual-alpha-calibration.json` and is frozen
before final evaluation.

## Mechanism

- `fine_memory` is train-only kNN interaction memory.
- train prior uses leave-one-out memory and leave-one-out drug marginal.
- blend weight is selected on validation only.
- frozen alpha shrinks the learned residual to prevent prior self-harm.
- `ResidualRefiner` is a PyTorch neural network trained on CUDA/AMP;
  MCSC-FrozenAlpha is therefore a deep learning model with a frozen
  memory-calibrated prior.
- KIBA cluster-cold uses sklearn `KMeans(n_init=10)` as the canonical split.

## GPU And Runtime Boundary

The active tensor path is CUDA-only: descriptor tensors, interaction-memory
retrieval, residual training, and batched inference run on GPU. The
implementation avoids per-label CPU/GPU writes in `InteractionMemory` and
preloads refiner inference indices on-device.

Canonical split construction, sklearn KMeans, dataset parsing, and cache loading
remain CPU-side by library/file-I/O design. CPU/GPU switching and runtime
monitoring are not exposed as model features.

## Current Evidence

Canonical artifacts:

- `doc/mcsc-mainline-results.json`
- `doc/mcsc-mainline-report.md`
- `doc/sota-evidence-results.json`
- `doc/sota-evidence-report.md`
- `experiments/model-comparison/mcsc-reproduced-frontier-sota-20260626/`
- `experiments/analysis/mcsc-mechanism-and-refiner-bottleneck-20260626/`

The allowed claim remains reproduced-frontier SOTA-level only, as scoped in
`doc/claim-boundary.md`.

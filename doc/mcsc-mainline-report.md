# MCSC Mainline Report

Current trainable mainline: **MCSC-FrozenAlpha**.

`final = prior + alpha * (refiner - prior)`

The residual refiner is trained by `python main.py mcsc --stage train`; alpha is loaded
from `config/residual-alpha-calibration.json` and frozen before final evaluation.

## Results

| split | target rep | alpha | prior R2 | full refiner R2 | MCSC R2 | delta vs prior | delta vs refiner | RMSE | Pearson | Spearman | worst-group |
|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| DAVIS/family-cold | ctriad | 0.25 | 0.2555 | 0.2823 | **0.2915** | +0.0360 [0.0217, 0.0538], 8/8 | +0.0092 [-0.0317, 0.0481], 5/8 | 0.7106 | 0.5673 | 0.5337 | 0.2050 |
| DAVIS/target-cold | ctriad | 0.50 | 0.4539 | 0.4886 | **0.4938** | +0.0398 [0.0362, 0.043], 8/8 | +0.0052 [0.002, 0.009], 7/8 | 0.6370 | 0.7344 | 0.5864 | 0.3955 |
| KIBA/cluster-cold | esm2_t30_150M_UR50D | 0.50 | 0.3455 | 0.3460 | **0.3722** | +0.0267 [0.0131, 0.0403], 7/8 | +0.0262 [0.0102, 0.041], 7/8 | 0.6335 | 0.6241 | 0.5408 | 0.0858 |
| KIBA/target-cold | esm2_t30_150M_UR50D | 0.50 | 0.4631 | 0.5020 | **0.5168** | +0.0537 [0.0415, 0.0658], 8/8 | +0.0147 [0.0015, 0.0273], 5/8 | 0.5757 | 0.7374 | 0.6463 | 0.3148 |

## Boundary

- This is the only current MCSC mainline.
- Older selectors, RCSC, dispersion, and full-refiner-only paths are retained only as failure/analysis history.
- SOTA-level wording is limited to reproduced frontier comparisons under this repository's identical protocol.

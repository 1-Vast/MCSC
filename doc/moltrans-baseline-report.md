# MolTrans-Adapted Baseline Report

Purpose: test a MolTrans-level transformer interaction baseline under the same local split, seed,
validation, and metric protocol. This does not change the MCSC mainline.

## Method Boundary

- Source snapshot: official MolTrans repository `https://github.com/kexinhuang12345/MolTrans`.
- Tokenization: official ESPF BPE codes and subword maps.
- Architecture: drug/protein token embeddings, two transformer encoder layers, interaction map, 2D conv, decoder.
- Profiles: `official` keeps MolTrans's original token lengths/width; `compact` keeps the same mechanism at a complete-run scale.
- Adaptation: binary classification loss/head is replaced by affinity-regression MSE.
- Boundary: this is a MolTrans-adapted regression baseline, not a paper-table comparison and not a claim of exact paper-faithful reproduction.

## Official-Profile Feasibility

- Official-profile smoke was run on DAVIS target-cold seed 1 with original MolTrans token lengths/width.
- Stable FP32 smoke: 2 epochs, batch 8, R2 0.1332, runtime 301.7 s.
- Faster AMP smoke: 1 epoch, batch 16, runtime 83.2 s, but numerically unstable for regression (R2 -63.585).
- Decision: official-shape 4x8x50-epoch regression reproduction is a compute/stability blocker on this workstation; compact profile is the complete same-protocol baseline.

## Results

| split | status | seeds | MolTrans R2 | CI95 | RMSE | Pearson | Spearman | worst-group | frozen alpha | DeepDTA | GraphDTA compact | XGBoost |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DAVIS/target-cold | complete | 8 | 0.3513 | [0.3283, 0.3743] | 0.7209 | 0.623 | 0.513 | 0.1424 | 0.4937 | 0.4666 | 0.3118 | 0.4837 |
| DAVIS/family-cold | complete | 8 | 0.2021 | [0.1525, 0.2417] | 0.7557 | 0.5101 | 0.4495 | 0.0177 | 0.2987 | 0.2541 | 0.2347 | 0.2836 |
| KIBA/target-cold | complete | 8 | -2.2175 | [-7.2294, 0.3139] | 1.0752 | 0.4888 | 0.4883 | -6.1524 | 0.5175 | 0.4148 | 0.3367 | 0.4672 |
| KIBA/cluster-cold | complete | 8 | -0.2205 | [-1.0577, 0.2196] | 0.8296 | 0.4399 | 0.4108 | -1.0098 | 0.3649 | 0.3012 | 0.2724 | 0.3465 |

## Claim Impact

- Compact profile status: compact with complete 4x8 split/seed coverage.
- Frozen alpha beats the complete compact MolTrans baseline on all four required cold splits.
- This supports reproduced-frontier SOTA-level wording, but not global SOTA over paper-faithful official MolTrans/DrugBAN/GraphDTA.

# GraphDTA Compact Baseline Report

Purpose: reproduce a graph-based deep DTA baseline under the same local split, seed,
validation, and metric protocol. This does not change the MCSC mainline.

## Method

- Drug encoder: RDKit molecular graph with a 3-layer GCN and global max pooling.
- Target encoder: protein character CNN with sequence length 1000.
- Selection: validation MSE only; test labels are evaluation-only.
- Boundary: compact GraphDTA-style reproduction, not paper-table comparison.

## Results

| split | status | seeds | GraphDTA R2 | CI95 | RMSE | Pearson | Spearman | worst-group | frozen alpha | DeepDTA | XGBoost |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| DAVIS/target-cold | complete | 8 | 0.3118 | [0.2891, 0.3386] | 0.7426 | 0.5616 | 0.5113 | 0.1381 | 0.4937 | 0.4666 | 0.4837 |
| DAVIS/family-cold | complete | 8 | 0.2347 | [0.2156, 0.2547] | 0.738 | 0.5112 | 0.4791 | 0.1614 | 0.2987 | 0.2541 | 0.2836 |
| KIBA/target-cold | complete | 8 | 0.3367 | [0.2962, 0.3785] | 0.674 | 0.5867 | 0.5283 | 0.1525 | 0.5175 | 0.4148 | 0.4672 |
| KIBA/cluster-cold | complete | 8 | 0.2724 | [0.1956, 0.343] | 0.6813 | 0.5446 | 0.4798 | -0.0431 | 0.3649 | 0.3012 | 0.3465 |

## Claim Boundary

- A complete baseline gate requires all four splits with seeds 1-8.
- Partial smoke results are implementation evidence only and cannot support superiority claims.
- SOTA remains forbidden unless the complete reproduced deep frontier is beaten under this protocol.

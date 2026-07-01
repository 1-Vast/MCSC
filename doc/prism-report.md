# PRISM Report

PRISM mainline: train-only GKN domain prototypes plus a DeepSeek-QC residual trust audit. DeepSeek is used as an offline quality signal, not as a direct affinity feature.

## Selected primary method: `DeepSeekQCSelective`

The public method is intentionally compact: residual prediction remains neural, while DeepSeek-QC calibrates residual trust and validation-selected selective defer. It does NOT claim that generated mechanism text predicts affinity. The name-only control collapses to prior-only, so the promoted claim is reliability calibration, not text enhancement.

## Results

| split | text source | prototypes | method | RMSE | MSE | Spearman | CI | worstgrp_R2 | harm_worse | gamma | route |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| KIBA/target-cold | llm-cache | 8 | Prior | 0.6159 | 0.3797 | 0.6157 | 0.7417 | 0.3268 | 0.0000 | 0.3411 | 0.4867 |
| KIBA/target-cold | llm-cache | 8 | DeepSeekQCSelective **[SELECTED]** | 0.6096 | 0.3721 | 0.6214 | 0.7444 | 0.3317 | 0.3229 | 0.3411 | 0.4867 |

Per-seed `harm_worse` for `DeepSeekQCSelective` on `KIBA/target-cold`: [0.3476, 0.3435, 0.3854, 0.1902, 0.3476]
All seeds <= 0.40: **True**.
Validation-selected coverages: [0.8, 0.8, 0.8, 0.8, 0.8].

## Rejected claims (kept explicit)

- Direct DeepSeek mechanism text/profile fusion is NOT supported by controls (name-only   family identity matched or beat the full mechanism profiles in round 7).
- Name-only family identity alone is NOT a valid mechanism-content claim.
- Conformal hard defer worsened both ranking and harm.
- Global shrink alone (fixed beta) improved mean harm but failed seed 3.
- beta >= 3 over-shrink was unstable because harm_worse conditions on moved samples;   the current harm-first beta selector caps beta at 2.5.

## Boundary

- Outputs are isolated under `outputs/prism/` and `doc/prism-*`.
- This enhanced workflow is CUDA-only; CPU fallback is rejected before train/infer.
- Mechanism graph/prototypes use inner-train targets only; test labels/statistics are final-evaluation only.
- LLM/API calls are not made during train or inference; cached text or public KB fallback is used.

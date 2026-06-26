# Leakage Review

Date: 2026-06-26. Scope: current MCSC-FrozenAlpha mainline, reproduced
baselines, and optional DeepSeek/LLM cache boundary.

## Current Guardrails

| surface | rule | guard |
|---|---|---|
| cold splits | held-out targets/families do not appear in train | `python main.py check` |
| normalization | fit on train-visible drugs/targets only | `split_norm` in `scripts/mcsc.py` |
| memory prior | train-only; train prior uses LOO | `exclude_self=True`, `marginal_loo` |
| blend weight | selected on validation only | `select_global_blend_weight` |
| residual alpha | frozen from calibration-only inner cold validation | `config/residual-alpha-calibration.json` |
| KIBA cluster-cold | sklearn `KMeans(n_init=10)` canonical split | `scripts/mcsc.py` |
| target representations | DAVIS ctriad and KIBA ESM are sequence-only | no labels or benchmark text |
| DeepSeek/LLM | excluded from MCSC; unsafe until reviewed | runtime audit + `targetTextSafety` |
| checkpoints | best state cloned, no later-epoch mutation | `python main.py check` |
| target-text preprocessing | KB/DeepSeek text audit is fail-closed before feature use | local `audit()` + `write_record()` in `scripts/runtime.py` |
| GPU train path | MCSC tensors, memory prior, refiner training, inference on CUDA | checkpoint metadata + source gate |

## Verdict

No active MCSC path uses test labels, held-out labels, split membership,
benchmark numbers, model predictions, or LLM-generated descriptions as target
features. Test rows are used only for final metric reporting.

DeepSeek remains optional audit material only. This round fixed the optional
target-text audit path so preprocessing calls the local leakage scanner directly
instead of a stale self-reference.

Run:

```powershell
python main.py check
python main.py verifygate
```

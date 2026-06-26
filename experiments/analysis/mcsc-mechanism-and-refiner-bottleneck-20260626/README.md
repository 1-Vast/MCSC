# MCSC Mechanism And Refiner Bottleneck Analysis

Date: 2026-06-26

Purpose: record the diagnosis explaining why the promoted MCSC frontier works
and where the previous full refiner failed.

## Diagnosis

The full residual refiner was the bottleneck: it degraded the prior on most cold
splits, especially DAVIS and KIBA cluster-cold. The promoted fix is not a new
architecture search; it is validation-frozen residual shrinkage:

`final = prior + alpha * (refiner - prior)`

Alpha is selected from calibration-only inner-cold validation and then frozen
per dataset/split before final evaluation.

## Mechanism Evidence

| split | delta vs prior | delta vs full refiner | harmful-correction reduction | worst-group delta |
|---|---|---|---|---|
| DAVIS/target-cold | +0.0398 [0.0362, 0.0430], 8/8 | +0.0051 [0.0018, 0.0089], 7/8 | +0.0193, 8/8 | +0.0141, 7/8 |
| DAVIS/family-cold | +0.0360 [0.0217, 0.0538], 8/8 | +0.0092 [-0.0318, 0.0481], 5/8 | +0.0183, 8/8 | +0.0541, 6/8 |
| KIBA/target-cold | +0.0537 [0.0415, 0.0658], 8/8 | +0.0147 [0.0015, 0.0273], 5/8 | +0.0359, 8/8 | +0.0934, 7/8 |
| KIBA/cluster-cold | +0.0267 [0.0131, 0.0403], 7/8 | +0.0262 [0.0102, 0.0410], 7/8 | +0.0285, 8/8 | +0.1155, 6/8 |

The strongest mechanism statement is therefore: frozen alpha improves the prior
on all four splits and reduces harmful correction on all four; improvement over
the full refiner is significant on three splits, while DAVIS family-cold mainly
shows safer correction rather than a significant R2 gain over the full refiner.

## Decision

- Promote frozen split-level residual alpha as the current MCSC frontier.
- Keep RCSC rejected/not promotable.
- Keep full refiner alone rejected for this claim because it can self-harm.
- Keep MCSC as the model name; use MCSC-FrozenAlpha only as a precise variant
  label in tables.

## Reproduce

Run from repository root:

```powershell
python main.py mcsc --stage full
python main.py sotaevidence
python main.py check
python main.py verifygate
python -m compileall -q main.py model scripts
```

## Canonical Artifacts

- `doc/mcsc-mainline-report.md`
- `doc/mcsc-mainline-results.json`
- `experiments/analysis/failed-directions-20260626/README.md`
- `config/residual-alpha-calibration.json`

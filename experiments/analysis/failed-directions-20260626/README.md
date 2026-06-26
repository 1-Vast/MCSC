# Failed Directions Record

Date: 2026-06-26

Purpose: preserve the lessons from old experiments while keeping the active
repository focused on the trainable MCSC mainline.

## Current Mainline

Only **MCSC-FrozenAlpha** is promotable:

`final = prior + alpha * (refiner - prior)`

- DAVIS target representation: ctriad.
- KIBA target representation: frozen ESM-2 150M.
- Prior: validation-global blend.
- Refiner: trained ResidualRefiner.
- Residual policy: frozen split-level alpha from calibration-only inner-cold validation.

Reproduce with:

```powershell
python main.py mcsc --stage full
python main.py sotaevidence
python main.py check
python main.py verifygate
```

## Rejected Or Retired Paths

| path | status | reason |
|---|---|---|
| dispersion prior | retired | validation-global blend was stronger and simpler on the promoted cold splits |
| selector search | retired | added moving parts without clearing the promotion gate |
| ctriad for KIBA mainline | rejected | prior-level signal did not justify replacing ESM-2 after e2e checks |
| full refiner alone | rejected | harmed the prior on most cold splits |
| RCSC | rejected | did not preserve the frozen-alpha frontier consistently |
| blanket prior-only | rejected | loses the KIBA target-cold residual gain |
| paper-table SOTA comparison | forbidden | not same split/seed/protocol |

The old artifacts should be read as failure evidence, not as active model
branches. New experiments should start from `python main.py mcsc`, not from the
retired scripts.

## Cleanup Decision

Second-round cleanup removed the executable branches for retired ideas. The
repository now keeps only:

- current trainable MCSC mainline;
- reproduced deep baselines;
- SOTA evidence builder;
- leakage/GPU/check gates;
- this failure record.

The detailed intermediate JSON files and large RCSC prediction tables were
removed because they duplicated conclusions and made failed branches look active.

## Lessons To Carry Forward

- Representation changes must clear e2e, not only prior-level probes.
- Residual corrections must be shrinkable or validation-frozen; full residual
  injection can degrade strong priors.
- Baseline comparisons must be reproduced under the same split and seed
  protocol.
- LLM/DeepSeek text is not part of the active mainline unless cached, audited,
  and free of benchmark/affinity leakage.
- GPU use is required for the current MCSC train/infer tensor path. Runtime
  monitoring and artificial utilization boosting are not model contributions and
  should stay outside the mainline interface.
- Optional target-text preprocessing must call the local leakage scanner before
  feature construction. A stale `leakage.audit` self-reference was fixed in the
  second cleanup round.

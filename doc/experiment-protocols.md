# Experiment Protocols

## Mainline Reproduction

```powershell
D:\anaconda\envs\drug\python.exe main.py mcsc --stage full
python main.py sotaevidence
python main.py check
python main.py verifygate
```

The required cells are:

- `DAVIS/target-cold`
- `DAVIS/family-cold`
- `KIBA/target-cold`
- `KIBA/cluster-cold`

Each promoted result uses seeds 1-8, validation-only model selection, and the
same split definitions across MCSC and reproduced baselines.

## Baseline Gate

```powershell
python main.py deepbaseline
python main.py graphbaseline
python main.py moltransbaseline
```

Baseline runs must use the shared split helpers and must not use paper-table
numbers as evidence.

## Promotion Gate

1. `python main.py mcsc --stage full` completes all 32 cells on the CUDA-only MCSC path.
2. `doc/mcsc-mainline-results.json` reports `MCSC-FrozenAlpha` with 8 seeds.
3. Deep baseline records are complete under the same splits and seeds.
4. `python main.py sotaevidence` marks every required cell `PASS`.
5. `python main.py check`, `python main.py verifygate`, and compileall pass.
6. Claim wording stays within `doc/claim-boundary.md`.

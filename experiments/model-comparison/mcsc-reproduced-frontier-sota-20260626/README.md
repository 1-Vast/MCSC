# MCSC Reproduced Frontier SOTA-Level Comparison

Date: 2026-06-26

Purpose: record and reproduce the local claim that the current MCSC frontier is
SOTA-level under this repository's identical reproduced-frontier protocol.

## Naming Decision

Do not rename the model. The paper/mainline name remains **MCSC**.

For tables, the exact promoted variant can be written as **MCSC-FrozenAlpha**:

`MCSC = global_blend prior + dataset-adaptive target representation + frozen split-level residual alpha`

This is a calibrated MCSC frontier, not a new model family.

## Claim Scope

Supported claim:

Current MCSC reaches **reproduced-frontier SOTA-level** under this repository's
same cold splits, 8 seeds, validation-only selection, and reproduced/adapted
baseline protocol.

Forbidden claims:

- No global SOTA claim.
- No paper-table SOTA comparison.
- No superiority claim over unreproduced paper-faithful official GraphDTA,
  MolTrans, or DrugBAN.
- No test-tuned residual alpha or feature-selection claim.

## Mainline Configuration

| component | DAVIS | KIBA |
|---|---|---|
| drug representation | Morgan | Morgan |
| target representation | ctriad | frozen ESM-2 150M |
| prior | validation-global blend | validation-global blend |
| residual policy | frozen split-level alpha | frozen split-level alpha |

Alpha source: `config/residual-alpha-calibration.json`.

## Frontier Results

| split | MCSC-FrozenAlpha R2 | DeepDTA delta | GraphDTA compact delta | MolTrans compact delta | XGBoost margin | decision |
|---|---:|---|---|---|---:|---|
| DAVIS/target-cold | 0.4938 | +0.0271 [0.0102, 0.0420], 7/8 | +0.1820, 8/8 | +0.1425, 8/8 | +0.0101 | PASS |
| DAVIS/family-cold | 0.2915 | +0.0375 [0.0048, 0.0673], 7/8 | +0.0568, 7/8 | +0.0895, 7/8 | +0.0079 | PASS |
| KIBA/target-cold | 0.5168 | +0.1020 [0.0855, 0.1166], 8/8 | +0.1801, 8/8 | +2.7343, 8/8 | +0.0496 | PASS |
| KIBA/cluster-cold | 0.3722 | +0.0711 [0.0365, 0.1112], 8/8 | +0.0998, 7/8 | +0.5927, 8/8 | +0.0257 | PASS |

The full CI fields and baseline records are canonical in
`doc/sota-evidence-results.json`.

## Reproduce

Run from repository root:

```powershell
python main.py mcsc --stage full
python main.py deepbaseline
python main.py graphbaseline
python main.py moltransbaseline
python main.py sotaevidence
python main.py check
python main.py verifygate
python -m compileall -q main.py model scripts
```

For a fast smoke rerun of deep baselines, pass one split and one seed, for
example:

```powershell
python main.py deepbaseline --splits DAVIS/target-cold --seeds 1
python main.py graphbaseline --splits DAVIS/target-cold --seeds 1
python main.py moltransbaseline --splits DAVIS/target-cold --seeds 1
```

## Canonical Artifacts

- `doc/sota-evidence-report.md`
- `doc/sota-evidence-results.json`
- `doc/mcsc-mainline-report.md`
- `doc/mcsc-mainline-results.json`
- `doc/deep-baseline-report.md`
- `doc/deep-baseline-results.json`
- `doc/graph-baseline-report.md`
- `doc/graph-baseline-results.json`
- `doc/moltrans-baseline-report.md`
- `doc/moltrans-baseline-results.json`
- `doc/claim-boundary.md`
- `config/residual-alpha-calibration.json`

Failure and mechanism lessons are consolidated in
`experiments/analysis/failed-directions-20260626/`.

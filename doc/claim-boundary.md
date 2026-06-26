# Claim Boundary

## Current MCSC Frontier

Current local frontier under the repository protocol: **frozen split-level residual alpha**.
RCSC status: **rejected / not promotable**.

Mainline ingredients:

- DAVIS target representation: ctriad.
- KIBA target representation: frozen ESM-2 150M.
- Prior: validation-global blend.
- Residual policy: frozen split-level alpha from calibration-only inner cold validation.

Allowed claims:

- Frozen residual alpha improves the prior on all four cold splits under 8 seeds.
- Frozen residual alpha reduces harmful correction on all four splits and improves the full refiner significantly on three of four splits; DAVIS family-cold is safer but not a significant R2 win over the full refiner.
- Frozen alpha outperforms reproduced DeepDTA, compact GraphDTA, compact MolTrans, and local XGBoost mean references on all four required cold splits.
- The current MCSC frontier is supported at the reproduced-frontier level under the repository's identical local split/seed/metric protocol.
- Mechanism evidence supports dataset-adaptive target representation plus validation-frozen residual shrinkage as a mitigation for the observed refiner self-harm bottleneck.

Forbidden claims:

- No global SOTA claim.
- No blanket superiority over all deep DTA baselines.
- No superiority claim over paper-faithful official GraphDTA, MolTrans, or DrugBAN until reproduced under identical splits/seeds.
- No paper-table comparison as evidence.
- No test-tuned residual alpha, threshold, fallback, or feature-set claim.

Evidence artifacts:

- `doc/mcsc-mainline-results.json`
- `doc/mcsc-mainline-report.md`
- `config/residual-alpha-calibration.json`
- `doc/deep-baseline-results.json`
- `doc/graph-baseline-results.json`
- `doc/moltrans-baseline-results.json`
- `doc/sota-evidence-results.json`
- `doc/sota-evidence-report.md`

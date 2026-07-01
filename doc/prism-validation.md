# PRISM Validation Record

Date: 2026-07-01

## Engineering Gate

- Compile: `python -m compileall -q main.py model scripts` passed.
- Integrity: `python main.py check` passed all PRISM checks.
- Leakage audit: `python main.py audit` passed 14/14 checks.
- Cache validation: `python main.py cache validate --splits KIBA/target-cold KIBA/cluster-cold --seeds 1 2 3 4 5 --device cuda` passed 10/10 caches with no leakage hits.
- GPU smoke inference: `python main.py prism --stage infer --splits KIBA/target-cold --seeds 1 --deterministic --strict-deterministic --device cuda` passed.
- GPU smoke training: `python main.py prism --stage train --splits KIBA/target-cold --seeds 1 --smoke --force --deterministic --strict-deterministic --device cuda` passed.
- CPU negative test: `python main.py prism --stage infer --splits KIBA/target-cold --seeds 1 --device cpu` failed closed as expected.
- Missing DeepSeek cache negative test: temporarily hiding the required seed-1
  cache made inference fail closed, then restoring the cache allowed inference.

Verified CUDA runtime:

```text
torch 2.6.0+cu124
NVIDIA GeForce RTX 4060 Laptop GPU
```

## Promoted Mainline

The promoted PRISM path is DeepSeek-QC/GKN selective affinity prediction. The
supported DeepSeek role is quality audit, residual confidence calibration, and
selective defer, not direct text-enhanced affinity prediction.

Canonical result files:

- `doc/prism-results.json`
- `doc/prism-report.md`

Historical MCSC/M3C-DTI/DTA-GKN files are retained as records only and are not
public entry points.

## Three-Round Audit

Round 1: source and public-boundary audit.

- Finding: `scripts/runtime.py` still contained a retired live DeepSeek client
  path even though `main.py` no longer exposed it.
- Fix: removed the retired runtime API path and strengthened
  `scripts/integritycheck.py` so live API client tokens are allowed only in
  `scripts/mechanismcache.py`.

Round 2: split, cache, and artifact audit.

- Independent checks across KIBA target-cold and cluster-cold seeds 1-5 found
  no fit/test target overlap, no val/test target overlap, and no pair overlap
  across evaluation boundaries.
- DeepSeek family members in every cache were verified to be subsets of the
  corresponding inner-train target set.
- Field-level cache scans over accepted profile summaries found no benchmark,
  label, affinity-value, metric, or prediction leakage patterns.
- `doc/prism-results.json` stores aggregate metrics and metadata only; it does
  not store test labels, row-level predictions, or split row identifiers.

Round 3: runtime fail-closed audit.

- GPU deterministic inference passed on the CUDA environment.
- CPU inference failed closed before model execution.
- Hiding the required DeepSeek profile cache made inference fail closed with an
  explicit rebuild instruction.
- Restoring the cache made inference pass again.

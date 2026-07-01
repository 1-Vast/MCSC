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

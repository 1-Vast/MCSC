# External Research Boundary

This directory is isolated from the PRISM mainline.

Contents:

- `thirdparty/`: downloaded third-party comparison repositories and archives.
- `experiments/`: retired experiments, model-comparison containers, and failed
  direction records.
- `runs/`: paper-innovation or exploratory research run logs.
- `scripts/`: retired or external helpers for LLM/API, PLM, TDC OOD,
  third-party SOTA comparison, old baselines, and GPU monitoring.
- `tmp/`: downloaded archives, expanded source caches, and temporary research
  material.

Rules:

- Code here is not imported by the PRISM mainline.
- Commands here are not public `main.py` entry points.
- Artifacts here cannot be used as base-model evidence unless promoted through
  the repository promotion gate.
- GPU monitoring and external SOTA work stay here as reproducibility evidence,
  not as model features.

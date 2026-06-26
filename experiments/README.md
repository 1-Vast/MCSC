# Experiments

This directory stores named experiment containers for reproducibility and review.
Canonical datasets, checkpoints, output records, and reports stay in their
repository-standard locations (`dataset/`, `outputs/`, `doc/`, and `config/`).

## Layout

- `model-comparison/`: same-protocol comparisons against reproduced or adapted
  baseline models.
- `analysis/`: mechanism, ablation, error-diagnosis, and decision-boundary
  experiments.

Each experiment folder should include a short `README.md`, a machine-readable
`manifest.json`, and optional thin scripts that call `python main.py ...`.
Large generated artifacts should be referenced rather than copied here.

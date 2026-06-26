# Data Contract

## Active Inputs

```text
dataset/davis/
dataset/kiba/
dataset/cache/
dataset/kb/
```

- Drug representation: Morgan fingerprints from SMILES.
- DAVIS target representation: sequence-only conjoint triad.
- KIBA target representation: frozen sequence-only ESM-2 150M cache generated
  from `dataset/kiba/proteins.txt`.
- Target text/DeepSeek descriptors are not part of the current MCSC mainline.

## Boundaries

- Dataset files and feature caches are not deleted during code cleanup.
- Source/config/JSON paths should stay repository-relative.
- Test rows and held-out target/family units are final evaluation only.
- Validation may select blend weight and checkpoints.
- Frozen residual alpha comes from calibration-only inner cold validation.
- DeepSeek descriptions, when generated, must remain cached, audited, and
  marked unsafe until reviewed.

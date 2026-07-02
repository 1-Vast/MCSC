# PRISM

**PRISM** is the active drug-target affinity mainline in this repository:
**Prototype-guided Reliability-Informed Selective Memory** prediction.

The method is intentionally compact. It combines a train-only interaction
memory prior, a neural residual affinity refiner, GKN target-domain prototypes,
and an offline DeepSeek-QC reliability audit that calibrates residual trust and
selective defer. DeepSeek is not used as a live predictor during training or
inference.

## Architecture

```text
prior = train-only interaction memory
pair = neural drug-target representation from drug/target/text/profile features
trust_context = [memory diagnostics, domain/QC distance]  (low-dimensional, train-only)
domain = train-only GKN target-domain prototypes
qc = offline DeepSeek mechanism quality audit
final = prior + gamma(pair, trust_context, qc) * residual(pair)
```

The mainline is deliberately two-branch: the cross-modal representation branch
predicts the residual, while memory/domain/QC signals reach the
`ResidualTrustGate` and validation-time calibration. Earlier shared-token
context injection was tested and rejected for promotion, then removed from the
public training path.

Core code lives under `model/`:

- `MemoryResidualRefiner`: memory-calibrated neural affinity refiner.
- `SelectiveAffinityRefiner`: mechanism text adapter, GKN prototype distances, and
  domain-aware defer gate.
- `TargetDomainGraphEncoder`, `ResidualTrustGate`, and mechanism-profile fusion are
  model components, not public scripts.

Tooling lives under `scripts/` and is called through `main.py`.

## Commands

Use the CUDA-enabled drug environment:

```powershell
D:\anaconda\envs\drug\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Public entry points:

```powershell
D:\anaconda\envs\drug\python.exe main.py prism --stage infer --splits KIBA/target-cold --seeds 1 --device cuda
D:\anaconda\envs\drug\python.exe main.py train --splits KIBA/target-cold --seeds 1 --device cuda
D:\anaconda\envs\drug\python.exe main.py infer --splits KIBA/target-cold --seeds 1 --device cuda
D:\anaconda\envs\drug\python.exe main.py cache validate --splits KIBA/target-cold --seeds 1 2 3 4 5 --device cuda
D:\anaconda\envs\drug\python.exe main.py audit
D:\anaconda\envs\drug\python.exe main.py check
```

Training and inference are GPU-only. CPU fallback is rejected before the model
path runs. Offline DeepSeek calls are allowed only through `main.py cache`; the
train/infer scripts load cached mechanism-QC records and fail closed if required
records are missing or partial.

## Evidence Boundary

PRISM promotes the DeepSeek-QC/GKN selective line only under these controls:

- GKN prototypes are built from inner-train targets only.
- Validation may select checkpoints, residual shrinkage, and selective defer.
- Test rows, held-out targets, and held-out families are final evaluation only.
- DeepSeek summaries must not include labels, benchmark membership, split
  membership, predictions, or affinity values.
- Name-only and shuffle controls are kept as evidence against overclaiming
  direct mechanism-text prediction.

Allowed claim: PRISM is a mechanism-grounded, reliability-calibrated DTA
architecture with explicit leakage controls and selective OOD behavior.

Rejected claim: raw LLM mechanism text alone improves affinity prediction. The
strongest supported role for DeepSeek is quality audit plus residual confidence
calibration and selective defer.

## Layout

```text
main.py             only public dispatcher
model/              PRISM architecture and neural components
scripts/            protocol runners, audits, cache builder, preprocessing
config/             lightweight default and calibration settings
requirements.txt    Python package requirements
```

Generated datasets, caches, checkpoints, reports, and retired experiment
records are intentionally ignored and not uploaded to the remote code branch.

# DrugTarget

This repository is a focused drug-target affinity project. The active trainable
mainline is **MCSC-FrozenAlpha**.

## Mainline

```text
prior = global_blend(fine_memory, drug_marginal; validation-selected w)
refiner = ResidualRefiner(drug_descriptor, target_descriptor, prior)
final = prior + alpha * (refiner - prior)
```

- DAVIS target representation: sequence-only conjoint triad.
- KIBA target representation: frozen sequence-only ESM-2 150M.
- Alpha: frozen per dataset/split from calibration-only inner cold validation.
- Model class: deep neural residual refiner in PyTorch, calibrated by a frozen
  memory prior.
- Device: CUDA-only for the active MCSC train/infer path; AMP is enabled.

The trainable component is a neural `ResidualRefiner`, so the current MCSC line
is a deep learning model rather than a pure kNN or shallow regressor. The memory
prior is frozen/calibrated and the neural residual learns the correction.

## GPU Boundary

`python main.py mcsc --stage train|infer|full` keeps the active MCSC tensor path
on CUDA: descriptor tensors, interaction-memory retrieval, residual-refiner
training, AMP forward/backward, and batched inference. The canonical split
construction, sklearn `KMeans(n_init=10)`, JSON loading, and feature-cache
loading remain CPU-side by library/file-I/O design, but the trainable model path
does not expose a CPU fallback.

On this Windows workstation the verified CUDA interpreter is:

```powershell
D:\anaconda\envs\drug\python.exe -c "import torch; print(torch.cuda.is_available())"
```

Using `C:\Python314\python.exe` will report CPU-only PyTorch and fail the GPU
gate. Run the reproduction commands with the CUDA environment when checking GPU
behavior.

The public MCSC command intentionally exposes only the key reproducibility
knobs: `--stage`, `--splits`, `--seeds`, `--force`, and `--batch-size`. Runtime
monitoring and CPU/GPU switching are not model features and are not part of the
mainline interface.

Retired routes such as dispersion selectors, RCSC, RA-MCSC, full-refiner-only
promotion, and selector search are summarized under
`experiments/analysis/failed-directions-20260626/`.

## Environment

Recommended Windows/conda setup:

```powershell
conda create -n mcsc python=3.11 -y
conda activate mcsc
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The existing local CUDA environment used for verification is:

```powershell
D:\anaconda\envs\drug\python.exe -m pip install -r requirements.txt
D:\anaconda\envs\drug\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Python `venv` alternative:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Reproduce

```powershell
D:\anaconda\envs\drug\python.exe main.py mcsc --stage full
D:\anaconda\envs\drug\python.exe main.py deepbaseline
D:\anaconda\envs\drug\python.exe main.py graphbaseline
D:\anaconda\envs\drug\python.exe main.py moltransbaseline
D:\anaconda\envs\drug\python.exe main.py sotaevidence
D:\anaconda\envs\drug\python.exe main.py check
D:\anaconda\envs\drug\python.exe main.py verifygate
D:\anaconda\envs\drug\python.exe -m compileall -q main.py model scripts
```

Convenience aliases:

```powershell
python main.py train   # mcsc --stage train
python main.py infer   # mcsc --stage infer
python main.py evidence
```

If the KIBA ESM-2 cache is missing:

```powershell
python main.py plmcache
```

## Claim Boundary

Allowed wording: current MCSC is supported at the reproduced-frontier level
under this repository's identical cold-split, 8-seed, validation-only protocol
against the reproduced/adapted local frontier.

Forbidden wording: global SOTA, paper-table comparisons, or superiority over
unreproduced paper-faithful official GraphDTA/MolTrans/DrugBAN.

## Optional LLM Boundary

`python main.py api` can generate cached DeepSeek target descriptions for
separate audits, but LLM/DeepSeek text is **not** part of the active MCSC
mainline. Any DeepSeek cache is marked unsafe until reviewed and is filtered by
the leakage audit before use in optional preprocessing paths.

## Layout

```text
main.py        public dispatcher
model/         InteractionMemory, ResidualRefiner, metrics, global-blend prior
scripts/       MCSC, baselines, PLM cache, checks, evidence builder
dataset/       source datasets, KB sources, regenerable caches
outputs/mcsc/  current MCSC checkpoints and manifest
doc/           current reports and protocol notes
experiments/   reproducibility containers and failure summaries
config/        frozen calibration and representation manifests
```

`outputs/archive/`, retired executable branches, and old probe reports are not
part of the current reproducible line; their lessons are summarized under
`experiments/analysis/failed-directions-20260626/`.

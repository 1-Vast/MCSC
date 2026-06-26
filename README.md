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
- Device: GPU-first (`--device auto` chooses CUDA when available); AMP is on by
  default.

The trainable component is a neural `ResidualRefiner`, so the current MCSC line
is a deep learning model rather than a pure kNN or shallow regressor. The memory
prior is frozen/calibrated and the neural residual learns the correction.

## GPU Boundary

`python main.py mcsc --stage train|infer|full --device cuda` keeps the active
MCSC tensor path on CUDA: descriptor tensors, interaction-memory retrieval,
residual-refiner training, AMP forward/backward, and batched inference. The
canonical split construction, sklearn `KMeans(n_init=10)`, JSON loading, and
feature-cache loading remain CPU-side by design.

On this Windows workstation the verified CUDA interpreter is:

```powershell
D:\anaconda\envs\drug\python.exe -c "import torch; print(torch.cuda.is_available())"
```

Using `C:\Python314\python.exe` will report CPU-only PyTorch and fail the GPU
gate. Run the reproduction commands with the CUDA environment when checking GPU
behavior.

Expected utilization is bursty on small DAVIS/KIBA cells: dense CUDA kernels
should run on GPU, but short epochs and CPU split/data preparation can leave
visible gaps in `nvidia-smi`. Do not add artificial GPU work to inflate
utilization. Use the default evidence settings for claim reproduction; increase
`--batch-size` or `--eval-batch-size` only as a throughput experiment and rerun
the full evidence gate if those settings are promoted.

For a real utilization audit, sample GPU state while running the model:

```powershell
D:\anaconda\envs\drug\python.exe main.py mcsc --stage full --device cuda --gpu-monitor outputs/mcsc/gpu-monitor.json
```

The monitor records `nvidia-smi` samples and summarizes utilization. Low average
utilization on a one-seed smoke run means the dataset cell is too small and too
short to saturate the GPU continuously; it is not evidence that the model fell
back to CPU. The gate checks CUDA checkpoint metadata, AMP, on-device
interaction memory, and CUDA availability.

Retired routes such as dispersion selectors, RCSC, RA-MCSC, full-refiner-only
promotion, and selector search are summarized under
`experiments/analysis/failed-directions-20260626/`.

## Reproduce

```powershell
D:\anaconda\envs\drug\python.exe main.py mcsc --stage full --device cuda
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

Allowed wording: current MCSC is **reproduced-frontier SOTA-level** under this
repository's identical cold-split, 8-seed, validation-only protocol against the
reproduced/adapted local frontier.

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

# DrugTarget Working Rules

These rules define the repository contract for the drug-target affinity
mainline. The previous cell perturbation direction is archived background, not
an active code path.

## Execution Boundary

- `main.py` is the only public dispatcher for commands such as `api`,
  `preprocess`, `train`, `infer`, `check`, `evidence`, and `experiment`.
- Core model code belongs under `model/`.
- Tooling, command wrappers, runtime helpers, and experiment dispatchers belong
  under `scripts/`.
- Dataset downloads and preprocessed feature caches belong under `dataset/`.
- Output records, checkpoints, reports, and archives belong under `outputs/`.
- Research ideas and external-model plans belong under `idea/`.
- `experiments/` is only a named experiment container, not a data/output store.
- Keep repository paths relative in source, configs, and JSON records.
- The default secret file is repository-root `.env`; `DRUGTARGET_ENV_FILE` may
  point to another dotenv file on another machine.

## Simplicity And Readability

- Prefer the minimum code that proves or falsifies the current DTI claim.
- Avoid speculative abstractions and unused experiment branches.
- Name new files with short semantic names; avoid underscores where practical.
- Keep concise English comments only where the code is not obvious.

## Mechanism Claim

The active contribution is a SCI-grade mechanism-grounded DTI architecture:

- drug side: offline chemistry descriptor;
- target side: public KB or DeepSeek mechanism text encoded once by hashing;
- prediction: interaction memory plus compact neural affinity refiner;
- evidence: saved inference metrics plus target-shuffle/name controls;
- rigor: frozen descriptors, train/infer separation, declared splits, and
  multi-seed reporting.

Do not claim a model improvement unless the mechanism control supports it.
Strong wording such as "SCI-grade" must refer to the protocol and architecture,
not to unverified performance.

## Leakage Control

- Validation may select refiner checkpoints.
- Test rows and held-out drug/target/family units are final evaluation only.
- Target descriptions must not include binding affinity, inhibition data,
  benchmark labels, split membership, or model predictions.
- DeepSeek summaries are optional and must be cached before training/evaluation.
- KB descriptors must come from declared public sources under `dataset/kb/`.

## Compatibility

- New DTI code should tolerate different DTI datasets when they satisfy the
  source contract.
- Do not hard-code DAVIS sizes, kinase names, target counts, or held-out panels
  in reusable logic.
- If CPU fallback is needed, make device behavior explicit.

## Promotion Gate

A candidate can enter the mainline only after:

1. short smoke run passes;
2. target-shuffle or name-only control is logged;
3. leakage review passes;
4. at least three seeds support the claimed task;
5. split-specific failures or high variance are reported honestly;
6. comparison against relevant DTI baselines is added or the result is labeled
   as an internal pilot.

## Karpathy Guidelines

- State assumptions before coding when the request is ambiguous.
- Make surgical changes: every changed line should trace to the task.
- Prefer simple code over speculative abstraction.
- Define verifiable success criteria and loop until they are checked.

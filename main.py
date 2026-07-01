"""DrugTarget dispatcher for the PRISM mainline."""
from __future__ import annotations

import runpy
import os
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

COMMANDS = {
    "prism": "scripts/selectiveaffinity.py",
    "cache": "scripts/mechanismcache.py",
    "audit": "scripts/leakageaudit.py",
    "preprocess": "scripts/preprocess.py",
    "train": "scripts/train.py",
    "infer": "scripts/infer.py",
    "check": "scripts/integritycheck.py",
}

HELP = """PRISM drug-target affinity commands:

  python main.py prism --stage full
  python main.py cache validate
  python main.py audit
  python main.py preprocess
  python main.py train      # alias: prism --stage train
  python main.py infer      # alias: prism --stage infer
  python main.py check

Optional:
  python main.py prism --stage infer --splits KIBA/target-cold --seeds 1 --device cuda

PRISM is the selected mainline: a memory-calibrated neural affinity refiner with
train-only GKN target-domain prototypes, offline DeepSeek-QC reliability audit,
and validation-selected selective defer. Train/infer is CUDA-only. DeepSeek/API
calls are allowed only in offline cache construction, never during train/infer.
Retired research routes are records only and are not public entry points.
Use `python main.py <command> --help` for options.
"""


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1].lower() in {"help", "-h", "--help"}:
        print(HELP)
        return
    command = sys.argv[1].lower()
    script = COMMANDS.get(command)
    if script is None:
        raise SystemExit(f"Unknown command: {command}\n\n{HELP}")

    old_argv = sys.argv
    old_entry = os.environ.get("DRUGTARGET_ENTRYPOINT")
    os.environ["DRUGTARGET_ENTRYPOINT"] = command
    sys.argv = [str(REPO / script), *sys.argv[2:]]
    try:
        runpy.run_path(str(REPO / script), run_name="__main__")
    finally:
        sys.argv = old_argv
        if old_entry is None:
            os.environ.pop("DRUGTARGET_ENTRYPOINT", None)
        else:
            os.environ["DRUGTARGET_ENTRYPOINT"] = old_entry


if __name__ == "__main__":
    main()

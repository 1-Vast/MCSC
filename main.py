"""DrugTarget dispatcher for the current MCSC line."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

COMMANDS = {
    "api": "scripts/api.py",
    "preprocess": "scripts/preprocess.py",
    "train": "scripts/train.py",
    "infer": "scripts/infer.py",
    "mcsc": "scripts/mcsc.py",
    "check": "scripts/checkfixes.py",
    "evidence": "scripts/evidence.py",
    "verifygate": "scripts/verifygate.py",
    "deepbaseline": "scripts/deep_baseline.py",
    "graphbaseline": "scripts/graph_baseline.py",
    "moltransbaseline": "scripts/moltrans_baseline.py",
    "plmcache": "scripts/representation_plm.py",
    "sotaevidence": "scripts/sota_evidence.py",
}

HELP = """DrugTarget commands:

  python main.py mcsc --stage full
  python main.py preprocess
  python main.py train      # alias: mcsc --stage train
  python main.py infer      # alias: mcsc --stage infer
  python main.py check
  python main.py evidence   # alias: sotaevidence
  python main.py verifygate
  python main.py deepbaseline --splits DAVIS/target-cold --seeds 1
  python main.py graphbaseline --splits DAVIS/target-cold --seeds 1
  python main.py moltransbaseline --splits DAVIS/target-cold --seeds 1
  python main.py plmcache   # regenerate KIBA ESM-2 target cache if missing
  python main.py sotaevidence

Optional:
  python main.py api --limit 30

The trainable mainline is MCSC-FrozenAlpha. Defaults use GPU automatically
when CUDA is available. Use `python main.py <command> --help` for options.
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
    sys.argv = [str(REPO / script), *sys.argv[2:]]
    try:
        runpy.run_path(str(REPO / script), run_name="__main__")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()

"""Alias for training the current MCSC mainline."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    old_argv = sys.argv
    sys.argv = [str(REPO / "scripts" / "mcsc.py"), "--stage", "train", *sys.argv[1:]]
    try:
        runpy.run_path(str(REPO / "scripts" / "mcsc.py"), run_name="__main__")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()

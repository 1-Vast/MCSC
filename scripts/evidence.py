"""Alias for the reproduced-frontier SOTA evidence builder."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    old_argv = sys.argv
    sys.argv = [str(REPO / "scripts" / "sota_evidence.py"), *sys.argv[1:]]
    try:
        runpy.run_path(str(REPO / "scripts" / "sota_evidence.py"), run_name="__main__")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()

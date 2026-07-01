"""Small helpers for script entrypoint wrappers."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def run_script(script: str, *prefix_args: str) -> None:
    old_argv = sys.argv
    sys.argv = [str(REPO / "scripts" / script), *prefix_args, *sys.argv[1:]]
    try:
        runpy.run_path(str(REPO / "scripts" / script), run_name="__main__")
    finally:
        sys.argv = old_argv

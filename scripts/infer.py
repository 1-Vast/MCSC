"""Alias for evaluating the PRISM mainline."""
from __future__ import annotations

import os

from scripts.dispatch import run_script


def main() -> None:
    old_entry = os.environ.get("DRUGTARGET_ENTRYPOINT")
    os.environ["DRUGTARGET_ENTRYPOINT"] = "prism"
    try:
        run_script("selectiveaffinity.py", "--stage", "infer")
    finally:
        if old_entry is None:
            os.environ.pop("DRUGTARGET_ENTRYPOINT", None)
        else:
            os.environ["DRUGTARGET_ENTRYPOINT"] = old_entry


if __name__ == "__main__":
    main()

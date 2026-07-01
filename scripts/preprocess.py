"""Preprocess DAVIS into frozen drug and target feature tensors."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DrugTarget features")
    parser.add_argument("--target", default="kb", choices=["kb"], help="Pure public target descriptor source.")
    parser.add_argument(
        "--drug",
        default="morgan",
        choices=["morgan", "hash"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda"],
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from scripts.runtime import save_features
    from scripts.runtime import RunSettings

    settings = RunSettings(device=args.device)
    settings.encoder.target = args.target
    settings.encoder.drug = args.drug
    path = save_features(settings)
    print(f"saved features: {path}")


if __name__ == "__main__":
    main()

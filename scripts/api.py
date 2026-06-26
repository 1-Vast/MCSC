"""API-only target description generation."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DeepSeek target descriptions")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of targets.")
    parser.add_argument("--env-file", default="", help="Optional dotenv file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from scripts.runtime import generate_deepseek_texts
    from scripts.runtime import deepseek_cache_path, load_data
    from scripts.runtime import RunSettings

    if args.env_file:
        os.environ["DRUGTARGET_ENV_FILE"] = args.env_file
    settings = RunSettings()
    data = load_data(settings)
    # Resolve to the existing cache (canonical preferred, known alternates honored) so
    # generation extends it instead of creating a parallel file.
    cache_path = deepseek_cache_path(settings)
    descriptions = generate_deepseek_texts(
        data["targetIds"],
        data["targetSeqs"],
        cache_path,
        limit=args.limit,
    )
    print(f"saved descriptions: {len(descriptions)} -> {cache_path}")


if __name__ == "__main__":
    main()

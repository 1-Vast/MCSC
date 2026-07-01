"""Integrity checks for the PRISM mainline."""
from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model import PrismMemoryRefiner, PrismSelectiveRefiner  # noqa: E402


MAINLINE_DIRS = {"config", "dataset", "doc", "externalresearch", "model", "outputs", "scripts"}
PUBLIC_COMMANDS = {"prism", "cache", "audit", "preprocess", "train", "infer", "check"}
RETIRED_COMMANDS = {
    "m" + "csc",
    "dta" + "qc",
    "dta" + "gkn",
    "dta" + "static",
    "deepseek" + "promptdta",
    "baseline",
}
REQUIRED_SCRIPTS = {
    "affinitydata.py",
    "affinityops.py",
    "dispatch.py",
    "infer.py",
    "integritycheck.py",
    "leakageaudit.py",
    "mechanismcache.py",
    "preprocess.py",
    "promptprofiles.py",
    "runtime.py",
    "selectiveaffinity.py",
    "seqdescriptors.py",
    "train.py",
}


def pass_(message: str) -> None:
    print(f"[PASS] {message}")


def read(path: str) -> str:
    return (REPO / path).read_text(encoding="utf-8")


def check_root_layout() -> None:
    root_dirs = {p.name for p in REPO.iterdir() if p.is_dir()}
    missing = sorted(MAINLINE_DIRS - root_dirs)
    assert not missing, f"mainline directories missing: {missing}"
    assert (REPO / "externalresearch" / "README.md").exists(), "external research boundary README missing"
    pass_("root layout keeps PRISM code, data, outputs, docs, and external records separated")


def check_scripts_layout() -> None:
    names = {p.name for p in (REPO / "scripts").glob("*.py")}
    missing = sorted(REQUIRED_SCRIPTS - names)
    assert not missing, f"required PRISM scripts missing: {missing}"
    retired = sorted(name for name in names if name in {f"{item}.py" for item in RETIRED_COMMANDS})
    assert not retired, f"retired scripts remain public under scripts/: {retired}"
    undernamed = sorted(p.name for p in (REPO / "scripts").glob("*.py") if "_" in p.stem)
    assert not undernamed, f"script filenames should avoid underscores: {undernamed}"
    pass_("scripts directory contains function-named PRISM tools only")


def check_dispatcher() -> None:
    text = read("main.py")
    for command in PUBLIC_COMMANDS:
        assert f'"{command}"' in text, f"main.py missing {command} route"
    lowered = text.lower()
    leaked = sorted(command for command in RETIRED_COMMANDS if command in lowered)
    assert not leaked, f"main.py exposes retired command names: {leaked}"
    pass_("main.py exposes only concise PRISM commands")


def check_model_modules() -> None:
    required = {
        "model/adapters.py",
        "model/attention.py",
        "model/defer.py",
        "model/enhanced.py",
        "model/graph.py",
        "model/memory.py",
        "model/promptfusion.py",
        "model/refiners.py",
        "model/text.py",
    }
    missing = sorted(path for path in required if not (REPO / path).exists())
    assert not missing, f"missing model modules: {missing}"
    undernamed = sorted(
        p.name for p in (REPO / "model").glob("*.py")
        if "_" in p.stem and p.name != "__init__.py"
    )
    assert not undernamed, f"model filenames should avoid underscores except __init__.py: {undernamed}"
    public_api = read("model/__init__.py")
    assert "PrismMemoryRefiner" in public_api and "PrismSelectiveRefiner" in public_api
    assert "M3C" not in public_api and "DTA" not in public_api
    pass_("model exposes PRISM memory and selective refiners with clean public names")


def check_deepseek_boundary() -> None:
    train_src = read("scripts/selectiveaffinity.py").lower()
    cache_src = read("scripts/mechanismcache.py").lower()
    forbidden = ("urllib.request", "chat/completions", "deepseek_api_key")
    leaked = [token for token in forbidden if token in train_src]
    assert not leaked, f"train/infer script contains live API tokens: {leaked}"
    for token in forbidden:
        assert token in cache_src, f"offline cache script missing expected DeepSeek client token: {token}"
    pass_("DeepSeek is isolated to offline mechanism cache construction")


def check_cuda_forward() -> None:
    assert torch.cuda.is_available(), "CUDA is required for PRISM integrity checks"
    device = torch.device("cuda")
    base = PrismMemoryRefiner(32, 24, d_model=64, n_heads=4, n_layers=1, ff_dim=128, mem_dim=5).to(device)
    base_out = base(
        torch.randn(8, 32, device=device),
        torch.randn(8, 24, device=device),
        torch.randn(8, device=device),
        torch.randn(8, 5, device=device),
    )
    assert base_out.shape == (8,), f"unexpected base shape: {tuple(base_out.shape)}"
    model = PrismSelectiveRefiner(
        32, 24, text_dim=16, domain_dim=4,
        d_model=64, n_heads=4, n_layers=1, ff_dim=128, mem_dim=5,
    ).to(device)
    out = model(
        torch.randn(8, 32, device=device),
        torch.randn(8, 24, device=device),
        torch.randn(8, device=device),
        torch.randn(8, 5, device=device),
        torch.randn(8, 16, device=device),
        torch.randn(8, 4, device=device),
    )
    assert out.shape == (8,), f"unexpected selective shape: {tuple(out.shape)}"
    pass_("PRISM CUDA forward path works for memory and selective refiners")


def main() -> None:
    check_root_layout()
    check_scripts_layout()
    check_dispatcher()
    check_model_modules()
    check_deepseek_boundary()
    check_cuda_forward()
    pass_("PRISM integrity checks completed")


if __name__ == "__main__":
    main()

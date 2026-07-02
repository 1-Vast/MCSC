"""Integrity checks for the PRISM mainline."""
from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model import MemoryResidualRefiner, SelectiveAffinityRefiner  # noqa: E402


MAINLINE_DIRS = {"config", "model", "scripts"}
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
    pass_("root layout keeps PRISM code and configuration separated")


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
        "model/domain.py",
        "model/fusion.py",
        "model/mechanismllm.py",
        "model/memory.py",
        "model/profiles.py",
        "model/residual.py",
        "model/selective.py",
        "model/text.py",
        "model/tokens.py",
    }
    missing = sorted(path for path in required if not (REPO / path).exists())
    assert not missing, f"missing model modules: {missing}"
    undernamed = sorted(
        p.name for p in (REPO / "model").glob("*.py")
        if "_" in p.stem and p.name != "__init__.py"
    )
    assert not undernamed, f"model filenames should avoid underscores except __init__.py: {undernamed}"
    public_api = read("model/__init__.py")
    assert "MemoryResidualRefiner" in public_api and "SelectiveAffinityRefiner" in public_api
    assert "M3C" not in public_api and "DTA" not in public_api
    pass_("model exposes function-named PRISM refiners")


def check_deepseek_boundary() -> None:
    api_tokens = ("urllib" + ".request", "requests" + ".post", "chat/" + "completions")
    offenders: list[str] = []
    # Scan both scripts/ and model/: the staged-generation orchestration/prompt/QC logic lives in
    # model/mechanismllm.py (importable from the train/infer path), so it must never gain a live
    # network client. Only scripts/mechanismcache.py may contain one.
    for directory in ("scripts", "model"):
        for path in (REPO / directory).glob("*.py"):
            if path.name == "mechanismcache.py":
                continue
            src = path.read_text(encoding="utf-8").lower()
            hits = [token for token in api_tokens if token in src]
            if hits:
                offenders.append(f"{directory}/{path.name}:{','.join(hits)}")
    assert not offenders, f"live API client tokens outside mechanismcache.py: {offenders}"
    train_src = read("scripts/selectiveaffinity.py").lower()
    mechanismllm_src = read("model/mechanismllm.py").lower()
    cache_src = read("scripts/mechanismcache.py").lower()
    forbidden = ("urllib" + ".request", "chat/" + "completions", "deepseek" + "_api_key")
    leaked = [token for token in forbidden if token in train_src]
    assert not leaked, f"train/infer script contains live API tokens: {leaked}"
    leaked_model = [token for token in forbidden if token in mechanismllm_src]
    assert not leaked_model, f"model/mechanismllm.py contains live API tokens: {leaked_model}"
    for token in forbidden:
        assert token in cache_src, f"offline cache script missing expected DeepSeek client token: {token}"
    pass_("DeepSeek is isolated to offline mechanism cache construction")


def check_cuda_forward() -> None:
    assert torch.cuda.is_available(), "CUDA is required for PRISM integrity checks"
    device = torch.device("cuda")
    base = MemoryResidualRefiner(32, 24, d_model=64, n_heads=4, n_layers=1, ff_dim=128, mem_dim=5).to(device)
    base_out = base(
        torch.randn(8, 32, device=device),
        torch.randn(8, 24, device=device),
        torch.randn(8, device=device),
        torch.randn(8, 5, device=device),
    )
    assert base_out.shape == (8,), f"unexpected base shape: {tuple(base_out.shape)}"
    model = SelectiveAffinityRefiner(
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


def check_selective_branch_separation() -> None:
    """Memory/domain context must affect trust gating, not the cross-modal residual branch."""
    device = torch.device("cuda")
    drug = torch.randn(6, 20, device=device)
    target = torch.randn(6, 18, device=device)
    mem = torch.randn(6, 5, device=device)
    domain = torch.randn(6, 4, device=device)

    model = SelectiveAffinityRefiner(
        20, 18, text_dim=0, domain_dim=4,
        d_model=32, n_heads=4, n_layers=1, ff_dim=64, mem_dim=5,
    ).to(device)
    model.eval()  # disable dropout so branch comparisons are not confounded by mask randomness
    with torch.no_grad():
        pair_with_context = model.pair_representation(drug, target, None, mem, domain)
        pair_without_context = model.pair_representation(drug, target, None, None, None)
        assert torch.allclose(pair_with_context, pair_without_context, atol=1e-5), (
            "memory/domain context must not enter the cross-modal residual representation"
        )
        _, gamma_with_context = model.residual_gate(drug, target, mem, None, domain)
        _, gamma_without_context = model.residual_gate(drug, target, None, None, None)
        assert not torch.allclose(gamma_with_context, gamma_without_context), (
            "memory/domain context must still affect the ResidualTrustGate"
        )
    assert not hasattr(model.base.space, "context_proj"), "mainline shared token must not allocate context injection"
    pass_("PRISM branch separation holds: residual representation is clean, trust gate uses context")


def main() -> None:
    check_root_layout()
    check_scripts_layout()
    check_dispatcher()
    check_model_modules()
    check_deepseek_boundary()
    check_cuda_forward()
    check_selective_branch_separation()
    pass_("PRISM integrity checks completed")


if __name__ == "__main__":
    main()

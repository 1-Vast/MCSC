"""Runtime support for command scripts.

This module owns repository paths, dataset loading, feature construction,
public mechanism text, DeepSeek cache handling, leakage auditing, and split
selection. Neural/model components stay in `model/`; scripts call this runtime.
"""
from __future__ import annotations

import ast
import gzip
import hashlib
import json
import os
import pickle
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

os.environ.setdefault("OMP_NUM_THREADS", "1")

from model.encode import HashSmilesEncoder, MechanismTextEncoder, MorganEncoder, split_normalize


# ---- settings.py ----
REPO = Path(__file__).resolve().parents[1]


def env_file() -> Path:
    value = os.getenv("DRUGTARGET_ENV_FILE", "").strip()
    if value:
        return Path(value).expanduser()
    return REPO / ".env"


def load_environment() -> Path:
    from dotenv import load_dotenv

    path = env_file()
    load_dotenv(path, override=False)
    return path


def repo_relative(path: str | Path) -> str:
    value = Path(path)
    try:
        return value.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return value.as_posix()


@dataclass
class EncoderSettings:
    drug: str = "morgan"  # morgan | hash
    morgan_radius: int = 2
    morgan_bits: int = 1024
    target: str = "kb"  # kb | deepseek | name
    target_dim: int = 256
    normalize: bool = True


@dataclass
class MemorySettings:
    k: int = 5
    temperature: float = 0.1
    mode: str = "both"  # drug | target | both


@dataclass
class RefinerSettings:
    hidden: Sequence[int] = (256, 128, 64)
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 300
    batch_size: int = 256


@dataclass
class RunSettings:
    encoder: EncoderSettings = field(default_factory=EncoderSettings)
    memory: MemorySettings = field(default_factory=MemorySettings)
    refiner: RefinerSettings = field(default_factory=RefinerSettings)
    seed: int = 42
    device: str = "auto"
    data_dir: str = "dataset/davis"
    cache_dir: str = "dataset/cache"
    record_dir: str = "outputs/prism"

    def repo_path(self, value: str | Path) -> Path:
        path = Path(value)
        resolved = path if path.is_absolute() else REPO / path
        return resolved

    @property
    def torch_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


default = RunSettings()

# ---- data.py ----
def load_davis(data_dir: Path) -> dict:
    """Load DAVIS and convert Kd in nM to pKd."""
    drug_dict = json.loads((data_dir / "ligands_can.txt").read_text())
    target_dict = json.loads((data_dir / "proteins.txt").read_text())

    drug_ids = list(drug_dict.keys())
    drug_smiles = [drug_dict[drug] for drug in drug_ids]
    target_ids = list(target_dict.keys())
    target_seqs = [target_dict[target] for target in target_ids]

    with open(data_dir / "Y", "rb") as handle:
        y_raw = pickle.load(handle, encoding="latin1")
    y_pkd = 9.0 - np.log10(np.maximum(y_raw, 1e-3))

    train_raw = ast.literal_eval((data_dir / "folds/train_fold_setting1.txt").read_text())
    test_raw = ast.literal_eval((data_dir / "folds/test_fold_setting1.txt").read_text())
    # train_fold_setting1.txt holds nested cross-validation folds; the default warm
    # training pool is ALL folds. (Previously only train_raw[0] was used, silently
    # dropping ~80% of the training pairs.) Flatten, then verify nothing was dropped.
    if train_raw and isinstance(train_raw[0], (list, tuple)):
        fold_sizes = [len(fold) for fold in train_raw]
        train_list = [idx for fold in train_raw for idx in fold]
    else:
        fold_sizes = [len(train_raw)]
        train_list = list(train_raw)
    train_idx = np.array(train_list, dtype=np.int64)
    if len(train_idx) != sum(fold_sizes):
        raise ValueError(
            f"DAVIS train folds dropped: kept {len(train_idx)} of {sum(fold_sizes)} pairs "
            f"across {len(fold_sizes)} folds"
        )
    test_idx = np.array(test_raw, dtype=np.int64)

    target_count = len(target_ids)
    train_drug = train_idx // target_count
    train_target = train_idx % target_count
    test_drug = test_idx // target_count
    test_target = test_idx % target_count

    print(f"[data] DAVIS: {len(drug_ids)} drugs x {len(target_ids)} targets")
    print(f"[data] train folds={len(fold_sizes)} sizes={fold_sizes} -> pool={len(train_idx)}")
    print(f"[data] pairs: train={len(train_idx)}, test={len(test_idx)}")
    print(f"[data] pKd range: [{y_pkd.min():.2f}, {y_pkd.max():.2f}]")

    return {
        "drugIds": drug_ids,
        "drugSmiles": drug_smiles,
        "targetIds": target_ids,
        "targetSeqs": target_seqs,
        "Ypkd": y_pkd,
        "trainD": train_drug,
        "trainT": train_target,
        "trainY": y_pkd[train_drug, train_target],
        "testD": test_drug,
        "testT": test_target,
        "testY": y_pkd[test_drug, test_target],
    }


def create_cold_splits(data: dict, cold_frac: float = 0.3, seed: int = 42) -> dict:
    """Create drug-cold and target-cold splits without overlapping held-out units."""
    drug_count = len(data["drugIds"])
    target_count = len(data["targetIds"])
    rng = np.random.RandomState(seed + 1)
    drug_grid, target_grid = np.meshgrid(
        np.arange(drug_count),
        np.arange(target_count),
        indexing="ij",
    )
    all_drug = drug_grid.ravel()
    all_target = target_grid.ravel()
    all_y = data["Ypkd"].ravel()

    all_drugs = np.arange(drug_count)
    cold_drugs = set(rng.choice(all_drugs, max(1, int(drug_count * cold_frac)), replace=False))
    drug_test_mask = np.isin(all_drug, list(cold_drugs))
    drug_cold = {
        "trainD": all_drug[~drug_test_mask],
        "trainT": all_target[~drug_test_mask],
        "trainY": all_y[~drug_test_mask],
        "testD": all_drug[drug_test_mask],
        "testT": all_target[drug_test_mask],
        "testY": all_y[drug_test_mask],
    }

    all_targets = np.arange(target_count)
    cold_targets = set(rng.choice(all_targets, max(1, int(target_count * cold_frac)), replace=False))
    target_test_mask = np.isin(all_target, list(cold_targets))
    target_cold = {
        "trainD": all_drug[~target_test_mask],
        "trainT": all_target[~target_test_mask],
        "trainY": all_y[~target_test_mask],
        "testD": all_drug[target_test_mask],
        "testT": all_target[target_test_mask],
        "testY": all_y[target_test_mask],
    }

    # Held-out isolation guards: a cold split must not share its held-out unit between
    # train and test (drug-cold: no shared drug; target-cold: no shared target).
    if set(np.unique(drug_cold["trainD"]).tolist()) & set(np.unique(drug_cold["testD"]).tolist()):
        raise ValueError("drug-cold split leaks: a drug appears in both train and test")
    if set(np.unique(target_cold["trainT"]).tolist()) & set(np.unique(target_cold["testT"]).tolist()):
        raise ValueError("target-cold split leaks: a target appears in both train and test")

    return {"drug-cold": drug_cold, "target-cold": target_cold}

# ---- leakage.py ----
# Kinase-inhibitor INNs (DAVIS is a kinase panel) + the -nib / -ciclib suffix catch-alls.
_DRUG_NAMES = [
    "imatinib", "dasatinib", "nilotinib", "bosutinib", "ponatinib", "gefitinib",
    "erlotinib", "afatinib", "lapatinib", "sorafenib", "sunitinib", "pazopanib",
    "vandetanib", "crizotinib", "vemurafenib", "dabrafenib", "ruxolitinib",
    "tofacitinib", "ibrutinib", "trametinib", "regorafenib", "cabozantinib",
    "axitinib", "nintedanib", "lenvatinib", "staurosporine",
]

# HARD = clear benchmark/drug/affinity-label leakage (not ordinary biology).
HARD_PATTERNS = {
    "drug_name": re.compile(r"\b(" + "|".join(_DRUG_NAMES) + r"|\w+tinib|\w+ciclib)\b", re.IGNORECASE),
    "affinity_metric": re.compile(r"\b(kd|ki|ic50|ec50|pkd)\b", re.IGNORECASE),
    "benchmark": re.compile(r"\b(davis|kiba|deepdta|bindingdb)\b", re.IGNORECASE),
    "protocol": re.compile(r"\b(train set|test set|held[- ]?out|data split|model prediction|predicted label)\b", re.IGNORECASE),
    "resistance": re.compile(r"\bresistance mutation\b", re.IGNORECASE),
}
# SOFT = could be legitimate biology (receptor-ligand affinity, endogenous inhibitor);
# counted as warnings only, never auto-excluded.
SOFT_PATTERN = re.compile(r"\b(inhibitor|binding affinity|nanomolar)\b", re.IGNORECASE)


def scan(text: str) -> dict[str, list[str]]:
    """Return hard-category -> sorted unique offending tokens for one text."""
    hits: dict[str, list[str]] = {}
    for category, pattern in HARD_PATTERNS.items():
        found = sorted({m.group(0).lower() for m in pattern.finditer(text)})
        if found:
            hits[category] = found
    return hits


def audit(ids: list[str], texts: list[str]) -> dict:
    """Audit per-target texts. Returns counts, soft warnings, and offender ids+hits."""
    offenders: dict[str, dict[str, list[str]]] = {}
    counts = {category: 0 for category in HARD_PATTERNS}
    soft = 0
    for tid, text in zip(ids, texts):
        hits = scan(text)
        if hits:
            offenders[tid] = hits
            for category in hits:
                counts[category] += 1
        if SOFT_PATTERN.search(text):
            soft += 1
    return {
        "nTargets": len(ids),
        "nOffenders": len(offenders),
        "hardCounts": counts,
        "softWarnings": soft,
        "offenderIds": sorted(offenders),
        "offenders": offenders,
    }


def write_record(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

# ---- knowledge.py ----
REPO = Path(__file__).resolve().parents[1]
KB = REPO / "dataset/kb"
STRING_INFO = KB / "raw/string/9606.protein.info.v12.0.txt.gz"
GO_GAF = KB / "raw/go/goa_human.gaf.gz"
GO_OBO = KB / "raw/go/go-basic.obo"
REACTOME_GMT = KB / "raw/reactome/reactome_symbols.gmt"


def _read_string_annotations() -> dict[str, str]:
    annotations: dict[str, str] = {}
    if not STRING_INFO.exists():
        return annotations
    with gzip.open(STRING_INFO, "rt") as handle:
        next(handle, None)
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                annotations[parts[1]] = parts[3]
    return annotations


def _read_go_terms(limit: int = 12) -> dict[str, list[str]]:
    if not GO_OBO.exists() or not GO_GAF.exists():
        return {}

    id_to_name: dict[str, str] = {}
    current: dict[str, str] | None = None
    with open(GO_OBO, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line == "[Term]":
                current = {}
            elif line.startswith("id: GO:") and current is not None:
                current["id"] = line[4:]
            elif line.startswith("name:") and current is not None:
                id_to_name[current.get("id", "")] = line[6:]

    terms: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    with gzip.open(GO_GAF, "rt") as handle:
        for line in handle:
            if line.startswith("!"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 8 and parts[8] in {"P", "F"}:
                symbol, go_id = parts[2], parts[4]
                if go_id in id_to_name and go_id not in seen[symbol]:
                    seen[symbol].add(go_id)
                    terms[symbol].append(id_to_name[go_id])
    return {gene: values[:limit] for gene, values in terms.items()}


def _read_reactome(limit: int = 10) -> dict[str, list[str]]:
    if not REACTOME_GMT.exists():
        return {}
    pathways: dict[str, list[str]] = defaultdict(list)
    with open(REACTOME_GMT, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = parts[0].replace("REACTOME_", "").replace("_", " ").title()
            for gene in parts[2:]:
                pathways[gene].append(name)
    return {gene: values[:limit] for gene, values in pathways.items()}


def build_knowledge_texts(target_names: list[str]) -> tuple[list[str], list[bool]]:
    """Build one public mechanism description per target symbol."""
    annotations = _read_string_annotations()
    go_terms = _read_go_terms()
    reactome = _read_reactome()

    texts: list[str] = []
    covered: list[bool] = []
    for name in target_names:
        parts = [annotations.get(name, f"Gene {name}.")]
        if name in go_terms:
            parts.append(f"GO: {', '.join(go_terms[name])}.")
        if name in reactome:
            parts.append(f"Reactome: {', '.join(reactome[name])}.")
        texts.append(" ".join(parts))
        covered.append(name in annotations or name in go_terms or name in reactome)
    return texts, covered


def build_knowledge_descriptors(
    target_names: list[str],
    seed: int = 7,
    target_dim: int = 256,
    device: str = "cpu",
) -> torch.Tensor:
    """Encode local KB target descriptions with deterministic hashing."""
    texts, covered = build_knowledge_texts(target_names)
    coverage = sum(covered) / max(len(covered), 1)
    print(f"[knowledge] targets with KB text: {sum(covered)}/{len(covered)} ({coverage:.1%})")
    return MechanismTextEncoder(target_dim, device).build(texts, center_mask=covered)

# ---- deepseek.py ----
PROMPT = """Describe the biological function of this human protein concisely.

Include:
- full protein name and family
- primary biological function
- key substrates, interactors, or pathways
- disease associations if public and well established

Use only public knowledge. Do not mention drugs, binding affinity, inhibition
data, benchmark datasets, split membership, or model predictions.

Protein: {name}
Sequence prefix: {seq}
"""


def call_deepseek(prompt: str, max_retries: int = 3) -> str:
    import requests

    env_path = load_environment()
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is missing. Put it in repository `.env` or set "
            f"DRUGTARGET_ENV_FILE. Checked: {repo_relative(env_path)}"
        )

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": int(os.getenv("DEEPSEEK_MAX_TOKENS", "400")),
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=90,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            if resp.status_code == 429:
                time.sleep(5)
                continue
        except Exception as exc:
            print(f"[deepseek] API error: {exc}")
        if attempt < max_retries - 1:
            time.sleep(2)
    return ""


def load_deepseek_texts(cache_path: Path) -> dict[str, str]:
    if not cache_path.exists():
        return {}
    return json.loads(cache_path.read_text(encoding="utf-8"))


def generate_deepseek_texts(
    target_names: list[str],
    target_seqs: list[str],
    cache_path: Path,
    limit: int | None = None,
) -> dict[str, str]:
    descriptions = load_deepseek_texts(cache_path)
    if descriptions:
        print(f"[deepseek] cached descriptions: {len(descriptions)}")

    pairs = list(zip(target_names, target_seqs))
    if limit is not None:
        pairs = pairs[:limit]
    missing = [(name, seq) for name, seq in pairs if name not in descriptions]
    if missing:
        print(f"[deepseek] querying {len(missing)} targets")
        for i, (name, seq) in enumerate(missing, start=1):
            desc = call_deepseek(PROMPT.format(name=name, seq=seq[:400]))
            descriptions[name] = desc if desc else f"Protein {name}. No public functional annotation."
            if i % 10 == 0:
                print(f"[deepseek] {i}/{len(missing)}")
            time.sleep(0.3)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(descriptions, indent=2, ensure_ascii=False), encoding="utf-8")
    return descriptions


def build_deepseek_descriptors(
    target_names: list[str],
    target_seqs: list[str],
    cache_path: Path,
    dim: int = 256,
    seed: int = 7,
    device: str = "cpu",
):
    descriptions = generate_deepseek_texts(target_names, target_seqs, cache_path)
    texts = [descriptions.get(name, f"Protein {name}.") for name in target_names]
    return MechanismTextEncoder(dim, device).build(texts)

# ---- pipeline.py ----




# Bump when the feature artifact format changes so stale caches are refused.
FEATURE_SCHEMA_VERSION = 2
# Keys that bind a checkpoint to the exact feature artifact it was trained on.
BINDING_KEYS = ("featureSchemaVersion", "drugIdsHash", "targetIdsHash", "drugFeatHash", "targetFeatHash")


def _hash_strings(items) -> str:
    digest = hashlib.sha1()
    for value in items:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()[:16]


def _hash_tensor(x: torch.Tensor) -> str:
    arr = x.detach().cpu().contiguous().to(torch.float32).numpy()
    return hashlib.sha1(arr.tobytes()).hexdigest()[:16]


def feature_binding(meta: dict) -> dict:
    """Extract the reproducibility binding from a feature- or checkpoint-settings dict."""
    return {key: meta.get(key) for key in BINDING_KEYS}


def load_data(settings: RunSettings) -> dict:
    return load_davis(settings.repo_path(settings.data_dir))


def feature_path(settings: RunSettings) -> Path:
    name = f"features-{settings.encoder.drug}-{settings.encoder.target}.pt"
    return settings.repo_path(settings.cache_dir) / name


def model_path(settings: RunSettings, seed: int, split_name: str = "warm", model_type: str = "direct") -> Path:
    name = f"model-{settings.encoder.drug}-{settings.encoder.target}-{split_name}-{model_type}-seed{seed}.pt"
    return settings.repo_path(settings.record_dir) / name


# Canonical (stable) DeepSeek cache name; older runs may have written an alternate name.
DEEPSEEK_CACHE_NAME = "deepseek-target-descriptions.json"
DEEPSEEK_CACHE_ALTERNATES = ("deepseekdescriptions30.json",)


def deepseek_cache_path(settings: RunSettings) -> Path:
    """Resolve the DeepSeek description cache. Prefer the canonical repository-relative
    name; fall back to a known existing alternate so a valid cache is never silently
    ignored (which would degrade `--target deepseek` to the name-only fallback). When no
    cache exists yet, return the canonical path for first-time generation."""
    cache_dir = settings.repo_path(settings.cache_dir)
    canonical = cache_dir / DEEPSEEK_CACHE_NAME
    if canonical.exists():
        return canonical
    for alt in DEEPSEEK_CACHE_ALTERNATES:
        candidate = cache_dir / alt
        if candidate.exists():
            return candidate
    return canonical


def build_drug_features(data: dict, settings: RunSettings) -> torch.Tensor:
    device = settings.torch_device
    if settings.encoder.drug == "morgan":
        encoder = MorganEncoder(
            settings.encoder.morgan_radius,
            settings.encoder.morgan_bits,
            device,
        )
    elif settings.encoder.drug == "hash":
        encoder = HashSmilesEncoder(settings.encoder.morgan_bits, device)
    else:
        raise ValueError(f"unknown drug encoder: {settings.encoder.drug}")
    return encoder.build(data["drugSmiles"])


def _audit_target_texts(settings, source, ids, texts, covered, exclude):
    """Audit target texts for benchmark/drug/affinity-label leakage, write a JSON record,
    and (for external DeepSeek text) exclude contaminated descriptions by replacing them
    with neutral text. KB text is warn-only so legitimate biology is not auto-deleted.

    Returns (texts, covered, summary) where summary captures the offender counts/action so
    build_features can stamp them into the feature metadata (no silent contamination)."""

    report = audit(ids, texts)
    report["source"] = source
    report["action"] = "exclude_offenders" if exclude else "warn_only"
    out = settings.repo_path(settings.cache_dir) / f"{source}-leak-audit.json"
    write_record(report, out)
    excluded = 0
    if report["nOffenders"]:
        print(f"[leakage] {source}: {report['nOffenders']}/{report['nTargets']} flagged "
              f"{report['hardCounts']} soft={report['softWarnings']} -> {repo_relative(out)}")
        if exclude:
            bad = set(report["offenderIds"])
            texts = [f"Protein {tid}." if tid in bad else t for tid, t in zip(ids, texts)]
            covered = [False if tid in bad else c for tid, c in zip(ids, covered)]
            excluded = len(bad)
            print(f"[leakage] {source}: excluded {excluded} contaminated descriptions (neutral fallback)")
    else:
        print(f"[leakage] {source}: clean (0 hard hits, {report['softWarnings']} soft) -> {repo_relative(out)}")
    summary = {
        "source": source,
        "action": report["action"],
        "nTargets": report["nTargets"],
        "nOffenders": report["nOffenders"],
        "hardCounts": report["hardCounts"],
        "softWarnings": report["softWarnings"],
        "excluded": excluded,
        "auditFile": repo_relative(out),
    }
    return texts, covered, summary


def target_texts(data: dict, settings: RunSettings) -> tuple[list[str], list[bool], dict]:
    ids = data["targetIds"]
    if settings.encoder.target == "kb":

        texts, covered = build_knowledge_texts(ids)
        return _audit_target_texts(settings, "kb", ids, texts, covered, exclude=False)
    if settings.encoder.target == "deepseek":

        cache_path = deepseek_cache_path(settings)
        descriptions = load_deepseek_texts(cache_path)
        if not descriptions:
            raise SystemExit(
                "--target deepseek requested but no cached descriptions found under "
                f"{repo_relative(settings.repo_path(settings.cache_dir))}/ "
                f"(expected {DEEPSEEK_CACHE_NAME}); run `python main.py api` first"
            )
        texts = [descriptions.get(name, f"Protein {name}.") for name in ids]
        covered = [name in descriptions for name in ids]
        return _audit_target_texts(settings, "deepseek", ids, texts, covered, exclude=True)
    if settings.encoder.target == "name":
        n = len(ids)
        summary = {"source": "name", "action": "none", "nTargets": n, "nOffenders": 0,
                   "hardCounts": {}, "softWarnings": 0, "excluded": 0, "auditFile": None}
        return [f"Protein {name}." for name in ids], [False] * n, summary
    raise ValueError(f"unknown target source: {settings.encoder.target}")


def build_target_features(data: dict, settings: RunSettings) -> tuple[torch.Tensor, list[bool], dict]:
    texts, covered, audit = target_texts(data, settings)
    feat = MechanismTextEncoder(settings.encoder.target_dim, settings.torch_device).build(
        texts,
        center_mask=covered,
    )
    return feat, covered, audit


def build_features(settings: RunSettings) -> dict:
    data = load_data(settings)
    drug_feat = build_drug_features(data, settings)
    target_feat, target_covered, target_audit = build_target_features(data, settings)
    # Features are saved RAW. Normalization is applied SPLIT-AWARE at train/infer (fit on
    # train-visible drugs/targets) so held-out cold units never influence the statistics.
    settings_meta = {
        "drug": settings.encoder.drug,
        "target": settings.encoder.target,
        "targetDim": settings.encoder.target_dim,
        "morganBits": settings.encoder.morgan_bits,
        "normalize": settings.encoder.normalize,
        "normalizeBasis": "split_train_visible (applied at train/infer)",
        "featuresRaw": True,
        "featureSchemaVersion": FEATURE_SCHEMA_VERSION,
        "drugIdsHash": _hash_strings(data["drugIds"]),
        "targetIdsHash": _hash_strings(data["targetIds"]),
        "drugFeatHash": _hash_tensor(drug_feat),
        "targetFeatHash": _hash_tensor(target_feat),
        # Leakage-audit summary + how many targets carry real (non-fallback) text. A silent
        # drop to name-only fallback is then visible in the record instead of hidden.
        "targetCoverage": f"{int(sum(target_covered))}/{len(target_covered)}",
        "targetAudit": target_audit,
    }
    if settings.encoder.target == "deepseek":
        # DeepSeek descriptions are model-generated external text: treat as UNSAFE until a
        # human review or clean regeneration. Hard offenders are excluded above; record the
        # source cache + safety state so no downstream claim treats them as vetted.
        settings_meta["deepseekCache"] = repo_relative(deepseek_cache_path(settings))
        settings_meta["targetTextSafety"] = "unsafe_until_reviewed"
    return {
        "drugFeat": drug_feat.cpu(),
        "targetFeat": target_feat.cpu(),
        "drugIds": data["drugIds"],
        "targetIds": data["targetIds"],
        "settings": settings_meta,
    }


def save_features(settings: RunSettings) -> Path:
    path = feature_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_features(settings)
    torch.save(payload, path)
    meta = {
        "featureFile": repo_relative(path),
        "drugCount": len(payload["drugIds"]),
        "targetCount": len(payload["targetIds"]),
        "settings": payload["settings"],
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def load_features(settings: RunSettings) -> dict:
    path = feature_path(settings)
    if not path.exists():
        raise SystemExit(f"missing features: {path}; run `python main.py preprocess` first")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    meta = payload.get("settings", {}) if isinstance(payload, dict) else {}
    # Refuse stale caches: they must declare raw features and the current schema version.
    # Old globally-normalized caches (no featuresRaw) would silently double-normalize.
    if not meta.get("featuresRaw") or meta.get("featureSchemaVersion") != FEATURE_SCHEMA_VERSION:
        raise SystemExit(
            f"stale/incompatible feature cache: {repo_relative(path)} "
            f"(featuresRaw={meta.get('featuresRaw')}, schema={meta.get('featureSchemaVersion')}, "
            f"expected featuresRaw=True schema={FEATURE_SCHEMA_VERSION}). Rerun: "
            f"python main.py preprocess --target {settings.encoder.target} --drug {settings.encoder.drug}"
        )
    payload["drugFeat"] = payload["drugFeat"].to(settings.torch_device)
    payload["targetFeat"] = payload["targetFeat"].to(settings.torch_device)
    return payload


def create_family_cold_split(data: dict, target_feat, n_families: int = 8,
                             cold_frac: float = 0.3, seed: int = 42) -> dict:
    """Hold out whole mechanism families from public target descriptors."""

    from sklearn.cluster import KMeans

    feats = target_feat.detach().cpu().numpy() if hasattr(target_feat, "detach") else np.asarray(target_feat)
    target_count = len(data["targetIds"])
    drug_count = len(data["drugIds"])
    k = int(min(n_families, max(2, target_count)))
    fam = KMeans(n_clusters=k, random_state=seed + 5, n_init=10).fit(feats).labels_
    rng = np.random.RandomState(seed + 5)
    n_hold = max(1, int(k * cold_frac))
    held = set(rng.choice(np.arange(k), n_hold, replace=False).tolist())
    test_targets = set(np.where(np.isin(fam, list(held)))[0].tolist())

    drug_grid, target_grid = np.meshgrid(np.arange(drug_count), np.arange(target_count), indexing="ij")
    all_d, all_t, all_y = drug_grid.ravel(), target_grid.ravel(), data["Ypkd"].ravel()
    test_mask = np.isin(all_t, list(test_targets))
    split = {
        "trainD": all_d[~test_mask], "trainT": all_t[~test_mask], "trainY": all_y[~test_mask],
        "testD": all_d[test_mask], "testT": all_t[test_mask], "testY": all_y[test_mask],
        "nFamilies": k, "heldFamilies": sorted(held), "nTestTargets": len(test_targets),
    }
    if set(np.unique(split["trainT"]).tolist()) & test_targets:
        raise ValueError("family-cold split leaks: a held-out family target appears in train")
    return split


def select_split(data: dict, split_name: str, seed: int, target_feat=None) -> dict:
    if split_name == "warm":
        return {
            "name": "warm",
            "trainD": data["trainD"],
            "trainT": data["trainT"],
            "trainY": data["trainY"],
            "testD": data["testD"],
            "testT": data["testT"],
            "testY": data["testY"],
        }
    if split_name == "family-cold":
        if target_feat is None:
            raise SystemExit("family-cold requires target features; pass target_feat to select_split")
        return {"name": "family-cold", **create_family_cold_split(data, target_feat, seed=seed)}
    cold_splits = create_cold_splits(data, seed=seed)
    if split_name not in cold_splits:
        raise ValueError(f"unknown split: {split_name}")
    split = cold_splits[split_name]
    return {"name": split_name, **split}


def normalize_for_split(drug_feat, target_feat, split: dict, do_norm: bool):
    """Split-aware normalization: fit the column mean on train-VISIBLE drugs/targets
    (unique ids appearing in the split's TRAIN pairs) and apply it to all rows. Held-out
    cold drugs/targets never enter the fitted statistics. Returns (drug, target, basis)."""
    if not do_norm:
        return drug_feat, target_feat, {"normalized": False}
    drug_rows = torch.from_numpy(np.unique(split["trainD"])).long().to(drug_feat.device)
    target_rows = torch.from_numpy(np.unique(split["trainT"])).long().to(target_feat.device)
    drug_norm = split_normalize(drug_feat, drug_rows)
    target_norm = split_normalize(target_feat, target_rows)
    basis = {
        "normalized": True,
        "fit": "split_train_visible",
        "drugFitRows": int(drug_rows.numel()),
        "drugTotal": int(drug_feat.shape[0]),
        "targetFitRows": int(target_rows.numel()),
        "targetTotal": int(target_feat.shape[0]),
    }
    return drug_norm, target_norm, basis


def validation_indices(split: dict, split_name: str, seed: int, val_frac: float = 0.2, target_feat=None):
    """Indices (into the split's train arrays) for an early-stopping validation set, matched
    to the split's generalization regime:
      - warm        -> pair-random
      - drug-cold   -> hold out train DRUG units
      - target-cold -> hold out train TARGET units
      - family-cold -> hold out whole MECHANISM FAMILIES of train targets (cold_family)
    Returns (train_idx, val_idx, basis). Uses train-visible targets + public KB text only."""
    n = len(split["trainD"])
    if split_name == "family-cold":
        from sklearn.cluster import KMeans
        from sklearn.model_selection import train_test_split

        # Family-aware validation: cluster TRAIN-visible targets' public KB descriptors into
        # families and hold out whole families, so validation reflects unseen-family
        # generalization (matching the family-cold TEST). No test rows/families/labels used.
        if target_feat is None:
            raise SystemExit("family-cold validation requires target_feat (train-visible KB descriptors)")

        train_targets = np.unique(split["trainT"])
        feats = target_feat.detach().cpu().numpy() if hasattr(target_feat, "detach") else np.asarray(target_feat)
        kfam = int(min(6, max(2, len(train_targets))))
        fam = KMeans(n_clusters=kfam, random_state=seed + 23, n_init=10).fit(feats[train_targets]).labels_
        rng = np.random.RandomState(seed + 23)
        n_hold = max(1, int(kfam * val_frac))
        held_fams = set(rng.choice(np.arange(kfam), n_hold, replace=False).tolist())
        held_targets = set(train_targets[np.isin(fam, list(held_fams))].tolist())
        val_mask = np.isin(split["trainT"], list(held_targets))
        train_idx = np.where(~val_mask)[0]
        val_idx = np.where(val_mask)[0]
        if len(train_idx) == 0 or len(val_idx) == 0:  # degenerate guard
            train_idx, val_idx = train_test_split(np.arange(n), test_size=val_frac, random_state=seed, shuffle=True)
            return train_idx, val_idx, "pair_random_fallback"
        return train_idx, val_idx, "cold_family"
    if split_name in ("drug-cold", "target-cold"):
        from sklearn.model_selection import train_test_split

        key = "trainD" if split_name == "drug-cold" else "trainT"
        units = np.unique(split[key])
        rng = np.random.RandomState(seed + 17)
        n_val = max(1, int(len(units) * val_frac))
        held = set(rng.choice(units, n_val, replace=False).tolist())
        val_mask = np.isin(split[key], list(held))
        train_idx = np.where(~val_mask)[0]
        val_idx = np.where(val_mask)[0]
        if len(train_idx) == 0 or len(val_idx) == 0:  # degenerate guard
            train_idx, val_idx = train_test_split(np.arange(n), test_size=val_frac, random_state=seed, shuffle=True)
            return train_idx, val_idx, "pair_random_fallback"
        return train_idx, val_idx, f"cold_{'drug' if split_name == 'drug-cold' else 'target'}"
    from sklearn.model_selection import train_test_split

    train_idx, val_idx = train_test_split(np.arange(n), test_size=val_frac, random_state=seed, shuffle=True)
    return train_idx, val_idx, "pair_random"


def numpy_seed(seed: int) -> np.random.RandomState:
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    return np.random.RandomState(seed)

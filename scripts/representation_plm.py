"""PLM representation probe (KIBA-focused). Compute frozen ESM-2 mean-pooled target embeddings
DIRECTLY from KIBA proteins.txt (guaranteed aligned; no dependence on external caches), cache with
a sha256 manifest, then prior-level probe vs current (aac_dip) and ctriad on KIBA cold splits.
Leakage-safe: ESM embeddings are deterministic functions of the protein sequence only (no labels).

Run: D:/anaconda/envs/drug/python.exe scripts/representation_plm.py
"""
from __future__ import annotations
import json, pickle, hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
import sys as _sys; _sys.path.insert(0, str(REPO))
from model.memory import InteractionMemory
from model import prior as ps
from model.metrics import compute_metrics
from sklearn.cluster import KMeans
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
R2 = lambda y, p: compute_metrics(y, p)["r2"]
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]
AA = "ACDEFGHIKLMNPQRSTVWY"; AA_I = {a: i for i, a in enumerate(AA)}
CACHE = REPO / "dataset" / "cache"; CACHE.mkdir(parents=True, exist_ok=True)
MANIFEST = REPO / "config" / "representation-manifest.json"


def l2(X): return (X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)).astype(np.float32)


def aac_dip(seqs):
    n = len(seqs); aac = np.zeros((n, 20), np.float32); dip = np.zeros((n, 400), np.float32)
    for i, s in enumerate(seqs):
        s = [c for c in s if c in AA_I]
        for c in s: aac[i, AA_I[c]] += 1
        for a, b in zip(s, s[1:]): dip[i, AA_I[a] * 20 + AA_I[b]] += 1
        if len(s): aac[i] /= len(s)
        if len(s) > 1: dip[i] /= (len(s) - 1)
    return l2(np.concatenate([aac, dip], 1))


CT = {**{a: 0 for a in "AGV"}, **{a: 1 for a in "ILFP"}, **{a: 2 for a in "YMTS"},
      **{a: 3 for a in "HNQW"}, **{a: 4 for a in "RK"}, **{a: 5 for a in "DE"}, **{a: 6 for a in "C"}}


def ctriad(seqs):
    n = len(seqs); X = np.zeros((n, 343), np.float32)
    for i, s in enumerate(seqs):
        cl = [CT[c] for c in s if c in CT]
        for a, b, c in zip(cl, cl[1:], cl[2:]): X[i, a * 49 + b * 7 + c] += 1
        if X[i].sum(): X[i] /= X[i].sum()
    return l2(X)


def esm_embed(seqs, model_id, max_len=1022):
    """Frozen ESM-2 mean-pooled embedding from sequence; cached. Returns (N, dim) float32."""
    key = hashlib.sha256((model_id + "|" + "|".join(s[:max_len] for s in seqs)).encode()).hexdigest()[:12]
    cache = CACHE / f"esm_{model_id.split('/')[-1]}_n{len(seqs)}_{key}.npy"
    if cache.exists():
        return np.load(cache), cache, key
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(DEV).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), 4):
            batch = [s[:max_len] for s in seqs[i:i + 4]]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(DEV)
            h = model(**enc).last_hidden_state  # (b, L, d)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (h * mask).sum(1) / mask.sum(1).clamp_min(1.0)
            out.append(pooled.cpu().numpy())
    emb = np.concatenate(out, 0).astype(np.float32)
    np.save(cache, emb)
    return emb, cache, key


def marg(D, Y, q, g):
    s, c = defaultdict(float), defaultdict(int)
    for d, y in zip(D, Y): s[int(d)] += float(y); c[int(d)] += 1
    return np.array([s[int(d)] / c[int(d)] if c[int(d)] else g for d in q])


def boot(d):
    d = np.asarray(d); b = np.random.RandomState(0).choice(d, (10000, len(d))).mean(1)
    return [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]


def main():
    print("device", DEV)
    K = REPO / "dataset" / "kiba"
    Y = np.asarray(pickle.load(open(K / "Y", "rb"), encoding="latin1"), dtype=float)
    seqs = list(json.loads((K / "proteins.txt").read_text()).values())
    kdrug = np.load(K / "morgan_cache_1024.npy").astype(np.float32)
    manifest = {"dataset": "kiba", "n_targets": len(seqs), "reps": {}}

    REPS = {"current(aac_dip)": aac_dip(seqs), "ctriad": ctriad(seqs)}
    for mid in ["facebook/esm2_t6_8M_UR50D", "facebook/esm2_t30_150M_UR50D"]:
        emb, cache, key = esm_embed(seqs, mid)
        name = "esm_" + mid.split("/")[-1].replace("esm2_", "")
        REPS[name] = l2(emb)
        manifest["reps"][name] = {"model": mid, "dim": int(emb.shape[1]), "cache": cache.name,
                                  "sha12": key, "pooling": "mean", "max_len": 1022, "source": "sequence-only frozen ESM-2"}
        print(f"{name}: {emb.shape} cache={cache.name}")
    # also concat current+best-esm later if esm wins
    di, ti = np.where(np.isfinite(Y)); out = {}
    for split in ["target-cold", "cluster-cold"]:
        per = {r: [] for r in REPS}; blends = {r: [] for r in REPS}; oheads = {r: [] for r in REPS}
        for seed in SEEDS:
            rng = np.random.RandomState(seed + 5); n_t = len(seqs)
            if split == "target-cold":
                held = set(rng.choice(n_t, max(1, int(n_t * 0.2)), replace=False).tolist())
            else:
                lab = KMeans(n_clusters=8, random_state=seed + 5, n_init=10).fit(REPS["current(aac_dip)"]).labels_
                hc = set(rng.choice(8, max(1, int(8 * 0.3)), replace=False).tolist())
                held = set(np.where(np.isin(lab, list(hc)))[0].tolist())
            tm = np.isin(ti, list(held)); trD, trT, teD, teT = di[~tm], ti[~tm], di[tm], ti[tm]
            trY, teY = Y[trD, trT], Y[teD, teT]
            rv = np.random.RandomState(seed + 99); trg = np.unique(trT)
            valt = set(rv.choice(trg, max(1, int(len(trg) * 0.2)), replace=False).tolist())
            vmsk = np.isin(trT, list(valt)); fD, fT, fY = trD[~vmsk], trT[~vmsk], trY[~vmsk]
            vD, vT, vY = trD[vmsk], trT[vmsk], trY[vmsk]
            g = float(trY.mean()); gtr = float(fY.mean())
            te_marg = marg(trD, trY, teD, g); v_marg = marg(fD, fY, vD, gtr)
            df = torch.from_numpy(kdrug)
            for r, T in REPS.items():
                tf = torch.from_numpy(T)
                mem_full = InteractionMemory(df, tf, trD, trT, trY, normalize=False)
                mem_fit = InteractionMemory(df, tf, fD, fT, fY, normalize=False)
                te_fine = mem_full.predict(teD, teT); v_fine = mem_fit.predict(vD, vT)
                w, _, _ = ps.select_blend_weight_on_validation(v_fine, v_marg, vY, R2)
                bl = ps.global_blend(te_fine, te_marg, w)
                orc = np.where(np.abs(te_fine - teY) <= np.abs(te_marg - teY), te_fine, te_marg)
                per[r].append(R2(teY, te_fine)); blends[r].append(R2(teY, bl)); oheads[r].append(R2(teY, orc) - R2(teY, bl))
        cur = np.array(blends["current(aac_dip)"])
        out[f"KIBA/{split}"] = {}
        print(f"\n=== KIBA/{split} (prior-level, n=8) ===")
        for r in REPS:
            b = np.array(blends[r]); d = b - cur
            out[f"KIBA/{split}"][r] = {"fineR2": round(float(np.mean(per[r])), 4), "blendR2": round(float(b.mean()), 4),
                "delta_vs_current": round(float(d.mean()), 4), "wins": f"{int((d > 0).sum())}/8",
                "ci95": boot(d), "oracle_headroom": round(float(np.mean(oheads[r])), 4)}
            print(f"  {r:22s} fine={np.mean(per[r]):.3f} blend={b.mean():.3f} dVcur={d.mean():+.4f} {out[f'KIBA/{split}'][r]['wins']} CI{out[f'KIBA/{split}'][r]['ci95']} oraclehead={out[f'KIBA/{split}'][r]['oracle_headroom']:.3f}")
        (REPO / "doc" / "representation-next-results.json").write_text(json.dumps(out, indent=2))
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print("\nwrote doc/representation-next-results.json + config/representation-manifest.json")


if __name__ == "__main__":
    main()

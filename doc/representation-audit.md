# Representation audit (consolidated)

Canonical audit entry-point. Full detail: `doc/representation-inventory.md` (Part A) and
`doc/representation-leakage-audit.md` (Part B).

## Verdict
All representations used this round are **leakage-safe, label-free**:
- **Morgan** (drug, RDKit from SMILES), **aac_dip**, **ctriad**, **ctd** (target, deterministic
  functions of the protein sequence) — never read affinity/Kd/Ki/IC50, benchmark labels, split
  membership, or model outputs. Only fitted statistic is split-aware mean centering on
  **train-visible targets only** + per-row L2.
- **KB mechanism-text hash** (DAVIS current): public STRING/GO/Reactome text → hash; warn-only scan,
  no hard label hits.
- **Prior signals** (fine memory, drug marginal, blend weight): train labels only, LOO/exclude_self,
  validation-only weight — leakage-controlled, not feature sources.

## Not used (flagged)
- **DeepSeek LLM target text**: unsafe-until-reviewed (11/30 drug-name flags); excluded this pass.
- **ESM/ProtT5 PLM embeddings**: safe-if-available but not installed; deferred to the next round
  (candidate KIBA lever where sequence-composition descriptors saturate).

## API-derived features
None used this round. The only API source (DeepSeek) remains unsafe-until-reviewed and is excluded.

Conclusion: the promoted candidate (conjoint triad) is sequence-only and passes the audit; `main.py
check` (14/14) and `verifygate` remain green.

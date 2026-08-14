---
name: per-gene-lr-strand-options
description: "TB-AST per-gene logistic-regression strand — what's built and the design options to discuss (extend coverage, Bacformer-free multivariate-LR benchmark, sparse/concat attention heads)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3240ca4c-5459-4b1e-97f4-9f2775c0f04c
---

The **per-gene logistic-regression strand** of Task-7 (snp_embeddings). For each gene, fit an L2 LR
on the gene's 960-d **ESM-C** vector predicting resistance; leakage-safe (5-fold out-of-fold on
train, full-fit for val/eval). Built by `build_per_gene_lr_store.py`. On the 1000-genome manifest
(core genes only, single-copy >95%): rpoB 0.9996 (circular — manifest defines R by rpoB), **katG
0.935 (INH), embB 0.908 (EMB)**, sharp falloff (15/1738 > 0.8) → co-resistance/MDR-lineage confound.
Already plumbed as an attention-head **panel channel** (`att_head`: steers the gate; `e2e`: into the
pooled value) — the supervised cousin of the surprisal panel.

**Design options the user wants to discuss (in a separate Claude *app* chat, 2026-06-16) — NOT yet
decided:**
1. **Extend coverage** — from core (>95%) to **all genes >10% prevalence**; allow a **0** (or a
   learned "absent" value) when the gene is absent → presence-absence-aware per-gene channel.
2. **Bacformer-free benchmark** — a **multivariate LR on the common genes** (one feature per common
   gene: its per-gene LR prob, or its pooled ESM-C vector). Bypasses Bacformer entirely. Weak on
   many fronts (relies on Prokka gene-name homology; within-species; poor on novel/divergent genes)
   but a clean **yardstick**: if the Bacformer versions can't beat it, contextualisation adds nothing
   for AST.
3. **Per-gene LR prob as an explicit attention-head panel** (implemented) — does the supervised
   channel help the head route to rpoB on the FULL eval? Cheaper to extend to full than the
   unsupervised surprisal panel (which needs the deferred ~1,800-GPU-h scan, see
   [[tb-mini-set-0977-confound-vs-bug]]).
4. **Selective / sparse attention heads** — drop meaningless genes so the head attends ≪4,000; or a
   structured read-out concatenating top-1/2/3 single-gene embeddings + mean(top-10) + mean(top-50)
   + overall mean → e.g. a 6×960 head. (`MultiheadAttentionPool` is already scaffolded alongside the
   gated-MIL pool in `tl/train/attention_pool.py`.)

**Core tension for the discussion:** *selection* (model learns to attend the causal gene — general,
but has so far failed to route) vs *injection* (model is handed the gene via a panel/LR channel —
works, but risks a species/drug-bound crutch). Benchmark target is open: beat the FT mean (0.905)?
match one-hot RRDR (0.960)? reach the SNP ceiling (~0.96–0.97)? Full briefing for the external chat:
`src/snp_embeddings/docs/readout_design_brief.md`. This (code) agent meanwhile follows the A/C
evals + the surprisal/attention-head path. See [[tb-vs-kp-chromosomal-hgt-contrast]].

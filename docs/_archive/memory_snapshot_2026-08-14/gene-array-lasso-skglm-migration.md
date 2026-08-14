---
name: gene-array-lasso-skglm-migration
description: gene_array_lasso fit engine migrated groupyr→skglm (2026-06-26); the handover plan + Phase-1 build order (A–D) and AGREED design choices
metadata: 
  node_type: memory
  type: project
  originSessionId: 3240ca4c-5459-4b1e-97f4-9f2775c0f04c
---

The gene_array_lasso fit engine is being migrated from groupyr to **skglm** (see [[skglm-sgl-api-stack]]),
because groupyr can't scale ([[groupyr-sgl-quadratic-memory-failure]]). skglm is the same estimator
(sparse-group lasso), just maintained/SOTA/scalable.

**Why:** A package change to a statistical method MUST be reasoned + discussed (see the global
"Statistical method is a decision" rule). This one was: the user authored the migration plan
`src/gene_array_lasso/skglm_sgl_handover_plan.md` (the source of truth). Contrast the earlier *celer* misstep —
I swapped to a different estimator (squared-loss group lasso) unilaterally; that was wrong and reverted.

**AGREED design choices** (from the handover doc):
- **Absence encoding = zero embedding block** for absent genes (sparse CSR, never stored). Group-L2 gives an
  all-zero block zero group-norm ⇒ ignored for *selection*. Do **not** mean/centroid-impute. Explicit
  presence/absence matrix is a downstream option, not Stage 1.
- **Grouping must be a SWAPPABLE input** (Panaroo orthogroups now; embedding clusters later — Stage 3). Never
  hard-wire Panaroo into the estimator.
- **>5% prevalence filter is scaffolding only** (fast smoke); biologically wrong long-term (target AMR genes are
  individually rare, collectively penetrant).
- Memory levers ranked: CSR sparse X (dominant) → float32 (validate, don't assume) → warm-started decreasing
  alpha path → grouping redefinition → out-of-core is LAST RESORT (skglm has none; explicit build).

**Phase-1 build order (A–D; E–F–G follow-on):** A verify min embedding L2-norm ≫ 0; B swappable group plumbing
+ off-by-960 hard test (highest-risk correctness); C Stage-1 smoke (100 genomes >5%) with 4 gates (gap closes /
indexing / float32==float64 / sparse==dense) + solver choice + de-risk read; D **memory-characterisation map**
(RSS-vs-nnz law, (N,p) cutpoints, alpha-vs-RSS; fresh process per cell; CONVERGED/OOM/MAXITER). Reuses the
already-built `build_gene_embedding_array.py` (CSR `X.npz`) + Panaroo GPAs + `colistin_p5` array.

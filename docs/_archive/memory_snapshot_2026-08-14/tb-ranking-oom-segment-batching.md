---
name: tb-ranking-oom-segment-batching
description: "TB per-segment LR ranking OOM'd (clonal→dense core matrices); fixed with opt-in segment batching (engine change a2db345)"
metadata: 
  node_type: memory
  type: project
  originSessionId: aac091b8-20e4-4661-ab5d-762fe4b1c697
  modified: 2026-08-06T20:41:16.296Z
---

**The per-segment LR ranking (`per_segment_lr` / `collect_segment_matrices`) OOM'd a 128 GB node on TB
(2026-08-06).** Root cause: the two-pass sweep materialised **all** in-band single-copy **core** segment
matrices at once. TB is **clonal**, so ~1946 core genes are near-ubiquitous → each matrix is a dense
`n_read×960` block (log: *"materialised 1946 core matrices over 4535 read genomes"* → Killed), ≈34 GB raw +
list/vstack + joblib workers → over 128 GB. **Kp did not OOM** because its **accessory** genome is sparser
(far fewer near-ubiquitous core genes) — so this only surfaced on TB. Both rifampin AND ethionamide per-gene
tasks OOM'd (rifampin at 1h01, ethionamide at ~40min) — not drug-specific, cohort-density specific.

**David's call:** don't throw RAM at it (rejected a 420 GB ask) — **batch it**. Durable engine fix.

**Fix — commit `a2db345`, branch `refactor/consolidate-engine` (shared engine; affects every organism):**
- Split the sweep in `engine/embedding/segment_embedding_extractor.py`: `sweep_core_prevalence` (pass 1,
  vector-free — core set + prevalence) + `collect_core_subset` (pass 2 restricted to a slice of core).
  `collect_segment_matrices` kept as a behavior-identical wrapper.
- `per_segment_lr.rank_segments` gains opt-in **`--segment-batch-size`**: materialise + fit the core set in
  slices (one genome scan per batch, only that batch's matrices held). Per-segment fits are **independent**,
  so `fitted`/`read_ids`/`n_core` are **IDENTICAL** to single-shot — proved by two tests (batched==single-shot
  k=1,2,3; core-partition==single-shot). Cost = K× genome I/O (K = ceil(n_core/batch)).
- Launchers: `build_per_gene_lr_ranking.sh` defaults **`SEG_BATCH=800`** (→ ~1 batch on Kp-scale, ~3 on TB);
  `build_per_igr/upstream/per_unit` accept `SEG_BATCH` (unset=off). 800-batch peaks ≈ `n_read×960×4 B ×800`.
- 128 GB stays the default node size; batching bounds peak, no RAM increase.

**Deployed:** fast-forwarded CSD3 checkout `~/workspace/BacPredict` → `a2db345` (clean; other agents' dirty
`uv.lock`/`pixi.lock`/untracked `src/kleb_ast` untouched — incoming commits touch none of them). Resubmitted
rif+eth per_gene (imputed+carrier) + upstream+per_igr (imputed_full, `BACLM_DIR=…/baclm_reembed`) all with
`SEG_BATCH=800` at the normal **128 G/32c/6h icelake-himem**. per_unit completed unbatched (fit fine).

Related: [[bacpredict-lr-scoring-audit]] (the TB fan-out this unblocks) · the ranking feeds the ladder + the
6-figure holdout suite.

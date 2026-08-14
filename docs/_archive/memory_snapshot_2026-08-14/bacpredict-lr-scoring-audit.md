---
name: bacpredict-lr-scoring-audit
description: ★POST-COMPACT ANCHOR — Kp AMR LR-panel scoring audit DONE+committed; head-vs-mean to add; TB fan-out next (Isambard re-certified)
metadata: 
  node_type: memory
  type: project
  originSessionId: aac091b8-20e4-4661-ab5d-762fe4b1c697
  modified: 2026-08-06T11:17:59.424Z
---

**Kp AMR figures were audited + corrected after David flagged surprising results (2026-08-05).** A 3-agent
read-only code trace resolved two concerns; fixes committed on branch `refactor/consolidate-engine`, **HEAD
`bdbd65f`** (pushed). The reference doc is **`src/bacpredict/docs/LR_SCORING_AUDIT.md`** (scope/imputation/
feature/split/metric per figure) — read it first.

**Audit verdict (both durable facts):**
- **Ladder is CORRECT.** Rung-1 `ft_mean` = a fresh L2 LR *re-probing the frozen FT genome-mean* (NOT the
  deployed head's logit) — `build_amr_ladder._score` → `fit_one_segment`. So `ft_mean ≈ head − 0.01…0.03` is
  the expected head-vs-mean gap, not a regression (cefotaxime 0.983 mean vs ~0.99 head). Near-zero gene/IGR
  lift = **genuine redundancy** with the FT-mean, PROVEN by the same pipeline giving large lift on
  non-redundant blocks: **isoniazid katG +0.028, ethionamide fabG1-promoter +0.057, kanamycin rrs +0.022**.
- **Catalogue comparison panel (was "driver panel") mis-compared** an all-sample presence one-hot vs a
  **carrier-only** embedding LR → penetrant HGT genes go single-class → blank/low baclm bar (artifact, not
  model failure). David chose: KEEP carrier-only, RELABEL as a within-carrier question (done).

**What's committed (bdbd65f):** LR_SCORING_AUDIT.md; catalogue panel renamed + relabelled (`plot_catalogue_
vs_embeddings.plot_drug_panel`); ranking figures (`plot_segment_ranking`, `plot_igr_ranking`,
`plot_causal_comparison`) now **DISPLAY deployment-holdout `eval_auroc_` while SELECTING on train-OOF
`lr_auroc_`**; causal_comparison design preserved + given the per-IGR ranking + bare-key `_match` so the ◆
(rung-2 gene) / ★ (rung-3 non-coding) markers land on **all 22** Kp drugs. Gallery **v3 republished, SAME URL**
`claude.ai/code/artifact/71445372-f035-4332-b246-5499ad9846e0`.

**NEXT (David, post-compact, in order):**
1. **Add a dedicated head-vs-mean comparison** — the gallery only shows ft_mean (mean re-probe); David wants
   ft_mean-vs-deployed-head made visible (summary-table column + small scatter). Earlier data: CSD3
   `hpc-work/head_vs_mean_panel.csv` (32 drugs, Δ ±0.03 symmetric/non-systematic = "head-vs-mean closed").
   cefotaxime wasn't cleanly in that panel (cache 95%-guard) → re-confirm it.
2. **TB fan-out (Full scope)** — **STARTED 2026-08-06.** ⚠ The pre-compaction gap list was STALE; live recon
   corrected it: on CSD3 **intergenic ALREADY complete (38 257)**, protein_sequences complete, baclm_reembed
   complete, **all 10 TB `ft_bacformer_cache` dirs present with `scope=trainholdout` + genome-mean npz** (so
   ft_mean rung ready — **bacformer store does NOT move, saves 574 GB + a GPU job**; caches were transferred
   not rebuilt, like Kp whose CSD3 bacformer link is empty), 10 TB FT checkpoints present. **The ONLY real gap
   = baclm-coding 30 892→38 257 = 7 365 files (~96 GB) — ✅ TRANSFER COMPLETE 2026-08-06 (CSD3 baclm=38257).**
   Done via rsync 2-leg laptop-bridge (`scratchpad/tb_baclm_rsync.sh`): Isambard→`$HOME/tb_baclm_stage`→CSD3,
   resumable. Direct Isambard↔CSD3 IMPOSSIBLE (CSD3 can't resolve the Isambard alias / has no cert; Isambard
   has no CSD3 host key) → laptop-bridge is the only route; tar-stream rejected (tar slow on shared Isambard
   login). Local stage `$HOME/tb_baclm_stage` (96 GB) still present — reclaim after TB figs verified. After transfer: rankings (per-gene baclm coding + IGR/upstream/unit) + 10 ladders
   (ceiling = committed `tbprofiler_gene_lr`, 9/10 present, rifabutin absent) + the same 6-figure suite (holdout
   display). No input_csv repoint needed — CSD3 CSVs already local-path.
3. **Curate** honest CSVs → `visualisations/{kp,tb}/<drug>/`; git rm contaminated; commit explicit paths.

**Operational:** local scratchpad `/private/tmp/claude-501/.../scratchpad/` is **pruned overnight** — re-stage
the Kp input CSVs from CSD3 with `csd3_restage.sh` (one ssh → tarball → pull → extract to `kp_viz/`), then
re-render fully LOCAL. CSD3 login rate-bans bursty ssh → batch into one base64-piped ssh. Related:
[[amr-ladder-fix-live-run-state]] · [[amr-ladder-descriptive-not-mechanism]] · [[dont-conflate-penetrance-with-lineage]].

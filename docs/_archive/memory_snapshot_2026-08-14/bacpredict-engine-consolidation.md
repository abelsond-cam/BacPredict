---
name: bacpredict-engine-consolidation
description: "BacPredict was consolidated from tb_ast/kleb_ast/pangena_predict/tl into one src/bacpredict/{engine,apps,_archive} engine on branch refactor/consolidate-engine"
metadata: 
  node_type: memory
  type: project
  originSessionId: 287d1555-8fdc-4241-9446-ecac54a396be
---

The BacPredict AST codebase was consolidated (2026-07) from four packages (`tb_ast`, `kleb_ast`,
`pangena_predict`, `tl`) into **one engine + thin per-organism apps** on branch
**`refactor/consolidate-engine`** (off `dev`). Plan: `~/.claude/plans/the-cambridge-hpc-and-dreamy-thacker.md`.
Started while both clusters were down; **Isambard is back up (2026-07-13)** and the branch is now
validated on-cluster. Branch HEAD `33df65a` (green: pytest 253; ruff 167 on cluster), pushed to
`origin/refactor/consolidate-engine`; **still NOT merged to `dev`** (user holds the merge until
full-cluster validation; local dev has not moved — will re-root onto the branch later). Session-2
work (2026-07-13) added on top of the original consolidation: **H1–H3 dual-cluster data paths, L
amr_over_time extraction, M figures-out-of-docs**, plus two module retirements. See the "Session 2"
section below.

**On-cluster testing uses a git worktree**, never the shared `$HOME/BacPredict` checkout (which stays
on `dev`): `$SCRATCHDIR/worktrees/consolidate` = `origin/refactor/consolidate-engine`. Loop: push
locally → `ssh u6fp.aip2.isambard 'git -C $HOME/BacPredict fetch origin refactor/consolidate-engine:refs/remotes/origin/refactor/consolidate-engine; git -C $SCRATCHDIR/worktrees/consolidate reset --hard origin/refactor/consolidate-engine'` → run with `PYTHONPATH=$WT/src $SCRATCHDIR/envs/bacpredict-gpu-venv/bin/python`.

## New layout — `src/bacpredict/`
- `engine/` (organism-agnostic): `config.py` (OrganismConfig/StorePaths/`store_paths(key)`), stages
  `labels` `download` `embedding` `finetune` `gene_lr` `concat` `catalogue` `plots`, plus `scripts/`.
- `apps/tb/` and `apps/kleb/` — organism specifics only (catalogue adapters, sidecar pipeline, metadata
  curation, epi plotter, download helpers).
- `_archive/tb_snp_diagnostic/` — the concluded rpoB/surprisal/attention diagnostic (10 modules incl. the
  split-off `snp_vs_esm_ladder.py`); **excluded from wheel/ruff/pytest**; README maps each module to its
  write-up. Frozen — its imports are not maintained.
- `docs/` — narrative docs only (Bacformer_FT_DEFICITS.md, baclm_*.md, readout_design_brief.md,
  _archive/, ISAMBARD_DATA.md, HPC_DATA.md). **Figures now live in `src/bacpredict/visualisations/{tb,kp}/`**
  (Phase M), out of docs.

`kleb_iso_source`, `bac_pyseer`, `gene_array_lasso`, `dp_short_read` stay top-level (import the engine).

## Key facts to know
- **Single trainer:** `bacpredict.engine.finetune.finetune_amr` — run `python -m bacpredict.engine.finetune.finetune_amr --task tb_ast|kleb_ast`. NOT by path (engine/finetune/`datasets.py` shadows HF `datasets` when the script dir is sys.path[0]; the `-m` form avoids it). Split-CSV prep is `engine.finetune.build_split_csv`.
- **Both organisms now train bf16** (was fp32 `dtype="auto"` for TB). TB fp32 checkpoints are SUPERSEDED — TB must be re-run under bf16 when a cluster returns. Kp unchanged. See [[tb_vs_kp_chromosomal_hgt_contrast]].
- Engine imports are `from bacpredict.engine.<stage>.<module>`; app imports `from bacpredict.apps.{tb,kleb}.<module>`.
- Promoted-to-public helpers (were private): `fit_one_gene`, `fit_one_gene_imputed`, `read_genome` (gene_lr.build_per_gene_lr_store); `real_protein_indices` (gene_lr.snp_vs_esm_prediction); `forward_inputs`, `load_model` (concat.bacformer_genome_vectors).
- Shared primitives extracted: `engine.catalogue.base.score_onehot_frame`, `engine.concat.concat_ingredients.{impute_block,load_ft_mean,load_ft_gene,load_frozen_gene}`, `engine.plots.labels.display_name`.
- `snp_vs_esm_prediction.py` (engine.gene_lr) keeps only the generic gene-LR primitives; its rpoB ladder driver is archived.

## Session 2 (2026-07-13) — data paths, amr_over_time, figures

- **H1–H3 dual-cluster data paths (DONE, cluster-validated).** `engine.config.resolve_data_root()`
  is now the single ROOT resolver; ~30 modules + ~58 shell scripts migrated off the dead CSD3 root;
  scripts on Isambard SBATCH. Full contract in [[bacpredict-dual-cluster-data-root]].
- **Kp store reconciled to per-task.** All Kp defaults point at `KP.data_root()/{esm,protein_sequences,
  bacformer}` = `$SCRATCHDIR/processed/train_kleb_ast/*` (live: 9,724; TB 38,257). The flat 80k
  `processed/klebsiella_esm_embeddings` is **CSD3-only, belongs to `kleb_iso_source`** — CLAUDE.md
  "Training data architecture" rewritten to say so.
- **L — `src/amr_over_time/` extracted** (commit `4a24d7e`): the predict-AST-across-80k →
  resistance-over-time subproject (`predict_amr_for_metadata`, `plot_resistance_over_time`, its slurm
  script + `resistance_over_time/` figs) moved to a new top-level package (flat code + `docs/` +
  `visualisations/`). Imports `bacpredict.engine` one-way like `kleb_iso_source`. It is the **primary
  `metadata_v2` consumer**; core now only touches the Kleborate-comparator + CARD mut/WT slice.
- **M — figures out of docs** (commit `a368aec`): both trees → **`src/bacpredict/visualisations/{tb,kp}/`**
  (tb_/kp_ prefixes DROPPED, `amr_per_abx/kp_<drug>` FLATTENED to `kp/<drug>`). New
  `config.visualisations_dir(org)` helper anchors every plot builder; `driver_panel`/
  `bacformer_gene_panel_vectors` discover per-drug dirs by driver-CSV presence (no folder-prefix).
  Wheel excludes `**/visualisations/**`.
- **Retired** (zero importers, superseded): `apps/kleb/find_missing_embeddings.py` +
  `add_paths_gff_fna_to_metadata.py`(+.sh). `annotate_amr_sidecar` is KEPT — it IS the CARD sidecar
  annotator the whole CARD ceiling reads.

## Remaining
- **Phase N DONE** (commit `33df65a`, pushed, pytest 253 unchanged): `tests/` renamed to mirror
  `bacpredict/` — `tl/embed`→`engine/embedding`, `tl/train`→`engine/finetune`, `pangena_predict`→`engine`,
  `tb_ast`→`apps/tb`, `kleb_ast`→`apps/kleb` (pure `git mv` + new `tests/apps/__init__.py`); one fix:
  `test_download_scripts.py` `REPO_ROOT` gained a `.parent`; two root-`CLAUDE.md` test-doc links repointed.
  **Plan `the-cambridge-hpc-and-dreamy-thacker.md` now fully executed (L+M+N).** Branch HEAD `33df65a`.
- Deferred done: by-path→`-m`, CSD3 defaults, figures split (items 1–3 of the old list) are all DONE.
  `--array=0-21` = 22 drugs, correct.
- NB: `ruff` is not installed in the **local** `.venv` (dev-dep absent) — lint must run on the cluster
  worktree or after `uv sync`; Phase N changed no `src/` lint surface so this didn't gate the commit.

## Post-cluster acceptance test (Tier 2, still pending)
Reproduce known numbers stage-by-stage (labels→embedding→finetune→gene_lr→concat/catalogue); Kp must
reproduce to the last figure, TB reproduces all except the bf16-affected finetune (record as new
baseline). Also stage `final/`+`metadata_v2` on Isambard (not present yet) before catalogue runs.

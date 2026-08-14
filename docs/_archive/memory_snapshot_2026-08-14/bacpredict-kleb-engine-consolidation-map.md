---
name: bacpredict-kleb-engine-consolidation-map
description: "Map for consolidating duplicated apps/kleb per-gene-LR/concat compute INTO the engine (Phase 2/4 standing rule); seam cut + ALL 8 relocations DONE (commits 1daba5f/3997ddd/9025fe9/849c245); GPU behaviour still validates post-FT"
metadata:
  node_type: memory
  type: project
  originSessionId: 287d1555-8fdc-4241-9446-ecac54a396be
---

The standing consolidation rule (plan the-cambridge-hpc-and-dreamy-thacker, Phases 2–6): at each stage
MOVE generic Kp compute into `src/bacpredict/engine/`, DELETE the redundant `apps/kleb` copy; KEEP
CARD/Kleborate **annotation** in apps/kleb. Engine must NEVER import `bacpredict.apps.*`. See
[[bacpredict-engine-consolidation]], [[bacpredict-isambard-validation-program]]. Map from a full read-only
survey (2026-07-15). **Note: the concat/cache modules are GPU/FT code — behaviourally validate only after
the FT checkpoints land (5661316/5661317); the moves themselves are behaviour-preserving by construction.**

## DONE — the seam cut (commit `82df586`)
`collect_reliable_amr` split. New **`engine/gene_lr/reliable_gene_vectors.py`**:
`collect_reliable_gene_vectors(eval_ids, esm_dir, parquet_dir, calls_fn)` + `GeneCall(label, flat_index,
source, tag_match)` + `MIN_CARRIERS` — generic, blind to how calls are made. `apps/kleb/per_gene_lr_from_annotation.py`
keeps the CARD half: `card_amr_calls(sidecar_dir, *, grain)` → calls_fn reading `{sid}_amr.parquet`, and a
thin back-compat `collect_reliable_amr` wrapper (renames tag_ids→bakta_ids) so `reliable_ft_concat` +
`gene_ingredient_concat` import UNCHANGED. 3 tests in `tests/engine/gene_lr/test_reliable_gene_vectors.py`.

## DONE — all 8 relocations (commits 1daba5f / 3997ddd / 9025fe9 / 849c245; 265 tests green, ruff clean)
Pattern used everywhere: **engine module takes the sidecar-agnostic `calls_fn` seam; the Kp CARD `calls_fn`
(`card_amr_calls`) + Kp data-root defaults live in a thin `apps/kleb` CLI of the SAME `-m` name** (so its
SLURM script is UNCHANGED). No-CARD modules moved wholesale + kleb copy deleted.
- **#1** `per_gene_esm_vs_ft_lr` → engine `gene_lr/per_gene_esm_vs_ft` (wholesale; kleb deleted). `collect_esm_vectors`
  is the ONE helper; `collect_esm_blocks` = its vstacked wrapper (imported by #7). Script `per_gene_esm_vs_ft_lr.sh` repointed.
- **#7** `concat_gene_panel_kleb` → engine `concat/concat_gene_panel` (wholesale; kleb deleted; imports #1's helper). `concat_gene_panel_kleb.sh` repointed.
- **#3** `reliable_ft_concat` → engine `concat/reliable_concat` (`run`+`_fit_metrics`, takes `calls_fn`); kleb = thin CLI. Script unchanged.
- **#4** `gene_ingredient_concat` → engine `concat/gene_ingredient_concat` (takes `calls_fn`); kleb = thin CLI. Script unchanged.
  `load_frozen_mean` folded into engine `concat_ingredients.load_genome_mean(prefix=)` (`load_ft_mean`/`load_frozen_mean` now wrappers).
- **#5** `aggregate_reliable_concat` → engine `concat/reliable_concat.aggregate`/`aggregate_run`; kleb = thin CLI (KP defaults).
- **#8** `cache_ft_bacformer_gene_embeddings` → engine `concat/cache_bacformer_gene_embeddings` (wholesale; kleb deleted).
  Both `cache_{ft,frozen}_bacformer_gene_embeddings.sh` repointed (frozen passes `--mode frozen`).
- **#9/#10 COLLAPSED** to engine `concat/bacformer_token_cache.run(mode, checkpoint, prefix, calls_fn)`. KEY INSIGHT:
  the CARD `GeneCall.tag_match` IS the bakta-match, and `card_amr_calls`' single-copy/source/in-range filter == the
  caches' inline logic → one calls_fn drives both. kleb `cache_{ft,frozen}_amr_proteins` = thin CLIs (mode/prefix ft|frozen).
  Output contract preserved (`ft_amr_emb`/`frozen_amr_emb`, `amr_gene_manifest_<drug>.csv` un-prefixed for ft).
- **#11** `cache_bacformer_genome_mean` → engine `concat/cache_genome_mean` (generic `run`+`main`); kleb = thin CLI (KP defaults).
- **#2 `per_gene_esm_vs_ft_card` KEPT** in apps/kleb (it's a PLOT; Phase 4).
- **render_card_figures.sh** stale `kleb_ast.*` (5 of them) → `bacpredict.apps.kleb.*` fixed.
- Doc `:mod:` cross-refs in `plot_per_gene_esm_vs_ft`/`per_gene_lr_from_annotation`/`plot_concat_gene_panel` repointed;
  `apps/kleb/CLAUDE.md` got a Concat/cache consolidation note.

**SMOKE-VALIDATED on-cluster (2026-07-15, job 5666339, CPU n=10):** #11 `cache_genome_mean`, #8
`cache_bacformer_gene_embeddings --mode frozen`, and #9 `bacformer_token_cache` (frozen, the #9/#10 collapse,
driven by 10 synthetic sidecars → exercises the `card_amr_calls` seam + token extraction) all PASS and write
the exact legacy output contract (`frozen_amr_emb/`, `frozen_genome_mean_<drug>.npz`, `frozen_amr_gene_manifest`).
#8's top-gene cache picked up gyrA+parC (cipro QRDR) — sanity ok. **ONLY #10 finetuned-mode (same primitive,
`mode="finetuned"`) is unexercised — purely the FT-checkpoint forward, waits on 5661317.**

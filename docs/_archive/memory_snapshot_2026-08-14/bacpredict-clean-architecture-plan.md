---
name: ""
metadata: 
  node_type: memory
  originSessionId: aac091b8-20e4-4661-ab5d-762fe4b1c697
---

**★★ ACTIVE FRONT = CLUSTER PIVOT (2026-07-29): migrate Isambard→CSD3, repoint, THEN regenerate (Step 5 below).**
Isambard unworkable (frequent outages incl. a ~24h+ one 2026-07-28/29; slow GH200 alloc for our small recurrent
CPU/GPU jobs). CSD3 (UoHPC) back up → new home. Plan **Step 0 (5 tracks) PREPENDED** to the plan file — read it.
- **NEW CSD3 layout (David):** `david/bacformer_processed/{esm,bacformer,baclm_reembed,protein_sequences,intergenic}/
  {klebsiella,tb}/` = SHARED embeddings home (invasive-phenotype/iso_source use these too). AST task root =
  `david/bac_ast_prediction/processed/train_{kleb,tb}_ast/<store>` → SYMLINKS into bacformer_processed;
  `BACPREDICT_DATA_ROOT`→bac_ast_prediction. `processed/` left for other work.
- **DONE this session (2026-07-29):** (1) **ESM relocated** Kp 87,719→`bacformer_processed/esm/klebsiella` + TB
  38,248→`esm/tb` via `mv` + BACK-COMPAT SYMLINKS at old `processed/{klebsiella_esm_embeddings,train_tb_ast/tb_esm_embeddings}`
  (`kleb_iso_source`×6 + `gene_array_lasso`×2 hardcode these — DON'T break). (2) **Canonical CSVs** → `bac_ast_prediction/processed/train_{kleb,tb}_ast/`.
  (3) **FT checkpoints** 22 Kp + 10 TB bf16 **weights-only** (`model.safetensors` only; DROPPED optimizer.pt + the 1 `_fp32_` rifampin dir)
  → `bac_ast_prediction/…/{models/finetune,checkpoints}/`, **sha-verified** (azithro/rifampin match), laptop cleaned.
  (4) **Code REPOINTED to CSD3 — committed+pushed `9705ea3`**: config `_CSD3_DATA_ROOT`→bac_ast_prediction; 62 launchers
  (37 CPU→icelake-himem/FLOTO-PROJECT-K-SL2-CPU · 25 GPU→ampere/FLOTO-SL2-GPU; venv→~/workspace/BacPredict/.venv, repo→~/workspace/BacPredict,
  DATA_ROOT default→bac_ast_prediction, logs→/rds/user/dca36/hpc-work/logs, dropped `--qos`). Env-var overrides preserved.
- **⚠ STORE INVENTORY (live-verified 2026-07-29 on Isambard `train_{kleb,tb}_ast/`) — CORRECTS 2 plan assumptions.** SEVEN full-cohort
  per-genome stores exist on Isambard; sizes (count×one-file, Kp 9724 / TB 38257 → GB): **esm** 188/597 (✅ on CSD3) · **baclm**(coding)
  166/481 · **baclm_reembed**(non-coding) 235/672 · **bacformer**(FROZEN) 191/595 · **intergenic** 8/21 · **protein_sequences** 17/51.
  (1) **Bacformer FROZEN is ON Isambard** — NOT only the down cold-storage tape → Track B = a transfer, NOT a tape-wait/50-GPU-h regen.
  (2) **TWO baclm stores, BOTH needed:** engine `store_paths().baclm_dir=root/"baclm"` (coding rung + default) vs non-coding launchers
  (`build_per_unit_lr_ranking.sh:70`) default `…/baclm_reembed`. baclm-coding≈esm-coding (prior finding) → its 647GB may be skippable — David decides.
  **TOTAL still-to-move (excl esm) ≈ 2.36 TB** = baclm 647 + baclm_reembed 907(~78 done) + bacformer 786 + intergenic 29 + protein_sequences 68.
- **RUNNING (HEALTHY):** `baclm_reembed` laptop transfer (`scratchpad/migrate_baclm.sh`, 40GB batches, RESUMABLE skip-CSD3-present, self-cleaning).
  Never died across compact — Kp batches 1–2 landed (3200/9724), batch 3+ pulling with the REFRESHED cert. ~40min/batch. Laptop-bridge (no Globus CLI on CSD3 login).
  ⚠ At ~2.4TB the laptop bridge ≈ 40h + many cert refreshes → **DECISION PENDING w/ David: investigate direct Isambard→CSD3 (Globus/rsync) vs keep laptop-bridge; prove headline first (ESM+FT already on CSD3) vs wait for full transfer.**
- **Track A DONE this session:** task-root **symlink skeleton** built + verified (`bac_ast_prediction/processed/train_{kleb,tb}_ast/{esm,baclm,baclm_reembed,bacformer,intergenic,protein_sequences}`→`../../../bacformer_processed/<store>/<org>` + `{raw,final}`→`../{raw,final}`) · **logs dir exists** · **CSD3 x86 venv = `bacotype`** (`.venv`→`/home/dca36/rds/hpc-work/.venv/bacotype`, torch 2.10+cu128/CUDA12.8, sklearn 1.8, pandas 2.3 — already built, NOT a rebuild needed). Remainder: land the 5 stores.
- **DAVID'S DECISIONS (2026-07-30):** transfer = **keep laptop-bridge** (no direct/Globus); sequencing = **prove headline now** (parallel to transfer); **move baclm coding too** (647GB). baclm_reembed DONE Kp (9724) + TB 25600/38257 → **PAUSED, needs cert refresh** to finish TB tail (~222GB) + baclm/bacformer/intergenic/protein_sequences.
- **★ 5A + Track D DONE (2026-07-30, CSD3) — HEADLINE PROVEN, ALL SPLITS VERIFIED.** CSD3 checkout synced to `9705ea3` (ff-pull; other agent's `pixi.lock`/`kleb_ast/` untouched). Ran `generate_kfold_splits` for ALL 32 drugs → `processed/train_{kleb,tb}_ast/splits/<drug>_split.csv`; **32/32 verify OK** (table holdout count == deployed `results.json` `n_evaluate` → canonical CSV faithfully reproduces every deployed holdout; **split-provenance cross-system question RESOLVED**). Honest FT deployed-holdout AUROC: **Kp azithromycin 0.799** (was 0.918 leaked ✓), rifampin 0.964, Kp carbapenems/cephalosporins 0.95–0.99, TB 0.79–0.96. Throwaway driver = `scratchpad/tmp_5a_fanout.py` (base64-piped to `hpc-work/tmp_5a_fanout.py`; not committed). **NEXT: 5D GPU re-cache** (`cache_bacformer_gene_embeddings --mode finetuned --scope trainholdout --split-table <d>_split.csv`) → **5E ladder** proves the LADDER pipeline itself yields 0.799 not 0.918.
- **★ FT-HEAD vs LR-ON-MEAN (David's Q, 2026-07-30) — RESOLVED + open comparison.** results.json `metrics.auroc` (rifampin 0.964) = the **FT HEAD via GPU forward pass** (`trainer.predict`→logits→sigmoid→AUROC, `finetune_amr.py:413-421`), **NOT** an LR on a frozen/genome mean. Deployed AST models use **`pooling="mean"`** = the **stock mask-normalised-mean genome head** (`finetune_amr.py:91,292` default; the gated-attention MIL pool is an UNUSED alt) → David's "head ≈ LR on the 960-dim mean, minus the layer-norm" framing is architecturally accurate (head = mean-pool→LayerNorm/linear vs LR = mean-pool→logistic). **FT genome-mean caches: only ciprofloxacin on Isambard (Kp ft_amr_cache/frozen_amr_cache; TB none) → REGENERATE ALL via 5D GPU** (agreed; caches are pipeline-recomputed, not transferred). **OPEN CHECK (in plan @5E):** the ladder's `ft_mean` rung (LR-on-FT-mean) vs the head 0.964 — if LR-on-mean **>** head, the stock head's LayerNorm(+trained linear) is shedding predictive signal (David's hypothesis) → then inspect the head in checkpoint `modeling_bacformer.py` + consider a head variant.
- **Track C transfer RE-LAUNCHED (2026-07-30, David recertified):** generalized `scratchpad/migrate_stores.sh` (all stores, resumable skip-present, 40GB batches, self-cleaning) running bg; order = finish TB baclm_reembed tail → protein_sequences → intergenic → baclm(coding) → bacformer(frozen). ~1.75TB; will stop rc=2 at next cert expiry → refresh + re-run.
- **COHORT TRUTH (Track D cross-check crux; CORRECTED 2026-07-29 from the actual CSV):** Isambard **AST cohort
  7,088 Kp / 36,692 TB = CANONICAL** (FT-trained; `binary_ast_with_split.csv`, now copied to CSD3 bac_ast_prediction).
  The **9,724 / 38,257** figures are the larger **embedding-store genome counts** (esm/baclm/bacformer .pt) — a SUPERSET,
  not the labeled cohort. CSD3 May CSVs **6,838/36,684** (legacy 70/10/20 train_val_eval) = DEPRECATED (Kp +250, TB +8 vs canonical).
  Old CSD3 FT eval = MAG-era+worse (TB rifampin 0.905 vs Isambard bf16 0.964; Kp azithro 0.827 ≈ honest ~0.80).
  CSD3 up=`login-q-1`, project RDS `/rds/project/rds-4k08a2yyQLw` 4.7TB free; laptop 149GB free; no Globus CLI on login.
- Tracks remaining: **E** repoint code · **B** cold recall · **C** laptop moves · **D** split/cohort cross-check → then Step 5 on CSD3.
  → [[bacpredict-dual-cluster-data-root]]

---

**APPROVED — now executing.** Full spec = the plan file `~/.claude/plans/inherited-doodling-peacock.md`
(co-designed with David over ~7 ExitPlanMode rounds; it IS the detailed design — read it before continuing).
Leak-fix live state in [[amr-ladder-fix-live-run-state]]; prior consolidation in [[bacpredict-engine-consolidation]].

**WHY (verified first-hand this session):** the train/test leak is a **CLASS**, not 2 incidents — 5 more
live sites score a *fine-tuned* genome-mean/token off the deployed holdout (`bacformer_token_cache:72`
producer + `reliable_concat:71`/`gene_ingredient_concat:59`/`concat_gene_panel:66`/`per_gene_esm_vs_ft:97`),
contaminating published CSVs (`kp_reliable_concat_summary` etc.). Root cause = split-resolution never
centralised + "gene" does 5 jobs + copy-forked code.

**THE DESIGN (locked with David — layered, one-way deps `plots→segment_amr_lr→embedding→splits`):**
- **`engine/splits/`** (top-level) — `generate_kfold_splits.build_split_table` writes `<drug>_split.csv`
  (`Sample, ast_label, split`) reproducing the deployed fold-0 partition; `load_splits.load_splits` is the ONE
  reader. Retires `resolve_clean_splits` + `resolve_deployed_holdout` (latter → one-time verify only). The
  trainer reads the same table → **leak impossible by construction.** Correctness spine: EVERY LR (FT + label-
  blind) fits on `train`, selects by train-OOF, evaluates on `holdout` (closes FT-leak AND selection-on-test).
- **`engine/embedding/`** — produce (ESM-C/baclm/Bacformer `cache_embeddings`) + `genome_mean_pool`/
  `real_protein_indices` (dedup 4 copies) + `segment_locator` + `segment_embedding_extractor`
  (`segment_embeddings(impute_missing)`) + `mean_genome_embedding` + `non_coding_segment_audit`.
- **`engine/segment_amr_lr/`** — `per_segment_lr` (folds in the 6 screen modules: build_per_{gene,igr,unit}/
  upstream + coding_amr_lr + igr_amr_lr) · `fit_lr` (the LR engine) · `concat/{concatenate, concat_lr}`.
- **`engine/plots/`** (top-level) — `ladder_of_results` (was build_amr_ladder) + driver_panel + figures.
- `finetune/` (trainer) · `ref_catalogues/` (was catalogue/) · `ast_labels/` (was labels/) · config.py (NOT renamed).
- **Vocab:** remove "gene" → **protein** (coding), **segment/segment_type** (generic), **ast_label** (phenotype,
  never bare `label`). External catalogue refs keep gene, prefixed: `_who_gene_onehot`, `card_gene_family`.
  Deny-list frozen: Bakta `gene_name`, `gene_emb/`, `ft_amr_emb/`, NPZ keys, `per_gene_lr_<drug>.csv`, scoped
  `ft_genome_mean_<drug>_<scope>.npz`. `pangena_predict` DATA-dir path = `TODO(data-move)`, not a code rename.
- **Archive** all ~24 unwired modules (list in plan §Archive); FLAG parse_ebi_ast_to_binary/tbprofiler_gene_lr/
  build_tb_input_csv/predict for David's veto.

**MIGRATION (5 steps, pytest+ruff green each; unify-first):** 1 splits+embedding primitives · 2 per_segment_lr
(fold in 6, same-split parity then switch to table) · 3 ladder+concat+producer through load_splits [CHECKPOINT]
· 4 archive + full rename [CHECKPOINT] · 5 GPU: materialize+verify tables vs results.json, re-cache, rebuild
honest, git-rm contaminated, prove azithro≈0.80/rifampin≈0.96, fan out ~7 Kp+5 TB.

**PROGRESS (branch refactor/consolidate-engine, HEAD 25e48cc, 388 green; b267696 pushed, 1bedd5c/25e48cc unpushed):**
- `eb39ce5` splits/ (build_split_table + load_splits + verify_table_matches_deployed; split_utils→shim).
- `914dfae` segment_amr_lr/fit_lr (fit_one_segment/_imputed/fit_per_segment + fit_score_step/LOGREG_KW). `fit_per_segment`
  ALREADY has `impute_absent_zero`(=impute_missing) + `eval_ids`(=holdout eval) — the two knobs the screens pivot on.
- `9861abb` embedding/protein_pooling (real_protein_indices + real_protein_rows guard + genome_mean_pool; deduped ALL 5
  Bacformer producers; deleted gene_lr/protein_rows.py).
- `47a1f72` **embedding/segment_locator** — SegmentLocator Protocol + Protein/Igr/Upstream/Unit locators, uniform
  seam (records() + discover_ids()). FACADE for now (lazy-delegates to the gene_lr.build_* record extractors; bodies
  relocate here in c3). KEY: discover_ids() None-vs-[] encodes the per-type prevalence DENOMINATOR with no flag
  (protein []=counts → reproduces discover_core_genes n=len(train_ids); non-coding None=skip → n=len(read_ids)). 10 tests.
- `c39f69a` **embedding/segment_embedding_extractor.collect_segment_matrices** — ONE two-pass sweep replacing all 4
  collectors. Pass 1 (fit-only, discover_ids) prevalence→(min,max] core; pass 2 (fit+eval, records) core-only vectors.
  read_ids = impute universe = fit∪eval read. 8 tests.
- **DECISION B (spine-mandated, NOTED behavior change):** read_ids spans fit∪eval for ALL types, so a non-coding
  *imputed* ranking with --eval-holdout now computes eval_auroc (was silently NaN — coding already included eval).
  Narrow: no existing test exercised impute+eval together. Aligns with "every LR evaluates on holdout."
- **PER-TYPE SURFACE is larger than 'locator+source+bands'** (found in deep read): presence-mode is NON-CODING only;
  panel-store + annotation are CODING only; per-type ranking-table + prevalence-CSV column schemas differ (incl.
  coding's gene_name-table-key vs gene-prevalence-key quirk). igr is the ONLY build_* without eval cols (A1). So the
  clean fold-in = one `rank_segments` core + per-type config (locator, id_column, extra table cols, offers_presence,
  coding panel/annotation hook). PRESERVE each legacy CSV schema exactly through the fold-in (downstream plots+concat
  read these by column name); the de-gene rename happens uniformly in STEP 4, not during the fold-in.
- **2 plan deviations (keep commits clean):** mean_genome_embedding + impute_block move WITH concat in STEP 3.
- **c1 DONE `587bc30`** — segment_amr_lr/per_segment_lr.py for the 3 NON-CODING types (folds in build_per_igr +
  build_upstream_region + build_per_unit). `run(segment_type,…)` + `rank_segments` core + per-type writer + `SEGMENT_TYPES`
  spec (id_column, extra_id_columns, needs_gff). A1 fix landed: igr table GAINS eval_auroc/n_eval/n_eval_pos cols
  (all 3 now uniform). unit prevalence re-enriched to feature_type/feature_name/n_present. 8 parity tests (monkeypatch
  `collect_segment_matrices`; EXACT columns + top segment + A1 eval population + empty-core header). build_* still coexist
  (additive). **DECISION (split source): per_segment_lr is BORN on `engine.splits.load_splits.load_splits(<drug>_split.csv)`
  — NOT a transitional finetune.holdout parity pass.** Fit on `train`, OOF-select, eval on `holdout` ALWAYS (no eval_holdout
  toggle; eval_ids=holdout_ids); `validate` reserved for the trainer/concat op-point, NOT in the screen's fit (literal spine
  "fits on train"). Numeric parity vs legacy is NOT expected/wanted (legacy CSV single-split ≠ deployed k-fold table — the
  leak); parity = OUTPUT-SCHEMA + ranking-machinery, split-source-isolated via the monkeypatched sweep. The 4th `run` axis
  `feature` (embedding/presence) + `impute_absent_zero` preserved. `subsample_balanced` still imported from
  build_per_gene_lr_store (transient; relocates in c3). **NEXT — per_segment_lr fold-in, remaining commits:**
  - **c2 DONE `113e1b9`** — coding (protein) type in per_segment_lr. `rank_segments`+ProteinLocator reproduce
    discover_core_genes+assemble_segment_matrices (ProteinLocator.discover_ids returns `[]` not None → denom
    =len(fit train) exactly). Coding branch on `spec.is_coding`: dual esm|baclm store; parquet `annotation` join
    (`_coding_annotation`); bespoke `gene_name`-keyed `per_gene_lr_<drug>.csv` writer (`_write_coding_table`,
    reproduces write_gene_drug_table); NO presence (fail-fast guard); optional `write_panels` (build_panels over
    train+validate+holdout, std on fit-train subsample — FT att-head channel the trainer consumes). Deny-listed
    filenames kept (per_gene_lr / gene_prevalence). `build_panels`+`_genome_segment_records` still imported from
    build_per_gene_lr_store (transient). 4 coding parity tests. 400 green. `SegmentTypeSpec` gained is_coding/
    offers_presence; run gained embed_dir/parquet_dir/store_kind/write_panels + baclm_dir now optional.
  - **c3 DONE `9426a62`** (383 green, ruff clean) — RELOCATED the record-extractor bodies (_genome_igr/upstream/
    unit_records + _flank_pair/_read_intergenic/_genes_by_seqid/_upstream_region_index/_read_features/_unit_key +
    coding read_genome/_embedding_rows/_genome_segment_records + EMBEDDING_STORES) into **segment_locator** (the
    locator classes call them directly; torch/pandas/GFF + flatten_proteins/_parse_gff kept **lazy in-function** →
    segment_locator imports **torch-free AND gene_lr-free**, verified). subsample_balanced → **splits/subsample.py**;
    build_panels(+_prob_for/_Standardizer1D/_write_sample/PANEL_COLUMNS) → **segment_amr_lr/panel_store.py**.
    **CORRECTION to the earlier importer list:** only **5** modules had CODE imports of build_* (concat_ingredients
    ×4 load_baclm_* blocks, cache_bacformer_gene_embeddings=subsample, per_gene_esm_vs_ft=read_genome,
    reliable_gene_vectors=read_genome, per_segment_lr=subsample/_genome_segment_records/build_panels) — all
    repointed; the rest (build_amr_ladder, concat_gene_panel, concatenate_*, plot_per_gene_lr_ranking, apps/*) were
    **docstring-only prose refs** (deferred). DELETED the 4 build_* + 4 test files; **ported** all body-logic tests
    (flank_pair/upstream_index/convergent/read_features/unit-mean-pool/embedding_rows → test_record_extractors;
    subsample → test_subsample; _prob_for/build_panels → test_panel_store; fit_per_segment → test_fit_lr). Net
    −1639 lines; 400→383 (−43 collect_*/run/legacy-load_splits tests of now-deleted code, +26 ported). Staged
    explicit paths (parked train_isolation_source.py showed ` M` from another agent — NOT staged).
    **c3 DEFERRED → step 4** (tracked in plan): 7 .sh launchers still -m deleted modules (need per_segment_lr CLI +
    <drug>_split.csv); lazy flatten_proteins/_parse_gff edge (drops when locate_gene/igr_amr_lr reorganize); prose
    refs. All low-risk (frozen HPC launchers / docstrings).
  - **(d) OPEN FORK — fold into step 3/4:** the 2 *_amr_lr probes (coding_amr_lr ESM-vs-baclm paired; igr_amr_lr
    promoter probe) share LOCATE+EXTRACT but use fit_score_step not fit_per_segment. Resolve: collapse into
    per_segment_lr, or a separate generic segment_vs_ft (renamed per_gene_esm_vs_ft)? DON'T drop the ESM-vs-baclm
    paired comparison. Given David's "clean codebase, no back-and-forth" → make the call when touching those modules
    (likely: per_segment_lr = ranking screen; segment_vs_ft = generic paired comparator), NOT a blocking ask.

**STEP 3 (FT-leak spine, 2026-07-26) — decomposed 3a/3b/3c after a full split-consumer map.** KEY finding:
it's NOT a split-source swap — the FT scorers OOF over the WHOLE FT-cache universe (train+holdout), so the leak
persists unless they FIT-TRAIN / EVAL-HOLDOUT. `fit_lr.fit_one_segment(_imputed)/fit_per_segment` ALREADY take
`eval_ids`; `build_amr_ladder._score` is the reference pattern (`fit_one_segment(all_ids, x, y, eval_ids=holdout_set)`
→ report `eval_auroc`). Files STAY PUT (moves+rename deferred to step 4 to keep the correctness diff clean).
- **3a DONE `b267696`** — producers cache_bacformer_gene_embeddings + bacformer_token_cache → `load_splits(split_table)`;
  token_cache gained scope=trainholdout (subsample_balanced train + full holdout) + scope-tagged mean
  `<prefix>_genome_mean_<drug>_<scope>.npz`; cache_summary key split_source→split_table + n_evaluate_expected=len(holdout).
  383 green, ruff clean.
- **3b DONE `1bedd5c`** (388 green, my diff ruff-clean) — reliable_concat / gene_ingredient_concat / concat_gene_panel /
  per_gene_esm_vs_ft ALL: `load_splits`→holdout_set; universe = load_ft_mean(scope=trainholdout); shared
  `assert_holdout_in_cache` extracted into concat_ingredients (build_amr_ladder's ≥0.9 coverage check + no-train-side
  check); `eval_ids=holdout_set` through every fit; report held-out `*_eval_auroc` (reworked `_fit_metrics`/`_score`),
  SELECT on train-OOF `*_lr_auroc`. ADDED `*_eval_auroc` cols to the 2 per-gene CSVs (schema-additive; delta now
  held-out). per_gene_esm_vs_ft moved off eval-only → trainholdout universe (collect_esm_vectors param eval_ids→
  sample_ids). CLI `--ast-sheet-path`→`--split-table`+`--scope` on the 2 with a main(). Tests: test_concat_ingredients
  (guard) + test_concat_gene_panel (fit-train/eval-holdout smoke, n_eval==holdout count).
- **3c DONE `25e48cc`** (388 green, ruff-clean) — build_amr_ladder: resolve_clean_splits(checkpoint_dir=run_dir)→
  load_splits(split_table); `_load_cache_summary`→`_cache_scope` (scope only; holdout from the table); adopted the shared
  guard; dropped --checkpoint + run()'s checkpoint param + ast_sheet→split_table; fixed stale build_per_gene_lr_store
  docstring ref. test_build_amr_ladder monkeypatches load_splits.
- **3 grep-proof DONE** — all 7 (2 producers + 4 scorers + ladder) clean of resolve_clean_splits/resolve_deployed_holdout/
  generate_kfold_splits/train_val_eval; all 7 import load_splits. (No test_cache_skip_existing resolve_clean_splits
  monkeypatch remained — 3a already converted it.) **← FT-leak spine CLOSED by construction.**
- **DEFERRED → step 4** (tracked in plan): concatenate_bacformer_genome_esm_protein_emb + bacformer_genome_vectors (rpoB
  probe PAIR — guarded-safe, consider ARCHIVE; its test_concatenate_* still monkeypatches resolve_clean_splits → update
  when converted); driver_panel (label_map-only + internal-kfold); coding_amr_lr/igr_amr_lr/kfold_probe (label-blind + 2d
  fork, fit_score_step path); module MOVES+rename; the .sh launchers. **Ripple flagged (call-broken until step 4):** (a)
  3a broke cache_ft_amr_proteins/cache_frozen_amr_proteins; (b) 3b broke apps/kleb reliable_ft_concat + gene_ingredient_concat
  (all call engine run(ast_sheet=…)); (c) 4 .sh pass --ast-sheet-path to the 2 scorer CLIs. All import-OK, not pytested,
  GPU-only — repoint to --split-table/--scope with the step-5 table path. Another agent owns apps/kleb → flag to David.

**STEP 4 DONE + PUSHED (2026-07-26) — 9 commits 234f381→5437e59, origin/refactor/consolidate-engine at 5437e59, 388 green ruff-clean throughout.** Approach: small green commits, stage explicit paths / verified `git add -u`, pytest+ruff green each.
- `234f381` archive 7 dead engine leaves (plots/{summarise_drug_sweep,build_drug_ladder_table,plot_ladder_barplot}, embedding/{explore_parquet_structure,build_ast_input_csv}, download/{download_bakrep_gbff_files,scripts/select_ast_cohort}) — verified no importer/test/.sh.
- `6922d63` catalogue/ → ref_catalogues/ (+3 apps importers: card/kleborate_determinant_lr, tbprofiler_gene_lr).
- `a5ff9fe` labels/ → ast_labels/ (parse_ebi stays, David: central); labels/audit_noncoding_regions → embedding/non_coding_segment_audit.py (+ test + .sh renamed). Data output dir left (data-move).
- `011160a` engine/concat/ → segment_amr_lr/concat/ (11 modules, mechanical `bacpredict.engine.concat`→`…segment_amr_lr.concat` across engine+apps+tests+.sh); concat_gene_panel→concat_segment_panel, gene_ingredient_concat→segment_ingredient_concat (ENGINE module; apps wrapper name + emitted CSV filename kept for step 5).
- `9d2657d` gene_lr/per_gene_esm_vs_ft→segment_vs_ft (in place); plots/plot_per_gene_lr_ranking→plot_segment_ranking (+tests). per_gene_esm_vs_ft_card/plot_per_gene_esm_vs_ft (distinct apps) untouched.
- `3ddb2a5` symbols: gene_flat_index→protein_index (VERIFIED in-memory-only — manifest carries gene_name/sanitized, not it), GeneTarget→ProteinTarget, GeneCall→ProteinCall, _gene_onehot→_who_gene_onehot. Deny-listed gene_name/flat_index/per_gene_lr_<drug>.csv untouched.
- `d185321` archive 11 dead apps modules (kleb: plot_amr_summary_panel, plot_concat_gene_panel, plot_kleb_per_gene_ranking, plot_kp_igr_rankings, build_isambard_amr_path_sheet, add_bakta_gbff_downloaded_flag, aggregate_reliable_concat, extract_anndata_with_bacformer_protein_embeddings, filter_esmc_embeddings_by_klebsiella, preprocess_ebi_amr_records; tb: plot_tb_igr_rankings) — verified no importer/test.
- `879e6d4` the 4 apps concat/cache wrappers (reliable_ft_concat, cache_ft/frozen_amr_proteins, gene_ingredient_concat) → run(split_table=,scope=), CLI --split-table(required)+--scope(default trainholdout), mirroring the engine convention. cache_bacformer_genome_mean left (cache_genome_mean.run still takes ast_sheet). Wrappers import end-to-end.
- `5437e59` docstring coherence (concat_ingredients driver names). Final sweep: no stale engine paths/symbols in the live tree.
- **NOT DONE by design** (David: "clean codebase but too much else being done" → scoped to high-value renames): gene_lr/ package still exists (dissolution deferred); build_amr_ladder still in segment_amr_lr/concat/ (plots move deferred); the 2 GPU cache producers not merged into embedding/cache_embeddings; amr_gene_family kept (David skip). All step-5 or later.

**CONSTRAINTS:** solo on the branch; stage EXPLICIT paths (parked `kleb_iso_source/train_isolation_source.py`
untouched — it forced the split_utils/linear_probe shims); Isambard worktree frozen at 9060617 (proof pending);
steps 1-4 local+GPU-independent. Never call a high-penetrance non-catalogue gene a "lineage correlate"
→ [[dont-conflate-penetrance-with-lineage]].

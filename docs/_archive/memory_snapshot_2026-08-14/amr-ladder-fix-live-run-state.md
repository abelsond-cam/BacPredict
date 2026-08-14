---
name: amr-ladder-fix-live-run-state
description: "★CHECKPOINT 2026-08-01 at top: Kp 34%-coverage coding bug FIXED (dedicated protein_sequences_B, 9724/9724 aligned); ALL Kp rankings + CARD ceiling regenerated at full coverage; ladder fanout job 32551666 running; azithro ft_mean 0.816 honest. NEXT = assemble_comparison.py over 22 → DESCRIPTIVE CARD→LR→ladder (no mechanism), then TB. Isambard cert expires ~hrs. Older 2026-07-30 GPU-proof state below is superseded."
metadata: 
  node_type: memory
  type: project
  originSessionId: aac091b8-20e4-4661-ab5d-762fe4b1c697
---

## ★ CHECKPOINT 2026-08-01 (CSD3) — the 34%-coverage coding bug FIXED; Kp ladders regenerating honestly

**Step 0 bug found+fixed.** Kp baclm coding COVERAGE was only 34% (3318/9724) — the shared `protein_sequences`
store is an A/B annotation MIX and baclm coding aligns 1:1 only with its own B-parquet (66% off by a
razor-constant **+3** → silently skipped, NOT a real result). FIX (David chose "pull B-parquets"): pulled the
9724 canonical B-parquets Isambard→CSD3 into a DEDICATED `train_kleb_ast/protein_sequences_B`; after-scan =
**9724/9724 aligned, 0 misaligned**. Non-destructive (FT-mean/5F keep the shared mix dir). Coding launcher gained a
`PARQUET_DIR` override.

**Kp rankings ALL regenerated at FULL holdout coverage (22/22 each):** coding (`per_gene_lr_ranking_imputed_baclm`,
baclm×protein_sequences_B), per_unit (`per_unit_lr_ranking_imputed`, FEATURE=imputed), upstream+igr
(`upstream_lr_ranking_imputed_full`/`per_igr_lr_ranking_imputed_full`, FEATURE=imputed_full,
BACLM_DIR=baclm_reembed), CARD ceiling (data-root `train_kleb_ast/card_ceiling/<drug>/card_determinant_lr_<drug>_{family,allele}.csv`),
FT caches (`pangena_predict/ft_bacformer_cache/<drug>/`). `$EVAL` bug fixed.

**Isambard pulls done (cert EXPIRES ~hrs — needs David refresh each window):** 307 missing GFFs → klebsiella_gff3
(`embedding_input.csv` now resolves 9724/9724); CARD `amr_annotation` (7089 sidecars + `amr_calls_all.parquet`).

**Ladders:** azithromycin smoke VALIDATED e2e — ft_mean **0.816** (honest; was 0.918 LEAKED), +gene(tadD)+0.000,
+nc(igr) −0.013, CARD ceiling 0.563. Full fanout = job **32551666** (--array=0-19,21; azithro=20 done) RUNNING.

**NEXT:** when 32551666 done → run `assemble_comparison.py` (CSD3 hpc-work + laptop scratchpad) over all 22 →
present the DESCRIPTIVE gene/weight comparison CARD→LR→ladder ([[amr-ladder-descriptive-not-mechanism]] — NO
causal/lineage/mechanism). Then Step E **TB** (needs Isambard: baclm-coding PARTIAL 30892/38257, TB GFFs ABSENT,
re-check TB coding parquet↔baclm alignment same as Kp), then Step F curate → visualisations/{kp,tb}/.

**OPEN Qs to David (asked, he said compact instead):** (1) descriptive format OK? (2) want an UNGATED per-gene LR
over just the CARD determinant set — the 10%-prevalence gate hides rare acquired determinants (azithro: only MphA
clears it, at LR#1262 lr=0.560)?

**Commits (branch refactor/consolidate-engine):** ea0b947 ($EVAL), 3ba1a53 (PARQUET_DIR+PYTHONPATH), 2656d0d
(ceiling→data-root + ladder `--catalogue-csv` Kp). Key scratchpad scripts: pull_bparquets_kleb.sh, pull_gffs_kleb.sh,
step0_full_scan.py/.sh, assemble_comparison.py, poll_kp_rankings.sh.

---

Resume point after /compact for the AMR-ladder train/test-leak fix (the bug + code fix are in
[[amr-ladder-holdout-leakage-csv-vs-kfold]]). Full plan: `~/.claude/plans/inherited-doodling-peacock.md`.

**★ CSD3 UPDATE (2026-07-30) — the 5D→5E proof now runs on CSD3, not Isambard** (cluster pivot; see
[[bacpredict-clean-architecture-plan]]). HEAD **`9705ea3`** (branch refactor/consolidate-engine, local +
CSD3 `~/workspace/BacPredict`). 5D re-cache is per-drug on **CSD3 ampere** (`--partition=ampere
--account=FLOTO-SL2-GPU`, logs `/rds/user/dca36/hpc-work/logs`, bacotype venv `~/workspace/BacPredict/.venv`
torch 2.10+cu128, `PYTHONPATH=src`, `BACPREDICT_REPO=$HOME/workspace/BacPredict`).
- **rifampin 5D = job `32392289`** (mean-only PROOF; scratch launcher `hpc-work/tb_ft_cache_rifampin_csd3.sh`;
  completion poller **bqlovtj9f**). Coverage 99.8% (clean); rifampin is also the head-vs-mean subject.
- **★ RESULT (2026-07-30):** rifampin 5D COMPLETED (1h25m, A100-80GB, 22 skipped). **Head-vs-mean** (driver
  `hpc-work/head_vs_mean.py` = ladder ft_mean rung `fit_one_segment` on FT-mean vs results.json head):
  **LR-on-FT-mean holdout AUROC 0.9582 (AUPRC 0.896) vs FT head 0.9642 (AUPRC 0.916), Δ −0.006 (~within 2 SE)
  → head ≥ LR-on-mean, so the pooling=mean head's LayerNorm is NOT shedding signal** (David's hypothesis
  unsupported for rifampin — the failure mode would be LR-on-mean ≫ head). Confirms **rifampin holds ~0.96
  honestly** (ft_mean rung ≈ 0.958 ≈ head 0.964). Still to check on a harder drug (azithromycin, the leak
  drug) once its coverage gap is filled. **⚠ bump 5D `--mem` to ~192G** (rifampin MaxRSS 155 GiB > 128 G req).
- **★★ AZITHROMYCIN LEAK-PROOF (2026-07-30, job 32400386, 20 min A100):** honest ladder `ft_mean` rung (LR-on-mean,
  deployed k-fold holdout) = **0.816** (AUPRC 0.951) vs the **leaked 0.918** → **THE LEAK IS FIXED** (~0.11 inflation
  gone; head = 0.799). Both headline drugs now proven (rifampin holds 0.96, azithromycin 0.918→~0.80). **Head-vs-mean
  TWIST: LR-on-mean 0.816 > head 0.799 (Δ +0.017) — OPPOSITE of rifampin** → David's LayerNorm-sheds-signal
  hypothesis MAY hold on harder drugs. **CAVEAT: azithro holdout is 333 R / 40 S (imbalanced) → AUROC noisy, +0.017
  may be within noise.** Do NOT inspect modeling_bacformer.py head yet — wait for the full 32-drug fan-out panel to
  see if "LR-on-mean > head" is SYSTEMATIC on harder drugs, then present to David (gather-then-hypothesise). Fan-out
  = Kp 32401437[0-19,21] + TB 32401438[0-7,9], mean-only, tight walls (Kp 1h/TB 2h); poller bxj8zgpge.
- **★★ FULL 32-DRUG HEAD-vs-MEAN PANEL DONE (2026-07-30, all fan-out COMPLETED, 31/32 scored).** `batch_head_vs_mean.py`
  → `hpc-work/head_vs_mean_panel.csv`. **VERDICT: head-vs-mean is NOT systematic — deltas scatter ±0.03 SYMMETRIC
  around 0 (7 LR>head, 13 head>LR, 11 tied), NO clustering on hard drugs; largest |Δ| land on low-n_neg/imbalanced
  holdouts (azithro nneg=40, colistin 200, kp-levo 97) = AUROC noise. The FT head ≈ a plain LR on the genome-mean
  everywhere → LayerNorm NOT shedding signal; azithro +0.017 was noise; NO head surgery.** Honest ft_mean recovered
  AUROCs (leak-proof at scale, none inflated): Kp carbapenems/cephalosporins 0.95–0.99, cipro 0.974, azithro 0.816
  (was leaked 0.918), colistin 0.887, tetracycline 0.865, aztreonam 0.842; TB rifampin 0.958, INH 0.894, EMB 0.909,
  PZA 0.910, moxi 0.789, ethionamide 0.815. **⚠ cefotaxime = ONLY guard-fail (cache holds 404/464 holdout <95%) due to
  ESM/parquet protein-count "misalignment"** (n_real>len(gene_names) skip; some Kp genomes' ESM store has more
  proteins than their protein_sequences parquet — ESM built off a different annotation than the pulled parquets).
  All 32 ft_genome_mean_<drug>_trainholdout.npz caches now on CSD3 (Kp 22 + TB 10) → ready for the full 5E ladders
  once baclm coding transfers.
- **David steer (2026-07-31):** (1) **head-vs-mean CLOSED** — confirmed not systematic, NO head surgery. (2)
  **INVESTIGATE cefotaxime** ESM/parquet misalignment. (3) **KICK OFF Kp ladders (all except cefotaxime) NOW, in
  parallel with cefotaxime reconciliation**, then add cefotaxime once reconciled. Transfer resumed 2026-07-31
  (job bb7icyho0) after an overnight CSD3 connection-rate block (baclm/tb stopped at 18/24; CSD3 auth self-recovered);
  baclm/kleb DONE → Kp coding rung ready; baclm/tb finishing (18-24) then frozen bacformer (for 5F).
- **Two mid-migration 5D mechanics:** (a) **stand-in header-only ranking**
  `hpc-work/standin_ranking/<drug>/per_gene_lr_<drug>.csv` (one sub-0.6 dummy row → empty top-set) because the
  `--ranking-csv` only drives the per-gene SIDE output (`gene_emb/`), regenerated in **5F** with the real baclm
  ranking once baclm transfers; the ladder's ft_mean rung is gene-agnostic. **⚠ so `--skip-existing` would
  wrongly skip a 5F rebuild — `rm` the cache dir in 5F.** (b) `--parquet-dir` = the relocated protein_sequences.
- **Track A parquet relocation DONE (David-approved):** native `processed/{klebsiella_protein_sequences 6418,
  train_tb_ast/tb_protein_sequences 38248}` → `bacformer_processed/protein_sequences/{klebsiella,tb}` +
  back-compat symlinks at old paths; task-root symlinks resolve. Transfer now fills only the 687 Kp parquet gap.
- **⚠ CSD3 whole-Kleb ESM store is NOT a strict superset of the Kp AST cohort** — **209/7088 (2.9%) missing**
  (IDs in `hpc-work/kp_missing_esm_ids.txt`); TB per-task esm ~14 short. Bites the whole Kp fan-out coverage
  (azithromycin holdout fell to 87.8% < the 95% guard → **azithromycin 5D DEFERRED**). Fix = add
  `esm:kleb:klebsiella` (+`esm:tb:tb`) to the next `migrate_stores.sh` re-run — its `comm -23` auto-pulls only
  the gap into `bacformer_processed/esm/<org>`.
- Canonical Kp cohort = **7088** (binary_ast_with_split.csv); 9,724 = embedding superset.

The Isambard specifics below (worktree@9060617, gpu-venv, `ssh u6fp`, jobs 5768108/9, monitor b2dn8d6v3) are
**SUPERSEDED for mechanics**; the git-rm curated-vs-untracked scope (step 4) + leak analysis remain valid.

**★ CSD3 5E-LADDER STATE (2026-07-31).** 5E pipeline (per_segment_lr CLI + ranking launchers + ceiling +
build_amr_ladder) is FULLY CSD3-ready at HEAD `5835f02` (another agent advanced our shared branch w/
bac_pyseer commits; local+CSD3 both 5835f02). baclm coding+reembed FULL coverage (9724 Kp). Kp ladder =
regenerate 4 imputed rankings (coding `build_per_gene_lr_ranking_imputed.sh EMBEDDING_STORE=baclm`;
upstream/igr `SPECIES=kp FEATURE=imputed_full BACLM_DIR=…/baclm_reembed`; unit `FEATURE=imputed`) + ceiling
(`build_amr_calls_store.sh` [⚠ its PYTHONPATH hardcodes stale `$HOME/BacPredict/src` — fix] → `card_determinant_lr.sh`)
→ `build_amr_ladder.sh SPECIES=kp --array=…`. Coding(32447368)+unit(32447369) rankings SUBMITTED (GFF-free).
- **★ MIGRATION ANNOTATION-MIX = root cause (David's off-by-3 debug).** CSD3 has TWO Bakta annotations mixed:
  **whole-Kleb "A"** (the CSD3 ESM 87k, native parquets, `raw/klebsiella_gff3` 84693) vs **Isambard per-task "B"**
  (baclm, baclm_reembed, the FT-training esm, `raw/kleb_ast/gff` 9724). Differ ~3 proteins/genome (sample
  SAMEA2609453: ESM-A 5321 vs parquet-B 5318). This one root cause produces BOTH the "off-by-3 parquet skip"
  (cefotaxime coverage-fail) AND the "307 GFF-less" (whole-Kleb A GFF covers only 9417/9724). Isambard **B GFFs
  = 9724 = full cohort** AND are what baclm_reembed was built from.
- **★ David's method call (2026-07-31): do NOT zero-impute data-missing genomes** (conflates missing-data with
  region-absent → biases non-coding rung) and **do NOT drop ~7%** (reviewer red flag). FIX THE DATA: **pull the
  Isambard B GFFs → CSD3, repoint input_csv to them (100% cohort), re-run on the FULL cohort.** input_csv backup
  at `…/train_kleb_ast/embedding_input.csv.isambard.bak`; my whole-Kleb-A repoint (9417/9724, zero-impute 3.2%)
  was REJECTED. Off-by-3 parquet: after the protein diff, likely also pull B parquets for coding-rung consistency.
- **Cache-skip fix `e61b6c2` (committed, clean FF, my file only):** tolerate n_real>n_gene by <=50 (mean is
  order-invariant), per-gene tokens only when exactly aligned. May become moot if B parquets are pulled.
- **⚠ CSD3 LOGIN RATE-BAN GOTCHA:** bursting interactive ssh to `login.hpc.cam.ac.uk` trips a fail2ban-style ban
  (`Permission denied (keyboard-interactive,hostbased)`) that persists 60+ min; a 5-min poll KEEPS it alive.
  Fix: go quiet (30-min gentle poll b1r1vjvfh), BATCH all CSD3 ops into single sessions. SLURM jobs + low-rate
  transfer pushes are unaffected. Transfer (bb7icyho0) stopped rc=3 at baclm/tb when the ban caught its push →
  re-launch when CSD3 clears (resumable).

**Git.** Branch `refactor/consolidate-engine`. Local + remote (abelsond-cam) HEAD = **`ece12bf`**
(`9060617` fix → `0d415ad` --skip-existing → `c33ffda` Phase-I hygiene [de-stage + fit_one_gene→
fit_one_segment rename, 65 sites] → `ece12bf` **2nd leak fix**). 352 tests green, ruff clean.
**Isambard origin is STALE at `9060617`** — the proof runs fine there (scripts already pass
`--scope trainholdout`), but `git -C $WT fetch` before the FAN-OUT. (Earlier "divergence" scare was the
same stale-fetch: 9734e8d is an ANCESTOR of the fix; my branch is the superset.)

**⚠ 2ND LEAK SITE — FIXED `ece12bf` (the same #1 bug, in the concat module).**
`concat/concatenate_bacformer_genome_esm_protein_emb.py::run_concat_probe` resolved its split via
`resolve_clean_splits(ast_sheet, drug)` with NO `checkpoint_dir` → CSV single-split. In **FT mode** the
FT genome-mean was scored on the CSV holdout (overlaps FT-train → leak); even `--kfold-on-eval-holdout`
restricted to those CSV evaluate ids (azithro CSV-eval shares only 69/370 with the deployed holdout). Fix:
new `_resolve_concat_splits` forces a finetuned mean onto the deployed `results.json` k-fold holdout
(`checkpoint_dir=<FT run>`) and FAILS LOUD if none given (never silent CSV fallback); FT run dir =
`--bacformer-checkpoint` or new `--holdout-checkpoint` (loaded FT NPZ). Frozen means unchanged (label-blind).
**This flips Part G**: the concat module is NOT stale (live test + 9 launchers, edited yesterday) → it was
FIXED, not archived. **It produced published concat rungs** (e.g. TB `rif_ladder_table.csv` "concat: FT
Bacformer mean + esm-rpoB" 0.9769, tagged `k5x3-evalhold` = the CSV holdout) → those are optimistic, add to
the step-3 git-rm/regenerate scope.

**Strategic direction (David, 2026-07-24, OPEN — plan after proof):** the leaks recurred because of accreted
merge/split code + overloaded "gene" vocab. Agreed target = a clean-core **`segment_amr_lr`** (segments =
coding-baclm / coding-ESM / intergenic / rRNA / upstream — one segment-type param) routing through the ONE
`holdout.py` resolver + one fit/score primitive, then DELETE the duplicate modules (`coding_amr_lr`+
`igr_amr_lr`+per-unit/upstream fold in). NOT a greenfield rewrite (keeps the forward-pass / ESM-baclm-load /
CARD-Kleborate parts) — a deliberate clean-core extraction = an upgrade of the plan's Phase II/III. Part
F-coding (coding_amr_lr's per-segment self-derived holdout) is label-blind (no leak) → fold into segment_amr_lr,
don't in-place patch. (David corrected my "don't unify, it shifts fabG1→ETH" — privileging a published number
over correctness contradicts the whole 0.918→0.799 ethos.)

**⚠ Isambard SSH auth EXPIRES (~work-day).** When it lapses, `ssh`/`squeue`/`sacct` AND any background poll
all fail with `Permission denied (publickey)` — this is what silently burned the last monitor's 12h cap
(it couldn't tell "still queued" from "can't connect"). Fix = user re-auths (they did, 2026-07-24; then
`ssh-add ~/.ssh/isambard`). New monitor `b2dn8d6v3` exits 3 (AUTH_EXPIRED) on publickey denial so it pings
fast instead of stalling. Script: `scratchpad/poll_cache_jobs.sh`.

**Isambard.** `ssh u6fp.aip2.isambard`. concat worktree (`$SCRATCHDIR/worktrees/concat`) is DETACHED at
`9060617` — the proof runs there; do NOT `git checkout`-advance it while 5764616/5764617 are pending/running
(memory foot-gun). Run env: `$SCRATCHDIR/envs/bacpredict-gpu-venv/bin/python` with
`BACPREDICT_REPO=$SCRATCHDIR/worktrees/concat` (sets PYTHONPATH).

**GPU jobs (PENDING, scope=trainholdout).** `5768108_[20]` = Kp azithromycin (deployed holdout n=384, wall
2h), `5768109` = TB rifampin (deployed holdout n=7127, wall 3h). Both deployed splits: kfold, n_folds=5,
fold=0, evaluate_seed=1. Monitor **b2dn8d6v3** fires when both leave the squeue. *(Resubmitted 2026-07-24
from the originals `5764616`/`5764617`, which sat PENDING ~12h on an 8h wall and never ran — the long wall
wrecked Isambard backfill; tight walls (~40–95 min of real GPU work) are the fix. Both cancelled.)*

**Cleanup already done:** cancelled the leaky `amr_ladder` array job 5758597; DELETED all 11 leaky
un-scoped `ft_bacformer_cache/*/ft_genome_mean_<drug>.npz` (6 Kp + 5 TB) so nothing leaky can be scored
(orphan `cache_summary_<drug>.json` from Jul 17 may linger — harmless; overwritten by the new run). The
corrected `ft_genome_mean_<drug>_trainholdout.npz` + `cache_summary_<drug>.json` are written by the jobs.

**NEXT when the caches land (monitor fires):**
1. Rebuild ladders from concat@9060617 (gpu-venv + BACPREDICT_REPO): azithromycin (kp) + rifampin (tb) via
   `python -m bacpredict.engine.concat.build_amr_ladder --species {kp,tb} --drug <d>
   --ft-cache-dir <root>/processed/train_{kleb,tb}_ast/pangena_predict/ft_bacformer_cache/<d>`. The ladder
   reads checkpoint+scope from cache_summary, resolves the deployed k-fold holdout, fits-on-FT-train /
   tests-on-FT-holdout, and GUARDS cache coverage.
2. VERIFY: azithro rung-1 `ft_mean` AUROC ≈ **0.799** (honest; was 0.918 leaked); rifampin ≈ **0.96**.
3. GAP to fix: `engine/scripts/build_amr_ladder.sh` pre-check still tests the un-scoped
   `ft_genome_mean_<drug>.npz` (now deleted) — update it to `*_trainholdout.npz` (or run the module directly
   for the proof).
4. `git rm` the contaminated azithro+rifampin ladder tables → visible-error commit → regenerate fresh →
   verify render (David's step 3). **Scope finding (2026-07-24, read-only):** the DIRECT fix target is the
   untracked auto-generated `visualisations/{kp/azithromycin,tb/rifampicin}/*_amr_ladder_table.csv` (the
   `build_amr_ladder.py` chain — regenerate freely). The TRACKED FT-bearing artifacts are OLDER curated
   CSD3-era tables from other code paths: Kp `azithromycin_card_ladder_table_family.csv` (BacF FT mean
   0.8215) + `azithromycin_card_ladder_family.png` + the `card_esm_vs_ft_*`/`esm_vs_ft_*` chain; TB
   `rif_ladder_table.csv`/`.md` (fine-tuned Bacformer mean-pool 0.9046, concat FT-mean 0.9769 tagged
   `k5x3-evalhold`) + `rifampicin_ladder_barplot.png`. These are hand-assembled publication tables (source
   col cites old job IDs + k5x3 SDs), NOT build_amr_ladder output — some may already use eval-holdout. Do
   NOT blanket-`git rm` them: after the proof lands, check each FT rung's code path/holdout and **confirm
   the removal scope with David before deleting curated publication tables** (concurrent-agent + reversibility
   caution). The label-blind rungs (CARD determinant LR, ESM/baclm per-gene, WHO one-hot, tbprofiler) are
   CLEAN — keep them.
5. Phase I remainder: Part F (coding_amr_lr holdout consistency), Parts G/H/I hygiene, core rename
   `fit_one_gene`→`fit_one_segment` (defer `gene_lr`→`segment_lr` dir to Phase III), then FAN OUT all ladder
   drugs (≈7 Kp + 5 TB) via the `--skip-existing` FT launchers + rebuild every ladder.

Never call a high-penetrance non-catalogue AMR gene a "lineage correlate" → [[dont-conflate-penetrance-with-lineage]].

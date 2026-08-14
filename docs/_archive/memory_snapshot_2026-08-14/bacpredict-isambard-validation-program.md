---
name: bacpredict-isambard-validation-program
description: "LIVE state of the Isambard pipeline-validation → baclm-concat-panels program (plan the-cambridge-hpc-and-dreamy-thacker): Phase 0 done, Phase 1 canonical finetune launched (TB rif 5661316, Kp cipro 5661317)"
metadata:
  node_type: memory
  type: project
  originSessionId: 287d1555-8fdc-4241-9446-ecac54a396be
---

The program after the L/M/N declutter: **validate the consolidated pipeline on Isambard, then build
the new AMR summary panels**. Plan file: `~/.claude/plans/the-cambridge-hpc-and-dreamy-thacker.md`
(phases 0–6). Branch `refactor/consolidate-engine` (unmerged; HEAD `2c226ba`). Deliverable = redo
`src/bacpredict/visualisations/{tb,kp}/amr_summary_panel.csv` as **red catalogue-ceiling** vs **blue
concat** built as TWO series: 2-way `bacformerFT-mean ⊕ baclm-best-gene`, then 3-way `+ baclm-best-IGR`.
See [[bacpredict-engine-consolidation]] for the code layout, [[bacpredict-dual-cluster-data-root]] for paths.

## Locked decisions (2026-07-15)
- Ceiling = run our OWN minimap calls on-cluster (existing `annotate_amr_sidecar` IS the Kleborate-style
  caller; refs in BacHGT sibling `$HOME/BacHGT/src/bac_kleborate/refs/kleb_amr/inputs/`). Build CARD +
  Kleborate-grain one-hots from the same sidecar; red = union/higher. No `metadata_v2` needed.
- IGR identity = **ordered 5′→3′ flanking-gene pair** `left_gene→right_gene` (ascending genome coord).
- Best gene / best IGR = single top by own-LR AUROC.
- Canonical drug first: Kp ciprofloxacin, TB rifampin(US spelling). TB bf16 IS compared to the ~0.9 fp32.

## Phase 0 — DONE
Weights present+complete (`macwiatrak/bacformer-large-masked-complete-genomes`, 1.73 GB safetensors,
snapshot `ab3a91a2`, on `$PROJECTDIR/david/cache/hf`). BacHGT ref panel present. Store filenames
confirmed & committed (`70535c0`): `bacformer/{Sample}_bacformer_embeddings.pt`,
`intergenic/{Sample}_intergenic.parquet`. Worktree refreshed.

## Phase 1 — canonical finetune RUNNING (verify when they finish)
- **Jobs:** TB rifampin `5661316`, Kp ciprofloxacin `5661317` — both `sbatch --array=0` (single split =
  k-fold fold0/seed1), `--time=24:00:00`, workq/brics.u6fp. **RUNNING as of 2026-07-15 ~13:5x**
  (nid010104 / nid010106) after ~2h PENDING for a GPU slot. Early-stop ~10-16h expected.
- **n=10 GPU smoke PASSED for both** first (loss→0, results.json written, ~88 s) — pipeline confirmed.
- **Repro checks when they land (async):** Kp cipro FT AUROC ≈ 0.979 ([[kp_cipro_beats_kleborate]]);
  TB rif bf16 vs fp32 ≈0.9 — compare, record delta. Results → `results.json` in each `--output-dir`:
  TB `…/processed/train_tb_ast/checkpoints/mycobacterium_tuberculosis_rifampin_lr_0.00015_finetuned*`,
  Kp `…/processed/train_kleb_ast/models/finetune/klebsiella_pneumoniae_ciprofloxacin_lr_0.00015_finetuned*`.
- Check: `ssh -o IdentitiesOnly=yes u6fp.aip2.isambard 'squeue -u dca36.u6fp; sacct -j 5661316,5661317'`;
  logs `$SCRATCHDIR/logs/{tb_rifampin,kp_ciprofloxacin}-<jobid>_0.out`.
- **✅ Kp cipro FT FINISHED — REPRO PASSES.** `results.json` on disk: **eval-holdout AUROC 0.9717 / AUPRC
  0.9863** (best `checkpoint-7750`); ≈ the ~0.979 target on the refreshed complete-genomes base ([[kp_cipro_beats_kleborate]]).
  Dir: `…/train_kleb_ast/models/finetune/klebsiella_pneumoniae_ciprofloxacin_lr_0.00015_finetuned_fold00_seed1`.
  **This is the concat test case** (see the concat-build next-stretch below). TB rif still training.
- **FT DIAGNOSTIC (2026-07-15 ~17:00):** Kp cipro **best val eval_auroc = 0.9848** at step 7750 (≈epoch 20;
  `best_model_checkpoint` in trainer_state) — repro check ESSENTIALLY PASSES (≈0.979 expected). Train
  loss→0 by ~epoch 20+ (memorises 3079-sample train set); HARMLESS because `load_best_model_at_end=True`
  restores the best ckpt for the eval-holdout results.json. It over-ran to ~epoch 39 only because
  patience=30 × eval_steps=250 = ~19 Kp epochs of tail (but only ~1.7 TB epochs — the mismatch is a fixed
  step count vs cohort size). TB genuinely still climbing (val 0.80→0.89, loss ~0.4, epoch 5) — fine.
- **FAN-OUT CONFIG FIX committed `c105e6e`** (future runs only, NOT the in-flight jobs): `finetune_amr`
  gained `max_epochs` (default 40) → effective max_steps=min(max_steps, steps_per_epoch×max_epochs) + warmup
  scaled off the cap (was flat 100k = ~260 Kp epochs, ~26-epoch warmup); `early_stopping_threshold` (0.005);
  patience 30→15. 86 finetune tests green.
- **⚠️ OPEN DECISION — reuse pre-trained models vs re-run fan-out.** User has "good finetuned models on the
  hpc" and wants to download→laptop→upload to Isambard to skip the ~22 Kp + ~9 TB fan-out (saves ~100s
  GPU-h). **NOT started — surfaced 2 blockers to confirm first:** (1) BASE PROVENANCE — §0.1 requires the
  REFRESHED complete-genomes base (the 2 canonical jobs use it); if the stored models are on the old
  MAG/pre-refresh base they're NOT comparable — must confirm. (2) LOCATION/REACHABILITY — if "the hpc" =
  CSD3: its RDS is **live again as of 2026-07-29** (outage over; only tape cold-storage still down), so
  downloads/transfers work — this blocker is cleared. Also need which drugs exist. Plan once confirmed: transfer
  as data (weights≠code, laptop round-trip OK) + load one ckpt on Isambard & re-eval holdout to verify AUROC
  before wiring into the panel.
- **IGR presence one-hot control DONE & CONFIRMED on disk** (jobs `5667241` TB / `5667242` Kp COMPLETED;
  TB 2517 pairs, Kp 6303 pairs in the 1-99% band) — see the Phase-3 presence block below; refuted the
  lineage-proxy guess for pptA→lamB.

## CRITICAL cluster gotchas (this program)
- **PYTHONPATH must point at the WORKTREE, not `$HOME/BacPredict/src`.** The shared checkout stays on
  `dev` (pre-consolidation, NO `bacpredict` package). The H3 scripts hardcode
  `PYTHONPATH="$HOME/BacPredict/src:…"`; dev has no `bacpredict` so it's harmless, but you MUST add the
  worktree: submit with `--export=ALL,PYTHONPATH=$SCRATCHDIR/worktrees/consolidate/src` (and run the
  worktree copy of the script). Worktree = `$SCRATCHDIR/worktrees/consolidate` @ origin/refactor/consolidate-engine.
- **QOS caps wall at 24h** (`workq_qos = 1-00:00:00`); scripts' `#SBATCH --time=36:00:00` is REJECTED
  (`QOSMaxWallDurationPerJobLimit`). Always submit with `--time=24:00:00` (CLI overrides the directive).
- **TB early-stopping was too tight** (patience 30 ≈ 2.5 epochs vs Kp's ~15, because TB cohort is ~10×).
  Fixed in `2c226ba`: TB script now `eval_steps=${EVAL_STEPS:-1000}`, `patience=${PATIENCE:-45}` ≈ 15 epochs.
- **Isambard bills wall USED not requested** → keep 24h on canonical runs (zero cost, GH200+early-stop
  finishes ~10–16h). workq was NOT congested (only our 2 jobs PD, idle GPUs on `mix`). Right-size the
  Phase-6 fan-out (~30 jobs) to ~1.5–2× the MEASURED canonical wall — that's where a shorter --time helps backfill.
- No GPU wasted on labelling: `binary_ast_with_split.csv` already carries labels+splits (TB rif 35,635
  labelled; Kp cipro 4,389); trainer injects labels at runtime, `.pt` I/O overlapped on 32 CPU workers.

## Phase 2 — baclm per-gene ranking DONE (best-gene input); 2-way concat WAITS on FT
- **Code (commit `fd9471f`, pushed):** `build_per_gene_lr_store.py` gained `--embedding-store {esm,baclm}`.
  New `_embedding_rows(store, store_kind, n_genes)` routes the reader: ESM = padded/interleaved +
  `real_protein_indices` (n_real ≤ n_genes); baclm = plain `[n_cds,dim]`, must match parquet CDS count
  exactly (else skip). `read_genome`/`assemble_gene_matrices`/`build_panels`/`run` thread `store_kind`;
  summary records `embedding_store`. 2 new Stage-A tests (both readers). 11/11 + 165 engine tests green.
  Both ranking SLURM scripts (engine TB `build_per_gene_lr_ranking.sh` + Kp
  `build_per_gene_lr_ranking_imputed.sh`) gained `EMBEDDING_STORE` env selector; output namespaced
  `per_gene_lr_ranking[_imputed]_<store>/<drug>` so esm/baclm never collide.
- **Reader validated on real baclm `.pt`** (login smoke, 30 genomes): rpoB OOF AUROC **0.9533** (leakage
  check passes, rpoB tops real core genes for rifampin — vs ESM's ~0.962). baclm stores present TB 38257 / Kp 9724.
- **Canonical baclm gene rankings DONE** (`5661818`/`5661820` COMPLETED). **Best baclm gene, validated:**
  TB rifampin → **rpoB AUROC 0.9625** (≈ ESM's 0.962), Kp ciprofloxacin → **gyrA 0.9156** (QRDR; parC also top).
  Both biologically correct. Tables: TB `…/train_tb_ast/pangena_predict/per_gene_lr_ranking_baclm/rifampin/per_gene_lr_rifampin.csv`,
  Kp `…/train_kleb_ast/pangena_predict/per_gene_lr_ranking_imputed_baclm/ciprofloxacin/per_gene_lr_ciprofloxacin.csv`.
  These are the best-gene blocks for the 2-way concat (**needs FT — waits**).
- **STILL TODO in P2:** consolidate+DELETE Kleb per-gene/concat compute into engine (cut `collect_reliable_amr`
  seam) — do when FT lands + 2-way concat is validated (concat is the thing being consolidated).

## Phase 3 — per-IGR LR ranking DONE (code); best-IGR input; 3-way concat WAITS on FT
- **New `engine/gene_lr/build_per_igr_lr_store.py`** (commits `de1a3ae`→`a34f80b`, pushed). Ranks baclm
  intergenic regions by out-of-fold AUROC; each region NAMED by its ordered 5′→3′ flanking-gene pair
  `left→right` (ascending genome coord, oblivious to strand — coarse; Nuna refines). A region is named only
  when both directly-abutting flanks are `gene=` symbols (boundary-tol bp). Reuses the gene store's
  `load_splits`/`subsample_balanced`/`fit_per_gene` + `igr_amr_lr._parse_gff`. NB `igr_amr_lr.py` is the
  TARGETED promoter probe (Phase-4 plots), NOT this discovery store. 7 Stage-A tests. Ranking script
  `engine/scripts/build_per_igr_lr_ranking.sh` (SPECIES tb|kp). GFF flank-join VALIDATED on real data
  (login smoke: 2566 named pairs / 30 genomes; serial n_jobs=1 wrote best pair).
- **AARCH64 GOTCHA (cost 3 debug rounds):** the per-IGR sweep does `torch.load(mmap=True)` per genome;
  parallelising it (mp.Pool OR joblib thread/loky) leaves torch/OpenMP multithreaded in the main process,
  and the subsequent FORK for the process-parallel numpy fit (`fit_per_gene`, n_jobs) SEGFAULTS on Grace.
  Fix = **serial sweep + process-parallel fit** (mirrors the gene store, proven at n_jobs=32 in SLURM).
  Also: **joblib n_jobs>1 fit SEGFAULTS on the LOGIN node but works in a SLURM allocation** — validate any
  n_jobs>1 CPU fit via `sbatch`, not login. (Login smoke: keep `--n-jobs 1`.)
- **Canonical TB IGR ranking `5664098` COMPLETED at scale** (2m09s, n_jobs=32 in SLURM — confirms the
  serial-sweep + process-fit works in an allocation; login-node crash was purely a loky limit). 12,894
  named pairs → 2,312 core (>10%) → all fitted. **Best rif IGR = `mlaD→mlaD` 0.71** — appropriately WEAK
  (rif = rpoB coding mutation, not promoter; the IGR/3-way block should add little for rif, more for
  fabG1/pncA-promoter drugs). n_filtered=0 at 0.8. Output
  `…/train_tb_ast/pangena_predict/per_igr_lr_ranking/rifampin/per_igr_lr_rifampin.csv`.
- **Kp cipro IGR ranking DONE** (job `5666340_5`; 4,757 named core pairs). **Best embedding = `pptA→lamB`
  0.974** (carriers-only, prev ~0.10), then `gatY→glpR` 0.944. Output
  `…/train_kleb_ast/pangena_predict/per_igr_lr_ranking/ciprofloxacin/per_igr_lr_ciprofloxacin.csv`.
- **PRESENCE/ABSENCE one-hot control DONE — REFUTES the "lineage proxy" guess** (new `--feature presence`,
  `87683cc`; TB `5667241`, Kp `5667242`; 1-D one-hot over the 1–99% band, out `…/per_igr_presence_lr_ranking/`).
  pptA→lamB **presence AUROC = 0.539** (≈ chance) vs its **embedding 0.974** → the signal is NOT the region's
  presence/phylogenetic distribution; it's baclm sequence content WITHIN the ~10% carriers. TB mlaD→mlaD
  presence 0.504 (embed 0.713). Presence is weak across the board (TB top 0.583 tatB→nagD; Kp top 0.665
  trnL→trnL, asnS→ompK35 0.646). Caveat: embedding fit carriers-only, presence fit all-genomes-imputed —
  complementary universes; the crude presence-lineage hypothesis is refuted, within-carrier sub-clonality
  not fully excluded. So the baclm IGR channel reads SEQUENCE, not synteny/lineage — reassuring for the concat.

## pptA→lamB RESOLVED + all-drug LR fan-out LAUNCHED (2026-07-15, `0da60d7`)
- **The 0.974 is a within-carrier, conditional-on-carriage AUROC — NOT whole-cohort.** The embedding
  IGR/gene screen calls `fit_per_gene(impute_absent_zero=False)` → the `else` branch labels from CARRIER
  IDs ONLY (`gene_matrices[g][0]`) and StratifiedKFold runs over just the carrier matrix; absent genomes are
  **dropped, not zero-coded**. On disk: pptA→lamB prev 0.1035, **n_train=207, n_pos=150** (only 57 sensitive
  carriers). Every top Kp pair is low-prev + resistant-enriched (lysr→iscr 217/197). TB shows none (top 0.71,
  balanced). So embedding (0.974, over 207) and presence (0.539, over all 2000) are DIFFERENT populations —
  not contradictory. **Per-gene rankings unaffected** (core genes gyrA n_train=2000/rpoB 1975; the existing
  `_imputed_baclm` run gives gyrA the SAME 0.9156 → impute≡drop for core). **User verdict: conclusive,
  candidate-causal/novel — causality test DROPPED** (0.974 OOF can't come from lineage).
- **Selection ≠ usage (deferred concat fix):** `concat_ingredients.impute_block` zero-imputes absent genomes,
  so best-IGR is *selected* by carrier-only AUROC but *used* as a block that's 0 for ~90% of genomes. When
  the concat is built (post-FT), decide whether to select best-IGR/gene by the zero-imputed WHOLE-COHORT
  AUROC (`--impute-absent-zero` flag already exists) — surface, don't switch silently.
- **FAN-OUT LAUNCHED (all remaining drugs, baclm, NO FT — user imports FT ckpts).** Script edit `0da60d7`
  added a `SPECIES=kp` branch + idempotency guard to `build_per_gene_lr_ranking.sh` (it was TB-only;
  build_per_igr already had both). 4 idempotent arrays submitted from the worktree with
  `PYTHONPATH=$WT/src`: **5669016** TB gene baclm (0-9) · **5669017** Kp gene baclm (0-21) · **5669018** TB
  IGR (0-9) · **5669019** Kp IGR (0-21). Remaining = TB gene ×9, Kp gene ×22, TB IGR ×9, Kp IGR ×21 = 61
  drug-jobs (canonical rif/cipro skip). Each ~3–12 min CPU, min-prev 0.10, 2000-genome balanced subsample.
  Output namespaced `per_gene_lr_ranking_baclm/<drug>` and `per_igr_lr_ranking/<drug>`. **CP-2 when they land:**
  assemble wide gene×drug + IGR×drug tables per organism, surface each drug's top gene/IGR (flag novel
  low-prev IGR hits like pptA→lamB with prevalence/n_carriers/n_pos).

## Consolidation — ALL relocations DONE (2026-07-15, HEAD `849c245`, pushed)
The full Kleb→engine relocation block (per [[bacpredict-kleb-engine-consolidation-map]]) is COMPLETE:
8 modules moved into `engine.{gene_lr,concat}` (per_gene_esm_vs_ft, concat_gene_panel, reliable_concat,
gene_ingredient_concat+aggregate, cache_bacformer_gene_embeddings, bacformer_token_cache [#9/#10 collapsed],
cache_genome_mean); CARD stays as thin apps/kleb CLIs via the `card_amr_calls` calls_fn seam; scripts +
render_card_figures repointed. Commits 1daba5f/3997ddd/9025fe9/849c245. 265 tests green, ruff clean.
**GPU behaviour-validation of the moved cachers still waits on FT** (behaviour-preserving by construction).

## CARD annotation RERUN on Isambard (2026-07-15) — the gating step for the combined plot
User confirmed: rerun CARD annotation on Isambard (CSD3 results stranded). **The combined-plot deliverable
uses the EXISTING relocated modules, not a new baclm_concat** — user steered me to `bacformer_token_cache`
(finetuned mode = the "one forward, save gene final-states + mean" module). Pipeline:
`annotate_amr_sidecar → build_amr_calls_store → cache_ft/frozen_amr_proteins (GPU) → reliable_ft_concat (CPU)
+ card_determinant_lr (ceiling) → combined plot`.
- **4 Isambard blockers found+fixed (all committed, pushed; HEAD `6973bb1`):**
  1. **metadata_v2 ABSENT on Isambard** → new `apps/kleb/build_isambard_amr_path_sheet.py` (`132ffa6`) walks
     `raw/kleb_ast/{assemblies/{S}.fa.gz, gff/<shard>/{S}/{S}.bakta.gff3.gz}` → 3-col path sheet
     (`Sample,sr_assembly_file,sr_gff_file`, absolute) that `annotate_amr_sidecar --metadata` eats unchanged.
     Built: **`processed/train_kleb_ast/amr_path_sheet.tsv` = 7088 AST-cohort genomes** (0 drops; the 9724
     esm/baclm files are a superset; split sheet = 7088).
  2. **amr-ref-dir default is stale CSD3 path** → `annotate_amr_parquet.sh` now passes
     `--amr-ref-dir $HOME/BacHGT/src/bac_kleborate/refs/kleb_amr/inputs` (refs confirmed present on Isambard).
  3. **aarch64 pixi** (`e2d9dab`): kleb `pixi.toml` was linux-64/osx-arm64 only → added `linux-aarch64`;
     `pixi install` → **minimap2 2.31** at `/projects/u6fp/david/envs/pixi/kleb-ast-amr-tools-*/envs/default/bin/minimap2`.
  4. **keep_internal_stop mismatch** (`85e217a`): Isambard protein/ESM store was built **keep_internal_stop=FALSE**
     (5035 rows), but `annotate_one` hardcoded True (5038) → ALL samples 'misaligned'. Made it a CLI flag
     (`--keep-internal-stop/--no-keep-internal-stop`, default True=CSD3); script passes `--no-keep-internal-stop`
     (KEEP_INTERNAL_STOP=1 to flip on CSD3).
- **3-genome smoke PASSED** (`--no-keep-internal-stop`, status ok×3): sensible calls incl **GyrA/ParC**
  (cipro chromosomal drivers) at real flat_index, OmpK35/36, PmrB, acquired SHV/CTX-M/Dfr.
- **Jobs QUEUED (workq congested — TB-rif + another agent's probe hold nodes):** annotation array **5669673**
  (`--array=0-35`, CHUNK=200, 7088 genomes, `--export=ALL,PYTHONPATH=$WT/src,MM2=<pixi minimap2>`),
  `build_amr_calls_store` **5669681** (afterok:5669673). Sidecar out = `processed/train_kleb_ast/amr_annotation/`.
- **New concat sbatch wrappers committed `6973bb1`** (the CARD-sidecar concat path had none):
  `scripts/cache_amr_tokens_kleb.sh` (GPU: FT+frozen token cache one job, DRUG-param, auto-globs best ckpt) +
  `scripts/reliable_ft_concat_kleb.sh` (CPU). Downstream chain (card_determinant_lr afterok:store; GPU
  token-cache afterok:annotation; concat afterok:token-cache) — submit pending SSH recovery (bg `b8ork14w2`;
  Isambard login threw sustained `Permission denied (publickey)` mid-session — transient, retry).
- **NEXT when annotation lands:** verify sidecar coverage (count + cipro-cohort GyrA presence) BEFORE the GPU
  token-cache fires (it's PENDING(Dependency) — cancel if thin). Then reliable_concat gives the blue
  FT-mean⊕best-gene number vs FT-alone 0.9717 + red CARD ceiling → **combined plot** (`engine.plots` has NO
  `amr_summary_panel` yet — needs porting). **baclm-gene/IGR 3-way (gyrA + pptA→lamB) = follow-on**, still open:
  the selection≠usage decision (carrier-only vs zero-imputed IGR pick).
- **CP-2 (background):** fan-out arrays `5669016-5669019` STILL PENDING (whole workq backed up).
- TB rif FT (`5661316`) still RUNNING (~7h).

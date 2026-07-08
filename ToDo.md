# BacPredict — live tracker

Single live status + plan tracker for all BacPredict experiments. Each task block
gives **current state** (a rolling summary) followed by **remaining milestones**
(tick as done). Step-by-step detail and dated running notes live in each task
folder's `CLAUDE.md`; this file is the cross-task summary and a backup for the
per-task agents. Global conventions: root [CLAUDE.md](CLAUDE.md) §0.

## Forward priorities (2026-06-12)

Current top-of-mind items for the coming days. These carry forward the relevant
streams of the retired
[`~/.claude/PROGRAM_PLAN_2026-05-30.md`](../../.claude/PROGRAM_PLAN_2026-05-30.md);
detail lives in the task blocks below.

1. **AMR prediction over time** — pending completion. Ties to
   [tb_ast](src/tb_ast/) / [kleb_ast](src/kleb_ast/) Stage C work.
2. **Defense predictor** — pending completion. Once results land, analyse them in
   BacHGT as reference vs short read, then decide whether to run the whole set or
   improve the model first. Links to the deferred Task 5 / DP-SR block below.
3. **Bacformer vs ESM rifampicin probe** — test ESM and Bacformer embeddings on
   rifampicin prediction. Identify rpoA/rpoB and other key ARG genes via GenBank,
   annotate them from Bakta, then train a linear head on the **frozen** embeddings
   of just those genes (for both ESM and Bacformer). Compare against the
   genome-average prediction using all genes vs a ~20-key-gene subset. (Carries
   forward the retired plan's SNP-representation probe, workstream E.)

## Shared infrastructure

- [x] Refresh Bacformer complete-genomes weights from Hugging Face — model ID
  `macwiatrak/bacformer-large-masked-complete-genomes`; HPC cache pinned to the
  2026-05-15 snapshot. Wired into `tl/embed/generate_embeddings.py` and the task
  train entrypoints.
- [x] Standardise the results JSON schema — `tl/train/metrics.py`
  (`build_results_payload`, `compute_full_metrics`, `write_results_json`) covers
  the §0.4 metric set (AUROC, AUPRC, sens, spec, balanced acc, F1, confusion
  matrix, calibration). Shared across tasks.
- [ ] Standardise the 36 h GPU SLURM template — per-task Stage C scripts exist
  (tb_ast, kleb_iso_source); not yet a single shared template. Known gotcha:
  `--max-steps` must fit the 36 h wall at ~3 s/step, or rely on early stopping.
- [ ] Standardise the smoke-test wrapper — **Stage A must run as a short GPU
  sbatch, not on the login node.** Bacformer-large + per-sample RDS embedding I/O
  exceeds login-node limits; CPU login Stage A silently produces empty tensorboard
  events. Each task has its own GPU smoke script; not yet unified.

## Task 1 — AST in TB ([src/tb_ast/](src/tb_ast/))

**State (2026-05-28).** ESM-C embeddings 100% complete (38,248 / 38,248). Stage A
smoke on rifampin PASS (job 29712625; loss → ~0, AUROC 1.0 on n=10). Split
regenerated at full coverage: 40,021 AST rows → 36,684 kept; rifampin non-NaN
train 24,977 / validate 3,574 / evaluate 7,075, ~31% resistant. **Stage C
launched** — rifampin, single split, SLURM job 29776879 (ampere, 36 h),
`results.json` expected in the checkpoint dir on completion. Stage B skipped by
design (the n=10 train=val Stage A run is the overfit check). NB drug column is
`rifampin` (US spelling).

- [ ] Stage C rifampin — collect `results.json`, report vs catalogue
- [ ] Stage C on pyrazinamide (flagship goldilocks drug), then EMB / MXF / LFX
- [ ] Compare results vs WHO V2 catalogue and CRyPTIC ML benchmarks
- [ ] **HGT-vs-vertical stratified performance** — mechanism via WHO V2; per-stratum AUROC/sens/spec + delta
- [ ] Decision: which drugs justify folds × seeds for publication

## Task 2 — AST in Klebsiella ([src/kleb_ast/](src/kleb_ast/))

**State.** Existing Kp AST models trained but **not formally evaluated**; all prior
training used the old MAG-trained weights. No run yet on the refreshed
complete-genomes model. Outstanding code fix: backport the `dtype="auto"` HF
loading idiom to `train_amr.py` (the `.to(torch.bfloat16)` cast pegs Stage A on
CPU — already fixed in tb_ast and kleb_iso_source).

- [ ] Evaluate existing Kp models against the refreshed model — save as the benchmark to beat
- [ ] Stage A/B/C on a canonical drug (meropenem or ceftriaxone) from the refreshed model
- [ ] Fan out across the Kp drug panel
- [ ] **HGT-vs-vertical stratified performance** — mechanism via AMRFinderPlus + Kleborate; per-stratum AUROC/sens/spec + delta (Kp is the strong test)
- [ ] One-paragraph MAG-vs-complete-genome model contrast
- [ ] Downstream (parked): held-out lineages; drug-class embeddings; Captum explainability; cross-training; Kp pre-training; read-depth gene-copy correction

## Task 3 — Isolation source in Klebsiella ([src/kleb_iso_source/](src/kleb_iso_source/))

**State (2026-05-25).** Re-baselined on the refreshed model + v2 metadata. Split
CSV regenerated (no stratification, no country cap): 25,879 unique samples; train
18,169 / validate 2,591 / evaluate 5,206; 13,289 blood (51.2%) / 12,677 faeces
(48.8%). **Stage A + Stage B both PASS** in one GPU run (job 29713406; train loss
→ 3.6e-5, eval AUROC 1.0 on n=10). **Stage C script ready but not submitted** —
fire with `sbatch src/kleb_iso_source/scripts/train_isolation_source_stage_c.sh`.

- [ ] Submit Stage C full run (blood-vs-faeces, single fold/seed, 36 h GPU)
- [ ] Compare against the prior 0.55–0.62 AUROC benchmark
- [ ] Downstream (parked): Kp pre-training first; complete-genomes-only subset; matched-pair SR-vs-CG contrast; Captum gene attribution; stepwise AUROC across modelling layers

## Task 6 — `predictHGT` embedding diagnostic ([src/predict_hgt/](src/predict_hgt/))

**State.** Not started — empty package stub. Diagnostic, runs in parallel; no
training for the main experiment (Aim 1). Needs only the refreshed Bacformer
weights + HGT/MGE annotations consumed from the sister `BacHGT` module.

- [ ] Pull HGT-region annotations from `BacHGT` (MOB-suite + ISEScan + geNomad prophages)
- [ ] Embed all proteins with the refreshed Bacformer
- [ ] Marker-protein nearest-neighbour analysis (KPC/NDM/OXA-48/*mcr-1*/*tetA*/*iutA*/*rmpA* + housekeeping controls)
- [ ] UMAP coloured by HGT vs chromosomal and by host species
- [ ] Centroid separation score: HGT-vs-chromosomal vs host-context baseline
- [ ] Layer-sensitivity scan (early vs late Bacformer layers)
- [ ] **Optional comparator:** raw ESM-C diagnostic — does contextualisation erase HGT identity?
- [ ] **Decision point:** Aim 1 outcome sets the embedding source for Aim 2 (Bacformer if HGT-preserving, DP-style if context-attractor); document the implication for cross-species HGT-aware work
- [ ] Boundary-detection head (Aim 2): pull ISEScan + MGEfinder ground truth from BacHGT; train per-protein head; evaluate held-out + on SR assemblies

## Task 7 — SNP embeddings: why TB AST is poor ([src/pangena_predict/](src/pangena_predict/))

**State (2026-06-13).** Re-planned + rebuilt; not yet run. Diagnostic task testing the central
hypothesis that Bacformer is blind to chromosomal point mutations because a **chain of two plain
means** (ESM-C residue→protein, then Bacformer protein→genome — both straight mask-normalised
means, no learned attention) dilutes the single causal RRDR residue. Positive control: TB rpoB /
`rifampin`. The first cut was rejected (it loaded all embeddings + used an ad-hoc `train_test_split`,
so its numbers weren't comparable to the deployed model); rebuilt on the repo infra. Full spec in
the task [CLAUDE.md](src/pangena_predict/CLAUDE.md); approved plan in
`~/.claude/plans/i-d-like-to-start-crystalline-allen.md`. On branch `dev` (per user).

- [x] Increment 0 — scaffold + docs (package stub, task CLAUDE.md, this block, root-doc entries)
- [x] **Code built + lint-clean (2026-06-13).** Three-step linear probes, all on the deployed
  model's canonical `binary_ast_with_split.csv` holdout (`tl.train.evaluate.resolve_holdouts`) +
  `tl.train.metrics`:
  [snp_vs_esm_prediction.py](src/pangena_predict/snp_vs_esm_prediction.py) (Step 1 one-hot RRDR,
  Step 2 frozen pooled ESM-C, Step 3a masked-marginal LLR, Step 2b Bacformer token; intersection
  head-line; optional reference-AUROC assertion),
  [rpob_genotype.py](src/pangena_predict/rpob_genotype.py) (RRDR allele + **rpoB-copy QC** +
  provenance docstring), [locate_gene.py](src/pangena_predict/locate_gene.py),
  [frozen_bacformer_rpob_vectors.py](src/pangena_predict/frozen_bacformer_rpob_vectors.py) (2b GPU),
  [geometry_probe.py](src/pangena_predict/geometry_probe.py) (3b),
  [reference_gene/rpoB_H37Rv.faa](src/pangena_predict/reference_gene/rpoB_H37Rv.faa),
  scripts `run_snp_vs_esm_prediction.sh` + `smoke_geometry_probe.sh`. Shared touch:
  [tl/embed/esm_residue_level.py](src/tl/embed/esm_residue_level.py) (residue-level ops) +
  [tl/embed/generate_embeddings.py](src/tl/embed/generate_embeddings.py) (extracted
  `load_bacformer_model` / `bacformer_last_hidden_state`, no behaviour change). Old
  `ceiling_ladder.py` SLURM job 30485091 superseded/dead.
- [ ] **Run Steps 1 + 2** (CPU sbatch — genotypes ~37k rpoB + mmap one-row `.pt` reads): the
  head-line `AUROC(Step 1) − AUROC(Step 2)` (info lost to the ESM-C residue→protein mean) on the
  common evaluate set, vs the deployed Bacformer ~0.9. Login-node `--max-samples` smoke first.
- [ ] **Run the Step-3 GPU pass:** 3a masked-marginal LLR (recoverable pre-pool?) + 3b geometry
  probe (`d_site ≫ d_pool`, best layer) + 2b bonus (Bacformer token ≈ Step 2 ⇒ loss sealed at ESM-C).
- [ ] *Fast-follow (deferred, not blocking):* **TB-Profiler `--fasta`** (assemblies only) →
  validate the sequence-derived RRDR calls (concordance %) + **lineage** + WHO-catalogue calls.
- [ ] **Gate call:** Representational (→ Remedy A then B, no retrain — expected) vs Absent (→ Remedy C).
- [ ] *Deferred (genome-wide, lineage-blocked splits):* Stage 1.3 causal ablations; Remedies A/B/C;
  pyseer oracle in parallel. Check `ebi_parsed_ast_metadata.csv` for lineage, else TB-Profiler.

## Pyseer GWAS ([src/bac_pyseer/](src/bac_pyseer/))

New package compartmentalising pyseer / GWAS analyses, **one subfolder per task**
(`kleb_iso_source` first; `tb_ast` and others to follow). Moved here from the BacHGT
tracker — this is variant-call / unitig / Panaroo-GPA GWAS work, not Bacformer
fine-tuning. Per-task detail:
[src/bac_pyseer/kleb_iso_source/CLAUDE.md](src/bac_pyseer/kleb_iso_source/CLAUDE.md).
An agent will plan the work in detail and start.

### kleb_iso_source — invasive disease (blood vs faeces first)

- [x] **Variant GWAS — blood vs faeces (DONE, 2026-06-17).** LMM (`--lmm`) is the method of
  record: **λ=0.562** (conservative; MDS fixed-effects gave λ=4.34, abandoned). **110 hits**,
  of which 18 blood/invasion — a cross-lineage **capsule (wzi) + fimbrial usher + iron/Fe-S**
  signature. This is the **chromosomal / core-allele axis** (variant calls are reference-anchored
  core, blind to accessory gain/loss). Figures + hit tables in
  [src/bac_pyseer/docs/visualise/lmm_model/](src/bac_pyseer/docs/visualise/lmm_model/);
  detail in the task CLAUDE.md Status.
- [x] **Faeces vs respiratory contrast (DONE, 2026-06-18).** Same pipeline/LMM, n=9,169,
  **λ=0.498**, 88 hits. Replication test (both contrasts share faeces as control, so
  faeces-direction recurrence is trivial): **only capsule *wzi* and the fimbrial usher recur
  in the *invasion* direction** (β>0 in both blood- and respiratory-vs-faeces, same SNP,
  cross-lineage) — a reproducible adhesion+capsule signature. **The iron theme does NOT
  replicate** (blood-specific/weak). Detail:
  [docs/visualise/faeces_resp_lmm_model/](src/bac_pyseer/docs/visualise/faeces_resp_lmm_model/).
- [ ] **Faeces vs liver/abscess contrast** — third niche; **blocked** on recurating the mixed
  liver/abscess `isolation_source_category` in BacHGT `metadata_curation.py` first. Then the
  same parameterised pipeline; check whether capsule/fimbrial/btuB recur again.
- **Collaborator summary:** [src/bac_pyseer/docs/invasion_gwas_collaborator_summary.md](src/bac_pyseer/docs/invasion_gwas_collaborator_summary.md)
  (both contrasts; reproducible adhesion+capsule+BtuB signature; coherent blood-specific iron-acquisition).
- [ ] **Hotspot rates by isolation source.** Compare per-source hotspot rates against
  the whole-population background mutation rate at each locus as control → Chi-sq for
  hotspots strongly associated with invasive disease. *Blocked on Aaron uploading
  hotspots to HPC.*
- [ ] **Pyseer unitig GWAS (KPSC-wide) — the accessory/HGT axis. ← NEXT priority.** Unitigs capture core SNPs
  *and* accessory sequence (the variant GWAS sees core only), so a unitig GWAS is both the
  whole-of-KPSC scan and the **acquired-vs-chromosomal test**: if the invasion signature recurs
  in an accessory-inclusive feature space, that's strong corroboration. From variant calls,
  tabulate mutation loci vs the reference per sample; filter low-frequency loci; pairwise
  Jaccard distances; combine with unitigs.
- [ ] **Pyseer presence/absence GWAS.** Same variant calls + the per-SL Panaroo we
  have → presence/absence GWAS.

## Deferred — Tasks 4 & 5

Paused until Tasks 1–3 are running. No task folders yet — recreate as
`src/admixture/` (Task 4) and `src/dp_short_read/` (Task 5) when work resumes.

**Task 4 — mixed / contaminated assemblies.** Use Bacformer's masked-gene loss to
detect admixtures of close-relative strains that differ in accessory genome (HGT,
IS, plasmid) — invisible to core-gene tools like CheckM2. Confirm masked (not
next-gene) objective with Maciej before committing.

- [ ] Build a fragmentation null model (loss as a function of N50 / contig count) — guards against locus-level loss just tracking contig breaks
- [ ] Whole-genome and locus-resolved masked-gene loss across SR assemblies
- [ ] If signal: map high-loss loci to independently-quantified HGT regions
- [ ] If signal: synthetic admixture experiment (mix reads in known ratios, re-assemble, does loss track the ratio?)

**Task 5 — DefensePredictor on short reads (DP-SR).** Test whether DP predicts
defence proteins better when *trained* on short-read assemblies rather than
trained on complete genomes and applied to SR with contig-border hacks —
defence systems sit at MGE boundaries, exactly where contig breaks occur.

- [ ] Translate CG defence-protein labels onto matched SR assemblies (minimap2)
- [ ] Baseline: DP-CG applied to SR — quantify the shortfall
- [ ] Retrain DP-SR from scratch on SR assemblies (same architecture)
- [ ] Add distance-to-contig-break as an input feature → DP-SR+break
- [ ] Compare DP-CG vs DP-SR vs DP-SR+break on a held-out SR test set

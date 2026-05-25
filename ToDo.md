# BacPredict — consolidated task tracker

Mirrors §7 of [BacPredict_Training_Plan.md](BacPredict_Training_Plan.md). Tick boxes as work completes. Per-task progress lives in each task folder's `CLAUDE.md` "Running notes" section.

## Shared infrastructure

- [ ] Refresh Bacformer complete-genomes weights from Hugging Face (blocks every task)
- [ ] Confirm SLURM scripts are current; standardise the 36 h GPU template
- [ ] Standardise the n=10 / CPU-only smoke-test wrapper so every subproject can call it
- [ ] Standardise the results JSON schema (AUROC, AUPRC, sens, spec, balanced acc, calibration, confusion matrix)

## Task 1 — AST in TB ([src/tb_ast/](src/tb_ast/))

- [ ] Kick off ESM-C embeddings on full TB protein set
- [ ] Stage A smoke test on rifampicin (local, n=10)
- [ ] Stage B overfit check on rifampicin (n=10)
- [ ] Stage C full run on rifampicin (HPC GPU, 36 h, 1 fold × 1 seed)
- [ ] Stage C on pyrazinamide (flagship goldilocks-zone drug)
- [ ] Stage C on ethambutol, moxifloxacin, levofloxacin
- [ ] Compare results vs WHO V2 catalogue and CRyPTIC ML benchmarks
- [ ] **HGT-vs-vertical stratified performance** — annotate mechanism via WHO V2; report per-stratum AUROC/sens/spec and the delta (central hypothesis test)
- [ ] Decision point: which drugs justify folds × seeds for publication

## Task 2 — AST in Klebsiella ([src/kleb_ast/](src/kleb_ast/))

- [ ] Evaluate existing Kp models against the refreshed model — save as benchmark
- [ ] Stage A/B/C on canonical drug from refreshed Bacformer
- [ ] Fan out across Kp drug panel
- [ ] **HGT-vs-vertical stratified performance** — annotate mechanism via AMRFinderPlus + Kleborate; report per-stratum AUROC/sens/spec and the delta vs catalogue (central hypothesis test; Kp is the strong test)
- [ ] One-paragraph MAG-vs-complete-genome model contrast
- [ ] Downstream (parked): held-out lineages; drug-class embeddings; Captum explainability; cross-training; Kp pre-training; read-depth gene-copy correction

## Task 3 — Isolation source in Klebsiella ([src/kleb_iso_source/](src/kleb_iso_source/))

- [ ] Regenerate train/val/eval splits for blood vs stool
- [ ] Stage A/B/C on blood-vs-stool from refreshed Bacformer complete-genomes model
- [ ] Compare against prior 0.55–0.62 AUROC benchmark
- [ ] Downstream (parked): Kp pre-training first; complete-genomes-only training; matched-pair SR-vs-CG contrast; Captum gene attribution; stepwise AUROC gain across modelling layers (ESM-C → frozen Bacformer → fine-tuned)

> **Tasks 4 + 5 deferred.** Mixed-assembly detection and DefensePredictor on short reads are paused until Tasks 1–3 are running. Full plans live in [BacPredict_Training_Plan.md](BacPredict_Training_Plan.md) §4 and §5. Recreate `src/admixture/` and `src/dp_short_read/` as task folders when work resumes.

## Task 6 — `predictHGT` embedding diagnostic (can run in parallel) ([src/predict_hgt/](src/predict_hgt/))

- [ ] Pull HGT-region annotations from the `BacHGT` sister module (MOB-suite + ISEScan + other annotation work — already done there, just consume the outputs)
- [ ] Embed all proteins with refreshed Bacformer
- [ ] Marker-protein nearest-neighbour analysis (KPC/NDM/OXA-48/*mcr-1*/*tetA*/*iutA*/*rmpA* + housekeeping controls)
- [ ] UMAP visualisation coloured by HGT vs chromosomal and by host species
- [ ] Centroid separation score: HGT-vs-chromosomal vs host-context baseline
- [ ] Layer-sensitivity scan (early vs late Bacformer layers)
- [ ] **Optional comparator:** raw ESM-C diagnostic — does contextualisation erase HGT identity?
- [ ] **Decision point:** Aim 1 outcome determines embedding source for Aim 2 — Bacformer if HGT-preserving, DefensePredictor-style (ESM-C + concat flanking + gene features) if context-attractor. Either way, document the implication that Bacformer is/isn't the right backbone for cross-species HGT-aware work.
- [ ] Boundary-detection head (Aim 2): pull ISEScan + MGEfinder ground truth from BacHGT; train a per-protein head to predict "within ±k of an HGT boundary"; evaluate held-out and on SR assemblies. Input-representation options: positional MGEfinder insertion vs IS-family gap tokens.

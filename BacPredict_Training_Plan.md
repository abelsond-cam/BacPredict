# BacPredict — Training Plan

**Owner:** David Abelson (Floto Lab, University of Cambridge) **Module:** `BacPredict` — fine-tuning Bacformer for downstream prediction tasks

---

## Scope & purpose

The code has been reorganised so we can run several prediction experiments in parallel, each as a self-contained subproject under `BacPredict/`. This document is the master plan. It will be split into:

- A root **`CLAUDE.md`** — orientation, global conventions, links to subprojects.  
- A root **`ToDo.md`** — task tracker (see §7 at the bottom).  
- Per-task **`CLAUDE.md`** files inside each `src/<task>/` subfolder — detailed workflow plan \+ running notes for that experiment.

A separate Claude Code agent will run each task independently.

---

## 0\. Global conventions (apply to every task)

### 0.1 Base model

- All experiments start from the **Bacformer complete-genomes model** (not the MAG-trained model).  
- **Refresh the Bacformer weights from Hugging Face first.** The previous weights we have locally had defects that the authors have since fixed. Until this is done, no benchmark we publish is comparable to current state-of-the-art.  
- Our previous runs were from the older MAG-trained model — every one of those benchmarks needs re-running once the refreshed complete-genomes model is in place.  
- Where comparison with the MAG-trained model is informative (we expect very little difference), we keep that as a small contrast experiment, not the default.

### 0.2 Three-stage testing protocol — apply to every task

Every experiment goes through these three stages in order. **Do not skip ahead.**

| Stage | Purpose | Scale | Folds × seeds | Where it runs |
| :---- | :---- | :---- | :---- | :---- |
| **A. Smoke test** | Confirm pipeline runs end-to-end, parameters are wired correctly | `n = 10` | 1 × 1 | MacBook M1 (CPU only — code must run with CUDA disabled). If too painful locally, run on HPC login node via SSH to avoid pulling data down. |
| **B. Overfit check** | Confirm the model can drive loss to \~0 on a tiny set — pipeline is learning | `n = 10`, train \= test | 1 × 1 | Local or HPC interactive node |
| **C. Full run** | Headline result | full dataset, one canonical drug/task first | 1 × 1 | GPU on HPC via existing SLURM scripts; \~36 h budget; early-stopping equivalent to \~15 epochs (back-calculated from total steps). |

Folds and seeds (≥5 of each) are an **advanced final step**, only when we are presenting findings externally. Do not burn compute on them during exploration.

### 0.3 Paths

- HPC root: `/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/`  
- Raw data: `project_k/david/raw/<task>/`  
- Processed data: `project_k/david/processed/<task>/`  
- Local dev: `/Users/davidabelson/developer/BacPredict/`

### 0.4 What we report at each milestone

For every full run: AUROC, AUPRC, sensitivity, specificity, balanced accuracy, confusion matrix, calibration curve, and per-drug / per-class breakdown. Save model checkpoint \+ a versioned results JSON so we can diff between runs.

**For AMR tasks specifically (Tasks 1 and 2), every report must additionally be stratified by resistance mechanism — HGT/acquired vs chromosomal point mutation.** This is the central hypothesis the whole programme is testing (see Task 1 §1 and Task 2 §2). Mechanism labels come from the WHO V2 catalogue in TB, and from AMRFinderPlus \+ Kleborate in Kp.

---

## 1\. Task 1 — AST in *M. tuberculosis*

### Aim

Predict antibiotic susceptibility in TB from genome embeddings, starting with ESM-C protein embeddings then fine-tuning from the refreshed Bacformer complete-genomes model. Establish whether Bacformer-based prediction can improve on the WHO 2nd-edition catalogue for drugs in the "goldilocks zone."

### Status

- AST labels downloaded from EBI for the full TB set.  
- Assemblies \+ GFFs already in `project_k/david/raw/tb/`.  
- Protein lists already extracted into `project_k/david/processed/tb/`.  
- Per-antibiotic resistance stats: `project_k/david/processed/tb/antibiotic_testing_stats.csv`.  
- **Not yet done:** ESM-C embeddings; Bacformer fine-tuning; refreshed-model run.

### The "goldilocks zone" — which drugs are worth training on

This was the upshot of our earlier deep-research conversation on TB AST prediction. The drugs cluster into three tiers; we should concentrate effort on the middle one.

**Tier 1 — saturated by catalogue (low ML headroom, but useful as positive-control sanity checks):**

| Drug | Best catalogue Sens / Spec | Comment |
| :---- | :---- | :---- |
| Rifampicin (RIF) | 93–97 % / 98.5–99 % | AUROC ≥ 0.97 already. Use as **canonical first test** — we expect to do well; failure here means the pipeline is wrong. Resistance is chromosomal (*rpoB*), not HGT — so this is also the harder of the two test cases for a transformer that we hypothesise should excel at HGT-borne resistance. |
| Isoniazid (INH) | 84.8–97 % / \~99 % | AUROC \~0.96. Same logic as RIF. |
| Amikacin / kanamycin / capreomycin | ≥85 % / ≥98 % | *rrs* 1401/1402/1484 essentially diagnostic. Skip for ML. |
| Streptomycin | 86–90 % / ≥97 % | *rpsL* dominates; small *gid* tail. Borderline. |

**Tier 2 — goldilocks zone (target these for the headline ML runs):**

| Drug | Catalogue Sens | ML AUROC reported | n resistant (CRyPTIC \+ EBI scale) | Why it's a good target |
| :---- | :---- | :---- | :---- | :---- |
| **Pyrazinamide (PZA)** | 26–66 % | \~0.90–0.93 (structure-aware ML) | \~2,500 | **Flagship.** \~600 distinct *pncA* LOF alleles, mostly singletons — exactly where a representation-learning approach beats lookup tables. |
| **Ethambutol (EMB)** | 80–94 % / 91–94 % | 0.88–0.92 | \~3,000 | *embB* M306V/I are MIC-shifters straddling the ECOFF. Quantitative ML \+ structural features add \~3–5 AUROC points. |
| **Moxifloxacin (MXF)** | \~70 % / \~92 % | \~0.90 | \~3,200 | *gyrA* 90/91/94 with very different MIC effects relative to the clinical breakpoint. |
| **Levofloxacin (LFX)** | \~73 % / \~94 % | \~0.91 | \~3,000 | Same story as MXF. |
| **Ethionamide** | moderate | – | \~ | Heterogeneous *ethA*/*ethR*/*inhA* — possible ML target. |

**Tier 3 — data-limited (parked until we have a better cohort):**

| Drug | EBI resistant n | Comment |
| :---- | :---- | :---- |
| Bedaquiline | \~150 | Below the \~1,000 threshold we need for ML to find complex patterns. Parked. Pooling external public cohorts could plausibly reach \~600–800 (4–5× expansion) but no single 1,000+ NEJM-style dataset exists. |
| Linezolid | \~200 | Same. Parked. |
| Clofazimine | low | Same. Parked. |
| Delamanid / pretomanid | low | Same. Parked. |

**The central hypothesis we're testing.** Bacformer should excel where resistance is driven by **HGT / gene acquisition** (a whole new gene appears in the genome — a context-token signal the transformer is built to see), and add much less where resistance is driven by **chromosomal point mutations in conserved core genes** (which the per-protein ESM-C embedding has to carry essentially alone — the genome context is unchanged). Demonstrating *this differential* is more interesting than any single headline AUROC. Every AMR result we report must therefore be **stratified by resistance mechanism**.

TB is unfortunately a near-worst case for that hypothesis — almost all TB resistance is chromosomal point mutation. So TB is a *conservative* test: if Bacformer adds even modest value here in the rare HGT-borne cases (e.g. acquired *eis* for kanamycin, some *rrs* alleles for streptomycin), that is informative, and we expect the gain to be much larger in Klebsiella (Task 2\) where HGT dominates.

### Workflow milestones (in order)

1. **Refresh Bacformer weights from Hugging Face.** (Shared infrastructure — do once, used by every task.)  
2. **Kick off ESM-C embeddings for the full TB protein set.** This is slow; start it immediately and let it run while we set up the rest.  
3. **Smoke test (Stage A) on rifampicin**, n \= 10, CPU local. Confirms pipeline.  
4. **Overfit check (Stage B) on rifampicin**, n \= 10\. Confirms model is learning.  
5. **Full run (Stage C) on rifampicin**, full data, single fold/seed, 36 h GPU SLURM job. Report AUROC etc. — should match or beat the catalogue benchmark above.  
6. **Fan out to all goldilocks-zone drugs with \> 1,000 resistant cases.** Single fold/seed each. PZA is the flagship; EMB/MXF/LFX next.  
7. **Compare against the WHO 2nd-edition catalogue and CRyPTIC ML benchmark** for each drug.  
8. **HGT-vs-vertical stratified performance — central hypothesis test.** For every resistant isolate, classify the resistance mechanism using the **WHO V2 catalogue** (point mutation in a core gene vs acquired allele / gene gain). Then re-compute AUROC, sensitivity and specificity **separately** for the two strata. Report the delta. We expect: small or no Bacformer gain over catalogue on the point-mutation stratum; larger gain on the HGT stratum, even though that stratum is small in TB. Isolates with mixed mechanisms get their own bucket. Where possible, hold out HGT-borne resistance from the training set and test transfer — this is the cleanest evidence for the hypothesis.  
9. **Decide whether to invest in folds × seeds** for any drug for publication.

### Open questions / parked

- The EBI AST labels have known inconsistencies (different DSTs, different breakpoints). Refining against curated datasets (CRyPTIC, WHO V2) is a **downstream** step, only after we have shown the approach is worth pursuing in TB.

---

## 2\. Task 2 — AST in *Klebsiella pneumoniae*

### Aim

Predict susceptibility for clinically relevant antibiotics in Kp, where we expect Bacformer to do best — resistance is heavily HGT-driven (carbapenemases, ESBLs, *mcr*) on plasmids and ICEs. This is the natural home for our AUROC 0.99 result, **and the strong test of the central HGT-vs-vertical hypothesis** (see Task 1 §1): unlike TB, Kp has both classes of mechanism well-represented in the same dataset, so a clean stratified comparison is possible.

### Status

- We already have trained Kp prediction models but have not formally **evaluated** them.  
- All previous training used the **older Bacformer weights** (now superseded) and the MAG-trained model.  
- Source notebook: `/Users/davidabelson/developer/BacHGT/docs/notebooks/amr_ebi_records.ipynb`.

### Three sub-steps (do in order)

1. **Evaluate current Kp models and save the results as the benchmark we are trying to beat.** No retraining yet — just produce the report.  
2. **Retrain from the refreshed Bacformer complete-genomes weights.** This is the main retraining. Compare against (1).  
3. **Retrain a small subset of drugs from the MAG-trained model.** Hypothesis: minimal difference. We just want a one-paragraph result confirming this for the paper, not a full benchmark.

### Workflow milestones

1. Run Stage A smoke test on the existing Kp pipeline against the refreshed model (one canonical drug — meropenem or ceftriaxone).  
2. Stage B overfit check.  
3. Stage C full run on canonical drug.  
4. Fan out across the drug panel.  
5. Side-by-side: old-Bacformer benchmark vs new-Bacformer vs MAG-model.  
6. **HGT-vs-vertical stratified performance — central hypothesis test.** For each isolate, run **AMRFinderPlus** and **Kleborate** to label every resistance determinant by its origin: acquired gene (HGT — carbapenemases like KPC/NDM/OXA-48, ESBLs, aminoglycoside-modifying enzymes, *mcr*, etc., typically plasmid- or ICE-borne) vs chromosomal point mutation (e.g. fluoroquinolone resistance via *gyrA*/*parC* QRDR, ramR/ramA-driven efflux changes, *ompK35*/*K36* porin loss, *pmrAB*/*phoPQ* mediated colistin resistance). For each drug: stratify and report AUROC, sensitivity, specificity **separately** for HGT-resistant vs vertically-resistant isolates. Mixed-mechanism isolates get their own bucket. The headline figure for the paper is the **delta** in Bacformer's gain over baseline between the two strata. The strongest test is held-out-by-mechanism: train on one stratum, test transfer to the other.

### Downstream / parked experiments (all on hold)

- **(i) Held-out lineages and held-out subspecies.** Test generalisation across ST/CG boundaries. The right way to ask "does the model learn AMR biology or lineage shortcuts?"  
- **(ii) Drug-class embedding.** Train a single model across all carbapenems (or aminoglycosides) with the drug as an input embedding, leveraging shared resistance mechanisms. More data per model, biologically natural.  
- **(iii) Explainability — find the underlying resistance genes.** Captum integrated gradients \+ feature ablation, ranking genes by contribution to each resistance call. Output: a gene-level attribution table per drug.  
- **(iv) Cross-trained model attribution** for explainability robustness.  
- **(v) Pre-train on Kp complete-genome masked-gene prediction, then fine-tune to AMR.** Plays into Task 3\. Lets the model learn Klebsiella-specific genomic grammar before the AMR head. We will track loss during this phase.  
- **(vi) Read-depth-aware gene copy correction.** Short-read assemblies collapse IS elements and repeat genes; we underestimate AMR gene dose. Plan: for each AMR gene, if read depth is \~2× the genome average, duplicate the protein string in the Bacformer input. **Advanced** — requires modifying how we construct the Bacformer input sequence, so genuinely downstream.

---

## 3\. Task 3 — Isolation source in *Klebsiella*

### Aim

Test whether Bacformer fine-tuned via BacPredict can pick up genomic markers that distinguish invasive from non-invasive infection. The canonical comparison is **blood vs stool** (the field has converged on this — it's the cleanest two-source comparison: largest n, least ambiguous microbiology, strongest biological interpretation). Liver abscess vs stool is an even stronger hvKp signal where it can be text-mined from metadata, even at modest n.

### Status

- We have trained models on isolation source already, achieving **AUROC 0.55–0.62 (poor)**.  
- Those models were trained on the **old Bacformer weights and from the MAG-trained model** — both of which we expect to be replaced.  
- The labels are noisy: \~50,000 human samples with isolation source, of which \~13k urine, \~13k blood, \~10k stool, \~7k respiratory, \~3k wound/abscess (a messy mixed category that includes liver abscess, ascites cultures, surgical drains, etc.).  
- Urine is likely a bad target (many samples are superficial UTIs; a minority are invasive) — leave for a downstream stratified analysis.

### Design decisions (deliberate simplifications for the first pass)

- **Use all samples from all countries.** No stratified sampling, no max 2:1 country ratio. We accept the bias and see how high we can push the model. Country-stratified sampling is a downstream experiment.  
- **Start with blood vs stool only.** Get a single clean number first.  
- Regenerate train/val/eval splits from scratch and re-label embeddings against the refreshed Bacformer model.

### Workflow milestones

1. Regenerate train/val/eval splits for blood vs stool.  
2. Smoke test (Stage A) on blood-vs-stool, n \= 10\.  
3. Overfit check (Stage B).  
4. Full run (Stage C) — single fold/seed, full data, 36 h GPU.  
5. Evaluate against the prior 0.55–0.62 AUROC benchmark. If meaningfully better, we have a result worth pursuing.

### Downstream / parked experiments

- **(i) Klebsiella-specific pre-training before isolation source.** Same idea as Task 2 downstream (v): masked next-gene prediction on Kp complete genomes, then fine-tune to isolation source. Hypothesis: this makes the model meaningfully more aware of Kp accessory architecture and phylogeny.  
- **(ii) Complete-genomes-only fine-tuning subset.** Train/val/eval drawn exclusively from complete genomes. Smaller dataset → expected to be worse all else equal (*ceteris paribus*), but possibly better because the inputs include the plasmid- and ICE-borne virulence factors that are exactly what we expect to drive invasion.  
- **(iii) Matched-pair short-read vs complete-genome comparison.** Are there isolates with both a complete genome and a short-read assembly from the same sample? If so, this is the only fully-controlled way to isolate the "assembly quality" effect from everything else.  
- **(iv) Gene-level explainability** (Captum integrated gradients \+ ablation). Expected hits: *rmpA/rmpA2*, *iuc*, *iro*, K1/K2 capsule loci, *ybt*/*clb* on the ICEKp.  
- **(v) Stepwise improvement in AUROC across modelling layers.** Predictions from raw ESM-C embeddings → frozen Bacformer embeddings → fine-tuned Bacformer. Quantifies the gain at each step — what does contextualisation buy, and what does fine-tuning add on top? Low priority.

### Caveat for framing

What we are predicting here is *not* "virulence" in the LD50 sense — it is "probability of isolation from a sterile site given gut carriage", i.e. invasion probability. That is a defensible and clinically relevant endpoint, but reviewers will (correctly) push back on calling source-of-isolation a virulence phenotype without acknowledging the indirection. Frame accordingly.

---

## 4\. Task 4 — Predicting mixed / contaminated assemblies

**Status: delayed until Tasks 1–3 are running.**

### Aim

Use Bacformer's masked-gene-prediction loss to detect and quantify admixtures of multiple closely-related strains with different accessory genomes in a single assembly.

### Background

Standard contamination tools (CheckM2, variant-calling-based pipelines) detect mixtures by looking for divergence in **core** genes against a reference. They are blind to admixtures of close-relative strains or clonal descendents that differ only in **accessory** content — HGT, ISE, plasmid load. These accessory differences are exactly what drives AMR and virulence switches in clinical contexts, and rare-gene sweeps from population admixture are a major route to AMR emergence. A tool that quantifies this would be very useful clinically.

### Experiments

**(i) Whole-genome and locus-resolved masked-gene loss across short-read assemblies.** Hypothesis: mixed assemblies show elevated masked-gene-prediction loss, both globally and at the loci where the accessory genome branches. Risk: locus-level loss may just track contig fragmentation (more contigs ⇒ more boundary artefacts ⇒ more loss), in which case the signal is uninformative. We need a fragmentation null model up front.

**(ii) If (i) is promising:** map the high-loss loci to independently-quantified HGT regions in Klebsiella. Concordance is the biological validation.

**(iii) If (ii) is promising:** synthetic admixture experiment.

- Take pairs of complete (hybrid-assembled) Kp genomes — the completeness is our guarantee of low contamination. Audit CheckM contamination scores on these to confirm even very low contig counts can carry true admixture signal that CheckM misses.  
- Take the original short reads from two strains with maximally different accessory genomes (quantified via Panaroo GPA).  
- Mix the reads in known ratios, re-assemble.  
- Measure Bacformer loss on the originals and the synthetic mixtures. Does loss track the mixing ratio? Does it localise to the accessory loci that differ?

### Methodological note

Use **masked-gene prediction** loss, not next-gene prediction — masked is the right objective for "is this gene out of place given its context?" Confirm this with the Bacformer authors (Maciej) before committing.

---

## 5\. Task 5 — Retrain DefensePredictor on short-read assemblies (DP-SR)

**Status: delayed until Tasks 1–4. New project directory, but listing here because it's the same ML workflow.**

### Aim

Test whether DefensePredictor (DP) would predict defence proteins more accurately on short-read assemblies if it were trained on short-read assemblies, rather than being trained on complete genomes and applied to short reads with hacks (zeroing genes at contig borders). The hypothesis is that contig breaks carry information the model can learn from — exactly because defence systems disproportionately sit at mobile-element boundaries, which is exactly where contig breaks occur. The current approach is therefore likely underperforming most where it matters most.

### Methods

1. **Generate the truth labels for short-read assemblies.** Take the complete genomes used to train the original DP. Translate their annotations onto the matched short-read assemblies — minimap2 is the easier option vs. Panaroo. Now we have per-gene defence-protein truth labels on the short-read assemblies.  
2. **Baseline: DP-CG applied to SR.** Run the existing complete-genome-trained DP on the short-read assemblies. Quantify the shortfall in AUROC and accuracy vs. the complete-genome test set. This is the gap we're trying to close.  
3. **Retrain DP from scratch on the short-read assemblies (DP-SR).** Same model architecture (target-centred \+ flanking-gene concatenated embeddings \+ gene-level features: distance, GC content). Just trained on SR rather than CG. Test on a held-out SR test set.  
4. **Compare DP-SR vs DP-CG on the SR test set.** If DP-SR wins, the hypothesis holds.  
5. **Add "distance to contig break" as an explicit input feature** and see whether it improves DP-SR further. This makes the contig-break signal explicit rather than implicit.

### Output

A directly comparable benchmark of DP-CG vs DP-SR vs DP-SR-with-break-distance on short-read assemblies. Result is the publication.

---

## 6\. Task 6 — `predictHGT`: do Bacformer embeddings preserve HGT identity?

**Status: diagnostic / exploratory. Can run in parallel with other tasks — only needs the refreshed Bacformer weights plus an HGT-annotation pipeline. No new training required for the main experiment.**

### Aim

Two linked questions, in priority order:

1. **(Main thrust — new framing.)** When a protein lives on a plasmid, ICE, or other HGT element, **what does its Bacformer embedding actually represent?** Two hypotheses:  
     
   - **HGT-preserving.** Bacformer recognises HGT-borne proteins as a distinct semantic class — their embeddings cluster with the same/similar proteins on HGT elements in other genomes, regardless of host. If true, Bacformer is the right tool for cross-species HGT-aware prediction.  
   - **Context-attractor.** The transformer's attention pulls each protein's embedding toward its local genomic neighbours, so a KPC on a plasmid in one isolate ends up looking like its chromosomal neighbours in that isolate, rather than like KPCs in other isolates. If true, Bacformer effectively *erases* HGT origin in favour of host context.

   

   Context-attractor is the more likely outcome given the architecture (the whole point of the model is to contextualise per-protein embeddings against their genomic neighbourhood). If it's true, **Bacformer is not the right LLM for HGT-specific work**, and any cross-species HGT-aware prediction should use DefensePredictor-style embeddings instead (raw ESM-C \+ concatenated flanking-gene embeddings \+ explicit gene-level features like distance and GC content). This conclusion would matter for how we frame all the AMR results — particularly the HGT-stratum performance in Tasks 1 and 2\.

   

2. **(Original framing — kept as a sub-aim.)** Train a head that predicts the **boundaries of HGT insertions** from per-protein embeddings (full plan in the "Mags and contigs with MGE" chat — IS-flanked context windows, MGEfinder integration, IS-family gap tokens). Brief summary below. This task **survives either outcome of Aim 1** — what changes is the embedding source:  
     
   - If **HGT-preserving** → use Bacformer embeddings as originally planned.  
   - If **context-attractor** → switch to DefensePredictor-style embeddings (raw ESM-C \+ concatenated flanking-gene embeddings \+ explicit gene-level features). The boundary task itself is no less interesting — we just route the signal through a representation that hasn't homogenised it away.

### Why this matters for the rest of the programme

- **AMR stratification (Tasks 1 & 2).** If Bacformer is a context-attractor, the gain we see on the HGT stratum will come from *genomic context that co-travels with the acquired gene* (replicon backbone, ICE scaffold, IS flanks), not from recognition of the resistance gene itself. Either is a valid signal, but they are *different findings*, and reviewers will ask which is which.  
- **Cross-species generalisation.** A context-attractor model will generalise poorly across species for HGT-borne phenotypes, because the host context changes. An HGT-preserving model would generalise well. This determines whether Bacformer is the right backbone for any future cross-genus work.  
- **Isolation source (Task 3).** Hypervirulence determinants (*rmpA/rmpA2*, *iuc*, *iro*, hvKp virulence plasmid backbone, ICE*Kp*) are HGT-borne. Same question, same implications.

### Methods — main experiment (Aim 1\)

1. **Pull HGT-region annotations from the sister `BacHGT` module** — the work is already done there (MOB-suite for plasmid replicons \+ ISEScan for IS elements \+ other annotation steps). Do not re-run these tools in BacPredict; just consume the BacHGT outputs. Per-protein labels we want from BacHGT: `chromosomal-core` / `plasmid` / `IS-flanked` / `other-MGE`, with element-level identifiers (replicon type, IS family) where available.  
     
2. **Embed all proteins with refreshed Bacformer** complete-genomes model. Save per-protein contextual embeddings.  
     
3. **The core diagnostic — nearest-neighbour analysis by ortholog group.** For a set of marker proteins that occur both HGT-borne and chromosomally across the corpus (good candidates: *bla*KPC, *bla*NDM, *bla*OXA-48, *mcr-1*, *tetA*, *iutA* from *iuc*, *rmpA*; plus a few core single-copy housekeeping proteins as negative controls): pull every instance from the corpus, compute pairwise embedding distances, and ask — for an HGT-borne instance, are its nearest neighbours in embedding space (a) other HGT-borne instances of the same protein in other genomes (= HGT-preserving), or (b) its chromosomal neighbours in the same genome (= context-attractor)?  
     
4. **UMAP visualisation** of the same marker proteins, coloured by HGT vs chromosomal location and by host species. Visual confirmation of the quantitative result above.  
     
5. **Centroid distance test.** For each marker, compute the centroid of all HGT-borne instances vs the centroid of all chromosomal instances. Distance between centroids ÷ within-group spread gives a clean separation score. Compare to a context-baseline: same calculation done on the protein's chromosomal neighbours. If the HGT-vs-chromosomal separation is small but the host-context separation is large, that is direct evidence of context-attractor behaviour.  
     
6. **Sensitivity to embedding layer.** Repeat (3)–(5) at multiple Bacformer layers, not just the final layer. Earlier layers may retain HGT identity even if later layers homogenise it — would inform which layer to take embeddings from for HGT-aware downstream tasks.  
     
7. **Optional comparator: ESM-C alone.** Run the same diagnostics on raw ESM-C protein embeddings (no genomic context). If ESM-C preserves HGT identity better than Bacformer, that is direct evidence that the *contextualisation step* in Bacformer is what erases HGT origin — and supports the DefensePredictor-style architecture as the right tool for HGT work.

### Methods — boundary labelling (Aim 2, brief)

We had previously planned a head that flags insertion boundaries in genome embeddings (full plan in the prior chat). In short: take complete genomes with ISEScan \+ MGEfinder ground truth (pulled from `BacHGT`), train a small head on per-protein embeddings to predict "is this within ±k genes of an HGT insertion boundary?", evaluate on held-out genomes, then apply to short-read assemblies. Two implementation options remain on the table for the input representation: (a) MGEfinder-recovered IS proteins inserted into the protein list at the correct genomic position (positional anchoring via flanking genes); (b) IS-family gap tokens — one learned embedding per IS family, simpler and works when MGEfinder fails on composite transposons.

**Embedding source is chosen by the outcome of Aim 1:** Bacformer embeddings if HGT-preserving, DefensePredictor-style embeddings (raw ESM-C \+ concatenated flanking \+ gene-level features) if context-attractor. Same architecture and training set either way; only the input representation changes.

### What we will and will not do

- **Will do:** the embedding diagnostic (Aim 1, steps 1–6). It is fast — no training — and the answer materially shapes how we frame everything else.  
- **Will do:** the boundary-detection head (Aim 2), with the embedding source selected by the outcome of Aim 1 (Bacformer if HGT-preserving, DefensePredictor-style if context-attractor).  
- **May do, depending on (1):** the ESM-C comparator (step 7).  
- **Won't do (explicitly out of scope):** *fixing* Bacformer to preserve HGT identity better — that would require retraining with HGT-aware objectives (separate special tokens for plasmid/ICE, contrastive loss against HGT-vs-chromosomal pairs, etc.). Important to be aware of as a future direction, but a much bigger lift than this PhD.

---

## 7\. ToDo.md — consolidated task tracker

Copy this section to `BacPredict/ToDo.md`. Tick boxes as work completes.

### Shared infrastructure

- [ ] Refresh Bacformer complete-genomes weights from Hugging Face (blocks every task)  
- [ ] Confirm SLURM scripts are current; standardise the 36 h GPU template  
- [ ] Standardise the n=10 / CPU-only smoke-test wrapper so every subproject can call it  
- [ ] Standardise the results JSON schema (AUROC, AUPRC, sens, spec, balanced acc, calibration, confusion matrix)

### Task 1 — AST in TB

- [ ] Kick off ESM-C embeddings on full TB protein set  
- [ ] Stage A smoke test on rifampicin (local, n=10)  
- [ ] Stage B overfit check on rifampicin (n=10)  
- [ ] Stage C full run on rifampicin (HPC GPU, 36 h, 1 fold × 1 seed)  
- [ ] Stage C on pyrazinamide (flagship goldilocks-zone drug)  
- [ ] Stage C on ethambutol, moxifloxacin, levofloxacin  
- [ ] Compare results vs WHO V2 catalogue and CRyPTIC ML benchmarks  
- [ ] **HGT-vs-vertical stratified performance** — annotate mechanism via WHO V2; report per-stratum AUROC/sens/spec and the delta (central hypothesis test)  
- [ ] Decision point: which drugs justify folds × seeds for publication

### Task 2 — AST in Klebsiella

- [ ] Evaluate existing Kp models against the refreshed model — save as benchmark  
- [ ] Stage A/B/C on canonical drug from refreshed Bacformer  
- [ ] Fan out across Kp drug panel  
- [ ] **HGT-vs-vertical stratified performance** — annotate mechanism via AMRFinderPlus \+ Kleborate; report per-stratum AUROC/sens/spec and the delta vs catalogue (central hypothesis test; Kp is the strong test)  
- [ ] One-paragraph MAG-vs-complete-genome model contrast  
- [ ] Downstream (parked): held-out lineages; drug-class embeddings; Captum explainability; cross-training; Kp pre-training; read-depth gene-copy correction

### Task 3 — Isolation source in Klebsiella

- [ ] Regenerate train/val/eval splits for blood vs stool  
- [ ] Stage A/B/C on blood-vs-stool from refreshed Bacformer complete-genomes model  
- [ ] Compare against prior 0.55–0.62 AUROC benchmark  
- [ ] Downstream (parked): Kp pre-training first; complete-genomes-only training; matched-pair SR-vs-CG contrast; Captum gene attribution; stepwise AUROC gain across modelling layers (ESM-C → frozen Bacformer → fine-tuned)

### Task 4 — Mixed / contaminated assemblies (delayed)

- [ ] Build fragmentation null model (loss as a function of N50 / contig count)  
- [ ] Whole-genome and locus-resolved masked-gene loss across SR assemblies  
- [ ] If signal: map high-loss loci to independently-quantified HGT regions  
- [ ] If signal: synthetic admixture experiment with read-mixing and re-assembly  
- [ ] Confirm masked vs next-gene objective with Maciej

### Task 5 — DefensePredictor on short reads (delayed)

- [ ] Translate CG defence-protein labels onto matched SR assemblies (minimap2)  
- [ ] Baseline: DP-CG on SR — quantify the shortfall  
- [ ] Retrain DP-SR from scratch on SR assemblies (same architecture)  
- [ ] Add distance-to-contig-break as input feature → DP-SR+break  
- [ ] Compare DP-CG vs DP-SR vs DP-SR+break on held-out SR test

### Task 6 — `predictHGT` embedding diagnostic (can run in parallel)

- [ ] Pull HGT-region annotations from the `BacHGT` sister module (MOB-suite \+ ISEScan \+ other annotation work — already done there, just consume the outputs)  
- [ ] Embed all proteins with refreshed Bacformer  
- [ ] Marker-protein nearest-neighbour analysis (KPC/NDM/OXA-48/*mcr-1*/*tetA*/*iutA*/*rmpA* \+ housekeeping controls)  
- [ ] UMAP visualisation coloured by HGT vs chromosomal and by host species  
- [ ] Centroid separation score: HGT-vs-chromosomal vs host-context baseline  
- [ ] Layer-sensitivity scan (early vs late Bacformer layers)  
- [ ] **Optional comparator:** raw ESM-C diagnostic — does contextualisation erase HGT identity?  
- [ ] **Decision point:** Aim 1 outcome determines embedding source for Aim 2 — Bacformer if HGT-preserving, DefensePredictor-style (ESM-C \+ concat flanking \+ gene features) if context-attractor. Either way, document the implication that Bacformer is/isn't the right backbone for cross-species HGT-aware work.  
- [ ] Boundary-detection head (Aim 2): pull ISEScan \+ MGEfinder ground truth from BacHGT; train a per-protein head to predict "within ±k of an HGT boundary"; evaluate held-out and on SR assemblies. Input-representation options: positional MGEfinder insertion vs IS-family gap tokens.

---

## 8\. Repo restructure — to be done in liaison with Claude Code

The current layout `BacPredict/src/bacpredict/...` has a redundant middle layer. Flatten to:

BacPredict/

├── CLAUDE.md      \# global orientation \+ §0 conventions

├── ToDo.md        \# §7

└── src/

    ├── tb\_ast/              \# Task 1

    ├── kleb\_ast/            \# Task 2

    ├── kleb\_iso\_source/     \# Task 3

    ├── admixture/           \# Task 4 (delayed)

    ├── dp\_short\_read/       \# Task 5 (delayed)

    └── predict\_hgt/         \# Task 6 (diagnostic, can run in parallel)

Each subfolder gets its own `CLAUDE.md` containing the task's training and labelling plan (lifted from the corresponding §1–§6 section of this document) plus a running-notes block the agent updates as work proceeds.

**Order of operations:** restructure the repo and write the per-subfolder `CLAUDE.md` files *first* — before starting any of the task work itself.  

# *P. aeruginosa* AST (`apps/pa`)

The *P. aeruginosa* (PsA) adapter for the multi-species AMR work: PsA first, then all Enterobacterales,
then every genome in the EBI AMR Portal. **Status, numbers and next steps live only in
[`PROJECT_STATE.md`](../../../../PROJECT_STATE.md)**; this file holds the mechanics.

The pipeline itself is the engine. This folder holds only what is PsA-specific: the adapter for the
AMR-GNN supplement and the run scripts.

## Where the data lives

All the multi-species AMR data is under one root on project_k (David, 2026-11-20):
`$BACPREDICT_AMR_ROOT`, default `~/rds/rds-floto-bacterial-4k08a2yyQLw/david/amr_prediction`, with `raw/`,
`processed/` and `final/`. Every script here sets it. **Nothing goes in `~/rds/hpc-work`**: that is David's
scratch, for SLURM logs and temporary files only.

| Kind | Where |
|---|---|
| **Per-sample files**, keyed by BioSample. Flat, shared across species | `raw/contigs/`, `raw/prodigal/`, `raw/bakta/`, `processed/protein_sequences/`, `processed/esm/` (`processed/esm_smoke/` for the smoke test) |
| **Per-species tables and batch outputs**, in a species folder | `processed/paeruginosa/` (labels, the ESM-C sample lists `esm_samples{,_smoke}.txt`, then `genomes/`: manifest, shards, CheckM2 + skani QC, Bakta rosters, `bakta_annotation.csv`; later splits and models); `processed/enterobacterales/`; `raw/paeruginosa/` (the AMR-GNN supplement) |
| **Multi-species inputs and evidence**, flat | `raw/{ebi_amr_portal,bakrep,bakta_db,allthebacteria,reference_genomes}/`; `processed/{ebi_amr_portal,bakrep_coverage_20261119,bakta_vs_bakrep_validation}/` |
| Superseded, pending deletion | `processed/_superseded_prodigal/` (the Prodigal-based protein parquets and ESM-C smoke) |
| Curated tables | `final/` (nothing yet) |

CheckM2's database is the one already on project_k (`david/raw/CheckM2/`), shared with the Kp work.

## Labels: `binary_ast.csv`

Built by `bacpredict.engine.ast_labels.build_binary_ast_labels`, which is organism-agnostic (species or
genus filter, drug list and extra sources are arguments). Run it with
[`scripts/build_pa_ast_labels.sh`](scripts/build_pa_ast_labels.sh). It is light enough for a CSD3 login node.
Output goes to `processed/paeruginosa/`.

**Two label sources, one rule.**

- **The EBI AMR Portal release 2026-07** (`phenotype.parquet`), read directly by the engine.
- **The AMR-GNN supplement** (Nguyen et al. 2026, *Nat Commun*), converted by
  [`amr_gnn_tests.py`](amr_gnn_tests.py) into the engine's long test schema
  (`ast_tests.TEST_COLUMNS`), keyed on the BioSample.

Every test from either source is called the same way (`engine/ast_labels/ast_tests.py`):

- **With an MIC (mg/L):** the test is called against **EUCAST v15.0**, and is resistant if and only if
  the MIC is above the R breakpoint. EUCAST's "I" counts as S. A censored MIC that straddles the
  breakpoint is dropped. It is not rescued by the source's own call.
- **Without an MIC** (disk zones, bare calls): the test keeps the **source's own call**, with the
  canonical parser's semantics: `resistant` → 1, `susceptible` → 0, anything else → no call.
- **A measurement with no unit that is not a disk test:** in the portal these are MIC dilution steps whose
  unit was never recorded (CDC AR Bank, CABBAGE E-tests). They are flagged as `mic_units_missing` and
  counted. By default they are treated as having no MIC, so the source's call stands.
  `--unitless-mic-as-mg-l` calls them against EUCAST as mg/L instead. Which of the two to use is a
  decision for David.

Then, for each (BioSample, drug), the calls are **averaged**. A fractional value means the tests conflict.
It stays in `binary_ast.csv` and is never scored, because `load_splits` keeps only 0/1.

**The breakpoint table** is data: `engine/ast_labels/eucast_v15_breakpoints.csv`. It holds the ten drugs
for the Pseudomonas and Enterobacterales sheets, with the *Salmonella* ciprofloxacin (R > 0.06) and
Morganellaceae imipenem rows. A more specific taxon row always wins. The table reproduces every one of
AMR-GNN's own EUCAST v15 calls, as shown in `call_agreement.csv`.

**Drugs** (the same ten for PsA and Enterobacterales): ciprofloxacin, ceftazidime, meropenem, tobramycin,
amikacin, imipenem, piperacillin-tazobactam, aztreonam, levofloxacin, cefepime.
- Gentamicin has no EUCAST PsA breakpoint and is dropped.
- Colistin is dropped.
- Ceftazidime-avibactam and ceftolozane-tazobactam are deferred.

**Outputs besides `binary_ast.csv`:**

| File | What it holds |
|---|---|
| `label_provenance.csv` | How each label was reached |
| `ast_tests_called.parquet` | Every test with its MIC, breakpoint and call |
| `cohort.csv` | One row per BioSample: sources, portal assembly, reads accession (a run is preferred over a sample accession), the paper's ST, a species-conflict flag between the two portal tables, and `genome_route` |
| `label_counts.csv` | Per-drug counts |
| `call_agreement.csv` | MIC call vs the source's own call |
| `manifest.json` | Input md5s, breakpoint md5, git SHA |

`genome_route` in `cohort.csv` is `assembly`, `reads` or `none`.

## MIC rather than R/S

[`scripts/train_pa_mic_models.sh`](scripts/train_pa_mic_models.sh) runs the four-arm comparison that asks
whether training on the **MIC** beats training on the call derived from it. Its header is the full mechanics;
the status, the cohort sizes and the decisions of record live in
[`PROJECT_STATE.md`](../../../../PROJECT_STATE.md) §3.8 (*MIC rather than R/S*) and §6.

The pieces are engine, not app, because nothing in them is PsA-specific:

| Module | What it does |
|---|---|
| [`engine/ast_labels/mic_intervals`](../../engine/ast_labels/mic_intervals.py) | Each `(Sample, drug)` MIC as the interval its latent log₂ MIC lies in, in doubling dilutions from the EUCAST R breakpoint. Intersects a cell's tests (which is also what deduplicates the two ingests of one assay), snaps to the doubling ladder, and checks its own call against the split table without ever redefining a label |
| [`engine/finetune/censored_mic_loss`](../../engine/finetune/censored_mic_loss.py) | The interval-censored Gaussian likelihood, the validate σ re-fit, and the MIC read-outs (coverage, absolute and essential agreement, Brier/ECE) |
| [`engine/finetune/mic_trainer`](../../engine/finetune/mic_trainer.py) | The three objectives on one network, and σ's life across a checkpoint |
| [`collect_mic_arms.py`](collect_mic_arms.py) | The arms side by side — each on its own holdout **and** on the intersection every arm scored |

**Interactive explainer of the likelihood** — slide μ and σ over the dilution ladder and watch the mass that
falls in the observed window, for an exact cell, a censored one and a bare R/S call, with the loss and
`∂loss/∂μ` live: <https://claude.ai/artifact/9gTtQnq5WA5uNPiZquWs1m?sk=OYBiyQhZGtk734bvHgCZ6w>
(worked on meropenem, EUCAST R > 8 mg/L; its arithmetic is cross-checked against
`interval_censored_gaussian_nll` to < 2e-4). ⚠ The `?sk=` is a share token — anyone who can read this repo
can open the page.

Two things to know before reading any of it:

- **A reported MIC is never a point.** `≤ 2` is an interval, `> 8` is an interval, and even `= 4` is `(2, 4]`
  because the assay reads a doubling ladder. Imputing a point throws that away and attenuates the fitted MIC
  scale by about half — invisibly to AUROC, which is rank-only.
- **σ is fixed at 0.4 in training and re-fitted on *validate*, once, at the selected checkpoint.** The
  training value is the assay's own measured reproducibility (below); the re-fit is the model's actual
  residual spread. Neither can move AUROC, AUPRC or the R/S call — σ only rescales `P(R) = Φ(μ/σ)`, which is
  monotone in μ and crosses 0.5 at μ = 0 for every σ.

Nothing in the binary path changes: without `--mic-table`, `finetune_amr` behaves exactly as before.

### Why σ = 0.4, and the references behind it

σ is held at the PsA MIC assay's **analytical reproducibility** — what a repeat test of the same isolate
would disagree by — rather than at the model's residual spread. Essential agreement (EA) compares two
independent reads, so their difference carries √2 × the single-read SD; inverting the published PsA rates:

| source | reported EA (within ±1 doubling dilution) | implied single-read σ |
|---|--:|--:|
| Chen *et al.*, "Variability of beta-lactam broth microdilution for *P. aeruginosa*", *Antimicrob. Agents Chemother.* 2021, [doi:10.1128/aac.00640-21](https://journals.asm.org/doi/10.1128/aac.00640-21) | ≥ 90 % for every beta-lactam tested | 0.430 |
| Digital dispense-assisted BMD for *P. aeruginosa*, 2021, [PMC8080699](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8080699/) | 92.7 % | 0.394 |
| Two readout methods, commercial panel, CF isolates, *Infect. Dis.* 2018, [doi:10.1080/23744235.2018.1500705](https://www.tandfonline.com/doi/full/10.1080/23744235.2018.1500705) | 98 % at 24 h (colistin; exact agreement only 82 %) | 0.304 |
| CLSI M07-A9 — acceptable reproducibility is within one twofold dilution, and "even under the best of controlled conditions, a dilution test may not yield the same end point each time" | — (the standard, not a rate) | — |

**0.4 spans the two PsA-specific figures and round-trips onto them**: two reads at σ = 0.4 agree within one
dilution 92.3 % of the time, against the 90–92.7 % measured. `tests/engine/finetune/test_mic_trainer.py`
asserts that round trip, so a constant that stops matching the literature fails the suite rather than leaving
this table as decoration. (The first value tried, 0.5 — from reading ±1 dilution as a ±2σ span — round-trips
to 84.3 % EA, looser than the assay actually is.)

⚠ **Say "analytical reproducibility", not "biological uncertainty".** The EA literature measures inoculum,
operator and reading variability. Biological variation and everything the genome does not determine show up
in the *re-fitted* σ, which arm A measured at 1.74–2.49 dilutions on held-out data — four to six times the
assay scale. Holding training-σ at the assay scale is therefore a deliberate weighting choice, not the MLE.

### How the MIC arms will be assessed (David, 2026-11-27)

Three read-outs, **in this order**. (c) is explicitly downstream of (a) and (b) and is not run before them.

**(a) The R/S call, as AUROC — the primary question, and the one the experiment exists to answer.**
Translate the predicted MIC into a EUCAST R/S call and score it against the binary-trained arms. *Does
training on the MIC produce a better predictor of the call than training on the call?* In this
parameterisation it costs nothing to compute: μ is in dilutions **from the EUCAST R breakpoint**, so μ > 0
*is* the R call and `P(R) = Φ(μ/σ)`. `collect_mic_arms` already reports it, against arms 0, 0′ and A, on each
arm's own holdout and on the intersection every arm scored. The intersection column is the comparable one.

**(b) Accuracy on the MIC scale itself, against a point model.** How well the predicted MIC tracks the
observed one, compared with least squares on an imputed point (arm A) and with Maciek's frozen point model.
`mic_agreement` reports **`pearson_r` and `slope_log2`** beside `bias_log2` and `rmse_log2`, over
`censor == "exact"` cells only — a censored cell has no MIC to correlate against.

**Read the slope before the correlation.** An attenuated model keeps its correlation while losing the scale,
and that is arm A's measured failure mode: 0.584 of the latent scale on known-truth simulation with its AUROC
still within 0.005 of the censored fit's. `tests/engine/finetune/test_censored_mic_loss.py` pins the case
directly — a prediction shrunk to half scale scores `r = 1.000` and `slope = 0.500`. A correlation quoted
without a slope would call that model fine.

**The comparator is a re-run, not a quoted number** (David, 2026-11-27). Maciek's frozen point model is run
on *our* holdout rather than compared against his published figure, so cohort, split and metric are identical
and the estimator is the only thing that differs. ⚠ Still to build: it needs the frozen checkpoint, an
inference pass writing the same npz schema (`mu`, `lo`, `hi`, `sigma`, `y_true`, `score`), and its own row in
`collect_mic_arms`. Nothing blocks it — it can be built while arm B trains, and because the npz files carry
everything per cell, every statistic here is recomputable over saved artifacts after the fact.

**(d) The ST-holdout gap, and (e) within-ST discrimination — the test of the lineage hypothesis**
(David, 2026-11-28; [`lineage_stratified_metrics.py`](lineage_stratified_metrics.py)).

The four things the censored arm is claimed to buy are not equally visible. Counting cells
(`PROJECT_STATE.md` §3.8, *what the train split is actually made of*) settles the first two: arm B trains on
**+18.1 %** more cells than arms 0′/A because a bare R/S call is an open interval to a censored likelihood and
nothing to a point model, and **31.9 %** of arm A's own cells are bounds it must invent a point for. Only
**57.7 %** of cells are genuinely exact. The third — flexibility about a prediction — is shared with any
continuous loss, which is exactly why arm A exists to control for it.

The fourth is the one worth the experiment: **a binary R/S label is well carried by population structure.** In
PsA a few high-risk clones are predominantly resistant, so a model that learned only "which lineage is this"
scores a respectable holdout AUROC having learned nothing about resistance — the failure this project has
already measured twice (the TB mini-set at 0.977 → 0.574 once its clonal marker went, and the invasion GWAS
whose bias vanished under within-sublineage shuffling). A graded MIC target cannot be supplied by a lineage
label, so training on it should push the model onto the genome.

⚠ **That gain can be entirely real and still not appear in read-out (a).** The main holdout carries the same
lineage structure as training, so a lineage-shortcut model still scores well on it and the two arms can tie on
overall AUROC while being very different models. A flat (a) is therefore not evidence of no effect; it is
evidence that (a) cannot see this one. Two read-outs that can, both pure analysis over artifacts that already
exist:

- **(d) the ST-holdout gap** — `holdout AUROC − held-out-ST AUROC`, per arm. ST155 is held out of training
  entirely and every arm already writes `eval_scores_holdout_st.npz`. An arm leaning on lineage has nothing to
  lean on for an unseen clone, so it should fall further. **The gap is the statistic**, not either number: the
  two sets differ in size and composition, so comparing raw ST-holdout AUROCs across arms compares cohorts.
- **(e) within-ST discrimination** — `P(a resistant genome outranks a susceptible one drawn from the same
  ST)`, plus the MIC slope after centring μ and the observed midpoint inside each ST. Lineage is constant in a
  stratum by construction, so whatever still discriminates is genomic. `st_key` is already in the split tables
  (936 STs; ST235 n=671, ST111 n=313, ST621 n=264). Strata are pooled by discordant-pair count
  `n_pos × n_neg` — the concordance rate over all within-ST pairs — so a three-genome ST cannot outvote
  ST235; the unweighted mean over strata is reported beside it for comparison.

⚠ **The within-ST MIC slope has a noise floor.** Within-clone MIC variation is bounded below by the assay
itself (σ ≈ 0.4 dilutions, above). The measured residual is ~2 dilutions so there is plenty of room, but the
ceiling on a within-ST slope is not exactly 1.0 and it should not be read as though it were.

**(c) The same, allowing ±1 dilution — essential agreement. A downstream check only.** It is the field's
convention and `collect_mic_arms` already computes it, so it costs nothing, **but it is a fudge**: it scores
as correct a prediction a clinician would not accept, and it flatters every arm by construction — the ceiling
it measures against is the assay's own reproducibility, which is precisely what σ was set to. Report it
because the field expects it; never lead with it, and never let it stand in for (a) or (b).


## Genomes

Every genome comes from one of two places:
- the portal's own annotation file, `genomes/{ERZ,GCA}/…/<acc>_annotations.gff.gz`, with its `##FASTA`
  contigs split off; or
- assemblies, from AllTheBacteria where it holds the sample, otherwise shovill on the ENA reads.

Where each one is (checked by HTTP against the whole cohort, 2026-11-18):

- **Portal annotation files.** Path:
  `https://ftp.ebi.ac.uk/pub/databases/amr_portal/genomes/{GCA|ERZ}/<d1-3>/<d4-6>/<acc>/<acc>_annotations.gff.gz`.
  - The two 3-digit groups are the accession's first six digits, taken after dropping `GCA_`/`ERZ` and the
    version.
  - It is one shared `genomes/` tree, not per release.
  - The files exist only for the **genotype-table** accession. The 195 Khaledi et al. genomes that have
    only a phenotype-table GCA have none.
  - **The portal's ERZ assemblies are AllTheBacteria's shovill assemblies**: their contigs are
    md5-identical.
- **AllTheBacteria.** One file per sample:
  `https://allthebacteria-assemblies.s3.eu-west-2.amazonaws.com/<BioSample>.fa.gz`.
  - The bucket's weekly file list is the index.
  - The OSF `file_list.all.latest.tsv.gz` lacks the 2026-09 batch.
  - The OSF batch tarballs hold about 4,000 genomes each, so don't go through them.
- **Reads.** The ENA portal API `filereport`. `cohort.reads_accession` is a run where one is known, and
  otherwise a sample accession (SRS/ERS), which `filereport` expands to its runs.
  - Shovill takes paired Illumina WGS only, so RNA-seq-only and ONT-only BioSamples are dropped.
  - Reads under 10× depth are not assembled.

Gene calls are **Bakta 1.9.4 with the full DB v5.1** on every genome (see *Annotation* below); the build's own
Prodigal 2.6.3 run is kept for provenance but is not the annotation of record. STs come from our own `mlst` run,
never from the paper's `st_reported`, which is kept only for cross-checking.

**How the build runs.** The organism-agnostic stage is `engine/genomes/`; tool binaries come from its
pixi env (shovill 1.1.0 with SPAdes 3.15, Prodigal 2.6.3, mlst, skani, plus CheckM2 in its own env).
[`scripts/build_pa_genomes.sh`](scripts/build_pa_genomes.sh) runs it in stages from a login node:

| Stage | What it does |
|---|---|
| `setup` | Installs the pixi envs and fetches the species references: PAO1 `GCF_000006765.1` and *P. paraeruginosa* PA7 `GCF_000017205.1` |
| `plan` | Writes `genomes/genome_manifest.csv`, one source per labelled genome |
| `smoke` | Builds 2 genomes per source |
| `submit` | Submits the full arrays |
| `qc` | Runs CheckM2 and skani |
| `collate` | Writes `genomes/genome_qc.csv`, `genomes/embedding_input.csv` and `genomes/st_counts_by_drug.csv` |

**Source priority** (even surface):

1. AllTheBacteria.
2. The portal's own assembly. These are submitter assemblies: `--prefer-reads` swaps them for shovill
   wherever usable reads exist.
3. Shovill on paired Illumina WGS/WGA reads at ≥10× depth. These use AllTheBacteria's shovill 1.1.0 defaults
   with two exceptions, both needed for the defaults to work at all (see `build_genomes.py`):
   - `--gsize 6.6e6`. Shovill's own size estimate collapses on low-depth reads and then discards almost all
     of them.
   - `OMP_NUM_THREADS` set to the core count. CSD3 exports 1, and SPAdes obeys it.
4. Drop, with a reason.

**Where the outputs go:**
- Contigs are renamed `<Sample>.contig00001` and stored at `raw/contigs/<Sample>.fna.gz`.
- Prodigal GFFs (not the annotation of record) are stored at `raw/prodigal/<Sample>.gff.gz`.
- The store is keyed by BioSample and shared across organisms.
- Each array task writes one row per genome to a shard under `genomes/shards/`. `collate` refuses to run
  if any genome is missing a row.

**QC** is AllTheBacteria's high-quality rule set:
- CheckM2 completeness ≥90% and contamination ≤5%;
- total length 100 kbp–15 Mbp;
- at most 2,000 contigs;
- N50 ≥2,000;
- a species check: the closest reference by skani ANI must be PAO1, at ≥95%.

**Two consequences for everything downstream:**
- **Embed with `--max-proteins 9000`.** `generate_embeddings` defaults to 6,000 and cuts each genome's
  flat protein list at that point. A *P. aeruginosa* genome has about 6,400 CDS, and contigs are
  length-sorted, so the default would silently drop the smallest contigs first: plasmids and integrons,
  where acquired carbapenemases and aminoglycoside-modifying enzymes sit. The trainer's own default is
  already 9,000. `collate` reports how many genomes exceed each cap.
- **Gene names come from Bakta**, as for TB and Kp: the extractor takes `gene=`, else the locus tag (a per-genome
  random prefix, so it never means the same gene across genomes). Bakta under-annotates some acquired AMR
  classes, so a name-keyed arm (per-gene LR, attribution's named-gene flag, catalogue ceilings, `locate_gene`)
  should take AMR gene identity from an AMRFinder/CARD alignment, as Kp does.

## Annotation (Bakta)

Bakta 1.9.4 + full DB v5.1 is the annotation of record (PROJECT_STATE §6). The engine code is
`engine/genomes/bakta.py`; [`scripts/annotate_pa_genomes.sh`](scripts/annotate_pa_genomes.sh) runs it:

| Stage | What it does |
|---|---|
| `plan` | Splits `embedding_input.csv` into `genomes/bakta/bakrep_roster.tsv` (ATB-sourced and in BakRep) and `run_roster.tsv` (the rest) |
| `ingest` | One CPU job, 8 BakRep streams: BakRep's GFF3 into `raw/bakta/`, byte for byte, with a `.bakta.ingest.json` |
| `submit` | The CPU array `engine/genomes/scripts/run_bakta_array.sh` (4 genomes per task, 8 cores, 52 GB) over the run roster plus every BakRep-roster genome ingest refused; each run writes a `.bakta.run.json` |
| `collate` | Writes `genomes/bakta_annotation.csv` only if every cohort genome has exactly one current annotation |

- **Why BakRep's files can be used at all:** our run reproduces BakRep's exactly (the gate,
  `engine/genomes/scripts/validate_bakta_against_bakrep.sh`), so the two origins are one annotation.
- **Ingest accepts a BakRep GFF3 only if** its header is `# Software: v1.9.4` / `# Database: v5.1, full` (a BakRep
  re-run with a newer Bakta is being prepared upstream) **and** its embedded `##FASTA` equals our contigs name for
  name and base for base. ⚠ Equal contig count, length and N50 do not imply equal names: where AllTheBacteria's
  contig numbering has gaps (`contig00130`, `contig00132`, …) our store renumbers consecutively, so BakRep's seqids
  name different contigs of ours. Such a genome is refused and annotated by us.
- **"Current"** means: the record's contigs md5 equals the cohort's contigs now, and, for our runs, the record's
  fingerprint (Bakta's flags, the tool env's exact package set, the DB's `version.json` and AMRFinderPlus DB)
  equals the env and DB collate is given. A rebuilt genome or a changed env therefore shows up as a problem in
  `bakta_annotation_problems.csv`, never as a silently stale GFF3.
- **AMRFinderPlus DB is 2024-07-22.1**, not the 2023-11-15.1 that DB v5.1 ships: AMRFinderPlus 3.12.8 (what
  Bakta 1.9.4 resolves to) refuses the older one. With it, AMR calls matched BakRep's on every gate genome.
- **Long arrays run from a pinned worktree** (`ARRAY_REPO=…/worktrees/BacPredict-<sha>`, with `.venv` symlinked to
  the shared env): other agents pull the shared checkout, and `EXPECT_SHA` would otherwise stop pending tasks. The
  worktree holds code only (every output goes under the AMR root) and is removed once the array is collated.
- **Memory:** size `--mem` from sacct (measured peaks: PROJECT_STATE §3.6), not from a run record's
  `max_child_rss_gb`, which sees only the largest single child process.

## Embeddings

[`scripts/build_pa_embeddings.sh`](scripts/build_pa_embeddings.sh) runs these stages in order:
`proteins`, `check-proteins`, `smoke`, `check-smoke`, `submit`, `check`.

- `proteins` is a CPU job that runs `preprocess_assemblies_to_protein_sequences` on `genomes/bakta_annotation.csv`
  (contigs + Bakta GFF3 per genome). It and `ingest` refuse to start if the shared checkout moved since submit.
- `smoke` and `submit` run the generic GPU array `engine/embedding/scripts/esm_embeddings_array.sh`, with
  `MAX_PROTEINS=9000`, `SAVE_DTYPE=bfloat16`, `NTASKS=16` and a sample list (below).
  - Both need `ARRAY_REPO`: a detached worktree at the SHA to run, `.venv` symlinked to the shared env. The tasks
    run its code and are pinned to its SHA (`EXPECT_SHA`).
  - Both refuse unless `ARRAY_REPO`'s `esm_embeddings_array.sh`, `generate_embeddings.py` and `sample_list.py`
    are byte-identical (`cmp`) to the driver's own checkout. The array is handed an export written for those
    copies.
  - `submit` refuses while a `pa-esm-embeddings` array is queued or running, and also if `squeue` itself fails: a
    failed `squeue` is not "none live".
- Every check counts against `bakta_annotation.csv`, not against the files that exist. The protein
  extractor logs a failed genome and still exits 0, so a count of existing files would shrink with the
  cohort and still read as complete.
- `check-proteins` also requires: each parquet's protein count = the GFF3's CDS − pseudogenes (a GFF3/contig
  name mismatch would otherwise drop proteins silently); the annotation table's genomes = `embedding_input.csv`'s;
  every genome's contigs unchanged since annotation (md5). Unreadable parquets are listed, not fatal to the check.
- `smoke` and `submit` refuse to start until every cohort genome has a protein parquet.
- **The ESM array takes a sample list, because the protein and ESM stores are shared across species.**
  - `submit` writes `processed/paeruginosa/esm_samples.txt`: the sorted Samples of `bakta_annotation.csv`, one
    per line. The file is rewritten atomically, and only when it changes.
  - `smoke` writes its own list, `processed/paeruginosa/esm_samples_smoke.txt` (the first 10 of those Samples),
    and never touches `esm_samples.txt`, which a live array may be pinned to. `check-smoke` checks those 10.
  - The array gets a list as `SAMPLES` plus its md5 as `SAMPLES_MD5`. Each task slices the list, never the store's
    listing, so Enterobacterales parquets in the same store are neither embedded nor able to shift the slices.
  - A task refuses to run (exit 3) if the list's md5 has changed since submit. `generate_embeddings` checks the
    md5 again (`--samples-md5`) against the very bytes it parses, so a list rewritten between the two reads is
    caught too. It exits before loading the model if a parquet in its slice is missing.
  - The list format (sorted, unique, blank lines ignored) is `engine/embedding/sample_list.py`.
    `generate_embeddings --samples-file` reads it the same way, and the array counts it with awk.
  - Re-running `submit` over a partly filled `processed/esm/` is safe: every task skips genomes whose `.pt`
    already exists.
  - Never run two arrays over the same samples into one ESM directory at once. `generate_embeddings` writes each
    `.pt` through a temporary `<name>.part`, and that name is the same for both arrays.
- The Enterobacterales driver (`apps/entero/scripts/build_entero_embeddings.sh`) uses the same array and list
  mechanism. Its checks are `engine/embedding/check_cohort_store.py`, which runs this file's `check_store` logic
  in a process pool; the inline version here is kept.

ESM-C in **bf16** in `processed/esm/`, half the fp32 size. The trainer runs in bf16 already. Four consumers call
`.numpy()` on embeddings and must call `.float()` first; they fail loudly, not silently:
`engine/concat/bacformer_token_cache.py`, `engine/concat/cache_bacformer_gene_embeddings.py`,
`engine/embedding/protein_pooling.py`, `engine/embedding/genome_assemblies_from_bacformer_embeddings.py`.

## Splits

`engine/splits/build_amr_holdout_splits.py` builds the split tables (its docstring has the mechanics). It is
light: a CSD3 login node is enough. Everything downstream reads the tables through `load_splits`, and reads
`holdout_st` through `load_split_ids(path, "holdout_st")`. Nothing else may resolve a holdout (root `CLAUDE.md`,
*Data-leakage guarantees*).

| Input | Where |
|---|---|
| Labels | `processed/paeruginosa/binary_ast.csv`, keyed by its first column `phenotype-BioSample_ID` (the writer's index; the builder also accepts `Sample`) |
| Eligible genomes | `processed/paeruginosa/genomes/bakta_annotation.csv`: the 3,677 annotated and embedded |
| ST (`mlst_scheme`, `st`) | `processed/paeruginosa/genomes/genome_qc.csv`. Keyed `<scheme>:<st>`, never `st` alone |
| Genomes left out | [`excluded_genomes.tsv`](excluded_genomes.tsv): the 15 Ecuador long-read portal assemblies |
| One isolate under two BioSamples | [`duplicate_isolate_groups.csv`](duplicate_isolate_groups.csv): the CDC strains S54485, U2504 and X24509. Each pair always shares a split |

**What it implements** (PROJECT_STATE §6, 2026-11-20):
- ST155 (`paeruginosa:155`) is held out whole as `holdout_st`. Membership is the exact ST, so a single-locus
  variant or an untyped genome stays in the random pool. **It was ST235 until 2026-11-21** (PROJECT_STATE §6):
  ST235 is 7.1–9.9 % of each drug's labelled set, which with the random 20 % put 26–28 % of genomes out of reach of
  training. The lineage holdout is now capped at 5 %; ST155 is 2.1–3.9 % and scores the same 8 of 10 drugs.
- A random 20 % of the remaining genomes is `holdout`.
- The rest is cut into 5 folds: fold 0 is `validate`, folds 1–4 are `train`. Seed 1.
- There is no per-class minimum.

**How the split is drawn:**
- Splits are assigned per genome, not per drug: a genome is in the same split in all ten tables. Each drug's
  table holds only the eligible genomes labelled for that drug. A fractional (conflicting) label keeps its row but
  is never scored.
- PsA has its own RNG, seeded from (seed, `paeruginosa`). Adding Enterobacterales to an arm therefore changes
  no PsA row.

**The two arms:**

| Arm | `--species` | Extra flags |
|---|---|---|
| PsA-only (baseline) | `paeruginosa` | none |
| PsA + Enterobacterales | both | `--assert-same-as paeruginosa=<PsA-only dir>` |

- The builder refuses the cross-species arm unless its PsA rows equal the PsA-only arm's, row for row, for every
  drug. That is what makes both arms score the same PsA holdouts.
- Pass this folder's exclusion and group files to the cross-species run as well, or the check fails.

**Reading one species from the cross-species arm.** That arm's `train`, `validate`, `holdout` and `holdout_st`
each pool PsA and Enterobacterales genomes. To score PsA alone, ask the reader for it:
`load_splits(path, species="paeruginosa")` and `load_split_ids(path, "holdout_st", species="paeruginosa")`.
Both return exactly the PsA-only arm's lists and labels (tested). Never filter the table's `species` column
yourself: that is a second reader. Without `species`, both read the whole table.

⚠ **Open, for David:** which `validate` the cross-species arm uses for early stopping and model selection. It
could be the pooled PsA + Enterobacterales set (`species` unset) or PsA only (`species="paeruginosa"`, the same
set the baseline uses). The builder writes both into the one table, so this is the trainer's choice and is not
decided yet.

**Outputs:**
- `<out-dir>/<drug>_split.csv` and `split_manifest.json`. The manifest records the arguments, input md5s, git
  SHA, each table's md5 and the counts per drug × species × split.
- The builder will not replace a table whose content differs unless given `--overwrite`. A new split invalidates
  every model trained on the old one.

The PsA-only arm, from the repo root (the out dir is a proposal, not yet a decision):

```bash
AMR=${BACPREDICT_AMR_ROOT:-$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/amr_prediction}
P=$AMR/processed/paeruginosa
uv run python -m bacpredict.engine.splits.build_amr_holdout_splits \
  --species paeruginosa=$P/binary_ast.csv,$P/genomes/bakta_annotation.csv,$P/genomes/genome_qc.csv \
  --holdout-st paeruginosa=paeruginosa:155 \
  --exclude src/bacpredict/apps/pa/excluded_genomes.tsv \
  --groups src/bacpredict/apps/pa/duplicate_isolate_groups.csv \
  --drugs ciprofloxacin ceftazidime meropenem tobramycin amikacin imipenem piperacillin-tazobactam \
          aztreonam levofloxacin cefepime \
  --out-dir $P/splits
```

⚠ The builder stops if a group spans the held-out ST and not — a CDC strain whose two BioSamples type
differently, with one of them the held-out ST. **Checked for ST155 against `genome_qc.csv`: no group spans it**, so
the builder does not stop. It was never checked for ST235, which grouped U2504's two BioSamples inside the ST.

## Training

One model per drug, trained by the shared trainer from that drug's split table (`<drug>_split.csv`, *Splits*).
[`scripts/train_pa_models.sh`](scripts/train_pa_models.sh) runs the PsA-only arm from a CSD3 login node:

| Stage | What it does |
|---|---|
| `splits` | Writes `splits/eligible_genomes.csv` (every `bakta_annotation.csv` genome; refuses if any lacks a `.pt`), then runs the builder with `HOLDOUT_ST` (ST155), the exclusion file and the group file |
| `check-splits` | Per drug, the trainer's own read of its table: genomes per split, % R, and any genome without a `.pt` |
| `smoke` | Stage A: one drug (`SMOKE_DRUG`, default ciprofloxacin), `--n-samples 10`, early stopping off (`PATIENCE=1000`), one GPU, 1 h wall. Output `models/smoke/<stamp>/<drug>/`; `models/smoke/LATEST` names it |
| `check-smoke` | Stage A passes if `results.json` holds `holdout` and `holdout_st`, both score files match them, a checkpoint was saved and `trainer_state.json` was written |
| `overfit` | Stage B, read from the same run by `engine/finetune/check_overfit`: the whole 100-step schedule ran, the logged loss reaches its end, the last logged train loss ≤ 0.1 (`OVERFIT_MAX_LOSS`) and train = holdout AUROC ≥ 0.99 |
| `submit` | Stage C: a GPU array, one task per drug, into `models/<drug>/`. Refuses while a `pa-finetune` job is live or if any `models/<drug>/` exists |
| `collect` | `models/holdout_metrics.csv`: a row per drug × holdout, with each run's `results.json` path. A drug without a `results.json` is a `missing` row |

**Stage A and B are one run.** The trainer's 10-genome path takes a class-balanced 10 of `train` (seeded) and uses
them as train, validate and holdout for 200 optimizer steps (1 per epoch at grad-accum 8) at a **constant** learning
rate, and scores the **final** model (no best-checkpoint reload); it also scores a balanced 10 of `holdout_st` so that
path runs. The first smoke (job `41545259`) ran a linear decay to 0 over 100 steps, which stalled the loss at ~0.10, and
reloaded the best checkpoint, which on train = validate was step 2 (AUROC 1.0 already), so it scored an untrained
model (every probability ~0.99, balanced accuracy 0.5). The smoke turns early stopping off: on train = validate the AUROC saturates in a
few epochs, and patience 15 would end the run before the loss has fallen.

**Stage B reads `<run>/trainer_state.json`, never `checkpoint-*/trainer_state.json`.** With `save_total_limit=1` and
`load_best_model_at_end`, HF deletes every checkpoint but the best when training ends. The best is the first
evaluation to reach the top AUROC, so its history stops there, often at the first logged loss. Split-table mode
therefore calls `Trainer.save_state()` after training, which writes the whole run's history beside `results.json`.

**What `finetune_amr --split-table` does** (`--task pa_ast`, which requires a split table):
- Train, validate, holdout and the labels come from the table alone, through `engine/splits` (`load_splits`,
  `load_split_ids`, `load_split_species`). There is no AST sheet, no k-fold and no `train_val_eval`. Only 0/1 labels
  are read, so a fractional (conflicting) label never trains or scores.
- Every genome of every split must have a `.pt` in `--embeddings-dir` (default `processed/esm/`). A missing one
  (a dangling symlink counts) stops the run before the model loads.
- After training, the best model (early stopping on validation AUROC) scores `holdout` and each `--extra-holdouts`
  split (default `holdout_st`). `results.json` keeps the holdout's metrics in `metrics` and adds `holdouts`, with
  per-species blocks when the table holds more than one species. Per-sample scores go to `eval_scores.npz` and
  `eval_scores_holdout_st.npz`. Field reference: `docs/results_schema.md`.
- The dataset casts every embedding to fp32, so the bf16 PsA files and any fp32 file symlinked from the Kp store
  reach the model as one dtype (bf16 → fp32 is exact). The model's master weights are bf16 either way.
- A cross-species arm uses a free task label (e.g. `pa_entero_ast`), which the trainer accepts only with explicit
  `--embeddings-dir` and `--split-table`.

**The GPU array** is `engine/finetune/scripts/finetune_amr_array.sh` (its header lists every setting):
- It runs `ARRAY_REPO`, a detached worktree at the SHA to run (`.venv` symlinked to the shared env), pinned by
  `EXPECT_SHA`. The driver refuses unless that worktree's trainer files are byte-identical to its own.
- Each task also checks `split_manifest.json`'s md5 against the one at submit: a rebuilt split stops pending tasks.
- It unsets `LD_LIBRARY_PATH`, `LIBRARY_PATH` and `LD_PRELOAD` and runs Python with `-X faulthandler` (the cclake
  login-node crash, PROJECT_STATE §3.6 *Caveats*). DataLoader workers = the allocation's cores.
- Hyperparameters are the deployed TB/Kp ones: lr 1.5e-4, batch 1 × grad-accum 8, warmup 0.1, bf16, evaluation every
  250 optimizer steps, early stopping after 15 evaluations without a 0.005 AUROC gain, at most 40 epochs.
  Patience counts evaluations, not epochs.
- Resources: 1 A100 on `ampere`, 32 cores, 250 GB (ampere's one-GPU ceiling), `FLOTO-SL2-GPU`, 35 h. A
  ~2,500-genome drug at ~3 s per step is ~16 min per epoch, so the 40-epoch cap is ~11 h of training plus
  validation passes; 35 h stays under the QOS's 36 h cap.
- A task refuses if its `results.json` already exists. A run killed at the wall resumes with
  `--resume-from-checkpoint <models>/<drug>/checkpoint-<step>`, not by resubmitting the array.

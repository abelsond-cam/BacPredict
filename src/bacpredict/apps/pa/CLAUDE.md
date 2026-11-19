# *P. aeruginosa* AST (`apps/pa`)

The *P. aeruginosa* (PsA) adapter for the multi-species AMR work: PsA first, then all Enterobacterales,
then every genome in the EBI AMR Portal. **Status, numbers and next steps live only in
[`PROJECT_STATE.md`](../../../../PROJECT_STATE.md)**; this file holds the mechanics.

The pipeline itself is the engine. This folder holds only what is PsA-specific: the adapter for the
AMR-GNN supplement and the run scripts.

## Labels: `binary_ast.csv`

Built by `bacpredict.engine.ast_labels.build_binary_ast_labels`, which is organism-agnostic (species or
genus filter, drug list and extra sources are arguments). Run it with
[`scripts/build_pa_ast_labels.sh`](scripts/build_pa_ast_labels.sh). It is light enough for a CSD3 login node.
Output goes to `<root>/processed/train_pa_ast/`.

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

Every genome then gets **one Prodigal 2.6.3 run with Prokka's flags**, so gene calls are uniform. STs
come from our own `mlst` run, never from the paper's `st_reported`, which is kept only for cross-checking.

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
3. Shovill on paired Illumina WGS/WGA reads at ≥10× depth.
4. Drop, with a reason.

**Where the outputs go:**
- Contigs are renamed `<Sample>.contig00001` and stored at `raw/ast_cohort_genomes/contigs/<Sample>.fna.gz`.
- Prodigal GFFs are stored at `raw/ast_cohort_genomes/prodigal/<Sample>.gff.gz`.
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
- **Prodigal GFFs carry no gene names.** The extractor falls back to Prodigal's positional `ID`
  (`1_1`, `1_2`, …), which repeats in every genome without meaning the same gene. Fine-tuning does not
  care. Before any name-keyed arm (per-gene LR, attribution's named-gene flag, catalogue ceilings,
  `locate_gene`) runs on this store, gene identity has to come from elsewhere, for example an
  AMRFinder/CARD alignment or nuna families.

## Embeddings

ESM-C in **bf16** on project_k, half the fp32 size. The trainer runs in bf16 already. Four consumers call
`.numpy()` on embeddings and must call `.float()` first; they fail loudly, not silently:
`engine/concat/bacformer_token_cache.py`, `engine/concat/cache_bacformer_gene_embeddings.py`,
`engine/embedding/protein_pooling.py`, `engine/embedding/genome_assemblies_from_bacformer_embeddings.py`.

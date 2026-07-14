# Gene Array Lasso — sub-project memory

This is a task folder under `src/`. See the root [CLAUDE.md](../../CLAUDE.md) for §0 global conventions
(base model, three-stage A/B/C protocol, paths, reporting, HPC resourcing) and
[ToDo.md](../../ToDo.md) for cross-task state. **All code, `/data`, `/visualisations` and `/docs` for this
work live under `src/gene_array_lasso/`.**

> **Status:** plan **approved** 2026-06-25, **building Step A**. Biology + full multi-stage programme:
> [`Bacformer_pangenome_aligned_head.md`](Bacformer_pangenome_aligned_head.md) (source of truth). This file is
> the concrete de-risking first sub-project that gates it. Approved plan snapshot:
> `~/.claude/plans/i-d-like-to-start-crystalline-allen.md`.

## Aim — de-risk the group-sparse approach on Panaroo GPA before building PangenomeFormer

The overarching design builds a **pangenome-aligned `genes × 960` Bacformer embedding matrix** fed to
**group-sparse penalised linear models** (sparse-group lasso / group elastic net), doubling as a
population-corrected GWAS method. It extends the [`../kleb_ast/`](../kleb_ast/) result — concatenating one
*hand-picked, Bakta-annotated* gene's embedding onto the fine-tuned genome-mean clears the CARD ceiling — by
letting a group-sparse penalty over **all** genes select the causal family itself.

**The biggest project risk is whether a group-sparse penalty works over our data at all** — learned attention
pooling did **not** (attention 0.868 lost to mean 0.905; routes to lineage hubs, not the SNP gene). Before
investing in PangenomeFormer (embedding-space clustering, Stage 0 of the doc), **test the sparse models on a
pangenome we can get cheaply: Panaroo's gene presence-absence (GPA).** Clean recovery of the causal family on
Panaroo nodes green-lights PangenomeFormer; a negative on all three drugs says the approach doesn't transfer.

## Decisions locked (2026-06-25)

| Decision | Choice |
|---|---|
| **Drugs (3 separate runs — all <3500 ⇒ all-in-one with a real in-run test)** | **imipenem** (~2,370; acquired carbapenemase KPC/OXA/NDM **+** porin loss ompK35/36 — mixed/HGT; kleb_ast AUROC 0.973), **tetracycline** (~1,945; acquired *tet* efflux — accessory/HGT, catalogue-gap, 0.914), **colistin** (~1,400; chromosomal mgrB/pmrB — hard SNP-localisation case, 0.807, heavy lineage confound). Counts ≈ 5× the per-drug evaluate holdout in `../bacpredict/visualisations/kp/eval/eval_summary.csv`. |
| **Embedding (Axis B)** | **B1 frozen ESM-C** (best single-residue localisation; already in the store, no GPU pass). Axis C (FT genome-mean concat) + B2 frozen-Bacformer are follow-ups once B1 works. |
| **Engine** | **`groupyr`** (sklearn-compatible Sparse Group Lasso, JOSS 2021, copt-backed, 2–10× faster than `group_lasso`, built-in CV). `group_lasso` (Moe 2020) = documented fallback. Both = Simon et al. 2013 SGL (**A1**); **A2** group elastic-net = same core, relaxed L1/L2 group ratio. Confirm `copt` installs under uv at Step D. |
| **Branch** | **None — stay on `dev`** (shared project space; pyseer + other work run here). |

**Run-scoping rule (Decision A).** Panaroo caps ~4000–4500 *genomes*/run; runs can't be linked. The 6500 figure
is the **union across drugs**; any one drug is far smaller. **If total labelled < 3500: one run over ALL its
samples** (train+validate+evaluate together) → evaluate is a **genuine in-run test** (gene columns shared).
**All three chosen drugs take this branch.** Otherwise (≥3500, e.g. cipro/mero later): train+val only, park the
20% evaluate as a separate run, report on validate. Reuse the **existing kleb_ast folds** (`train_val_eval` in
the split CSV) throughout.

**Test priority + variant ladder (user, 2026-06-25).** The first **>5%** array fit is a *preliminary*
pipeline/feasibility test — **the >1% cutoff is the important test** (the headline; draw conclusions there).
Beyond the embedding ladder for the gene blocks — **B1 frozen ESM-C → B2 frozen Bacformer → Bacformer-FT**
(swap "Bacformer instead of ESM" and FT embeddings are explicit planned tests, not just follow-ups) — keep
Axis C (⊕ FT genome-mean) in view. **Why group lasso at all:** the parked attention-head attempt
([`../tl/train/attention_pool.py`](../tl/train/attention_pool.py), gated-attention MIL pool) *lost* to the
mean (~0.905 → ~0.868) because it treats the genome as an unordered bag of millions of dims with no
ortholog/synteny structure. The structured group lasso exists precisely to impose that known per-gene
grouping — this is the motivation of record.

## Resolved facts (from the code, 2026-06-25)

- **Embeddings derive from the SR assembly (Open #1 closed).** `tl/embed/preprocess_assemblies_to_protein_sequences.py`
  reads **only** `sr_gff_file`/`sr_assembly_file` — no long-read path. So every embedded `Sample` has an SR
  Bakta assembly; the ESM store + `klebsiella_protein_sequences` parquet both come from it.
- **One-genome-per-Sample = keep SR, null LR.** BacHGT `panaroo_run_strain.py` emits an SR genome
  (`panaroo_label = sample_accession`, `assembly_type="sr"`) **and** an LRA genome (`panaroo_label = Sample`)
  whenever each one's GFF+assembly exist. To get a 1:1 GPA-column ↔ `Sample` ↔ embedding map, the subset TSV
  **nulls `lr_gff_file`/`lr_assembly_file`** so only the SR genome (matching the embedding) is emitted.
- **The column→Sample join is `panaroo_genomes.tsv`** (`panaroo_label`,`Sample`,`assembly_type`,
  `sample_accession`), written by the runner. SR `panaroo_label` = `sample_accession` (e.g. GCF_/GCA_/SAM…),
  *not* the BioSample `Sample` — always remap via this TSV, never assume the column header is our `Sample`.
- **locus_tag join (Open #2) expected to hold:** Panaroo clusters the *same* SR Bakta GFF the protein parquet
  was extracted from → per-cell locus tags match the parquet/ESM protein order. **Still spot-check on imipenem**
  (Step C verification) and drop any per-sample join failures.
- **kpsc_final_list filter:** the runner drops rows where `kpsc_final_list != True` unless `--non-kpsc-species`.
  The builder **forces `kpsc_final_list=True` on the selected AST rows** (they *are* the KPSC AST cohort) so
  none are silently dropped. Join AST `Sample` (= `phenotype-BioSample_ID`) → metadata_v2 `Sample`.

## Pipeline — four steps, per drug (×3)

**A) Sample set → Panaroo.** `build_panaroo_sample_tsv.py`: read `processed/train_kleb_ast/binary_ast_with_split.csv`
   (`Sample`,`<drug>`,`train_val_eval`) → rows with non-NaN `<drug>` (all-in-one) → join to
   `final/metadata_v2_all_samples_and_columns.tsv` on `Sample` → **null `lr_*`, force `kpsc_final_list=True`** →
   write `data/panaroo_tsv/<drug>_sample_metadata.tsv` (+ a `<drug>_splits.csv` carrying `Sample,train_val_eval`).
   Launch BacHGT `slurm_scripts/panaroo_run_strain.sh --sample-metadata-file <tsv>` (no `--clonal-group`/
   `--sublineage` ⇒ uses all rows in the TSV), one job/drug → `processed/gene_array_lasso/panaroo/<drug>/`
   (`gene_presence_absence.csv`, `panaroo_genomes.tsv`, `pan_genome_reference.fa`, `final_graph.gml`).
   *(Builder is light — two CSV reads + a join, no embedding stat — login node is fine.)*

**B) node→gene→sample + filter.** Parse the **rich** `gene_presence_absence.csv` (cells = locus_tag(s)) +
   `panaroo_genomes.tsv`. **Filter clusters present in >1% of genomes — lower bound ONLY, no upper bound** (keep
   core genes). The **>1% universe is the primary, solid result**; a **>5%** variant is a compute/feasibility
   check only (**no conclusions** from it). `min_prevalence` default 0.01; no `max_prevalence`.

**C) frozen-ESM-C array (B1).** Per present `(Sample, cluster)`: locus_tag → protein index → frozen ESM-C
   960-dim vector from `{Sample}_esm_embeddings.pt`; absent → zero block. `X` = `n_samples × (n_genes × 960)`,
   **block-sparse** (groups = genes). Paralogue cells (≥2 loci) → **mean the copies**. Axis C off.
   `build_gene_embedding_array.py`; reuse kleb_ast per-gene LR machinery for the locus_tag→index join.

**D) fit + score.** `groupyr`, groups = genes. **A1 sparse-group lasso** + **A2 group elastic net** (one core,
   exposed L1/L2 ratio). Fit on train, tune on validate, **report on the in-run evaluate test**.
   `fit_group_sparse.py` + `aggregate_selection.py`. **Gate:** beat genome **mean-pool** AND recover the causal
   family — imipenem → carbapenemase (KPC/OXA/NDM)/porin (ompK35/36); colistin → mgrB/pmrB (constant-presence
   SNP case — the strongest test); tetracycline → *tet* efflux. Benchmark vs kleb_ast CARD ceiling +
   best-Bacformer concat.

## Open items (resolve during build)

3. **Universe size vs engine scale** — >1% can give ~25k groups × 960 ≈ 24M features; dense infeasible. Rely on
   the block-sparse store (most accessory blocks zero); use the >5% variant to characterise load; himem node.
   (#1 SR-assembly and #2 locus_tag join resolved above — #2 still spot-checked on imipenem.)
- **Parked evaluate / cross-run node alignment** = deferred to PangenomeFormer; not in this sub-project.

## Directory layout (under `src/gene_array_lasso/`)

`build_panaroo_sample_tsv.py` (A) · `build_gene_embedding_array.py` (B+C) · `fit_group_sparse.py` +
`aggregate_selection.py` (D) · `scripts/` (one sbatch per step) · `data/` (small manifests/TSVs/universe CSVs;
big arrays on RDS `processed/gene_array_lasso/{panaroo,gene_arrays}/<drug>/`) · `visualisations/` · `docs/`
(the design doc + this file). Name files for the action, not the step.

## Milestones

1. **A** — three Panaroo runs (imipenem, tetracycline, colistin), all-in-one → GPA + `panaroo_genomes.tsv`.
2. **B+C** — gene-embedding array (frozen ESM-C, >1%) for **imipenem first**, block-sparse on himem.
3. **D** — A1/A2 fit: imipenem (beat mean-pool + carbapenemase/porin) → colistin (mgrB/pmrB SNP thesis) → tetra.
4. **Gate** — clear positive ≥1 drug (ideally imipenem + colistin) → write up prelim, green-light
   PangenomeFormer (Stage 0) + take to John Lees. Negative → reconsider the sparse approach.

## Pointers

- Overarching design + biology + Stage 0 clustering + LMM/pyseer A3 + non-coding:
  [`Bacformer_pangenome_aligned_head.md`](Bacformer_pangenome_aligned_head.md).
- Panaroo: `~/developer/BacHGT/src/bac_panaroo/` — runner `slurm_scripts/panaroo_run_strain.sh` →
  `run_panaroo/panaroo_run_strain.py` (`--sample-metadata-file`; emits SR `sample_accession` + LRA `Sample`
  genomes; writes `panaroo_genomes.tsv`). metadata_v2 default `…/david/final/metadata_v2_all_samples_and_columns.tsv`.
- Splits: [`../tl/train/split_utils.py`](../tl/train/split_utils.py) (`add_splits` 70/10/20); split-CSV producer
  [`../kleb_ast/prepare_esmc_embeddings_and_labels_to_finetune_amr.py`](../kleb_ast/prepare_esmc_embeddings_and_labels_to_finetune_amr.py).
- Embeddings: SR-only extractor [`../tl/embed/preprocess_assemblies_to_protein_sequences.py`](../tl/embed/preprocess_assemblies_to_protein_sequences.py);
  `{Sample}_esm_embeddings.pt` store + `klebsiella_protein_sequences/{Sample}_protein_sequences.parquet`. Upstream
  success we extend: kleb_ast Plot #1 / ladder / panel under `../bacpredict/visualisations/kp/`.

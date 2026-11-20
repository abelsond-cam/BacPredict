# HPC (CSD3/UoHPC) data map — BacPredict

Where BacPredict's data lives on **CSD3/UoHPC** (Cambridge), and how the data-root resolver maps onto
it. Sibling doc: [`ISAMBARD_DATA.md`](ISAMBARD_DATA.md) (live-verified). Cluster reference:
[`~/.claude/cluster_uohpc.md`](~/.claude/cluster_uohpc.md).

> **CSD3 has been operational again since 29 Jul 2026** (SSH, RDS, `ampere` GPU and CPU jobs; only
> cold-storage *tape* files remain inaccessible). The "CSD3 is down, seeded not verified" banner that
> used to sit here was itself stale — it is the reason parts of this doc were treated as unusable long
> after they could have been checked.
>
> **Verified live 2026-08-14:** the canonical AST root and cohorts below, the Kp `card_ceiling/` tree
> (22 drugs), and the TB `tbprofiler_gene_lr/` tree (9 drugs). **Not re-verified:** per-store file
> counts for the embedding stores. Treat an unverified count as a claim, not a fact.
>
> **The canonical AST root is `$R/bac_ast_prediction/`** — Kp **7,088** / TB **36,692**.
> `$R/processed/` is the **deprecated May cohort** (6,838 / 36,684); the embedding stores are a
> *superset* again (9,724 / 38,257). These three numbers are different things — see
> [`PROJECT_STATE.md`](../../../PROJECT_STATE.md) §5 and never write a bare cohort count.

## Root — the single knob

```bash
export BACPREDICT_DATA_ROOT=~/rds/rds-floto-bacterial-4k08a2yyQLw/david   # = project_k/david
```

BacPredict's working data is the **`project_k/david` RDS allocation** (20 TB, the main working-data
home). The code resolves the root via `bacpredict.engine.config.resolve_data_root()`: `--data-root`
arg → `$BACPREDICT_DATA_ROOT` → `$SCRATCHDIR` → **this CSD3 path if it exists on disk** → error. On
CSD3 there is no `$SCRATCHDIR` in the Isambard sense, so **set `$BACPREDICT_DATA_ROOT` explicitly**
in `.bashrc` (the autodetect fallback works only because the path exists, but the export is clearer).

## Storage tiers

| Tier | Path | Size | Holds for BacPredict |
|---|---|---|---|
| Home | `~/workspace` = `/home/dca36/workspace/` | ~52 GB (all `/home`) | **code only** — the `BacPredict` checkout (chronically near-full; code + dotfiles only) |
| `project_k` (RDS) | `~/rds/rds-floto-bacterial-4k08a2yyQLw/david` | 20 TB (shared) | **all working data** — raw/ processed/ final/ (stay in `david/`) |
| `personal_rds` (RDS) | `~/rds/hpc-work/` | 1 TB | scratch — SLURM logs, caches, staged scripts, intermediates |
| `bacformer_rds` (RDS) | `~/rds/rds-flotolab-9X9gY1OFt4M/` | 13 TB (shared) | Bacformer mount — **hands off**, shared lab mount |
| `cold_storage` (RCS) | `~/rcs/rcs-vgm23-lcms/David/` | ~50 TB | backups + published outputs (archival; write tarballs, not many small files) |

Caches/env off `/home` via `.bashrc` (`UV_CACHE_DIR`, `HF_HOME`, `TORCH_HOME`, `TMPDIR=~/rds/hpc-work/tmp`, …).

## Layout under `project_k/david` (the root)

Same three-part shape as Isambard: `raw/<task>/`, `processed/<task>/`, `final/`. The per-organism
processed stores mirror `ISAMBARD_DATA.md` — `processed/train_tb_ast/` and `processed/train_kleb_ast/`
each with `esm/ baclm/ bacformer/ protein_sequences/ intergenic/` plus `binary_ast_with_split.csv`,
`embedding_input.csv`, etc. **TODO (confirm):** exact contents, file counts, and whether the
pre-outage CSD3 stores are the same cohort as the Isambard re-download.

## Multi-species AMR — `project_k/david/amr_prediction` (`$AMR`)

The EBI AMR Portal work (*P. aeruginosa*, then Enterobacterales) has its own root, separate from the TB/Kp
tree above (David, 2026-11-20): `$AMR` = `$BACPREDICT_AMR_ROOT`, default
`~/rds/rds-floto-bacterial-4k08a2yyQLw/david/amr_prediction`, with `raw/`, `processed/`, `final/`. Its scripts
(`apps/pa/scripts/`, `apps/entero/scripts/`, `engine/genomes/scripts/validate_bakta_against_bakrep.sh`) set it
themselves; they do not read `$BACPREDICT_DATA_ROOT`. **Per-sample files are flat and keyed by BioSample; per-species
tables sit in a species folder** (`processed/paeruginosa/`, `processed/enterobacterales/`, `raw/paeruginosa/`).
Full table: `apps/pa/CLAUDE.md` *Where the data lives*.

### EBI AMR Portal — the all-species AST + genotype tables

`$AMR/raw/ebi_amr_portal/<release>/` holds the portal's FTP release, fetched from
`https://ftp.ebi.ac.uk/pub/databases/amr_portal/releases/<release>/` (`releases.yml` there names the
latest) and md5-checked against its `md5sums`:

- **`phenotype.parquet` — the AST tests** (one row per BioSample × drug × test, all species). This is
  the multi-species label source. Columns are unprefixed (`BioSample_ID`, `antibiotic_name`,
  `resistance_phenotype`, `species`, …) unlike the older `phenotype-*`-prefixed per-species extracts.
- `genotype.parquet` — AMRFinderPlus hits on every assembly the portal holds (GCA *and* EBI's own ERZ
  assemblies); its `BioSample_ID → assembly_ID` keys are the portal's genome set.
- ⚠ **`$AMR/raw/ebi_amr_portal/ebi_amr_records_<date>.csv` is a portal-UI export of the *genotype* table**
  (`genotype-*` columns), not AST — despite the name.

**Labels built from it**, one folder per species: `$AMR/processed/paeruginosa/` (*P. aeruginosa*:
`binary_ast.csv`, `cohort.csv`, …; built by `apps/pa/scripts/build_pa_ast_labels.sh`, mechanics in
`apps/pa/CLAUDE.md`) and `$AMR/processed/enterobacterales/`.

**Genomes built for the AST cohorts** (`engine/genomes/`, organism-agnostic, keyed by BioSample):
`$AMR/raw/{contigs,prodigal}/<Sample>.{fna,gff}.gz` — contigs renamed `<Sample>.contig00001…`, one Prodigal 2.6.3
GFF per genome (not the annotation of record) — and `$AMR/raw/bakta/<Sample>.bakta.gff3.gz`, the annotation of
record (Bakta 1.9.4, DB v5.1 full): BakRep's file beside a `.bakta.ingest.json`, or our run's beside a
`.bakta.run.json` (+ `.faa`, `.txt`, `.log`). Proteins and ESM-C: `$AMR/processed/{protein_sequences,esm}/`.
Our Bakta DB: `$AMR/raw/bakta_db/v5.1_full/` (AMRFinderPlus DB `latest` → 2024-07-22.1; md5s in `v5.1_full.md5`).
The ours-vs-BakRep gate: `$AMR/processed/bakta_vs_bakrep_validation/`; BakRep's export and the coverage join:
`$AMR/raw/bakrep/`, `$AMR/processed/bakrep_coverage_20261119/`. Per-species manifests, shards and QC (CheckM2,
skani) sit in `$AMR/processed/<species>/genomes/`. Inputs staged once: `$AMR/raw/allthebacteria/assemblies_list_<date>/`
(the ATB S3 inventory used to plan) and `$AMR/raw/reference_genomes/<GCF>.fna.gz` (species-check references).
CheckM2's DB is the shared one: `$R/raw/CheckM2/Zenodo_db/CheckM2_database/uniref100.KO.1.dmnd`.

Per-genome annotation (Prokka/Prodigal GFF with `##FASTA` contigs appended) is on the same FTP under
`https://ftp.ebi.ac.uk/pub/databases/amr_portal/genomes/{ERZ,GCA}/<3>/<3>/<accession>/<accession>_annotations.gff.gz`
(one shared tree, not per release; genotype-table accessions only). Scoping counts per release:
`$AMR/processed/ebi_amr_portal/scoping_<release>/` (script alongside).

### AMR-GNN *P. aeruginosa* set (Nguyen et al. 2026, Nat Commun, doi 10.1038/s41467-026-69934-8)

`$AMR/raw/paeruginosa/amr_gnn_nguyen2026/` — 2,515 isolates, 12 drugs, MIC + EUCAST v15 S/R. `MOESM4_ESM.xlsx` is
the paper's Supplementary Data 1 as published; `amr_gnn_long.csv` is it tidied (one row per isolate × drug);
`amr_gnn_isolate_biosample.csv` maps every isolate to its BioSample (runs via ENA `read_run`; the Kos
et al. `AZPAE…` IDs are *strain names*, mapped via the WGS-set BioSamples' `strain`); `ena/` holds the
ENA tables behind that map.

## Compute (SBATCH)

- **CPU:** `--account=FLOTO-PROJECT-K-SL2-CPU --partition=icelake-himem` (icelake-himem is less
  oversubscribed; ~76 cores/node, ~6.7 GB/core). Avoid the small personal `FLOTO-SL2-CPU` (budget holds).
- **GPU:** `--account=FLOTO-SL2-GPU --partition=ampere` (~£0.55/GPU-h — keep saturated) or
  `FLOTO-SL3-GPU` (free, low priority). No project_k GPU account (project_k is CPU-only).
- Logs → `~/rds/hpc-work/logs/` (a `#SBATCH` line needs the literal path). Checkout at `~/workspace/BacPredict`.
- **Gotcha:** `module load` of a spack tool leaks python-3.9 onto `PYTHONPATH` and breaks `uv` — use a
  pixi env for tool binaries, or `unset PYTHONPATH PYTHONHOME` after `module load`.

**TODO (confirm on return):** aarch64-vs-x86 env (CSD3 is x86 `icelake`/`ampere` — a *different* uv
solve from the Isambard aarch64 env); the actual env path; per-store file counts; `final/` contents.

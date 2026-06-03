# Task 5 — DefensePredictor on short- vs long-read assemblies (DP-SR)

Run [DefensePredictor](https://github.com/PeterDeWeirdt/defense_predictor) (DeWeirdt et al.,
*Science* 2024) across the paired **long-read (LR) vs short-read (SR)** Klebsiella cohort and
compare per-genome defence-protein calls between the two assembly types. Defence systems
cluster at MGE / contig boundaries — exactly where SR assemblies fragment — so the **LR-vs-SR
delta** is the headline quantity. Output lands in `processed/defence_predictor/`.

This is the baseline arm of Task 5 in [ToDo.md](../../ToDo.md) ("DP-CG applied to SR — quantify
the shortfall"). Retraining DP on SR and adding a distance-to-contig-break feature are later
milestones, not built yet.

## What DefensePredictor actually is (the wrinkles)

It is **not** a Bacformer/ESM-C model and **cannot reuse our ESM-C embedding store**:

- A **5-fold LightGBM ensemble** (`beaker_fold_{0..4}.pkl`) over a feature matrix of: **ESM2
  `esm2_t30_150M_UR50D`** embeddings (640-dim, layer 30) of each gene **and its ±2 same-contig
  neighbours**, plus nucleotide composition, protein length, neighbour co-directionality, and
  inter-gene distances. Embeddings are regenerated internally per genome (GPU forward pass).
- Output: per-protein log-odds (`mean_log_odds`); call defensive at **`mean_log_odds ≥ 4`**.
- Because features come from the ±2 gene window on a contig, **gene order + contig membership
  are load-bearing**. SR contig breaks blank out neighbours (sentinel-filled) → expected
  degradation. That degradation *is* the experiment.

### Input format
DP's `--gff` route needs a **Prokka-style GFF3 with an embedded `##FASTA` section**. Our Bakta
GFFs don't carry the FASTA (assembly is a separate file), so we fuse GFF + assembly using the
**same `convert_bakta_to_prokka_gff.convert`** that BacHGT's Panaroo pipeline uses
([panaroo_run_strain.py:463](../../../BacHGT/src/bac_panaroo/run_panaroo/panaroo_run_strain.py)) —
so DP sees the identical gene models Panaroo does. `--ignore-overlapping` is on, matching Panaroo.

## Isolated environment

DP pins `torch >=2.5.1,<2.6` + `fair-esm` + `lightgbm`, conflicting with the Bacformer env, so
it gets its **own uv venv** at `.venv-dp/` — invoke with `.venv-dp/bin/python`, **never `uv run`**.
The venv also carries `gffutils` + `biopython` for the convert script.

```bash
bash src/dp_short_read/scripts/setup_dp_env.sh   # login node, once (PyPI + weight downloads)
```

## Cohort & paths

- **Cohort:** `processed/complete_vs_sr_genomes/paired_index.tsv` (~2,911 LR/SR pairs; 748 with
  `lra_is_reference_genome=True`). Smoke = 10 reference genomes, both arms.
- **File paths** (Bakta GFF + assembly per arm) come from `metadata_v2`, columns
  `lr_gff_file`/`lr_assembly_file` + `sr_gff_file`/`sr_assembly_file` (loader also accepts the
  pre-rename legacy names `lra_*`/`gff_file`/`assembly_file` — METADATA_v2_README §12). SR paths
  are project_k-relative, LR paths absolute; resolved in `build_dp_cohort._abs_path`.
- `project_k` root: `/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw`
- Output: `<project_k>/david/processed/defence_predictor/{smoke,full}/`
- Panaroo fork (convert script): `/home/dca36/workspace/panaroo`

## Pipeline

| Step | Module | Where |
|---|---|---|
| Build manifest (paired_index → v2 join → LR+SR genome records) | `build_dp_cohort.py` | login node (2 TSV reads) |
| Convert GFF+assembly → combined GFF, run DP, write per-protein CSV | `run_defense_predictor.py` | GPU (ampere) |

Output layout: `combined_gff/{lr,sr}/<label>.gff` (cached), `predictions/{lr,sr}/<label>.csv`,
`run_manifest_shard*.tsv` (status + timing). Labels: `Sample` (LR) / `sample_accession` (SR).

## Run order (three-stage protocol §0.2)

1. **Setup** (once): `bash scripts/setup_dp_env.sh` on the login node.
2. **Stage A smoke**: `sbatch scripts/smoke_dp.sh` — 10 ref genomes × 2 arms; validate output +
   measure per-genome seconds.
3. **Full**: build the full manifest (cmd in `scripts/run_dp_cohort.sh` header), size `--array`
   / `--time` from the smoke timing, then `sbatch scripts/run_dp_cohort.sh` (round-robin shards).

## Status

- **2026-06-03** — **Stage A smoke PASS** (job 30014777). 17/17 arms scored (10 LR + 7 SR over 10
  reference genomes). End-to-end on ampere GPU; isolated `.venv-dp` (torch 2.5.1+cu124, bundled
  CUDA — no system module). **~82 s/arm** (23 min for 17 arms). Sensible output (top defensive
  log-odds 10–12; ~25–30 of ~4,800 proteins ≥4 cutoff). Two fixes en route: drop unavailable
  `cuda/12.4` module; `uniquify_locus_tags` (RefSeq dup locus_tag crashed ESM).
  - ⚠️ **Reference-set anchor is degenerate as-is.** All 7 LR/SR pairs in the smoke were the
    `sr_assembly_file == lr_assembly_file` collision (METADATA bug — see below), so LR and SR
    predictions came out **bit-for-bit identical** (zero contrast). 957/2,749 paired rows (83% of
    reference genomes) hit this. Real LR-vs-SR contrast needs the **1,792 rows with a distinct SR
    assembly**. Upstream fix belongs in BacHGT `bac_metadata` (`add_paths_gff_fna_to_metadata.py`).
- **2026-06-02** — Package scaffolded (loader, runner, env setup, smoke + full SLURM scripts).

### Full-run sizing (from smoke)
~82 s/arm × ~5,000 arms ≈ 114 GPU-hours → ~5.7 h/shard across the 20-shard array
(`run_dp_cohort.sh`, `--time=10:00:00`). Comfortable within budget; 80k extension still optional.

## Open checks (resolve at smoke)

- Confirm `metadata_v2` carries **both** `sr_*` and `lr_*` path columns on the LR `Sample` row
  (the loader assumes one paired row per Sample, mirroring Panaroo's `_genome_records_for_row`).
- Confirm the Bakta `seqid`s match the assembly FASTA headers (else convert raises
  "Mismatch between fasta and GFF!" → that arm is skipped + logged in the run manifest).
- Confirm DP's per-genome wall time on `esm2_t30_150M` to size the full array.

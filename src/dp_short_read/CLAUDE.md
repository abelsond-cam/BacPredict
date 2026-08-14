# Task 5 — DefensePredictor on short- vs long-read assemblies (DP-SR)

Run [DefensePredictor](https://github.com/PeterDeWeirdt/defense_predictor) (DeWeirdt et al.,
*Science* 2024) across the paired **long-read (LR) vs short-read (SR)** Klebsiella cohort and
compare per-genome defence-protein calls between the two assembly types. Defence systems
cluster at MGE / contig boundaries — exactly where SR assemblies fragment — so the **LR-vs-SR
delta** is the headline quantity. Output lands in `processed/defence_predictor/`.

This is the baseline arm of Task 5 in [PROJECT_STATE.md](../../PROJECT_STATE.md) ("DP-CG applied to SR — quantify
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

- **Cohort source = `metadata_v2` read directly.** `build_dp_cohort.py` selects rows that carry
  **all four** distinct file paths — `lr_assembly_file`+`lr_gff_file` (LR arm),
  `sr_assembly_file`+`sr_gff_file` (SR partner) — then filters `--cohort {complete,reference,all}`.
  No `paired_index.tsv` join and no `sr_shadow` lookup (both were detours; see history below).
- **Counts (v2 of 2026-06-03):** all-4-files ∧ `is_complete` = **1,454 pairs** (0 collisions);
  `is_reference_genome` subset = **709**. Smoke = `--cohort reference --limit 10`.
- SR paths are project_k-relative, LR paths absolute; resolved in `build_dp_cohort._abs_path`.
  LR arm labelled by `Sample` (GCF/GCA), SR arm by `sr_biosample` (SAMN…, matches the SR FASTA).
- `project_k` root: `/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw`
- Output: `<project_k>/david/processed/defence_predictor/{smoke,full}/`
- Panaroo fork (convert script): `/home/dca36/workspace/panaroo`

## Pipeline

| Step | Module | Where |
|---|---|---|
| Build manifest (v2 → all-4-files + cohort filter → LR+SR genome records) | `build_dp_cohort.py` | login node (1 TSV read) |
| Convert GFF+assembly → combined GFF, run DP, write per-protein CSV | `run_defense_predictor.py` | GPU (ampere) |

Output layout: `combined_gff/{lr,sr}/<label>.gff` (cached), `predictions/{lr,sr}/<label>.csv`,
`run_manifest_shard*.tsv` (status + timing). Each manifest row carries `Sample` (shared by both
arms = the pairing key), `sr_biosample`, and `is_reference`. Pair LR↔SR predictions on `Sample`.

## Run order (three-stage protocol §0.2)

1. **Setup** (once): `bash scripts/setup_dp_env.sh` on the login node.
2. **Stage A smoke**: `sbatch scripts/smoke_dp.sh` — 10 ref genomes × 2 arms; validate output +
   measure per-genome seconds.
3. **Full**: build the full manifest (cmd in `scripts/run_dp_cohort.sh` header), size `--array`
   / `--time` from the smoke timing, then `sbatch scripts/run_dp_cohort.sh` (round-robin shards).

## Status

- **2026-06-12 — Full sweep + matched-recovery analysis DONE.** `--cohort complete` sweep
  (job 30459693): 2,744/2,908 arms scored, ~72 s/arm; **1,290 usable pairs** (554 reference).
  164 LR arms failed `fasta_gff_mismatch` — all the RefSeq/GenBank `NZ_` contig-prefix mismatch
  (GCA-namespace GFF vs GCF-namespace FASTA); left as-is per user. Matched-protein recovery
  (`analyze_lr_vs_sr.py`, exact-AA match, LR calls as truth/AUROC 0.975):
  - **recovery on shared proteins = 94.6% (94.3% reference)** — when a defence protein is intact
    in the SR assembly, SR-DP reproduces the LR call ~95% of the time. Model barely degrades.
  - **lost-to-assembly = 20.0% (19.9% reference)** — 1 in 5 LR-defensive proteins has no exact SR
    match (dropped/frameshifted by SR fragmentation at MGE/contig borders).
  - End-to-end ≈ 0.946 × 0.80 ≈ **76%** of LR defence calls recovered; gap dominated by assembly.
  - Outputs: `full/analysis/lr_vs_sr_recovery_{per_pair.tsv,summary.json}`.
  - ⚠️ Exact-AA match is a conservative lower bound (SR seq errors block exact matches even for
    present genes) — RBH/identity-threshold matching would tighten recovery vs lost-to-assembly.
- **2026-06-03 (corrected) — Stage A smoke PASS with genuine LR-vs-SR contrast** (job 30040166).
  10 reference pairs, 20/20 arms scored, ~80 s/arm. **LR and SR predictions now differ for every
  pair** (e.g. GCF_001718115.2: LR 4770 prot/30 defensive vs SR 4121/10) — the SR arm consistently
  has fewer proteins (fragmented assembly) and a diverging defensive count. This is the intended
  signal; pair LR↔SR on `Sample` for the comparison.
- **Root cause of the earlier degenerate run (resolved):** the first smoke (job 30014777) showed
  bit-for-bit-identical LR/SR because `sr_assembly_file` pointed at the LR/complete FASTA for the
  merged RefSeq pairs (the "957 SR+RefSeq merge"). **Not** a loader bug and **not** unfixable
  metadata — BacHGT rebuilt metadata_v2 (2026-06-03 17:05) to repopulate `sr_assembly_file` with
  the real SR draft. The loader was then simplified to read v2 directly (all-4-files + cohort).
- Fixes carried in: drop unavailable `cuda/12.4` module (venv bundles CUDA); `uniquify_locus_tags`
  (RefSeq dup locus_tag crashed ESM); isolated `.venv-dp` (torch 2.5.1+cu124).
- **2026-06-02** — Package scaffolded (loader, runner, env setup, smoke + full SLURM scripts).

### Full-run sizing (from smoke)
`--cohort complete` ≈ 1,454 pairs → ~2,900 arms × ~80 s ≈ 64 GPU-hours → ~3.5 h/shard across the
20-shard array (`run_dp_cohort.sh`, `--time=10:00:00`). Comfortable within budget.

## Resolved checks
- ✅ metadata_v2 carries distinct `sr_*` vs `lr_*` paths after the 2026-06-03 rebuild.
- ✅ Bakta `seqid`s match assembly FASTA headers (convert succeeded on all 20 arms).
- ✅ DP wall time on `esm2_t30_150M` ≈ 80 s/arm.

## Remaining milestones

Carried out of the retired `ToDo.md` (Task 5) on 2026-08-14, where they were the only record. The
work is **deferred**; this list exists so picking it up does not mean re-planning it.

The premise: DefensePredictor is trained on complete genomes and applied to short-read assemblies
with contig-border hacks. Defence systems sit at MGE boundaries — exactly where short-read contigs
break — so training *on* short reads should do better than patching a complete-genome model.

- [x] Translate CG defence-protein labels onto matched SR assemblies (minimap2)
- [ ] **Baseline: DP-CG applied to SR — quantify the shortfall.** This is the arm the package
      currently implements, and the LR↔SR delta is its headline quantity
- [ ] Retrain **DP-SR** from scratch on SR assemblies, same architecture
- [ ] Add **distance-to-contig-break** as an input feature → DP-SR+break
- [ ] Compare DP-CG vs DP-SR vs DP-SR+break on a held-out SR test set

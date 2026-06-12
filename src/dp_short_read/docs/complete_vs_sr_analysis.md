# Complete-genome vs short-read DefensePredictor analysis

**Question.** How well does [DefensePredictor](https://github.com/PeterDeWeirdt/defense_predictor)
(DeWeirdt et al., anti-phage defence-protein classifier) predict defence proteins on **short-read
(SR) assemblies** vs the matched **long-read / complete (LR) assemblies** of the same isolate?
Defence systems cluster at MGE/contig boundaries — exactly where SR assemblies fragment — so the
LR→SR delta estimates how much short reads cost us. The LR arm is the reference: DefensePredictor's
published AUROC is **0.975**, so LR calls are treated as ground truth.

Run date: **2026-06-12**. Cohort: complete-genome paired LR/SR (`--cohort complete`).

---

## TL;DR

| metric | all pairs | reference subset |
|---|---|---|
| usable pairs (scored on both arms) | **1,290** | **554** |
| proteins per genome (LR / SR, mean) | 4,607 / 4,556 | — |
| defensive calls per genome @ log-odds ≥ 4 (LR / SR, mean) | 26.6 / 26.4 | — |
| **recovery on shared proteins** (SR reproduces LR call) | **94.6%** | **94.3%** |
| **lost to assembly** (LR-defensive with no SR match) | **20.0%** | **19.9%** |
| end-to-end recovery (≈ recovery × retained) | **≈ 76%** | ≈ 76% |

**Two separable findings:**
1. **The model barely degrades.** When a defence protein is *intact* (exact AA sequence) in the SR
   assembly, SR-DefensePredictor reproduces the LR defensive call **~95%** of the time.
2. **The assembly loses ~20% of defence proteins outright** — dropped or frameshifted by short-read
   fragmentation/error at MGE/contig borders. This, not the model, dominates the end-to-end gap.

---

## Absolute gene counts (addressing "only ~25 of 4,500?")

DefensePredictor scores **every** protein (~4,600/genome) and emits a continuous
`mean_log_odds`. "~25" is the count at DP's **recommended stringent cutoff, `log-odds ≥ 4`**
(≈ 98% confidence) — a high-precision threshold for discovering *novel* defence systems, **not**
the total count of defence-associated genes. The count scales strongly with the threshold:

**Mean defensive-called proteins per genome (LR arm, n=1,290), by cutoff:**

| `mean_log_odds ≥` | prob ≥ | mean / genome | median |
|---:|---:|---:|---:|
| 0 | 0.50 | **54.5** | 51 |
| 1 | 0.73 | 42.8 | 40 |
| 2 | 0.88 | 36.3 | 34 |
| 3 | 0.95 | 31.6 | 30 |
| **4** (DP default) | **0.98** | **26.6** | **25** |
| 6 | 0.998 | 19.0 | 18 |
| 8 | 0.9997 | 12.9 | 12 |

So a *Klebsiella* genome carries on the order of **~50 defence-associated proteins at prob > 0.5**,
narrowing to **~25 high-confidence** ones at the strict cutoff. The strict number is what the
recovery table above uses; loosen the cutoff if a more inclusive defence-gene census is wanted.

- Total proteins/genome: LR mean **4,607** (median 4,592, range 4,054–5,523); SR mean 4,556.
- Defensive @ ≥4 per genome: LR mean **26.6** (median 25, range **7–73**); SR mean 26.4 (range 7–60).

---

## What was run

| Cohort stage | value |
|---|---|
| `--cohort complete` = `is_complete` ∧ all 4 file paths present | **1,454 pairs** (709 reference) |
| arms scored | **2,744 / 2,908** (94.4%), ~72 s/arm |
| failures | 164 LR arms, all `fasta_gff_mismatch` (RefSeq `NZ_` vs GenBank contig-name prefix) |
| usable pairs (both arms ok) | **1,290** (554 reference) |

The 164 failures are a RefSeq-vs-GenBank contig-naming mismatch (GFF uses `CP073788.1`, FASTA uses
`NZ_CP073788.1`); recoverable later with a one-line prefix normalize in the convert step.

**Recovery method.** Per pair, proteins are matched between arms by **exact amino-acid sequence**
(re-derived from the cached combined GFFs via DefensePredictor's own GFF parser). With LR calls as
truth: *recovery* = fraction of LR-defensive proteins (that also appear in SR) which SR also calls
defensive; *lost-to-assembly* = LR-defensive proteins with no exact SR match. Exact matching is a
**conservative lower bound** — SR sequencing errors block exact matches even for present genes, so
true recovery is ≥ 94.6% and true assembly loss ≤ 20%. An identity-threshold / reciprocal-best-hit
match would tighten both.

---

## Where everything lives (to reconstruct later)

**HPC root:** `/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/`

**Inputs**
- Cohort + file paths: `final/metadata_v2_all_samples_and_columns.tsv` — columns
  `lr_assembly_file`, `lr_gff_file`, `sr_assembly_file`, `sr_gff_file`, `is_complete`,
  `is_reference_genome`, `Sample`, `sr_biosample`. (Definition: BacHGT `METADATA_v2_README.md`.)

**Outputs** — under `processed/defence_predictor/full/`
- `dp_manifest_full.tsv` — per-arm manifest (`Sample` = pairing key, `arm`, `sr_biosample`, `is_reference`, `gff_abs`, `assembly_abs`).
- `predictions/{lr,sr}/<label>.csv` — per-protein DP output (`product_accession`, `mean_log_odds`, `predicted_defensive`, …). LR label = `Sample`; SR label = `sr_biosample`.
- `combined_gff/{lr,sr}/<label>.gff` — cached Prokka-style GFF+FASTA fed to DP.
- `run_manifest_shard*.tsv` — per-arm status + `n_proteins` + `n_defensive` + timing.
- `analysis/lr_vs_sr_recovery_per_pair.tsv` — per-pair recovery (the source for finer slicing, e.g. vs assembly N50/contig count).
- `analysis/lr_vs_sr_recovery_summary.json` — pooled all / reference summary.

**Code** ([src/dp_short_read/](../)) — run in the isolated `.venv-dp` (`scripts/setup_dp_env.sh`):
- [build_dp_cohort.py](../build_dp_cohort.py) — metadata_v2 → manifest (all-4-files + `--cohort`).
- [run_defense_predictor.py](../run_defense_predictor.py) — convert GFF+assembly → DP → per-protein CSV (`--n-shards`/`--shard-index`).
- [analyze_lr_vs_sr.py](../analyze_lr_vs_sr.py) — matched-protein recovery (exact-AA), per-pair TSV + summary JSON.
- SLURM: [scripts/run_dp_cohort.sh](../scripts/run_dp_cohort.sh) (ampere array, full sweep = job 30459693), [scripts/analyze_lr_vs_sr.sh](../scripts/analyze_lr_vs_sr.sh) (icelake; also runs on the login node in ~1.5 min at `--workers 8`).

**Reproduce the gene-count threshold sweep:** read `mean_log_odds` from each
`predictions/lr/*.csv` and count `>= {0,1,2,3,4,6,8}` per genome (see the table above).

See the task tracker [CLAUDE.md](../CLAUDE.md) for status and the design rationale (why DP needs a
combined GFF+FASTA, the metadata_v2 `sr_assembly_file` fix, the `uniquify_locus_tags` fix).

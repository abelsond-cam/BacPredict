# Unitig GWAS → LR baseline for AMR, vs Bacformer fine-tuning

> **Working plan for [`src/bac_pyseer/ast_gwas/`](../CLAUDE.md).** Status and layout live in that
> `CLAUDE.md`; this file holds the *why*, the decisions and their alternatives, the leakage
> argument, the scale analysis, and the sequenced run order with its `[look]` checkpoints.
>
> **State (2026-08-11, updated):** the package is written and tested — `ruff` clean, Stage A smoke
> green (CPU-only, synthetic fixtures). **Step 4's gate has now passed, on CSD3 rather than
> Isambard** — see *Cluster: CSD3, not Isambard* below, which supersedes Decision 1. The next
> action is step 5, the Kp cohort build. PR [#1](https://github.com/abelsond-cam/BacPredict/pull/1).

## Cluster: CSD3, not Isambard — supersedes Decision 1

Decision 1 chose Isambard because that is where the assemblies and split tables were, and flagged
aarch64 `ggcat` as the step-4 risk. **The work is running on CSD3 instead.** Verified live
2026-08-11:

- **The toolchain risk evaporates.** CSD3 is `linux-64`, exactly what `src/bac_pyseer/pixi.toml`
  already targets, and every binary resolves in the installed env: pyseer 1.4.1, ggcat 2.2.0,
  unitig-caller 1.3.0, mash 2.3, bcftools/samtools 1.23.1. No `platforms` edit, no `cargo` fallback.
- **The real gate turned out to be data availability, and it passes:** Kp 7,080/7,088 genomes
  resolve, TB 36,692/36,692 — and against the *canonical* cohorts (7,088 / 36,692), matching
  Isambard. Sampled paths all dereference.
- **One adaptation was needed.** The module's flat-BioSample-keyed-directory assumption is
  Isambard's. It holds for TB on CSD3 but not for Kp, whose AST genomes are sharded across
  `seb/assemblies_2/` batch directories while `raw/assemblies` holds a GCA-keyed whole-*Klebsiella*
  store. CSD3 ships the join as `raw/assemblies_file_list.tsv`, in the same `Sample<TAB>path` format
  this package emits — so `--file-list` filters it rather than adding a second resolution strategy.

The consequence for the plan is small: `metadata_v2` is *available* on CSD3, so debt item 1
(mash-derived lineage clusters standing in for curated Kp `Sublineage`) could be closed earlier
than expected. Not done here — it would change the population-structure correction mid-pilot, and
that is a statistical decision to take deliberately rather than in passing.

---

## Context

We have two independent lines of work in this repo that have never been compared on the same
footing:

- **`src/bac_pyseer/`** — a mature pyseer GWAS pipeline (GGCAT unitigs → sharded LMM →
  Bonferroni-on-patterns → hit table → Aho-Corasick placement). It has only ever been run on the
  *Klebsiella* **isolation-source** phenotype (blood vs faeces, blood vs respiratory), never on AMR.
- **`src/bacpredict/engine/`** — Bacformer fine-tuning for **AMR**, per antibiotic, for *K.
  pneumoniae* and *M. tuberculosis*, with a fixed 20 % holdout and results in a versioned JSON
  schema.

The question this plan answers: **how good is the Bacformer AMR work, really?** A unitig GWAS is
the right yardstick because it is mechanism-agnostic and sees things a protein-embedding model
structurally cannot — promoter SNPs, IS-element insertions, truncations, plasmid backbone. The
existing CARD/WHO catalogue ceilings only tell us how well *known* determinants do; a unitig
screen tells us how much signal is in the genome at all.

The comparison is only meaningful if both arms are scored on **identical labels and identical
splits**, so this plan pins the unitig-LR arm to the same `<drug>_split.csv` tables the fine-tuned
checkpoints were evaluated on.

Phase 1 (this plan) is a 2 + 2 correctness pilot chosen to bracket the performance range:

| Organism | Drug | Bacformer FT AUROC / AUPRC | Catalogue ceiling AUROC | Why this drug |
|---|---|---|---|---|
| Kp | `ertapenem` | 0.9882 / 0.9937 | 0.9828 (CARD) | Saturated. Positive control — unitigs must also hit ~0.98 or the pipeline is broken. |
| Kp | `colistin` | 0.9094 / 0.8330 | 0.6563 (CARD) | **Largest FT-over-catalogue gap in Kp (+0.253).** Chromosomal `mgrB` truncation / IS insertion — the exact thing a unitig sees and an embedding does not. |
| TB | `rifampin` | 0.9642 / 0.9160 | 0.9666 (WHO) | FT sits *at* the catalogue ceiling. The TB positive control. |
| TB | `ethionamide` | 0.8097 / 0.5962 | 0.8706 (WHO) | FT *underperforms* the catalogue (−0.061), lowest AUPRC of the pilot. `inhA` **promoter** SNP + `ethA` LoF — invisible to a protein-only model. |

> **⚠ Corrected 2026-08-13 — the previous version of this table was wrong on all four FT numbers**
> (ertapenem 0.9870, colistin 0.8072, rifampin 0.9046, ethionamide 0.7742). They predate the
> July re-runs. The errors were not uniform: colistin was understated by +0.10 and rifampin by
> +0.06, which **inverted two of the four selection rationales** — colistin is not the worst Kp
> drug (that is `azithromycin`, 0.7993) and rifampin does not underperform its catalogue, it
> matches it. The drug choices survive on corrected reasoning; the stated reasons did not.
>
> Numbers now come from the deployed checkpoints' own `results.json` — 32 runs, 15–21 Jul 2026,
> all `kfold` fold 0 / seed 1 on `bacformer-large-masked-complete-genomes`. See
> *Comparator provenance* in [`../CLAUDE.md`](../CLAUDE.md) for why a summary panel must never be
> the source, and the three silent failure modes that were found and fixed in
> `collect_comparison` while chasing this.

Note the TB AST column is **`rifampin`** (US spelling), not `rifampicin`. Only the figure
*directory* uses `rifampicin`.

Phase 2 fans out to all 22 Kp and 10 TB drugs. That is why the unitig set and the kinship are
built **once per organism over the whole AST cohort** and reused per drug — the per-drug cost
collapses to one pyseer run plus a cached submatrix extraction.

## Decisions taken (each is a switch, not baked in)

Recorded with their alternatives so they can be revisited without re-deriving the reasoning. None
is load-bearing on the others.

1. **Cluster: Isambard (aarch64).** That is where the AST assemblies (Kp 9,891 / TB 39,494),
   labels and split tables physically are. CSD3 has been down since 27 Jun 2026 and every existing
   GWAS artifact there is cohort-specific to the isolation-source run anyway, so nothing is lost by
   rebuilding. **Risk:** `src/bac_pyseer/pixi.toml` pins `pyseer`/`ggcat`/`unitig-caller` to
   `linux-64`. Step 0 below is a platform probe with a documented fallback.
2. **GWAS scope: train+validate only.** Feature selection never sees the holdout. See
   *Leakage control* — this is the single most important design point.
3. **Population structure: mash/ANI for both organisms**, kinship *and* lineage clusters derived
   from the same distance matrix. Self-contained; no CSD3 `metadata_v2` and no 39k-genome
   TB-Profiler run required. This is the ANI approach requested.
   *Worth knowing:* the invasion GWAS used core-SNP kinship in production, and the docs record
   mash as *"a trade-off, not a fix"* for that phenotype. The concern is that a whole-genome
   k-mer kinship partly absorbs accessory/HGT structure — which for AMR is signal, not nuisance.
   The within-lineage permutation null (step 10) is what tells us whether that bit us.
4. **Branch: `claude/unitig-amr-bacformer-comparison-y1xvch`, based on
   `refactor/consolidate-engine`** — not `main`, which predates the July 2026 consolidation and
   contains none of this code.
5. **Two unitig→LR implementations are kept, not unified.** A sibling agent built the same
   comparison for the **invasion** phenotype (`kleb_iso_source/{subset_cohort_trainval,
   unitig_presence_model}.py`); both arrived independently at the same leakage conclusion. Theirs
   stays the invasion comparator, this the AMR one. Their findings are ported across by import
   (`DEFAULT_C_GRID`, `paired_delta_ci`) and pattern (CSC accumulation) — see the `CLAUDE.md`
   section on that relationship. Unifying is the right pre-publication move but would rewrite a
   live sibling module, so it is logged as debt instead.

Per repo `CLAUDE.md` §0.5 (3–4 agents share the checkout), work here stages only paths under
`src/bac_pyseer/ast_gwas/` and `tests/bac_pyseer/` — never `git add -A`.

## Pre-flight fixes and technical-debt register

### Pre-flight: `pheno_var` was unreachable from the CLI — **fixed** (`53b2d5d`)

`pyseer_postprocess.significant_hits()` computes the ranking key

```python
hits["var_explained_pct"] = af * (1 - af) * beta**2 / pheno_var * 100
```

with `pheno_var: float = 0.249`. Audit findings:

- **0.249 is correct for the isolation-source work it was written for.** Both contrasts were
  balance-sampled: blood/faeces 7,176 / 6,426 → p=0.5276, p(1−p)=0.2492; faeces/respiratory
  4,432 / 4,737 → p=0.4834, p(1−p)=0.2497. The docstring says so explicitly. And since it is a
  constant divisor on a monotone ranking key, it cannot reorder hits — **no existing iso-source
  conclusion is affected**, including the Manhattan marker sizes (`ve * 12 + 15`, also a constant
  rescale).
- **The defect is that it is unreachable.** `run()` calls
  `significant_hits(assoc, threshold, pval_col, pos_label, neg_label)` at line 546 without
  `pheno_var`, and the argparse block has no `--pheno-var`. By contrast
  `build_blood_resp_concordance.py:144` *does* expose `--pheno-var`. So the one place it matters
  most cannot be corrected without a code change.
- **For AMR it is materially wrong.** Prevalence per drug: ertapenem 0.634 → p(1−p)=0.232;
  colistin 0.278 → 0.201; TB rifampin ~0.31 → 0.214. Using 0.249 understates VE by ~7 % for
  ertapenem and ~20 % for colistin, and distorts any cross-drug VE comparison — which is precisely
  what Phase 2's 32-drug league table is.

**Fixed by** adding `--phenotype-tsv` / `--phenotype-column` to `pyseer_postprocess.py`, computing
`pheno_var = p(1−p)` from the *actual samples used in that GWAS* (i.e. train+validate only), thread
it through `run()` → `significant_hits()`, and record the value used in
`<stem>_gwas_summary.json`. Keep the literal `0.249` as the fallback when no phenotype file is
supplied, so existing iso-source outputs remain byte-reproducible rather than silently changing.
Add `--pheno-var` as an explicit override. Unit-test that the computed value matches `p(1−p)` and
that the fallback path is unchanged.

### Also found in the same audit (lower priority, recorded so they are not lost)

| Issue | Where | Impact on this work |
|---|---|---|
| `--gff` was `required=True` even when `--feature-mode unitigs`, where it is unused — **fixed** (`53b2d5d`) | `pyseer_postprocess.py` | Would otherwise have needed a dummy GFF on every AMR run. |
| Virulence cross-reference is a *Klebsiella hypervirulence* panel (`rmp`, `iuc`, `ybt`, `clb`, capsule keywords…) | `pyseer_postprocess.py` `VIRULENCE_*` | Dormant in unitig mode (gene mapping is skipped). Becomes wrong in Phase 2 if we map hits to genes — wants a CARD / WHO-V2 panel for AMR. |
| `pattern_group` keys perfect-LD blocks on rounded `(af, β, p)` strings (`round(6)` + `f"{p:.3e}"`) | `pyseer_postprocess.py` | Distinct variants can collide into one block, deflating the hit count. Matters because `pattern_group` is one of our LR comparators. Worth a tolerance-based grouping later. |
| `DEFAULT_CONTIG = "NC_009648"` / `DEFAULT_CONTIG_LEN = 5_315_120` (Kp MGH 78578) | `pyseer_postprocess.py:41-42` | CLI-overridable, unused in unitig mode. No action. |

### Technical debt to clean up before publishing the code base

Recorded per your instruction — fine for now, but must not ship as-is:

1. **Mash-derived lineage clusters stand in for curated lineage labels.** We cluster the mash
   distance matrix because Kp `Sublineage`/`Clonal group` live only in `metadata_v2` on CSD3 and TB
   lineages do not exist at all. The publishable version uses Kleborate sublineages for Kp and
   TB-Profiler `main_lineage`/`sub_lineage` for TB. This affects `--lineage-clusters` and the
   permutation null, so it is a methods-section item, not just tidiness.
2. **`linux-aarch64` added to `src/bac_pyseer/pixi.toml`** (and possibly a `cargo`-built GGCAT
   outside the pixi env) to get the toolchain onto Isambard. A published environment should be a
   single clean solve, and the lock file is already stale relative to the toml (`ggcat` and
   `unitig-caller` do not appear in `pixi.lock`).
3. **`pheno_var` fallback of `0.249`** retained for backward compatibility — should become a
   required input once the iso-source outputs have been regenerated.
4. **Two near-duplicate unitig→LR implementations** — `kleb_iso_source/unitig_presence_model.py`
   (invasion) and this package (AMR). They now share a `C` grid and the paired bootstrap by import,
   but the matrix build and the fit are still written twice. Unify into one engine with thin
   per-phenotype callers before publication; needs the sibling agent coordinated with.

## Module layout

One new subpackage, organism-agnostic, because one codebase must serve 2 organisms × 32 drugs.
This follows `src/bac_pyseer/CLAUDE.md`'s "one subfolder per GWAS task" convention while keeping
`--organism {kp,tb}` as a flag rather than forking the package.

```
src/bac_pyseer/ast_gwas/
├── CLAUDE.md                     # task notes, status, running log (repo convention)
├── __init__.py
├── resolve_ast_assemblies.py     # AST cohort → Sample<TAB>assembly_path TSV (flat-dir)
├── build_ast_phenotype.py        # <drug>_split.csv → pyseer phenotype.tsv (train+validate only)
├── mash_kinship.py               # sketch/triangle orchestration + per-drug similarity subsetting
├── lineage_from_distances.py     # mash distances → Sample<TAB>cluster (hierarchical)
├── unitig_design_matrix.py       # hit unitigs → sparse CSR presence over ALL samples
├── unitig_lr.py                  # sparse LR → results.json (schema v1.2) + eval_scores.npz
├── collect_comparison.py         # unitig-LR + FT + catalogue → one comparison table
└── scripts/
    ├── probe_toolchain.sh        # step 0: can pyseer/ggcat install on aarch64?
    ├── build_cohort_once.sh      # per organism: reflist → GGCAT → mash sketch/triangle → clusters
    └── run_drug.sh               # per drug: phenotype → kinship → sharded LMM → design → LR
```

### Existing modules extended (flags, not forks)

| File | Change | Why |
|---|---|---|
| `src/bac_pyseer/kleb_iso_source/ggcat_to_pyseer.py` | add `--max-samples N` to `convert()` | Drops near-universal unitigs (af > 0.99) that pyseer cannot test anyway but which dominate file size. Critical for the highly clonal TB cohort, where core unitigs carried by ~all 38k samples would otherwise blow the matrix up. |
| `src/bac_pyseer/kleb_iso_source/scripts/unitig_lmm_sharded_job.sh` | accept `CLUSTERS_TSV` env var instead of the inline `Sublineage` heredoc | The heredoc hard-codes Kp metadata columns. Everything else in this script is already env-parameterised. |
| `src/bac_pyseer/kleb_iso_source/pyseer_postprocess.py` | add `--phenotype-tsv` / `--phenotype-column` / `--pheno-var`; thread through `run()` → `significant_hits()`; record the value in the summary JSON; make `--gff` optional in unitig mode | See *Pre-flight fixes* above. Blocking for AMR — the 0.249 default is currently unreachable and wrong for every drug. |

### Existing modules reused unchanged (library calls, not CLI)

- `unitig_placement.extract_hit_submatrix()` — the one-time `pigz | awk` streaming hash-join over
  the giant matrix, cached forever (returns `-1` if the output exists). Called as a **library
  function**, deliberately bypassing `unitig_placement.py --phase select`, whose CLI pulls in
  `resolve_assembly_paths.resolve` and with it the CSD3 `metadata_v2` hard-wiring. We need the
  submatrix, not the Aho-Corasick placement.
- `unitig_placement.shard_expected()` → `dict[Sample, set[unitig_idx]]`, which *is* the sparse
  presence matrix.
- `pyseer_postprocess.py --feature-mode unitigs` — `count_unique_patterns`,
  `bonferroni_threshold`, `genomic_inflation`, `significant_hits` (ranked by `var_explained_pct`,
  with `pattern_group`/`n_in_pattern` perfect-LD blocks). Already organism-agnostic in unitig mode.
- `mash_dist_to_kinship.parse_triangle()` / `to_similarity()` — already organism-agnostic
  (sample id = filename stem).
- `sample_nonsig_unitigs.py` — af-matched non-significant unitigs, as the negative control.
- `permute_phenotype_within_lineage.py`, `genomic_inflation_by_af.py` — the calibration protocol.
- From `bacpredict.engine`: `splits.load_splits.load_splits`, `config.resolve_data_root` /
  `OrganismConfig`, `finetune.metrics.{compute_full_metrics, youden_threshold,
  build_results_payload, write_results_json}`, `segment_amr_lr.fit_lr.LOGREG_KW`, and the sparse
  `sp.hstack(..., format="csr")` pattern from `finetune.linear_baselines._build_design_matrix`.

**Do not** call `fit_lr.fit_score_step` or `ref_catalogues.base.score_onehot_frame` on the unitig
matrix — both densify via `.to_numpy()` and will not survive 10⁴–10⁶ columns. We reuse their
*estimator settings and metric block* so the numbers stay comparable, but with a CSR design matrix.

## Pipeline

Everything lands under `$DATA/processed/pyseer_ast/<organism>/`, mirroring the existing
`pyseer_iso_source/` layout.

```
processed/pyseer_ast/<kp|tb>/
├── cohort/
│   ├── assembly_refs.txt              # Sample<TAB>path
│   └── cohort_manifest.json
├── unitigs/                           # BUILT ONCE PER ORGANISM
│   ├── unitigs_ggcat.fa.gz  + .colors.dat
│   ├── color_names.jsonl  + colormap_ranges.csv
│   └── unitigs.pyseer.gz              # the pyseer --kmers matrix
├── structure/                         # BUILT ONCE PER ORGANISM
│   ├── mash_sketch.msh
│   ├── mash_triangle.txt
│   ├── similarity.tsv                 # pyseer --similarity
│   └── lineage_clusters.tsv           # Sample<TAB>cluster
└── <drug>/
    ├── phenotype.tsv                  # train+validate ONLY
    ├── gwas/  <drug>.assoc, patterns.txt, lmm_cache.npz, shards/
    ├── <drug>_hits_annotated.tsv, <drug>_gwas_summary.json, pyseer_qq.png
    ├── design/  hits_submatrix.tsv (cached), id_map.tsv, presence.npz
    └── lr/  results.json, eval_scores.npz, coefficients.tsv
```

**Step 0 — toolchain probe** (`scripts/probe_toolchain.sh`, login node, minutes). Add
`linux-aarch64` to `src/bac_pyseer/pixi.toml` `platforms` and try to solve. `pyseer` is pure
Python and should resolve; its deps (numpy/scipy/scikit-learn/statsmodels/pysam/DendroPy) all have
aarch64 builds. **`ggcat` is a Rust binary and bioconda's aarch64 coverage is the open risk** —
fallback is `cargo install ggcat` (Rust cross-compiles to aarch64 cleanly, and the existing
scripts already put `$HOME/.cargo/bin` on `PATH`). Precedent: `src/bacpredict/apps/kleb/pixi.toml`
already lists `linux-aarch64`. **Gate: do not size any compute until this passes.**

**Step 1 — cohort + assembly list.** `resolve_ast_assemblies.py --organism kp --check-exists`
takes the universe from `<root>/processed/train_kleb_ast/binary_ast_with_split.csv` (already
pruned to samples with embeddings) and resolves `<root>/raw/{kleb_ast,tb}/assemblies/<Sample>.fa.gz`.
Emits `Sample<TAB>path`, the only contract `run_ggcat_unitigs.sh` needs. Records the drop count.

**Step 2 — GGCAT unitig build, once per organism.** Reuses `run_ggcat_unitigs.sh` with Isambard
headers. `K=31`, `SVAL=2`, `--min-samples` = 1 % of cohort, **`--max-samples` = 99 %** (new).

**Step 3 — mash kinship, once per organism.** `mash sketch` over the assembly list then
`mash triangle` → `mash_dist_to_kinship.py` → `similarity.tsv`. The repo has no `mash sketch`
sbatch (it was run by hand for the invasion cohort), so `mash_kinship.py sketch` is new; the
`build_cohort_once.sh` driver submits it.

**Step 4 — lineage clusters, once per organism.** `lineage_from_distances.py` — average-linkage
hierarchical clustering of the mash distance matrix, cut to give clusters of ≥ `--min-size 100`
(matching the `min_sl_size=100` used in production), everything smaller collapsed to `other`.
Emits the same `Sample<TAB>cluster` format `build_lineage_clusters.py` produces, so it is a
drop-in for `pyseer --lineage-clusters` and for the permutation null.

**Step 5 — per-drug phenotype.** `build_ast_phenotype.py --organism kp --drug ertapenem` reads
`splits/<drug>_split.csv` via `load_splits()`, keeps **`train_ids + validate_ids` only**, and
writes the pyseer format (first column literally `samples`, second `<drug>_label`). Asserts
`holdout_ids ∩ phenotype == ∅` and writes the assertion into the manifest.

**Step 6 — sharded pyseer LMM, per drug.** `run_drug.sh` wraps the proven three-phase
`prep → array → combine` chain with `--dependency=afterok`. Shard count scales with matrix size.

**Step 7 — postprocess.** `pyseer_postprocess.py --feature-mode unitigs` → Bonferroni threshold on
unique patterns, λ, QQ plot, `<drug>_hits_annotated.tsv` ranked by `var_explained_pct`.

**Step 8 — design matrix.** `unitig_design_matrix.py` takes the hit sequences, calls
`extract_hit_submatrix()` (one pass over the big matrix, cached), then `shard_expected()` over
**all** cohort samples — train, validate *and* holdout — to build a CSR of shape
`(n_samples, n_hit_unitigs)`. Holdout genomes get feature *values* here; they never contributed to
feature *selection*. Also builds the af-matched non-significant control set via
`sample_nonsig_unitigs.py`.

**Step 9 — logistic regression.** `unitig_lr.py`: fit on `train_ids` with `C` swept on
`validate_ids` (see the module docstring for why the repo-pinned `C=1.0` overfits unitig features;
the pinned fit is kept as a secondary), pick the Youden threshold on `validate_ids`, score
`holdout_ids` with
`compute_full_metrics`. Emit `results.json` via `build_results_payload` /`write_results_json`
(schema v1.2, `model.name_or_path = "unitig_lr"`, `split.source = "split_table"`, `extra` carrying
`n_unitigs`, `n_patterns`, `bonferroni_threshold`, `lambda_gc`) plus `eval_scores.npz` in the same
shape `evaluate.py` writes, so existing plotting code reads it directly.

Also fit two comparators on the identical splits: **(a)** the af-matched non-significant unitig
set — should be ~0.5, and if it is not, the splits leak population structure; **(b)** one-hot of
`pattern_group` representatives only (one feature per perfect-LD block), which controls for hit
count being inflated by LD.

**Step 10 — calibration.** `genomic_inflation_by_af.py` for λ per af bin, and
`permute_phenotype_within_lineage.py` for the within-lineage permutation null. On the invasion
GWAS observed λ reached 23.8 at high af while the permutation null stayed ≈1 at every af, which is
what established that the inflation was genuine within-lineage signal rather than structure. The
same test decides whether the mash kinship is adequate here.

**Step 11 — comparison.** `collect_comparison.py` joins the unitig-LR `results.json` against the
recorded FT and catalogue numbers into one table and appends a `unitig-LR` rung to the existing
per-drug ladder CSVs, so `engine/plots/plot_amr_ladder.py` renders it next to `BacF FT mean` and
`CARD all-determinant ceiling` with no plotting changes.

## Leakage control

This is what makes the comparison honest. Every stage, and what it sees:

| Stage | Sees holdout **genomes**? | Sees holdout **labels**? | Note |
|---|---|---|---|
| GGCAT unitig build | Yes | **No** | Unsupervised. Building features over the whole cohort is standard and is exactly what the embedding arm does. |
| mash kinship / lineage clusters | Yes | **No** | Unsupervised. |
| pyseer phenotype file | **No** | **No** | Restricted to `train + validate`; asserted. |
| af / MAF filter | **No** | — | pyseer computes af over the phenotyped samples only, so the filter is train-only. |
| Bonferroni pattern count | **No** | **No** | Patterns come from the train+validate run, so the threshold is train-derived. |
| Hit selection | **No** | **No** | The whole point. |
| LR fit | train only | train only | |
| Operating threshold | validate only | validate only | Youden on validate, as `fit_score_step` does. |
| Final scoring | holdout | holdout | Once, at the end. |

No standardisation is applied (binary features), so there is no scaler to fit on the wrong split.

**Caveat to state in the write-up, not to fix:** the repo's splits are a uniform random shuffle
with **no lineage or clonal-group grouping** — near-clonal genomes can straddle train and holdout.
Both arms inherit this, so the *comparison* is fair, but both absolute numbers are optimistic
relative to a lineage-blocked split. The lineage clusters from step 4 make a secondary
lineage-blocked evaluation cheap, and it is worth reporting as a sensitivity analysis once the
headline numbers exist.

## Scale and risk

| | Kp | TB |
|---|---|---|
| Cohort (assemblies) | 9,891 → ~9,724 with labels | 39,494 → ~38,257 |
| Genome size | ~5.3 Mb, large accessory | ~4.4 Mb, near-clonal |
| Expected unitigs | ~4–6 M (13.6k Kp genomes gave 6.28 M) | far fewer, but carrier lists ~38k long |
| Matrix size | ~40–60 GB | **the open question** — `--max-samples` is what keeps this sane |
| Per-drug labelled n (train+val) | ertapenem ~1.7k, colistin ~1.1k | rifampin ~29k, ethionamide ~8k |

The Kp drugs are *small* — a few thousand phenotyped samples means the LMM is trivial and the cost
is dominated by streaming the matrix. **TB `rifampin` at n≈29k is the genuine risk:** peak LMM RAM
goes as `cpu × n²`, so at `--cpu 8` that is ~55 GB for the rotation matrices alone, on top of
parsing a 29k² similarity matrix. Mitigations, in order of preference: subset the kinship per drug
before writing it (avoids ever materialising 38k²); request a full Isambard node rather than the
default per-socket allocation; drop `--cpu` and add shards; and if all else fails, subsample
rifampin's train+validate to ~10k stratified on phenotype, documented in `extra`. Run
**ethionamide before rifampin** so the TB pipeline is proven at n≈8k first.

Other things that could go wrong: `ggcat` has no aarch64 conda build (step 0 gates this);
TB unitig carrier lists make the matrix larger than Kp's despite lower diversity; mash kinship
under-corrects for Kp AMR because plasmid content correlates with both kinship and phenotype
(step 10 detects this — if the permutation null is inflated, we need a core-genome kinship and
CSD3 or a fresh reference alignment); and ertapenem failing to reach ~0.98 would mean a pipeline
bug rather than a scientific result.

## Testing

Tests live under `tests/bac_pyseer/ast_gwas/`, following `tests/bac_pyseer/test_gwas.py`'s style of
testing pure logic against small inline fixtures:

- `test_pipeline.py` — holdout exclusion (the leakage guard, as an assertion) and the fractional
  `ast_label` drop; the design matrix's CSR construction, sample ordering, all-zero rows for
  non-carriers, `--dedupe-patterns`, and join-failure vs out-of-cohort accounting; the LR recovering
  a planted signal at AUROC 1.0, the `C` sweep selecting on validate, the pinned-`C` secondary, and
  the results JSON validating against the v1.2 required-key sets.
- `test_cohort_and_structure.py` — assembly resolution (including the `raw/kleb_ast` vs `raw/tb`
  naming asymmetry) and lineage clustering (block recovery, small clusters collapsing to `other`).
- `test_comparison_and_kinship.py` — the per-drug kinship/distance subsets, the hard error on a
  sample absent from the kinship, and the comparison table including the paired CI aligning by
  sample id rather than position.
- `tests/bac_pyseer/test_ggcat_to_pyseer.py` — the `--max-samples` cap drops near-universal unitigs
  and nothing else.

Extending the existing `tests/bac_pyseer/test_gwas.py`:

- `pheno_var` computed from a phenotype file equals `p(1−p)` for that file's samples; the
  no-phenotype fallback still yields exactly the current 0.249 behaviour (so existing iso-source
  outputs are unchanged); an explicit `--pheno-var` overrides both; and the value used is written
  into the summary JSON.
- `--gff` omitted in `--feature-mode unitigs` no longer raises.

**Stage A smoke** (repo §0.2, must run CPU-only): the Python stages run end-to-end against a
synthetic 10-genome pyseer-format matrix fixture with no cluster and no GGCAT, giving a real
`results.json`. Then a 50-genome HPC smoke exercising GGCAT + mash + pyseer before any full run.

**End-to-end verification:** `ruff check src/bac_pyseer/`, `pytest tests/bac_pyseer/`, then the
Kp `ertapenem` full run as the positive control — if unitig-LR does not reach ~0.98 there,
stop and debug rather than proceeding to the other three drugs.

## Sequenced work plan

Checkpoints marked **[look]** are where a human should read results before continuing.

- [x] **1. Branch based on `refactor/consolidate-engine`**, with the updated base merged in.
- [x] **2. Pre-flight fixes** — `pheno_var` reachable + optional `--gff` (`53b2d5d`),
      `--max-samples` cap (`a877215`), torch-free engine metrics (`8709b85`).
- [x] **3. Package + tests + Stage A smoke** (`5d54117`, `2bd56bd`), reconciled with the sibling
      invasion comparator (`45122af`). 200 tests, CPU-only, no cluster.
- [x] **4. Toolchain probe — passed on CSD3** (2026-08-11), not Isambard. The aarch64 `ggcat` risk
      did not apply; the binding gate was data availability, which also passed (Kp 7,080/7,088,
      TB 36,692/36,692). Required one adaptation — `--file-list` for Kp assembly resolution. See
      *Cluster: CSD3, not Isambard* above.
- [ ] **5. Kp cohort build** — `ORGANISM=kp FILE_LIST=<rds>/david/raw/assemblies_file_list.tsv
      scripts/build_cohort_once.sh` (reflist → GGCAT → mash → clusters). **[look]** unitig count,
      matrix size, how many near-universal unitigs the `--max-samples` cap dropped, cluster size
      distribution.
- [ ] **6. Kp `ertapenem`** — `ORGANISM=kp DRUG=ertapenem scripts/run_drug.sh`, through to
      `results.json`. **[look]** the positive control: expect ~0.98. Anything much lower is a
      pipeline bug, not a result — stop and debug rather than running the other three.
- [x] **7. Kp `colistin`.** **[look]** DONE. ⚠ The numbers this step was written against were retracted: Bacformer is **0.9094** (not 0.807) and the CARD ceiling **0.6563** (not 0.649). Result: unitig-LR 0.9188 vs FT 0.9100, delta +0.0088 [−0.0171, +0.0347] — **a tie**. See `PROJECT_STATE.md` §3.3.
- [ ] **8. TB cohort build, then `ethionamide`** (smaller n first, ~8k vs rifampin's ~29k). **[look]**
- [ ] **9. TB `rifampin`**, with the scale mitigations above. **[look]**
- [ ] **10. Calibration** — λ by allele frequency + the within-lineage permutation null — and the
      comparison table across all four drugs. **[look]** decide whether the mash kinship held up.
      If λ_perm is inflated, the hits are structure and the kinship needs replacing.
- [ ] **11. Fan out** to the remaining 20 Kp + 8 TB drugs as SLURM arrays.

> **This requires a machine with SSH access to the cluster** (CSD3 in practice — see the note
> superseding Decision 1). Steps 4 onward cannot run from a Claude
> Code cloud session: that sandbox has no `ssh`/`scp`/`rsync` binary, an empty `~/.ssh`, and proxied
> egress. Teleport the session to a local terminal
> (`claude --teleport <session-id>`) or drive the scripts by hand.

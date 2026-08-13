# ast_gwas — unitig GWAS → LR baseline for AMR

Task folder under [bac_pyseer](../CLAUDE.md). Unlike `kleb_iso_source` (one phenotype, one
organism), this one is **organism-agnostic** — `--organism {kp,tb}` — because the same pipeline has
to serve 2 organisms × 32 antibiotics. See the root [CLAUDE.md](../../../CLAUDE.md) for §0
conventions.

> **The working plan is [`docs/PLAN.md`](docs/PLAN.md)** — the decisions and their alternatives, the
> full leakage argument, the scale/risk analysis, and the sequenced run order with its `[look]`
> checkpoints. Start there before running anything; this file is the layout and status summary.

## Why

To answer *how good is the Bacformer AMR fine-tuning, really?* A unitig GWAS is the right yardstick
because it is mechanism-agnostic: it sees promoter SNPs, IS insertions, truncations and plasmid
backbone that a protein-embedding model structurally cannot. The CARD/WHO catalogue ceilings only
say how far *known* determinants get; a unitig screen says how much signal is in the genome at all.

Phase 1 is a 2 + 2 pilot bracketing the performance range:

| Organism | Drug | Bacformer FT AUROC | Catalogue ceiling | Why |
|---|---|---|---|---|
| Kp | `ertapenem` | 0.9870 | 0.977 (CARD) | Saturated — the positive control. If unitig-LR is not ~0.98 here, the pipeline is broken. |
| Kp | `colistin` | 0.8072 | 0.649 (CARD) | Worst Kp drug; chromosomal `mgrB` truncation / IS insertion is exactly what a unitig sees. |
| TB | `rifampin` | 0.9046 | 0.967 (WHO) | FT *underperforms* the catalogue; unitig-LR should land near the RRDR one-hot (0.960). |
| TB | `ethionamide` | 0.7742 | 0.871 (WHO) | Worst TB drug, widest catalogue gap; `inhA` **promoter** + `ethA` LoF are invisible to a protein-only model. |

Then fan out to all 22 Kp and 10 TB drugs.

> **The TB AST column is `rifampin` (US spelling), not `rifampicin`.** Only the figure *directory*
> uses `rifampicin`.

## The one thing that must not break

The GWAS phenotype carries **train + validate only**. Unitig selection therefore never sees a
holdout label — not through the af filter, not through the unique-pattern count behind the
Bonferroni threshold, not through the betas. Holdout genomes re-enter only in the design matrix, as
unsupervised presence/absence, and are scored once at the end.

If that ever breaks, the LR is fitted on features chosen with knowledge of its own test set and its
AUROC is **not** comparable to the fine-tune's. `build_ast_phenotype` asserts it twice (in the
manifest and by re-reading what it wrote), and
`tests/bac_pyseer/ast_gwas/test_pipeline.py::test_phenotype_never_contains_holdout` guards it.

`--splits train,validate,holdout` deliberately reproduces the leaky classic-GWAS framing; the
manifest flags it loudly. Worth running once to quantify the gap — never worth reporting as the
headline.

## Layout

| Module | Does |
|---|---|
| `resolve_ast_assemblies.py` | AST cohort → `Sample<TAB>assembly_path` (flat, BioSample-keyed; no `metadata_v2`) |
| `build_ast_phenotype.py` | `<drug>_split.csv` → pyseer `--phenotypes` TSV, train+validate only |
| `mash_kinship.py` | `mash sketch`/`triangle` once per organism; per-drug similarity + distance subsets |
| `lineage_from_distances.py` | mash distances → `Sample<TAB>cluster` for `--lineage` and the permutation null |
| `unitig_design_matrix.py` | hits → sparse genomes × unitigs CSR over **all** split genomes |
| `unitig_lr.py` | fit train → Youden on validate → score holdout → `results.json` (schema v1.2) |
| `collect_comparison.py` | unitig-LR + FT + catalogue → one table per organism |

Scripts: `probe_toolchain.sh` (step 0 gate) · `build_cohort_once.sh` (per organism) ·
`run_drug.sh` (per drug).

## Reuse, not reimplementation

The GWAS itself is `kleb_iso_source`'s, driven with different env vars: `run_ggcat_unitigs.sh` for
the build, `run_unitig_lmm_sharded.sh` for the sharded LMM, `pyseer_postprocess.py` for the
threshold/λ/hits, and `unitig_placement.extract_hit_submatrix` for the cached big-matrix pass. Those
files gained env-overridable roots (`REPO`/`PYSEER`/`OUT_DIR`/`TMP`/`MATRIX`/`GWAS_DIR`/`SIM`/
`DIST`/`CLUSTERS_TSV`) with defaults preserving their existing behaviour — no forks.

The read-out reuses the engine: `load_splits`, `LOGREG_KW` (L2, lbfgs, no class weight),
`compute_full_metrics`, `build_results_payload`/`write_results_json`. The design matrix stays
**sparse** — `fit_score_step` and `score_onehot_frame` densify and will not survive 10⁴–10⁶ columns —
but the estimator settings and metric block are shared, so the numbers stay comparable.

### Relationship to `kleb_iso_source/unitig_presence_model.py`

A sibling agent built the same comparison for the **invasion** phenotype. Both arrived independently
at the same leakage conclusion (their `subset_cohort_trainval.py`, my `build_ast_phenotype.py`).
Rather than fork or rewrite, this package **imports** from theirs:

- `DEFAULT_C_GRID` — so both comparators sweep the same grid.
- `paired_delta_ci` — the paired bootstrap on the model-vs-model AUROC delta.

and adopts their CSC accumulation pattern in `unitig_design_matrix` (can't call their builder
directly: my column order is the GWAS rank order in `id_map.tsv`, which is what lets a coefficient
be traced back to its GWAS row).

**Their measured finding changed this package's estimator.** They found that ~33k correlated binary
unitig columns against ~9.5k training rows overfit badly at the repo-pinned `C=1.0`. So `unitig_lr`
sweeps `C` on **validate** and reports that as the headline, while also fitting the pinned `C=1.0`
as a secondary (`extra.pinned_C_metrics`) so the comparison against the catalogue ceilings — which
are fitted at that `C` — stays like-for-like.

## Status

- [x] `pheno_var` made reachable in `pyseer_postprocess` (it was hardcoded at 0.249 and unreachable;
      correct for the ~50:50 iso-source cohorts, wrong for every AMR drug — ertapenem 0.232,
      colistin 0.201)
- [x] `--max-samples` cap in `ggcat_to_pyseer` (near-universal unitigs are untestable and dominate
      matrix size; matters most for near-clonal TB)
- [x] Package + Stage A smoke (CPU-only, synthetic fixtures, no GGCAT/pyseer/cluster)
- [x] Reconciled with the sibling invasion comparator: C swept on validate (their measured
      overfitting finding), paired bootstrap CI in `collect_comparison`, CSC accumulation
- [x] Step 0 toolchain probe — **passed on CSD3, 2026-08-11** (see *Running on CSD3*). The aarch64
      `ggcat` risk was Isambard-specific and does not apply: CSD3 is `linux-64`, exactly what
      `pixi.toml` targets. pyseer 1.4.1 · ggcat 2.2.0 · unitig-caller 1.3.0 · mash 2.3.
- [x] Step 0b cohort resolution on CSD3 — Kp 7,080/7,088, TB 36,692/36,692
- [ ] Kp cohort build → `ertapenem` (positive control) → `colistin`
- [ ] TB cohort build → `ethionamide` (smaller n first) → `rifampin`
- [ ] Calibration: λ by af + within-lineage permutation null
- [ ] Fan out to the remaining 20 Kp + 8 TB drugs

## Running on CSD3

The plan was written for Isambard. CSD3 differs in three ways that matter, all verified live on
2026-08-11.

**1. The canonical task root is `bac_ast_prediction/`, not `processed/`.** Both exist; the cohort
sizes disambiguate. Set `BACPREDICT_DATA_ROOT` to the former or you silently get the deprecated
May cohort.

| Sheet | Rows | |
|---|---|---|
| `<rds>/david/bac_ast_prediction/processed/train_{kleb,tb}_ast/binary_ast_with_split.csv` | 7,088 / 36,692 | **canonical**, matches Isambard |
| `<rds>/david/processed/train_{kleb,tb}_ast/binary_ast_with_split.csv` | 6,838 / 36,684 | deprecated |

**2. Kp assemblies need `--file-list`; TB does not.** TB is flat and BioSample-keyed exactly as the
module assumes (`raw/tb/assemblies/SAMEA*.fa.gz`, 36,692/36,692 resolve on the default path). Kp has
no such directory on CSD3 — `raw/assemblies` is the whole-*Klebsiella* 81k store keyed by **GCA
accession**, so a filename join resolves *zero*. The AST genomes are sharded across
`seb/assemblies_2/klebsiella_pneumoniae__NN/<BioSample>.fa.gz`, and CSD3 ships the join already made
as `raw/assemblies_file_list.tsv` (95,131 rows, `Sample<TAB>path` — the format this module emits).
So pass `FILE_LIST=<rds>/david/raw/assemblies_file_list.tsv`: 7,080/7,088 resolve, 8 missing.

**3. Cluster knobs.** `ACCT=FLOTO-PROJECT-K-SL2-CPU PART=icelake-himem QOS=`. Prefer a node
*fraction* — himem nodes are 76 cores at 6,760 MB/core, and requesting all 76 (or ~all memory)
forces whole-node placement and a slower start.

Input volume by count × one-file: Kp ≈ 10.6 GB (7,080 × ~1.5 MB), TB ≈ 51 GB (36,692 × ~1.4 MB).

> `mash` is **not** declared in `src/bac_pyseer/pixi.toml` — it arrives as a transitive dependency of
> `pyseer`, whose conda recipe requires it (alongside `bedops`/`bedtools`/`bwa`/`pysam`/`dendropy`).
> It is pinned in `pixi.lock`, so this is not a live break, but we call `mash` directly and should
> declare what we use. Left alone for now because `pixi.toml` is shared and adding a dependency
> would force a lock re-solve while a sibling agent has uncommitted lock changes.

## Known technical debt (clean up before publishing)

1. **Mash-derived lineage clusters stand in for curated labels.** Kp `Sublineage` lives only in
   `metadata_v2` on CSD3; TB lineages do not exist until TB-Profiler is run over ~39k assemblies.
   The publishable version uses Kleborate sublineages (Kp) and TB-Profiler lineage (TB). This is a
   methods-section item, not just tidiness.
2. **`src/bac_pyseer/pixi.toml` needs tidying** — only if Isambard is used again, plus one item
   that applies regardless. `linux-aarch64` would have to be added to `platforms` (and possibly a
   `cargo`-built GGCAT alongside) to get the toolchain onto Isambard; **on CSD3 none of that is
   needed**. Regardless of cluster, `mash` should be declared explicitly rather than relied on as a
   pyseer transitive (see *Running on CSD3*). The "`ggcat`/`unitig-caller` missing from `pixi.lock`"
   note is now stale — a sibling agent re-solved the lock and both are present, though that
   re-solve is still uncommitted.
3. **`pheno_var` keeps a 0.249 fallback** for backward compatibility; should become required once
   the iso-source outputs have been regenerated.
4. **Two near-duplicate unitig→LR implementations** — `kleb_iso_source/unitig_presence_model.py`
   (invasion) and this package (AMR). They now share a grid and the bootstrap by import, but the
   matrix build and the fit are still written twice. Unify into one engine with thin per-phenotype
   callers before publication; needs the sibling agent coordinated with.

## Watch out for

- **TB `rifampin` is the scale risk.** ~29k train+validate genomes; peak LMM RAM ≈ `cpu × n²`, so
  ~55 GB at `--cpu 8` for the rotation matrices alone. Run `ethionamide` (~8k) first. Mitigations:
  the per-drug kinship subset (avoids ever materialising 38k²), a whole node, more shards at lower
  `--cpu`, and — last resort — a documented stratified subsample.
- **Mash kinship may under-correct for AMR**, because plasmid content correlates with both kinship
  and phenotype. The within-lineage permutation null is the check, and it is not optional: if
  λ_perm is inflated, the hits are structure, not signal.
- **Splits are not lineage-blocked.** Both arms inherit this, so the comparison is fair, but both
  absolute numbers are optimistic. The lineage clusters make a lineage-blocked sensitivity analysis
  cheap once the headline numbers exist.

# ast_gwas — unitig GWAS → LR baseline for AMR

Task folder under [bac_pyseer](../CLAUDE.md). Unlike `kleb_iso_source` (one phenotype, one
organism), this one is **organism-agnostic** — `--organism {kp,tb}` — because the same pipeline has
to serve 2 organisms × 32 antibiotics. See the root [CLAUDE.md](../../../CLAUDE.md) for §0
conventions.

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

The read-out reuses the engine: `load_splits`, `LOGREG_KW` (C=1.0, L2, lbfgs, no class weight),
`compute_full_metrics`, `build_results_payload`/`write_results_json`. The design matrix stays
**sparse** — `fit_score_step` and `score_onehot_frame` densify and will not survive 10⁴–10⁶ columns —
but the estimator settings and metric block are shared, so the numbers stay comparable.

## Status

- [x] `pheno_var` made reachable in `pyseer_postprocess` (it was hardcoded at 0.249 and unreachable;
      correct for the ~50:50 iso-source cohorts, wrong for every AMR drug — ertapenem 0.232,
      colistin 0.201)
- [x] `--max-samples` cap in `ggcat_to_pyseer` (near-universal unitigs are untestable and dominate
      matrix size; matters most for near-clonal TB)
- [x] Package + Stage A smoke (CPU-only, synthetic fixtures, no GGCAT/pyseer/cluster)
- [ ] Step 0 toolchain probe on the active cluster — **gates everything below**
- [ ] Kp cohort build → `ertapenem` (positive control) → `colistin`
- [ ] TB cohort build → `ethionamide` (smaller n first) → `rifampin`
- [ ] Calibration: λ by af + within-lineage permutation null
- [ ] Fan out to the remaining 20 Kp + 8 TB drugs

## Known technical debt (clean up before publishing)

1. **Mash-derived lineage clusters stand in for curated labels.** Kp `Sublineage` lives only in
   `metadata_v2` on CSD3; TB lineages do not exist until TB-Profiler is run over ~39k assemblies.
   The publishable version uses Kleborate sublineages (Kp) and TB-Profiler lineage (TB). This is a
   methods-section item, not just tidiness.
2. **`linux-aarch64` added to `src/bac_pyseer/pixi.toml`** (and possibly a `cargo`-built GGCAT
   outside the pixi env) to get the toolchain onto Isambard. The lock is already stale relative to
   the toml — `ggcat` and `unitig-caller` do not appear in `pixi.lock`.
3. **`pheno_var` keeps a 0.249 fallback** for backward compatibility; should become required once
   the iso-source outputs have been regenerated.

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

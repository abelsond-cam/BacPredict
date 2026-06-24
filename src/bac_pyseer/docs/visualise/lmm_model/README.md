# blood vs faeces — variant LMM (figures & tables)

Per-contrast artifacts for the **blood/faeces** variant axis. The narrative + interpretation live in the
single report [`../../PROGRESS.md`](../../PROGRESS.md); this directory is a store, not a write-up.

- `blood_vs_faeces_hits_annotated.tsv` — the 110 hits, invasion-oriented (`invasive_af`, `abs_beta`,
  `var_explained_pct`, `consequence`, `display_name`, `lineage`).
- `blood_vs_faeces_gwas_summary.json` — n=13,602, λ=0.562, Bonferroni threshold, hit counts.
- `manhattan_lmm_bigsl_af1*.png`, `qq_lmm_bigsl_af1.png` — Manhattan + QQ diagnostics.

Reproduce: `kleb_iso_source/scripts/run_pyseer.sh` (the GWAS) → `scripts/regen_progress.sh` (tables + figures).

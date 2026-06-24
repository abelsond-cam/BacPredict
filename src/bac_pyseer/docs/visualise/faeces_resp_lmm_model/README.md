# respiratory vs faeces — variant LMM + cross-contrast (figures & tables)

Per-contrast and cross-contrast artifacts. Narrative + interpretation: [`../../PROGRESS.md`](../../PROGRESS.md).

- `respiratory_vs_faeces_hits_annotated.tsv` (88 hits), `respiratory_vs_faeces_gwas_summary.json`
  (n=9,169, λ=0.498), `manhattan_resp_faeces_lmm.png`, `qq_resp_faeces_lmm.png`.
- `blood_resp_concordance_union.tsv` — **§2 replication**: the union of Bonferroni hits in either niche
  looked up in *both* associations (85/86 independent patterns concordant, p≈1.1×10⁻²⁴, r²=0.78).
- `lineage_breadth.tsv` — **§3** per-hit sub-lineage breadth (species-wide / single-SL / few / rare).
- `cross_contrast_overlap_blood_vs_resp.tsv` — the 33 genes significant in both contrasts (feeds the
  cross-axis `replicates_invasion` flag).

Reproduce: `scripts/run_pyseer.sh` (`PAIR=faeces_respiratory`) → `build_blood_resp_concordance.py` +
`build_lineage_breadth.py` (HPC) → `scripts/regen_progress.sh`.

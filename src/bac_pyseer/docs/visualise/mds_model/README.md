# MDS fixed-effects model — blood-vs-faeces GWAS (abandoned approach)

These are the diagnostics from the **fixed-effects + MDS** structure correction
(`pyseer --distances jaccard_distances.tsv --max-dimensions 10`), run on the
country-balanced `sampled_country_2_1_all` cohort (n=13,602; 372,239 variants at
af 1–99%; big-SL `--lineage`, ≥100-sample Sublineages kept, run `gwas_bigsl`, job 30658211).

**Verdict: MDS fixed-effects is the wrong method for this cohort — abandoned in favour of LMM.**

## Why (the numbers)

- **Genomic inflation λ = 4.34** (well-calibrated ≈ 1) — severe under-correction.
  The QQ plot shows the *entire* distribution lifted above the null, not just the tail.
- The **scree** explains it: the top MDS axis captures only **0.33%** of relatedness
  variance, and the spectrum flattens almost immediately. Cumulative variance captured:

  | K (MDS axes) | cumulative variance |
  |---|---|
  | 10 (used) | **1.6%** |
  | 50 | 4.2% |
  | 100 | 7.3% |
  | 200 | 12.8% |

  With K=10 we corrected for 1.6% of the structure and let ~98% leak into every test.
  Relatedness here is spread across thousands of comparable dimensions (~1,345 sublineages),
  so no low-K projection captures it — fixed-effects MDS cannot work for this cohort.
- The 6,657 "hits" are therefore dominated by lineage confounding (e.g. 1,475 attribute to
  hypervirulent SL23). The capsule/siderophore signal is biologically plausible but
  **not interpretable at λ=4.34** — kept here only as a record, not a result.

## Files

- `qq_bigsl_af1_k10.png` — QQ of structure-adjusted p (λ=4.34, genome-wide inflation).
- `manhattan_bigsl_af1_k10.png` — Manhattan on NC_009648.
- `scree_mds_bigsl.png` — MDS eigenvalue scree (reconstructed from the saved projection;
  sum-of-squares per axis = eigenvalue). The smoking gun.
- `blood_vs_faeces_hits_annotated.tsv` — the 6,657 significant hits, gene-mapped +
  virulence cross-ref (kept for reference; **not a validated result**).
- `blood_vs_faeces_gwas_summary.json` — λ, thresholds, counts.

## Next

LMM (`pyseer --lmm --similarity <kinship>`, FaST-LMM) — uses the full kinship as a random
effect (all 4,712 dimensions, no K truncation). The right tool for clonal data.
See the `lmm_model/` outputs once that run completes.

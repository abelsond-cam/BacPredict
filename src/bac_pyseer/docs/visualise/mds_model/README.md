# MDS structure correction — abandoned (kept as a negative-result record)

The fixed-effects MDS correction (`pyseer --distances` + `--max-dimensions K`) was tried for the
blood-vs-faeces variant GWAS and **abandoned**: at K=10 the genomic-inflation λ = 4.34 (severe
under-correction), and the scree (`scree_mds_bigsl.png`) shows K=10 captures only 1.6 % of the
relatedness (K=200 only 12.8 %) — no low-K MDS projection captures this clonal cohort's structure.
The 6,657 "hits" were lineage-confounded and are not used.

**Method of record is the LMM** (`pyseer --lmm --similarity <core-SNP kinship>`, FaST-LMM): it uses
the full kinship as a random effect (no K truncation) — the right tool for clonal data. See
[`../lmm_model/`](../lmm_model/) and the hub [`../../PROGRESS.md`](../../PROGRESS.md).

Kept here: this README + `scree_mds_bigsl.png` (the diagnostic that shows why MDS fails). The
unvalidated hit table, summary JSON, and Manhattan/QQ figures were removed (recoverable from git
history).

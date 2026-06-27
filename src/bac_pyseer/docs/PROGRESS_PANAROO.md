# *Klebsiella* invasion GWAS — gene presence/absence (Panaroo) axis

> **Axis docs.** Gene presence/absence axis. Variant axis: [`PROGRESS_VARIANTS.md`](PROGRESS_VARIANTS.md);
> unitig axis: [`PROGRESS_UNITIGS.md`](PROGRESS_UNITIGS.md); cross-axis overview: [`PROGRESS.md`](PROGRESS.md).

**Status: placeholder — may not run.** A Panaroo gene-presence/absence (GPA) GWAS (`pyseer --pres` over a
pangenome gene × sample matrix) would test invasion association at the **gene gain/loss** level —
complementary to the unitig axis, which sees the same accessory content at finer (sub-genic, sequence)
resolution.

**Two reasons it may not happen:**
1. **Panaroo doesn't scale** to this set — pangenome construction is the bottleneck at ~80k genomes (and is
   heavy even at the ~14k cohort); we do not currently have a pangenome for this cohort. A per-sublineage
   Panaroo + merge would be the only tractable route.
2. **The unitig axis may already capture the accessory signal** ([`PROGRESS_UNITIGS.md`](PROGRESS_UNITIGS.md)).
   Gene gain/loss is a coarser view of what unitigs resolve at sequence level, so GPA may add little once the
   unitig hits are mapped to plasmid/chromosome (geNomad).

**Decision: deferred.** Revisit only if the unitig + geNomad mapping leaves a specific **gene-level** question
GPA would answer, *and* a scalable pangenome (or per-sublineage Panaroo) becomes available. If run, it reuses
the same pyseer LMM + calibration/reliability protocol (locus filter → af-stratified λ → within-lineage
permutation) as the other axes. Inputs TBD. Cross-task tracker: [`../../../ToDo.md`](../../../ToDo.md).

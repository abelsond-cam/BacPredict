# Parked — detecting mixed / contaminated assemblies with masked-gene loss

> **Parked, never started.** Carried out of the retired `ToDo.md` (Task 4) on 2026-08-14. No package
> exists; recreate as `src/admixture/` if this is picked up.

## The idea

Use Bacformer's **masked-gene loss** to detect admixtures of close-relative strains that differ in
their accessory genome (HGT, IS elements, plasmids). Core-gene tools such as CheckM2 are blind to
this by construction: two strains of the same species have near-identical core genes, so a mixture
looks clean. The accessory genome is where they differ, and a genome model that predicts genes from
their neighbours should find such a mixture *surprising*.

**Open question that gates the work:** confirm with Maciej that the objective is genuinely masked-gene
and not next-gene. The whole approach depends on which it is.

## Milestones

- [ ] Build a **fragmentation null model** — loss as a function of N50 and contig count. This is the
      guard that stops locus-level loss from merely tracking contig breaks, which is the obvious
      confound in short-read assemblies
- [ ] Whole-genome and locus-resolved masked-gene loss across the short-read assemblies
- [ ] If there is signal: map high-loss loci to independently quantified HGT regions
- [ ] If there is signal: a **synthetic admixture experiment** — mix reads in known ratios, re-assemble,
      and test whether loss tracks the ratio. This is what would turn a correlation into a usable
      detector

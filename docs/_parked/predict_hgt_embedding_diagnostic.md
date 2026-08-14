# Parked — `predictHGT`: does Bacformer preserve HGT identity?

> **Parked, never started.** Carried out of the retired `ToDo.md` (Task 6) on 2026-08-14 so the plan
> survives the file it lived in. The `src/predict_hgt/` package stub was retired in the July 2026
> consolidation — recreate it if this is picked up. Paths below are pre-consolidation; see
> `PROJECT_STATE.md` §2.

## Why it was worth doing

A diagnostic, not a training task. It asks whether Bacformer's contextualised embedding **preserves
or erases** the identity of horizontally acquired genes. That matters because the whole programme
hypothesis is that Bacformer reads HGT well — this would test the representation directly rather
than inferring it from downstream AUROC.

It needs only the refreshed Bacformer weights plus HGT/MGE annotations consumed from the sibling
`BacHGT` repo, so it can run in parallel with anything else.

## Aim 1 — is HGT identity preserved?

- [ ] Pull HGT-region annotations from `BacHGT` (MOB-suite + ISEScan + geNomad prophages)
- [ ] Embed all proteins with the refreshed Bacformer
- [ ] Marker-protein nearest-neighbour analysis — KPC / NDM / OXA-48 / *mcr-1* / *tetA* / *iutA* /
      *rmpA*, against housekeeping controls
- [ ] UMAP coloured by HGT vs chromosomal, and by host species
- [ ] Centroid separation score: HGT-vs-chromosomal against a host-context baseline
- [ ] Layer-sensitivity scan — early vs late Bacformer layers
- [ ] **Optional comparator:** raw ESM-C. Does contextualisation *erase* HGT identity?

## The decision this feeds

- [ ] **Decision point.** The Aim 1 outcome sets the embedding source for Aim 2: Bacformer if it is
      HGT-preserving, a DefensePredictor-style representation if it turns out to be a context
      attractor. Document the implication for cross-species HGT-aware work either way.

## Aim 2 — boundary detection

- [ ] Pull ISEScan + MGEfinder ground truth from `BacHGT`; train a per-protein head; evaluate on a
      held-out set and on short-read assemblies

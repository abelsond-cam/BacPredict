# Task 6 — `predictHGT`: do Bacformer embeddings preserve HGT identity?

**Status: diagnostic / exploratory. Can run in parallel with other tasks** — only needs the refreshed Bacformer weights plus an HGT-annotation pipeline (the latter pulled from the sister `BacHGT` module, not re-run here). **No new training for the main experiment.**

See the root [CLAUDE.md](../../CLAUDE.md) for §0 global conventions, and [BacPredict_Training_Plan.md](../../BacPredict_Training_Plan.md) §6 for the long-form plan.

## Aim

Two linked questions, in priority order:

1. **(Main thrust — new framing.)** When a protein lives on a plasmid, ICE, or other HGT element, **what does its Bacformer embedding actually represent?** Two hypotheses:
   - **HGT-preserving.** Bacformer recognises HGT-borne proteins as a distinct semantic class — their embeddings cluster with the same/similar proteins on HGT elements in other genomes, regardless of host. → Bacformer is the right tool for cross-species HGT-aware prediction.
   - **Context-attractor.** The transformer's attention pulls each protein's embedding toward its local genomic neighbours, so a KPC on a plasmid in one isolate ends up looking like its chromosomal neighbours in that isolate, rather than like KPCs in other isolates. → Bacformer effectively *erases* HGT origin in favour of host context.

   Context-attractor is the more likely outcome given the architecture. If true, **Bacformer is not the right LLM for HGT-specific work**, and any cross-species HGT-aware prediction should use DefensePredictor-style embeddings instead (raw ESM-C + concatenated flanking-gene embeddings + explicit gene-level features). This conclusion would matter for how we frame all the AMR results — particularly the HGT-stratum performance in Tasks 1 and 2.

2. **(Original framing — kept as a sub-aim.)** Train a head that predicts the **boundaries of HGT insertions** from per-protein embeddings (IS-flanked context windows, MGEfinder integration, IS-family gap tokens). This task **survives either outcome of Aim 1** — what changes is the embedding source:
   - If **HGT-preserving** → use Bacformer embeddings as originally planned.
   - If **context-attractor** → switch to DefensePredictor-style embeddings. The boundary task itself is no less interesting — we just route through a representation that hasn't homogenised the signal away.

## Why this matters for the rest of the programme

- **AMR stratification (Tasks 1 & 2).** If Bacformer is a context-attractor, the gain on the HGT stratum will come from *genomic context that co-travels with the acquired gene* (replicon backbone, ICE scaffold, IS flanks), not from recognition of the resistance gene itself. Either is a valid signal but they are *different findings* — reviewers will ask which is which.
- **Cross-species generalisation.** A context-attractor model generalises poorly across species for HGT-borne phenotypes (host context changes). An HGT-preserving model would generalise well. Determines whether Bacformer is the right backbone for future cross-genus work.
- **Isolation source (Task 3).** Hypervirulence determinants (*rmpA/rmpA2*, *iuc*, *iro*, hvKp virulence plasmid backbone, ICE*Kp*) are HGT-borne. Same question, same implications.

## Plan / workflow milestones — Aim 1 (main experiment)

1. **Pull HGT-region annotations from the sister `BacHGT` module** — work already done there (MOB-suite for plasmid replicons + ISEScan for IS elements + other annotation steps). Do **not** re-run these tools in BacPredict; just consume the BacHGT outputs. Per-protein labels wanted: `chromosomal-core` / `plasmid` / `IS-flanked` / `other-MGE`, with element-level identifiers (replicon type, IS family) where available.
2. **Embed all proteins with refreshed Bacformer** complete-genomes model. Save per-protein contextual embeddings.
3. **Core diagnostic — nearest-neighbour analysis by ortholog group.** For a set of marker proteins that occur both HGT-borne and chromosomally across the corpus (candidates: *bla*KPC, *bla*NDM, *bla*OXA-48, *mcr-1*, *tetA*, *iutA* from *iuc*, *rmpA*; plus a few core single-copy housekeeping proteins as negative controls): pull every instance from the corpus, compute pairwise embedding distances. For an HGT-borne instance, are its nearest neighbours in embedding space (a) other HGT-borne instances of the same protein in other genomes (= HGT-preserving) or (b) its chromosomal neighbours in the same genome (= context-attractor)?
4. **UMAP visualisation** of the same marker proteins, coloured by HGT vs chromosomal location and by host species. Visual confirmation of the quantitative result.
5. **Centroid distance test.** For each marker, compute centroid of all HGT-borne instances vs centroid of all chromosomal instances. Distance ÷ within-group spread = clean separation score. Compare to a context-baseline: same calculation done on the protein's chromosomal neighbours. If HGT-vs-chromosomal separation is small but host-context separation is large → direct evidence of context-attractor behaviour.
6. **Sensitivity to embedding layer.** Repeat (3)–(5) at multiple Bacformer layers, not just the final layer. Earlier layers may retain HGT identity even if later layers homogenise it — informs which layer to take embeddings from for HGT-aware downstream tasks.
7. **Optional comparator: ESM-C alone.** Run the same diagnostics on raw ESM-C protein embeddings (no genomic context). If ESM-C preserves HGT identity better than Bacformer, that's direct evidence the *contextualisation step* in Bacformer is what erases HGT origin — supports the DefensePredictor-style architecture as the right tool for HGT work.

## Plan / workflow milestones — Aim 2 (boundary labelling, brief)

Pull ISEScan + MGEfinder ground truth from `BacHGT`. Train a small head on per-protein embeddings to predict "is this within ±k genes of an HGT insertion boundary?" Evaluate on held-out genomes, then apply to short-read assemblies. Two input-representation options on the table: (a) MGEfinder-recovered IS proteins inserted into the protein list at the correct genomic position (positional anchoring via flanking genes); (b) IS-family gap tokens — one learned embedding per IS family, simpler and works when MGEfinder fails on composite transposons.

**Embedding source chosen by the outcome of Aim 1:** Bacformer if HGT-preserving, DefensePredictor-style (raw ESM-C + concat flanking + gene features) if context-attractor. Same architecture and training set either way; only the input representation changes.

## What we will and will not do

- **Will do:** the embedding diagnostic (Aim 1, steps 1–6). Fast — no training — and the answer materially shapes how we frame everything else.
- **Will do:** the boundary-detection head (Aim 2), with the embedding source selected by the outcome of Aim 1.
- **May do, depending on Aim 1:** the ESM-C comparator (step 7).
- **Won't do (explicitly out of scope):** *fixing* Bacformer to preserve HGT identity better — would require retraining with HGT-aware objectives (separate special tokens for plasmid/ICE, contrastive loss against HGT-vs-chromosomal pairs, etc.). Important to flag as a future direction, but a much bigger lift than this PhD.

## Three-stage testing protocol (recap of root §0.2)

Mostly N/A for Aim 1 (no training). For Aim 2 once it begins:

| Stage | Scale | Folds × seeds | Where |
| :-- | :-- | :-- | :-- |
| **A. Smoke** | n=10 | 1 × 1 | MacBook M1 CPU |
| **B. Overfit** | n=10, train=test | 1 × 1 | Local or HPC interactive |
| **C. Full** | full data | 1 × 1 | GPU HPC SLURM, ~36 h |

## Reporting

Aim 1: nearest-neighbour purity per marker, UMAP, centroid separation scores, per-layer sensitivity table, optional ESM-C comparator. **Decision point**: explicit declaration of Bacformer's behaviour (HGT-preserving vs context-attractor) and the implication for cross-species HGT-aware work.

Aim 2 (boundary task): per root §0.4 — AUROC, AUPRC, sens, spec for "within ±k of boundary" classification.

## Files in this folder

None yet. Empty Python package stub.

## Running notes

<!-- Agent appends here as work proceeds. -->

---
name: syntology-synteny-map-project
description: "Sibling project (nuna → being renamed syntology) = bakta-independent homology+synteny classification, to anchor non-coding regions"
metadata: 
  node_type: memory
  type: project
  originSessionId: 287d1555-8fdc-4241-9446-ecac54a396be
---

A **sibling project in a separate folder, `nuna` (being renamed `syntology`)**, is developing a
**bakta-independent classification system based on homology + synteny** — the goal is to name/anchor
genomic regions (genes AND non-coding) by conserved synteny/homology rather than by whatever unstable
`gene=` symbol bakta happens to assign. David flagged it as directly relevant to the BacPredict
non-coding capture problem.

**Why it matters here (the inhA-promoter capture bug — see [[baclm-build-defects]]):** the mabA-inhA
operon promoter (ethionamide/isoniazid −15) is the 59 bp region 5′ of `fabG1`, but its other flank is an
**unnamed AbiEi antitoxin CDS** → our per-IGR store's "both flanks must be consistently-named core genes"
filter DROPS it, and `fabg1`/`inha` never appear as flanks in the TB rankings. The region is embedded but
un-rankable and un-locatable. **David's chosen fix (2026-07-16):** name a regulatory region by the GENE
IT SITS UPSTREAM OF (TSS-anchored, keyed `upstream:<gene>`), not by the flank pair — "the most flexible,
lets us evaluate more regions we're currently dropping." **syntology** is the longer-term, bakta-independent
way to do this anchoring robustly across genomes. Keep it in mind as the eventual home of the synteny map;
for now BacPredict does a lightweight gene-upstream-window version itself.

Decisions locked with David (2026-07-16), plan the-cambridge-hpc-and-dreamy-thacker:
- Embed regulatory regions **both** ways: (i) **fragments** matching bakta's catalogue-call granularity,
  and (ii) **whole_igr** — an important comparison. Mean-pool (NOT max/concat) — pooling kept simple.
- Immediate goal: **prove on canonical examples (inhA promoter) that the embeddings are meaningful** when
  we capture + name the region correctly — before the full re-embed. The full upstream-window re-embed
  waits for the GPU (CP-B); build the pipeline now.

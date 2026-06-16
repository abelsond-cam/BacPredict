# TB-AST read-out design brief — pooling, attention heads, and per-gene logistic regression

*A self-contained hand-off for a design discussion (e.g. a Claude app chat). It states where the
science stands and lays out the architecture options to weigh. No prior context needed.*

---

## 1. The problem

We predict antibiotic susceptibility (**AST**) from bacterial genomes using **Bacformer** genome
embeddings. The programme hypothesis: **Bacformer reads HGT / gene-acquisition resistance well but is
comparatively blind to chromosomal point mutations (SNPs).**

- ***Klebsiella* (Kp) AST is strong** — resistance is often HGT/acquired (a whole gene appears).
  Bacformer Kp ciprofloxacin AUROC ≈ **0.979**, ~9 pp above Kleborate's published marker model.
- ***M. tuberculosis* (TB) rifampicin AST underperforms** — resistance is a **single rpoB RRDR point
  mutation**. Deployed model ≈ **0.905**, below a SNP-only ceiling of ~0.96–0.97.

TB rifampicin is the clean test case: one causal residue, one gene (*rpoB*), in a genome of ~4,000
proteins.

## 2. The pipeline (two means in series)

1. **ESM-C** embeds each protein's residues → a 960-d **per-protein** vector (residue→protein **mean**
   pool inside ESM-C). These are precomputed and saved.
2. **Bacformer** is a transformer over the genome's ~4,000 protein vectors (genes as "words"),
   producing **contextualised** per-protein tokens.
3. **Genome head**: a mask-normalised **mean** of the ~4,000 tokens → one genome vector → linear →
   AST logit.

So a single causal protein is diluted by **two** averaging steps; the protein→genome mean is ~1/4,000.

## 3. Where the signal is — the localization ladder (TB rifampicin, full ~38k eval AUROC)

| Representation | AUROC | Reading |
|---|---|---|
| SNP-only ceiling | ~0.96–0.97 | what a perfect rpoB read achieves |
| one-hot RRDR genotype | **0.960** | the causal variant is (almost) fully predictive |
| frozen pooled **ESM-C rpoB** vector | **0.971** | ESM-C **encodes the mutation** in the rpoB protein vector |
| frozen **Bacformer rpoB token** | **0.953** | the contextualised rpoB token still carries it |
| frozen **genome mean** | **0.788** | **averaging 4,000 proteins destroys it** |
| **mean-pool fine-tuned** (deployed) | **0.905** | FT partly recovers the mean |
| learned **gated-MIL attention pool** (e2e) | **0.868** | **worse than the dumb FT mean** — the paradox |
| frozen gated-MIL | ~0.78 | |

**Headline:** the monogenic SNP signal is *present* at the protein level (0.95–0.97). It is destroyed
by **pooling**. The failure is in the **read-out**, not the embedding.

## 4. The attention paradox

- Bacformer's **own internal self-attention** *does* concentrate on rpoB (rpoB lands in the top
  ~0.2% of attended proteins, especially in resistant genomes). The backbone "knows" rpoB matters.
- But the prediction **head's learned pooling attention apparently does *not* route to rpoB** — it
  collapses to something mean-like, which is why the learned gated-MIL pool (0.868) *loses* to the
  plain fine-tuned mean (0.905).
- A diagnostic measuring the head-pool's rpoB weight directly is running now (see §8).

## 5. Why HGT survives the mean but a SNP does not

- **HGT/acquired** = a whole gene present/absent → shifts *many* features and is reflected in the
  genome mean (and is essentially a presence/absence signal). → Kp AST is strong.
- **SNP** = a tiny perturbation of *one* protein among ~4,000 → the mean washes it out (~1/4,000).

So a fix must do one of: **(a) select** the causal gene (attention that actually routes), **(b)
inject** an explicit per-gene signal, or **(c) drop** the ~3,990 irrelevant genes.

## 6. The 1000-genome manifest confound (a caveat that shapes evaluation)

A balanced 1,000-genome set (≈500 rpoB-mutant *R* + 500 WT) scored **0.977** — but this is
**confounded**: rpoB resistance is **lineage-clonal**, so selecting on rpoB genotype segregates
lineage → R vs WT differ genome-wide, and the model separates them on *population structure*, not
rpoB. Evidence: a per-gene LR on this set "predicts" rifampicin from **katG (an INH gene, 0.935)** and
**embB (an EMB gene, 0.908)** — co-resistance/MDR-lineage markers. **Lesson for any design: judge it
on the full, naturally-distributed eval, never a genotype-balanced subset.**

## 7. The per-gene logistic-regression strand (built)

For each gene, an L2 **logistic regression** on the gene's 960-d **ESM-C** vector predicting
resistance, leakage-safe (5-fold **out-of-fold** on train; full-fit for val/eval). On the manifest
(core genes, single-copy >95% prevalence): rpoB 0.9996 (circular — the set defines R by rpoB), then
katG 0.935, embB 0.908, sharp falloff (only 15/1,738 genes clear 0.8). It is already wired as an
optional **attention-head panel channel** (steers the gate, or enters the pooled value) — the
supervised cousin of an unsupervised "surprisal" panel.

## 8. The design space to discuss

**A. Extend per-gene LR coverage.** Move from core genes (>95% prevalence) to **all genes >10%
prevalence**, allowing a **0** (or a learned "absent" embedding) when the gene is absent →
presence-absence-aware per-gene channel.

**B. Bacformer-free benchmark.** A **multivariate LR on the common genes** (one feature per common
gene — its per-gene LR prob, or its pooled ESM-C vector). **Bypasses Bacformer entirely.** Weak on
many fronts — relies on **Prokka gene-name homology**, is **within-species**, and is blind to
novel/divergent genes — but a clean **yardstick**: if the Bacformer models can't beat it,
contextualisation adds nothing for AST.

**C. Per-gene LR prob as an explicit attention-head panel** (implemented). Does the supervised channel
help the head route to the causal gene on the *full* eval? (Cheaper to scale to the full cohort than
the unsupervised surprisal panel, which needs a ~1,800-GPU-h genome-wide scan — currently deferred.)

**D. Selective / sparse / structured heads.**
- **Drop meaningless genes** so the head attends ≪4,000 (hard top-K, or a learned soft gate that
  zeros most genes).
- **Structured concatenation read-out**: top-1, top-2, top-3 individual gene embeddings **+** mean of
  top-10 **+** mean of top-50 **+** overall mean → e.g. a **6×960** head feeding the classifier.
  Explicitly captures "the few causal genes + the background."
- **Multi-head pool** (a learnable-query `nn.MultiheadAttention`, already scaffolded) vs the
  single-query gated-MIL pool.

## 9. Tensions / open questions for the discussion

- **Selection vs injection.** Should the model *learn to select* the causal gene (attention — general,
  but has so far failed to route), or be *handed* it (panel/LR channel — works, but risks a
  species/drug-bound crutch that won't generalise)?
- **Generalisation.** Per-gene LR and the multivariate-LR benchmark are Prokka-homology- and
  species-bound. Bacformer's promise is cross-species / novel-gene generalisation — a benchmark that
  wins within-species may miss the point.
- **Leakage.** Any supervised per-gene step must be out-of-fold cross-fit (we do this).
- **Target metric.** Beat the FT mean (0.905)? Match one-hot RRDR (0.960)? Reach the SNP ceiling
  (~0.96–0.97)? And on the *full* eval, not the confounded 1,000.
- **Mechanism stratification.** The programme wants HGT-vs-chromosomal stratified reporting
  (Kleborate / WHO catalogue labels) — the chosen design should make that legible.

## 10. In flight right now (this code agent is following these)

- **A** — score the manifest baseline checkpoint on the **full 7,074-genome** eval. Stays ~0.97 → the
  read-out is genuinely that good; drops to ~0.8 → the 0.977 was the lineage confound.
- **C** — measure whether that checkpoint's head-pool actually attends rpoB or sits mean-like.

These two settle whether the 0.977 is real and whether the gate routes to rpoB — directly informing
which §8 option is worth pursuing.

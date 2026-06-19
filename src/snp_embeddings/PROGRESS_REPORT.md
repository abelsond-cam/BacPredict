# TB-AST read-out — progress report

*Shareable write-up of Task 7 (`snp_embeddings`): where Bacformer's mean-embedding AMR prediction loses
signal, what fixes it, and the precise map of what it still can't reach. Operational detail (paths, models,
file map) is in [`CLAUDE.md`](CLAUDE.md); the living plan is
`~/.claude/plans/i-d-like-to-start-crystalline-allen.md`. Per-drug figures live in
[`docs/visualisations/tb_<drug>/`](docs/visualisations/).*

## The question

*M. tuberculosis* rifampicin AST underperforms (deployed Bacformer eval AUROC ~0.905) while *Klebsiella*
AST is strong. Programme hypothesis: **Bacformer reads HGT / gene-acquisition resistance well but is
comparatively blind to chromosomal point mutations and non-coding resistance** — TB's regime (the
rpoB/RRDR SNP). This task finds *where* the single-residue signal is lost, what fixes it, and exactly which
mechanisms remain out of reach.

---

## 1. The read-out gap, and the fix (rifampicin deep dive)

1. **The signal is present in the protein embeddings, and destroyed by pooling.** Frozen ESM-C rpoB scores
   **0.97** and Bacformer's contextualised rpoB token **0.95**; the protein→genome **mean-pool collapses it
   to 0.79**. Fine-tuning the mean recovers it only to **0.905**.
2. **A learned attention pool does not rescue it (0.868, *below* the mean).** On honest full data the
   gated-MIL head never concentrates — it collapses to a ~uniform mean. On a balanced mini-set it *does*
   concentrate sharply, but onto **lineage / accessory-genome markers**, still suppressing rpoB. The head
   works mechanically; the training label alone can't steer it to the causal gene — it takes the
   phylogenetic shortcut.
3. **So inject the causal-gene vector directly.** Concatenate the ESM-C rpoB vector to the Bacformer
   genome-mean → plain logistic regression: **AUROC 0.975 (frozen mean) / 0.977 (fine-tuned mean)** on the
   full eval. k-fold × m-seed confirms it: concat beats ESM-rpoB-alone **15/15** runs (frozen Δ+0.0023; an
   honest eval-holdout FT k-fold Δ+0.0065). **The read-out, not the embedding, was the bottleneck.**

### The rifampicin ladder

![ladder](docs/visualisations/tb_rifampicin/rifampicin_ladder_barplot.png)

frozen mean **0.788** → FT+attention **0.868** → FT mean-pool **0.905** → frozen Bacformer rpoB **0.953** →
one-hot RRDR **0.960** → frozen ESM-C rpoB **0.971** → **concat 0.975 (frozen) / 0.977 (FT)**.

---

## 2. It generalises across drugs

The recipe **"causal gene ⊕ genome mean → LR"** holds across the TB panel; the *size* of the lift scales
with how much headroom the single gene leaves (auto-discovered: a per-gene ESM-LR screen finds each drug's
causal gene — rpoB/rif, katG/inh, embB/emb, pncA/pza, gyrA/fluoroquinolone — 7/10 land on the known gene).

| drug (gene) | gene alone | concat | lift |
|---|---:|---:|---:|
| isoniazid (katG) | 0.914 | **0.940** | **+0.026** |
| pyrazinamide (pncA) | 0.916 | **0.928** | +0.011 |
| ethambutol (embB) | 0.933 | **0.939** | +0.006 |
| rifampin (rpoB) | 0.970 | **0.972** | +0.002 |
| moxifloxacin (gyrA) | 0.921 | 0.920 | wash (gyrA already saturates) |

---

## 3. Embedding vs catalogue — the WHO one-hot ceiling

We ran **TB-Profiler (WHO v2 catalogue) over all 36,684 genomes** and built, per drug, a one-hot of its
WHO resistance mutations → LR. The **ceiling** = *all WHO mutations combined*. Comparing the protein-embedding
concat to this catalogue ceiling splits the drugs into three clean regimes:

- **Embedding BEATS the catalogue — pyrazinamide: concat 0.928 ≫ WHO one-hot 0.854 (+0.074).** pncA
  resistance is *hundreds of diverse loss-of-function mutations*; the one-hot can only encode catalogued
  ones and misses rare/novel variants, but the ESM embedding captures the **functional consequence of any
  damaging mutation** → it *generalises to mutations the catalogue has never seen*. The single strongest
  argument for embeddings.
- **Tie — rifampin (0.972 vs 0.967), ethambutol, fluoroquinolones.** Both capture the coding hotspot
  (rpoB RRDR, embB, gyrA QRDR).
- **Catalogue BEATS the embedding — the resistance is non-coding:**
  - **isoniazid** (one-hot 0.958 > concat 0.940): **87% of inhA-mediated resistance is a promoter SNP**
    (`c.-777C>T` etc., 4,361 non-coding vs 641 coding genomes) that leaves the inhA *protein* WT — so the
    ESM embedding is structurally blind (inhA-protein LR **0.526 = chance**) while the mutation one-hot sees
    it (0.650). Not mean-dilution: there is *no protein signal to dilute*.
  - **kanamycin** (ceiling 0.899 ≫ every embedding method ~0.833): the dominant cause **rrs (16S rRNA)** has
    no protein to embed at all.

---

## 4. The gap map — where Bacformer is blind

The per-drug **WHO one-hot histograms** split every WHO site into **coding (embeddable, solid)** vs
**promoter / non-coding (hatched)** vs **rRNA (hatched)**, against the combined-WHO ceiling. They are a
literal map of the protein-embedding blind spot:

| ![isoniazid](docs/visualisations/tb_isoniazid/isoniazid_WHO_one_hot_histogram.png) | ![kanamycin](docs/visualisations/tb_kanamycin/kanamycin_WHO_one_hot_histogram.png) |
|---|---|
| isoniazid: **inhA (promoter)** cross-hatched at 0.646 beside embeddable katG 0.893 | kanamycin: **rrs** (rRNA) 0.778 towers over the best embeddable pick |

**The protein embedding (ESM-C, and the current protein-only Bacformer) is structurally blind to
non-coding resistance — promoters (inhA), rRNA (rrs/rrl).** These are exactly the mechanisms a
nucleotide/genome-aware Bacformer would need to recover; the gap between each drug's embedding methods and
its WHO ceiling quantifies the headroom.

### The figure set
Each drug has, in `docs/visualisations/tb_<drug>/`: an **ESM-screen histogram** (per-gene ESM-embedding LR),
a **WHO one-hot histogram** (per-site mutation LR, hatched by mechanism), and a **ladder** (frozen mean ·
FT mean-pool · WHO top gene · ESM gene · concat, with the WHO ceiling line).

---

## 5. Forward

- **A nucleotide / genome-aware Bacformer** is the route to the non-coding gap (inhA promoter, rrs/rrl) —
  the chart already marks the targets and the recoverable headroom per drug.
- **Port to *Klebsiella*** (the HGT-driven complement): use **Kleborate** (CARD-derived, already run) as the
  gold-standard determinant ceiling, and run the same screen / concat / ladder / gap-map across all 22 Kp
  drugs — weakest first (**colistin 0.807, azithromycin 0.827**, …), the chromosomal/efflux/regulatory cases
  that mirror TB. Does concat surface those weak drugs? See the plan file.
- **What we drop:** the learned attention head (it takes the phylogenetic shortcut) and fine-tuning in the
  ladder are not pursued — concat is the simpler, stronger read-out.

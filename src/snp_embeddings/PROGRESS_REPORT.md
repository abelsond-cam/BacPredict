# TB AST signal-loss diagnostic — progress report

**Project.** BacPredict Task 7 — `src/snp_embeddings/`.
**Question.** Bacformer predicts *Klebsiella* AMR well but *Mycobacterium tuberculosis* rifampicin
resistance poorly. **Where, mechanistically, is the causal-mutation signal lost — and what is the
remedy?**
**Status.** Active diagnostic, as of **2026-06-15**. Author: David Abelson (with Claude Code).

This document is the shareable write-up: headline numbers, what we have concluded, and what is
still open. The companion [`CLAUDE.md`](CLAUDE.md) is the lean operational reference (paths, models,
file map) for an agent picking up the work; the living task tracker is [`../../ToDo.md`](../../ToDo.md).

---

## 1. Summary

Resistance to rifampicin in *M. tuberculosis* is caused almost entirely by **point mutations in a
short window of the *rpoB* gene** (the rifampicin-resistance-determining region, RRDR) — a single
amino-acid change. This is the opposite regime to *Klebsiella*, where resistance is largely
**acquired genes** (horizontal gene transfer). The programme hypothesis is that **Bacformer reads
gene gain/loss well but is comparatively blind to single chromosomal point mutations**; TB
rifampicin is the cleanest test of that.

We traced the signal through every pooling step of the model with a ladder of linear probes on the
**same held-out test set the deployed model uses**. The result is unambiguous:

> **The causal mutation is fully present in Bacformer's contextualised *rpoB* protein token
> (AUROC 0.953), and is then destroyed by the mean-pool that collapses ~4,000 protein tokens into
> one genome vector (AUROC 0.788).** Fine-tuning partly compensates (0.905) but never reaches the
> token-level ceiling, and — surprisingly — a *learned* attention pool we trained to replace the
> mean does **worse than the mean** (0.868). Yet the model's attention is **not** the problem in the
> way that result suggests: Bacformer's *internal* self-attention already concentrates on *rpoB*
> (top ~0.2% of all proteins). The signal is sitting in the tokens, attended to inside the network —
> so the failure is specifically in the prediction **head's** pooling attention, which apparently
> does *not* route to *rpoB*. That is exactly what we are now diagnosing directly.

---

## 2. Background and hypothesis

**The chain of averaging.** The model reaches a genome-level prediction through two mean-pools in
series, each of which dilutes a single causal residue:

1. **Residue → protein (ESM-C).** ESM-C mean-pools the ~1,178 residues of *rpoB* into one protein
   vector. One RRDR substitution is 1 residue in ~1,178.
2. **Protein → genome (Bacformer).** Bacformer mean-pools ~4,000 protein tokens into one genome
   vector. *rpoB* is 1 protein in ~4,000.

A mutation that is obvious per-residue can be near-invisible after two averages. The first pool is
**frozen and non-invertible**, so if the signal were lost there no amount of Bacformer fine-tuning
could recover it. Where the model still predicts correctly, it might be reading **lineage /
accessory-genome structure** (a phylogenetic shortcut) rather than the causal SNP. The diagnostic's
job was to find **which** pool loses the signal, and whether it is recoverable.

**Positive control.** *rpoB* / rifampicin (US spelling **`rifampin`** in the label table). RRDR
panel: **S450L, H445Y, D435V, S441L** (Mtb numbering; UniProt P9WGY9 is +6-offset, anchored on the
conserved motif `DQNNPLSGLTHKRR` and asserted WT at every codon so a wrong reference fails loudly).

---

## 3. Methods (brief)

- **Cohort.** 38,758 rifampicin-labelled TB genomes (26,147 susceptible / 12,595 resistant; 16
  ambiguous `0.5` labels dropped). Assemblies only — no reads. Genotypes are read **directly from
  the translated CDS in the assembly** (the sequence ESM-C actually saw), single-copy *rpoB* only.
- **Comparable AUROCs.** Every probe is an L2 logistic regression fit on the **train** split and
  scored on the **evaluate** split of the *same* canonical 70/10/20 holdout the deployed Bacformer
  model trained on (`binary_ast_with_split.csv` via `tl.train.evaluate.resolve_holdouts`), so all
  numbers — including the model's own — sit in one table. Headline AUROCs are on the **intersection**
  of samples every probe covers (n ≈ 6.9k evaluate genomes). Metrics via `tl.train.metrics`.
- **Surprisal scan.** A label-blind per-residue **surprisal** (−log P under ESM-C's masked LM)
  flags "an anomaly is here" without seeing the resistance label. Validated cheap *unmasked*
  surprisal as a faithful proxy for the gold-standard masked-marginal ablation, then scanned all
  ~4,000 proteins of 1,000 genomes for a per-protein statistic an attention pool could exploit.
- **Attention-pool head.** A gated-attention MIL pool (and a panel-augmented variant) replacing the
  mean, in `tl/train/attention_pool.py`, trained end-to-end on the canonical split.
- **Intrinsic-attention diagnostic.** Reads Bacformer's **own** self-attention
  (`return_attn_weights=True`, the method from the model author's operon-prediction code), to ask
  what the network attends to internally — distinct from what the prediction *head* pools.

---

## 4. Results

### 4.1 The localization ladder — the headline

RIF **evaluate** AUROC (n ≈ 6.9k), each row isolating one stage of the pipeline:

| Representation | AUROC | What it isolates |
|---|---:|---|
| One-hot RRDR codon genotype | **0.960** | The 4 panel codons alone — a SNP reference point |
| Frozen ESM-C pooled *rpoB* protein vector | **0.971** | ESM-C **keeps** the signal through the residue→protein mean |
| Frozen Bacformer contextualised *rpoB* **token** | **0.953** | Bacformer's *rpoB* token still carries it |
| Frozen Bacformer **genome mean** | **0.788** | …**destroyed** by the protein→genome mean |
| Fine-tuned **mean-pool** model (deployed) | **0.905** | Fine-tuning partly recovers — but below the token ceiling |
| Learned gated-MIL attention pool (end-to-end) | **0.868** | A *learned* pool does **worse than the mean** |
| Frozen gated-MIL attention pool | ~0.78 | …and frozen ≈ the frozen mean |

**Reading the ladder.**
- The signal is **not absent**. ESM-C's pooled *rpoB* vector (0.971) actually exceeds the 4-codon
  one-hot ceiling (0.960) — it encodes the whole protein, not just the panel codons. So the
  residue→protein mean is **not** the bottleneck here: the representational world, not the "absent"
  world.
- The collapse happens at the **protein→genome mean**: 0.953 (the *rpoB* token) → 0.788 (the genome
  mean). This is the single largest drop on the ladder and it is the crux.
- Fine-tuning the backbone through the mean-pool head recovers some of it (0.788 → 0.905) but
  **cannot reach** the 0.953 token ceiling — consistent with the backbone learning to smear the
  signal across many tokens so the mean catches a little of it, rather than the head reading *rpoB*
  directly.

### 4.2 The mutation is a razor-thin single-residue spike

Masked per-residue surprisal across *rpoB*: the resistance SNP is a **single sharp outlier** — mean
surprisal **8.14** at the mutated codon versus **0.11 / 0.19** at ±1 / ±2 neighbours and ~0.50 gene
background. The 2nd/3rd most-surprising residues are **not** the SNP's neighbours (within ±2 aa in
only 9% / 3% of genomes). Genome-wide spatial autocorrelation confirms neighbours carry no extra
information. **Implication:** a single-residue **peak** statistic suffices to flag the causal site;
no windowing needed. Cheap *unmasked* surprisal reproduces the masked-ablation signal, so the flag
can be precomputed at scale without the expensive per-residue masking.

### 4.3 A learned attention pool underperforms the mean

Replacing the mean with a gated-attention MIL pool and training end-to-end gives **0.868** — *below*
the 0.905 mean-pool baseline, and far below the 0.953 token ceiling. The pooling mechanism we hoped
would fix the read-out is currently the **weak link**: a pool that learned to route to *rpoB* could
not lose to the mean, so it has not learned to. (A surprisal **panel** — a 9-feature per-protein
"anomaly here" vector glued onto each output token, 960→969 — is built and tested as an explicit
steer for the pool; full-cohort evaluation pending the architecture decision below.)

### 4.4 Bacformer *internally* attends to *rpoB* — but that is not the head

The intrinsic-attention diagnostic shows the frozen Bacformer's own self-attention places *rpoB* in
the **top ~0.2%** of attention received (the single most-attended protein at several deep layers).
But this attention is **SNP-blind** — resistant and wild-type *rpoB* are attended essentially
equally (percentile 0.99792 vs 0.99791) — i.e. the model attends *rpoB* because it is a conserved
core gene (RNA polymerase β), not because of the mutation.

**The critical distinction** (and the current focus): this is attention *inside* the backbone,
between protein tokens — it explains why the *rpoB* **token** is well-formed (0.953). It is **not**
the prediction **head's** pooling, which is what collapses the tokens into the classifier's input.
The deployed model's head is a plain **mean**; our learned pool is a separate attention that —
per §4.3 — is failing to route to *rpoB*.

This is the apparent paradox the diagnostic must resolve: **the network attends to *rpoB*
internally, the learned pool still made prediction worse, so the head's pooling is evidently *not*
attending to *rpoB*.** The internal-attention finding tells us the signal is sitting in the tokens,
ready to be read, if only the head would attend to it — and whether the trained head's pool actually
does is exactly what we are now measuring (D1, §6). The earlier diagnostic answered the *internal*
attention question; we had not yet measured the *head's*.

---

## 5. Conclusions so far

1. **The signal is present, not absent.** It survives ESM-C's residue→protein pool (0.971) and lives
   in Bacformer's contextualised *rpoB* token (0.953). This is the *representational* regime — the
   remedy is a better read-out, **not** continued pretraining.
2. **The protein→genome mean-pool is the bottleneck.** It is where 0.953 collapses to 0.788; it is
   the chain-of-averaging's fatal link for a single-residue cause.
3. **Fine-tuning the mean is a partial patch (0.905), not a fix.** It never reaches the token ceiling.
4. **Bacformer attends to *rpoB* internally, but the prediction head apparently does not.** The
   backbone's own self-attention places *rpoB* in the top ~0.2% of proteins — the signal is in the
   tokens — yet a naive learned attention pool still underperforms the mean (0.868). So the head's
   *pooling* attention is evidently not routing to *rpoB*, even though the network internally does.
   **We are now diagnosing the head's attention directly** (D1) to confirm this is the precise
   failure — the previous attention diagnostic measured the *internal* attention, not the head's.
5. **The mutation is a clean single-residue spike**, cheaply flaggable by unmasked surprisal — which
   is what motivates giving the pool an explicit per-protein anomaly signal (the panel) and/or
   selecting candidate proteins by attention before the head reads them.

These findings are the first empirical support for the broader programme hypothesis: a chromosomal
point-mutation phenotype (TB rifampicin) is exactly where the genome-pooling architecture is weakest,
in contrast to the acquired-gene phenotypes (*Klebsiella*) where it is strong.

---

## 6. Open questions and current work

**In flight (2026-06-15) — three label-blind diagnostics over the 1,000-genome manifest:**

- **D1 — Does the prediction *head's* pool attend to *rpoB*?** Read the trained pool's per-protein
  weights (`last_attention_weights`) on the 0.868 / frozen checkpoints and rank *rpoB*. A plain mean
  is uniform (percentile 0.5); clustering at 0.5 with no resistant-vs-WT gap would confirm the head
  never routed to *rpoB* — the mechanistic cause of 0.868 < 0.905.
- **D2 — Does a *fine-tuned* backbone keep *rpoB* attended internally while the mean obliterates it?**
  Run the intrinsic-attention diagnostic on the 0.868 and the 0.905 backbones (not just the frozen
  model). Expectation: *rpoB* stays attended internally in all three → the signal is in the tokens
  and it is the **mean** that destroys it.
- **D3 — Is *rpoB* *by far* the strongest-attended gene, and what else is up there?** Name the top-K
  most-attended genes per genome and measure how reliably *rpoB* is captured — the evidence for a
  top-K-attended-gene selection head.

**The architecture decision (next regroup), informed by D1–D3.** Candidate read-outs to fix the pool:
- a PyTorch multi-head attention pool (more expressive than the single-query gated-MIL);
- the **surprisal panel** steering the pool to flagged proteins (built, tested);
- a **top-K-attended-gene selection head** — use intrinsic attention to pick the few most-attended
  proteins, then classify on *their* tokens (+ the mean), bypassing a single learned pool. (Design
  note: select an order-invariant *set*, not a rank-ordered concatenation, since rank-k is a
  different gene in different genomes.)

**Longer-horizon questions.**
- **Intrinsic attention vs Captum.** The 4-minute intrinsic-attention pass surfaces *rpoB* where
  Captum ablations (hours of GPU) also would — but the two measure different things (information flow
  vs causal attribution) and would **diverge** for accessory/HGT causal genes that are not conserved
  hubs. Intrinsic attention is a fast first-pass screen for *chromosomal-core* causal genes, not a
  general Captum replacement — and that divergence is itself a probe of the chromosomal-vs-HGT
  contrast.
- **Generalisation.** Does the same read-out fix carry to other TB drugs (katG/inhA, gyrA, embB,
  pncA, rrs) and to *Klebsiella* chromosomal mechanisms? Does the attention/selection approach behave
  differently for acquired-gene resistance (where the causal protein is *not* a conserved hub)?
- **Lineage shortcut.** The locus-restricted probes cannot use a phylogenetic shortcut, but any
  genome-wide head can; cross-lineage transfer (not in-distribution AUROC) is the acceptance test for
  the eventual fix.

---

## 7. Reproduction pointers

- **Ladder:** `snp_vs_esm_prediction.py` (Steps 1/2/3a) + `frozen_bacformer_rpob_vectors.py`
  (the frozen *rpoB* token + genome mean), on `resolve_holdouts`.
- **Surprisal:** `llr_distribution_probe.py` (per-residue + per-protein surprisal) →
  `unmasked_surprisal_scan.py` (genome-wide scan) → `surprisal_analysis.py` (figures, read-only).
- **Pool + panel:** `tl/train/attention_pool.py`, `build_surprisal_store.py`,
  `tl/train/datasets.py` (`PanelInjectingFileDataset`), `tb_ast/train_amr.py`.
- **Attention diagnostics:** `intrinsic_attention_probe.py` (internal, ± `--checkpoint-dir` and
  top-K gene naming) and `head_pool_attention_probe.py` (the head's pool), with
  `scripts/phase1_{intrinsic,head_pool}_attention.sh`.
- All AUROC artefacts are versioned JSON under
  `…/processed/train_tb_ast/snp_embeddings/` (one subfolder per analysis), schema per `tl/train/metrics`.

# Task 7 — SNP-embedding signal-loss diagnostic — Progress Report

**Date:** 2026-06-15 · **Branch:** `dev` · **Status:** diagnostic complete; the learned attention pool was built and **underperformed the mean** (0.868 / ~0.78 < 0.905) — pivoting to embedding augmentation (a per-protein ESM-C surprisal panel).

This report summarises the diagnostic phase of [`src/snp_embeddings/`](../) — why TB AST
(rifampicin) prediction underperforms, what we measured, the AUROC numbers, and why the
attention-pool remedy proved necessary but not sufficient — driving the pivot to embedding augmentation. The
authoritative running record is [the task `CLAUDE.md`](../CLAUDE.md); the per-figure numbers are
pinned in [`docs/surprisal_analysis.json`](surprisal_analysis.json).

---

## 1. The question

TB AST prediction underperforms badly — rifampicin Stage-C validation AUROC ~0.88 against a
WHO-catalogue ceiling ≥0.97 — while *Klebsiella* AST is strong. The programme hypothesis is that
**Bacformer is strong on HGT/gene-acquisition resistance but blind to chromosomal point
mutations** — exactly TB's regime (a single RRDR codon in *rpoB* confers rifampicin resistance).

The suspected mechanism was a **chain of two averages** that dilutes one causal residue:

1. **ESM-C residue → protein pool** — ~1,178 *rpoB* residues mean-pooled into one 960-vector
   (≈1/L dilution; one RRDR substitution is one residue in ~1,178).
2. **Bacformer protein → genome pool** — ~4,000 protein tokens mean-pooled into one genome
   vector (≈1/N dilution).

This task tested **which** average (if either) destroys the signal, and **whether** the signal is
recoverable at all — to choose between a representational remedy (re-pool) and a "signal absent"
remedy (re-pretrain).

---

## 2. What we ran

All probes are linear (`sklearn.LogisticRegression`, `C=1.0`, L2) fit on the **train** split and
scored on the **evaluate** split of the **same canonical 70/10/20 holdout the deployed Bacformer
model trained on** (`binary_ast_with_split.csv` via `tl.train.evaluate.resolve_holdouts`) — so
every AUROC below, including the deployed model's own, sits in one comparable table. Positive
control: *M. tuberculosis* *rpoB* / rifampicin (US spelling `rifampin` in the AST table), RRDR
panel S450L / H445Y / D435V / S441L.

| Probe | Module | Compute | What it isolates |
|---|---|---|---|
| One-hot RRDR genotype | [`snp_vs_esm_prediction.py`](../snp_vs_esm_prediction.py) | CPU | the SNP **ceiling** |
| Frozen pooled ESM-C *rpoB* vector | [`snp_vs_esm_prediction.py`](../snp_vs_esm_prediction.py) | CPU | loss at the residue→protein mean |
| Frozen Bacformer *rpoB* token | [`frozen_bacformer_rpob_vectors.py`](../frozen_bacformer_rpob_vectors.py) | GPU | signal in the contextualised token |
| Frozen Bacformer genome mean | (genome-mean probe) | GPU | loss at the protein→genome mean — **the deployed input** |
| Masked-marginal / unmasked surprisal | [`llr_distribution_probe.py`](../llr_distribution_probe.py) | GPU | is the residue in ESM-C *at all*, and where |

Data verified on HPC: `rifampin` is the canonical RIF column, 38,758 labelled (26,147 S /
12,595 R; 16 ambiguous `0.5` dropped); 38,248 protein-sequence parquets + 38,248 ESM-C `.pt`
under `processed/train_tb_ast/`. The localization ladder is on the RIF **evaluate** split
(n ≈ 6.9k).

---

## 3. Headline result — the localization ladder

Each rung is a linear probe over a different point in the production pipeline, all on the same RIF
evaluate split:

| Pipeline stage | Representation the classifier sees | RIF eval AUROC |
|---|---|---:|
| **SNP genotype** (ceiling) | one-hot RRDR codon | **0.960** |
| ESM-C protein pool | frozen mean-pooled *rpoB* 960-vector | **0.971** |
| Bacformer protein token | frozen contextualised *rpoB* token | **0.953** |
| **Bacformer genome pool** (deployed input) | frozen mask-mean over ~4,000 tokens | **0.788** |
| Deployed model | fine-tuned mean-pool head | **0.905** |
| Learned attention pool (e2e gated-MIL) | gated-attention over the ~4,000 tokens | **0.868** |
| Learned attention pool (frozen gated-MIL) | gated-attention, frozen backbone | **~0.78** |

**The signal survives ESM-C, survives the contextualised token, and collapses only at the genome
mean-pool.** Replacing that mean-pool with a learned attention pool (last two rows) did **not** help —
it lost to the mean; §6 diagnoses why.

- The **residue→protein average is NOT the culprit** — the frozen pooled ESM-C *rpoB* vector
  scores **0.971**, at/above the one-hot ceiling (it also carries other *rpoB* residues / lineage
  context). This refines the original two-averages hypothesis: the first average is innocent.
- The **protein→genome average IS the culprit** — the frozen *rpoB* token (0.953) collapses to
  **0.788** once it is mean-pooled into ~4,000 other proteins. That single step costs **0.165
  AUROC**.
- **Fine-tuning the mean-pool head cannot fix it** — the deployed model reaches only **0.905**,
  *below* the frozen *rpoB* token (0.953). A learned classifier on top of a destructive pool
  cannot recover what the pool threw away.

The natural remedy — **replace the genome mean-pool with a learned attention pool** over the
per-protein tokens — was built and **underperformed the mean** (0.868 / ~0.78 < 0.905). Since a
uniform set of pool weights *is* the mean, losing to it means the learned head never concentrated on
rpoB. §6 diagnoses why (the gate does not route to rpoB; fine-tuning erodes rpoB's internal salience)
and motivates the embedding-augmentation pivot: a per-protein **surprisal panel** (§4) that hands the
gate an explicit pointer to the anomalous protein.

→ Figure: [`figures/esm_surprisal.png`](figures/esm_surprisal.png) (per-residue vs per-protein
surprisal histograms) and [`figures/surprisal_vs_ablation.png`](figures/surprisal_vs_ablation.png).

---

## 4. Surprisal sub-experiments (toward the pre-pool feature channel)

"Surprisal" = −log P (information-theoretic). The **masked-marginal** surprisal (mask each
residue, read its log-prob) is the gold standard but costs L forwards per protein; the
**unmasked** surprisal is one forward per protein and is deployable genome-wide. Phase 0 asked
(0A) whether the cheap proxy is faithful, and (0B) whether a per-protein surprisal statistic
pushes the mutated *rpoB* into a high tail an attention pool could attend to.

### 4A — the cheap unmasked proxy is faithful (n=100 isolates, 17 genotypes)

- Pooled **Pearson 0.948**, Spearman 0.974 over 117,200 residues; per-isolate mean r 0.948
  (sd 0.001) — extremely stable across isolates.
- At the SNP site specifically, unmasked vs masked Pearson 0.893.
- The resistance residue is the **#1 unmasked surprisal anomaly in *rpoB* for 70%** of isolates
  (top-3 for 83%).

**Conclusion:** the unmasked single-forward surprisal is a faithful stand-in for the expensive
masked ablation — so a genome-wide scan is affordable.
→ Figure: [`figures/supp_proxy_r_hist.png`](figures/supp_proxy_r_hist.png).

### 4B — the mutated protein sits in the per-protein high tail (pilot, 6 genomes)

Across all ~4,000 proteins per genome, where does *rpoB* rank by each candidate statistic
(resistant vs WT genomes, mean percentile; n=3 each — a **small pilot**, hence the scaled scan in
§6):

| Statistic | Resistant *rpoB* %ile | WT *rpoB* %ile | R−WT gap |
|---|---:|---:|---:|
| `mean_top3` | 98.5 | 96.4 | 2.1 |
| `hotspot_z_floored` (robust max-z) | 96.8 | 93.2 | 3.6 |
| `max_minus_p95` | 95.7 | 90.6 | 5.1 |
| `max_surprisal` | 93.7 | 86.2 | 7.5 |
| `max_minus_p99` | 91.3 | 84.7 | 6.6 |
| `top1_minus_top2` | 58.9 | 1.0 | **57.9** |

Two readings: by **absolute rank**, `mean_top3` / `hotspot_z` put *rpoB* highest (~97–99th pct);
by **resistant-vs-WT separation**, `top1_minus_top2` is by far the most discriminating (the SNP
creates a large gap between the top residue and the rest), though at a lower absolute percentile.
This is exactly the magnitude-vs-concentration tension the scaled feature-selection scan is
designed to resolve. (`hotspot_z` is recomputed with a MAD floor of 0.732 to remove MAD→0
blow-ups in the raw statistic.)

---

## 5. The SNP is a razor single-residue spike — neighbours don't matter

We tested whether the 2nd/3rd most-surprising residues are the SNP's sequence neighbours (which
would justify a top-3 / windowed feature). They are not.

**Distance profile (masked, n=100), mean surprisal by signed distance from the SNP:**

| Distance from SNP | 0 | ±1 | ±2 | ±5 | ±10 | gene-wide background |
|---|---:|---:|---:|---:|---:|---:|
| Mean masked surprisal | **8.14** | 0.11 | 0.19 | 0.16 | 0.23 | 0.50 |

The signal is a sharp single-residue spike — the immediate neighbours are *below* the gene-wide
background.

**Neighbour-rank test (masked, n=100):**

- The SNP is the **top-1** masked anomaly in **72%** of isolates.
- The 2nd / 3rd most-surprising residues lie within ±2 aa of the SNP in only **9% / 3%** of
  isolates (rank-2 median distance **617 aa**); in **0%** of isolates are all of the top-3
  within ±2 aa.

**Conclusion:** a **single-residue peak statistic suffices** — "use top-3" is *not* justified by a
neighbour smear. `mean_top3` ranks *rpoB* highly only via *intrinsic* high-surprisal residues
elsewhere in the protein (poor SNP-specificity), so it is retained only as an orthogonal
*magnitude* channel, not as a neighbour proxy.
→ Figure: [`figures/snp_distance_profile.png`](figures/snp_distance_profile.png).

---

## 6. The attention pool was built — and the head still can't find rpoB

Acting on §3, we replaced the genome mean-pool with a **learned gated-attention MIL pool** (Ilse 2018):
one scalar weight per protein token, softmaxed over the ~4,000 tokens, then a weighted sum. Trained
freeze-backbone-first then end-to-end on rifampicin. It **lost to the mean** (e2e **0.868**, frozen
**~0.78** vs the mean-pool's **0.905**). Because a uniform set of weights *is* the mean, losing to it
means the learned pool collapsed toward uniform (or weighted the wrong genes) rather than concentrating
on rpoB.

Three label-blind diagnostics over the 1000-genome manifest separate Bacformer's **internal**
self-attention (between protein tokens, building the representation) from the predictive **head's** pool
(what collapses ~4,000 tokens → 1 vector), and ask where rpoB lands in each:

| model | backbone internal attention on rpoB | head pool weight on rpoB | eval AUROC |
|---|---|---|---|
| **frozen** (pretrained) | 99.8th pct · top-20 **98.8%** · gap 0.82 | — | (rpoB token 0.953) |
| **mean-pool FT** | 86.7th pct · top-20 **0%** · gap 0.006 | uniform (≡ mean) | **0.905** |
| **e2e gated-MIL FT** | 99.5th pct but only **rank ~20** (top-20 27%) | **68th pct (R 0.68 / WT 0.66), never #1** | **0.868** |
| **frozen gated-MIL** | (frozen backbone, as row 1) | 82nd pct, **no R/WT gap** | ~0.78 |

Three findings:

1. **rpoB is a strong attention hub only because it is a conserved core gene.** The *frozen* (pretrained)
   model attends rpoB in the top ~0.2% — but R ≈ WT, so it attends the *gene*, not the *mutation*. That is
   structural hub-ness inherited from masked-genome pretraining, not a learned resistance signal.
2. **Fine-tuning erodes it.** Training degrades rpoB's internal salience — the mean-pool drops it out of
   the top-20 entirely (gap 0.82 → 0.006); the e2e gated-MIL pushes it down to ~rank 20. "Fine-tuning
   surfaces the causal gene" is *false* here; if anything it buries it.
3. **The head never routes to rpoB.** Even a learned pool that *can* concentrate on one gene places rpoB
   at only the ~68th weight-percentile (never #1), with a faint R>WT difference that is most likely noise.
   The head cannot, from the resistance label alone, find the one SNP-bearing protein among ~4,000.

**Implication.** The signal is in the tokens (frozen rpoB token 0.953) but unreachable by the head — a
better *pool* is not the fix. The head needs an explicit, label-blind per-protein **pointer** to the
anomalous protein: the **surprisal panel** (§4) — feed each token a 9-dim ESM-C surprisal vector so the
gate can route to rpoB. (A heavier alternative — a per-gene logistic-regression probability channel over
core genes, with train-only cross-fitting to stay leakage-free — is recorded for later.)

---

## 7. Conclusions

1. **The defect is representational, not absent.** The causal residue is present and linearly
   decodable at every stage up to and including the frozen Bacformer *rpoB* token (0.953). It is
   the **protein→genome mean-pool** that destroys it (→ 0.788), and fine-tuning the mean-pool
   head cannot undo that (0.905 < 0.953).
2. **The first average is innocent.** ESM-C's residue→protein pool preserves the signal (0.971) —
   the original "two averages" suspicion narrows to the *second* average alone.
3. **A learned attention pool is necessary but not sufficient (§6).** We built it; it *lost to the
   mean* (0.868 / ~0.78 < 0.905) because the head's gate never routes to rpoB (it sits at the ~68th
   weight-percentile, never #1) and fine-tuning erodes rpoB's internal salience. A better pool alone
   cannot recover the signal.
4. **The remedy is the surprisal panel (§4/§6).** Hand each token a 9-dim ESM-C surprisal vector so the
   gate can find the mutated protein — a label-blind, leakage-free pointer. Unmasked single-forward
   surprisal tracks the masked gold standard (r 0.948) and flags the mutated protein into the high tail.
   Now training on the 1000-genome manifest (att_head panel vs a panel-less baseline).
5. **One residue, not a window.** Feature design should use a single-residue peak / contrast
   statistic, not a neighbour window.

This is the **first direct empirical support for the central HGT-vs-chromosomal hypothesis**: the
mean-pool that works for distributed accessory-genome (HGT) signal is precisely what erases a
single chromosomal point mutation.

---

## 8. Figures

All saved under [`docs/figures/`](figures/); numbers pinned in
[`docs/surprisal_analysis.json`](surprisal_analysis.json).

| Figure | What it shows |
|---|---|
| [`surprisal_vs_ablation.png`](figures/surprisal_vs_ablation.png) | 0A proxy scatter: unmasked vs masked surprisal, resistance residues in red |
| [`esm_surprisal.png`](figures/esm_surprisal.png) | 2-panel surprisal histogram — per-residue within *rpoB* \| per-protein across the genome |
| [`snp_distance_profile.png`](figures/snp_distance_profile.png) | masked surprisal vs distance from the SNP + neighbour-rank test |
| [`supp_proxy_r_hist.png`](figures/supp_proxy_r_hist.png) | per-isolate proxy correlation distribution |
| [`supp_rankcurve_max_surprisal.png`](figures/supp_rankcurve_max_surprisal.png) | *rpoB* rank curve by `max_surprisal` |
| [`supp_rankcurve_hotspot_z_floored.png`](figures/supp_rankcurve_hotspot_z_floored.png) | *rpoB* rank curve by robust max-z |
| [`supp_rpob_percentile_by_stat.png`](figures/supp_rpob_percentile_by_stat.png) | *rpoB* percentile by statistic, R vs WT |

(The first three are also mirrored into [`src/tb_ast/docs/figures/`](../../tb_ast/docs/figures/).)

---

## 9. Status & next steps

- **Done:** the 1000-genome unmasked-surprisal scan + the per-sample **panel store**
  (`tb_surprisal_panel/`, 9-panel + train standardisation); the gated-attention pool (built —
  *underperformed*, §6); the panel→head wiring + the n=10 overfit **smoke** (att_head & e2e, both
  AUROC 1.0).
- **In flight — the 1000-genome panel run:** gated-MIL `--panel-mode att_head` (panel) vs a
  panel-less baseline on a 700/100/200 manifest split (jobs `30602029` / `30602030`), read out by
  the D1 **head-pool probe** — does the panel lift rpoB's head-pool percentile from ~0.68 toward
  ~0.99 (R > WT) and beat the baseline? The manifest is a balanced, rpoB-enriched subset, so this is
  a *routing diagnostic*, not a headline AUROC.
- **Next (go/no-go):** if the panel routes the gate to rpoB, run the unmasked-surprisal scan over all
  ~38k genomes, build the full panel store (`build_surprisal_store.py --source parquet`), and train
  Stage C `--panel-mode att_head` vs the 0.905 mean-pool / 0.868 gated-MIL baselines, stratified by
  mechanism.
- **Recorded for later (option 2):** a per-gene logistic-regression probability channel over core
  genes (Prokka/Bakta-adjudged), fed to the head like the panel — heavier, and needs train-only
  cross-fitting to stay leakage-free.

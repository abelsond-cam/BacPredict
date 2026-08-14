---
name: tb-mini-set-0977-confound-vs-bug
description: "TB-AST 1000-genome manifest hit eval AUROC 0.9768 vs ~0.78-0.86 on full 38k — open question (lineage confound vs earlier bug), with the tests to settle it"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3240ca4c-5459-4b1e-97f4-9f2775c0f04c
---

On 2026-06-16 the Task-7 (snp_embeddings) 1000-genome **manifest** runs scored far above the
full-38k localization ladder: frozen gated-MIL **no-panel** baseline eval AUROC **0.9768** (job
30602029); surprisal-panel att_head val ~0.967 (job 30611005). The full-38k eval for the *same*
frozen gated-MIL config is only **~0.78** (ladder: mean 0.788, mean-pool FT 0.905, e2e gated-MIL
0.868, frozen gated-MIL ~0.78). A 0.78→0.977 jump for the same architecture is the open question.

**It is NOT a balancing artifact** — AUROC is prevalence-independent, so the 50/50 manifest split
cannot lift it. Two live hypotheses:
- **H1 (confound, most likely):** the manifest's ~500 R were selected as rpoB-RRDR carriers and
  ~500 WT as non-carriers. TB rpoB resistance is lineage-clonal, so selecting on rpoB genotype
  inadvertently selects on lineage → R vs WT differ genome-wide across lineage-marker proteins, and
  Bacformer's genome representation separates them on that population-structure shortcut, not on
  rpoB. The 200-genome held-out eval shares the same selection bias, so it does NOT break the
  confound — only the full 38k eval does.
- **H2 (bug/underbaked):** the earlier full-eval gated-MIL runs (0.78/0.868) were depressed by a bug
  or undertraining and the true attention-head perf is ~0.97. If true, TB AST is largely solved and
  the programme premise shifts. (The panel cap-fix does NOT touch a no-panel run, so a jump would
  implicate a *different* earlier bug.)

**2026-06-16 evidence — strongly supports H1 (confound):**
- Surprisal-panel att_head eval **0.9666** (job 30611005) is *below* the panel-less baseline **0.9768**
  (30602029) on the same manifest — the panel adds nothing here; the ~0.97 is not an rpoB/panel win.
- Per-gene-LR store (1738 core genes, single-copy >95%): top out-of-fold AUROC genes are **rpoB 0.9996,
  katG 0.935 (INH gene), embB 0.908 (EMB gene)** then a tail to 0.82 (only 15/1738 clear 0.8). katG/embB
  predicting *rifampin* resistance is the co-resistance/MDR-lineage confound smoking gun. rpoB ~1.0 is
  NOT a CV leak — it's circular (the manifest defines R = rpoB-RRDR genotype); the sharp falloff (15/1738)
  confirms the OOF cross-fitting is leakage-free. (The "expect 0.95-0.97 not 1.0" check is calibrated for
  the FULL eval, not the circular manifest.)
- Implication: manifest per-gene-LR / surprisal trainings are low-value (everything ~0.97 via lineage);
  the per-gene-LR rpoB channel and the panel must be judged on the FULL 38k eval.

**Tests to settle it (cheap → definitive):** A) eval the existing mini checkpoint (30602029) on the
full 38k eval — stays ~0.97 = not an artifact, drops ~0.8 = confound; B) retrain frozen gated-MIL
no-panel on full 27k → full eval (definitive); C) D1 head-pool readout on the mini checkpoints — does
the gate attend rpoB or lineage genes (we know the full-eval gate does NOT route to rpoB); D) the
full 38k surprisal scan (long pole, never launched — only the 1000-manifest scan ran) to run the
surprisal panel on the real eval. A+C are the fast disambiguator; B is definitive; D unblocks
downstream. Read all mini-set AUROCs as routing diagnostics on a confounded distribution; the
full-eval numbers are the real scoreboard. See [[tb-vs-kp-chromosomal-hgt-contrast]].

**2026-06-16 — A + C launched (full eval).** A = `eval_attn_pool_on_full_split.py` scoring the
no-panel manifest baseline (`30602029`, checkpoint-1600, panel_mode=none) on the **full 7,074-genome**
evaluate fold (R=2,206 / S=4,868, prev 0.312) — job **30615548**. Confirmed the manifest's 1000 are
drawn from the train fold: **0 overlap** with the 7,074, so A is leakage-clean with no exclusions; it
self-checks by first reproducing the manifest's own 200-genome eval (~0.9768). Stays ~0.97 → H2,
drops ~0.8 → H1. C = head-pool probe on the same checkpoint — job **30615551**. Cost reconciliation:
A is **one Bacformer forward per genome** (ESM-C inputs are precomputed/saved — NOT re-embedded),
~0.3 s/genome on A100 → ~0.5–1 GPU-h for 7,074 (I/O-bound), NOT the 5–7 s/genome ESM/surprisal rate.
Results pending. The per-gene-LR follow-on options are in [[per-gene-lr-strand-options]].

**2026-06-16 — RESOLVED: H1 (confound) confirmed, H2 (solved/bug) DEAD.** A (job 30615548) finished:
manifest baseline `30602029` scored **full-eval AUROC 0.574** (AUPRC 0.399, n=7074, prevalence 0.31,
0 manifest-overlap excluded) vs **0.9764** on the manifest's own 200-genome eval — reproduced *inside the
same script/path*, so the 0.977→0.574 collapse is real, not a scorer artifact. The 0.977 was entirely the
lineage/selection confound; the checkpoint barely beats chance on the natural distribution. **Caveat:** this
checkpoint also trained on only 700 manifest genomes, so the *magnitude* of the drop entangles confound with
small-train — does NOT rescue H2 (refuted regardless), but **Plan B** (retrain no-panel gated-MIL on the
full 27k → full eval, expect ~0.78 ladder number) is the clean separator. Mechanism corroborated by C: the
head-pool **suppresses rpoB** (enrichment 0.1–0.2×, rank ~196 of ~4000) — see [[per-gene-lr-strand-options]]
and the cumulative-attention figure work in `head_pool_attention_probe.py`. Reporting fix: the full-eval
scorer no longer scores the validate split (commit 920d88d) — AUROC lands the instant eval scoring finishes.

---
name: invasion-comparators-2026-08
description: "Blood-vs-faeces (invasion) comparator results 2026-08-11 — unitig tie, per-SL holds within clone, Kleborate virulence_score at chance; plus which AUROC to quote"
metadata: 
  node_type: memory
  type: project
  originSessionId: 65fba51f-bd25-41d2-813a-24d9abb4255d
  modified: 2026-08-11T21:47:57.959Z
---

**Which number to quote for the invasion model: 0.786**, on the country-controlled
`sampled_country_2_1_all` cohort (n=14,119; holdout 2,822). There is **no ~0.85 Bacformer
result** — 0.827/0.835 are the country-**confounded** `all_samples` cohort, which sits *below*
its own linear metadata baseline (0.857) and is not defensible. 0.786 is also the cohort the
pyseer unitig GWAS ran on, so it is the right comparator everywhere.

Three comparators, all on that identical holdout (2026-08-11):

1. **Unitig GWAS model = a TIE.** L2 LR on the 33,039 significant hit unitigs scored **0.781**
   vs Bacformer **0.787** on the same 2,715 genomes — paired bootstrap delta **+0.0055,
   95% CI [-0.0110, +0.0230], so the CI SPANS ZERO: a statistical tie, not a win. The unitig model held a *selection
   advantage* (its features were chosen by an LMM fitted over the whole cohort incl. the
   holdout), so "Bacformer matches a leakage-advantaged accessory model" is the claim —
   **not** "Bacformer wins". An honest train-only-selection re-run is still outstanding.
2. **Per-sublineage: the signal holds WITHIN every major clone** — SL258 0.858, SL15 0.841,
   SL307 0.815, SL17 0.806, SL147 0.738, rare-SL bucket 0.759, vs pooled 0.786. Four of five
   at or above pooled; SL258 significantly so. So lineage identity is not what the model reads.
   **Mechanism is an open hypothesis** (within-clone accessory content? plasmid?) — David has
   not yet given his read; do not plan downstream work off an interpretation.
   **Extended 2026-08-11 by whole-cohort scoring** (all 14,119 genomes, job `33494112`): evaluate
   reproduces 0.7858 exactly, but train is **0.9590** vs validate 0.7943 — the model memorises
   hard and train is 70% of the cohort, so **all-splits per-clone AUROCs are mostly recall of
   fitted rows and cannot show a clone generalises**. Hence the `heldout` scope (validate+evaluate,
   n=4,234, nothing fitted on): SL15 0.890, SL258 0.844, SL307 0.828, SL17 0.814, SL147 0.739,
   **SL37 0.662** (weakest, wide CI), other 0.762, pooled 0.788. At `all` scope 20 SLs clear
   n≥100 and every one scores 0.849–0.966 — no clone fails. Clonal groups: 15 at `all`, only 4 at
   `heldout` (too fine-grained for the holdout). Spread 0.662–0.890 is **not interpreted**.
3. **Kleborate's total `virulence_score` predicts invasion at CHANCE (0.489).** Virulence
   one-hot (6 loci) 0.552; virulence+AMR 0.638; all-Kleborate 0.640. Notably **AMR annotation
   beats virulence annotation** (0.617 vs 0.552) — probably a healthcare-association confound,
   not a virulence result.

Reusable code (all committed on `refactor/consolidate-engine`):
`bacpredict.engine.finetune.stratified_metrics` (organism-agnostic per-stratum §0.4 metrics +
bootstrap CIs; also answers the long-open TB/Kp per-stratum item) ·
`bacpredict.engine.plots.plot_stratified_auroc` · `bac_pyseer.kleb_iso_source.unitig_presence_model`
(+ `scripts/run_unitig_presence_model.sh`).

→ [[invasion-model-was-fp32-not-bf16]] · [[dont-conflate-penetrance-with-lineage]]

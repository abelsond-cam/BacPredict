"""Gene-level attribution of a deployed AMR fine-tune — captum ablation, catalogue join, statistics.

One protein token is ablated at a time and the shift in the model's output recorded, giving every
gene in every genome an attribution score. The question it answers is **which genes the model
reads**, so that a model predicting rifampicin resistance at 0.96 AUROC can be checked against
``rpoB``, and one predicting ciprofloxacin against ``gyrA`` / ``parC`` and the acquired ``qnr`` family.

Stages, each its own module and each runnable alone:

- :mod:`~bacpredict.engine.explain.select_attribution_cohort` — pick the genomes, through
  :func:`bacpredict.engine.splits.load_splits.load_splits` and no other route.
- :mod:`~bacpredict.engine.explain.gene_ablation_attribution` — the GPU ablation itself.

**The stored quantity is the shift in the logit, plus each genome's base logit.** The probability-
scale score is then recoverable exactly as ``sigmoid(base) - sigmoid(base - delta)``, so both scales
come out of one run. Attributing the probability directly saturates on a confident model — the
published Bacformer run lost 1,895 of 4,242 genomes that way — and a saturated genome carries no
information at all.

**Attribution is descriptive.** That a model attributes heavily to ``rpoB`` is not evidence it
learned causality until the signal is shown to survive stratification by lineage; a high-attribution
gene outside the catalogue is not thereby a lineage correlate either. Both directions are open
hypotheses — see ``PROJECT_STATE.md`` §6.
"""

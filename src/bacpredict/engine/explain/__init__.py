"""Gene-level attribution of a deployed AMR fine-tune — captum ablation, catalogue join, statistics.

One protein token is ablated at a time and the shift in the model's output recorded, giving every
gene in every genome an attribution score. The question it answers is **which genes the model
reads**, so that a model predicting rifampicin resistance at 0.96 AUROC can be checked against
``rpoB``, and one predicting ciprofloxacin against ``gyrA`` / ``parC`` and the acquired ``qnr`` family.

Stages, each its own module and each runnable alone:

- :mod:`~bacpredict.engine.explain.select_attribution_cohort` — pick the genomes, through
  :func:`bacpredict.engine.splits.load_splits.load_splits` and no other route.
- :mod:`~bacpredict.engine.explain.gene_attribution` — the GPU attribution itself, by deletion
  (``--method ablation``) or by gradient (``integrated_gradients`` / ``input_x_gradient``).
- :mod:`~bacpredict.engine.explain.annotate_attributions` — flat index → gene → catalogue.

**The stored quantity is the shift in the logit, plus each genome's base logit.** The probability-
scale score is then recoverable exactly as ``sigmoid(base) - sigmoid(base - delta)``, so both scales
come out of one run. Attributing the probability directly saturates on a confident model — the
published Bacformer run lost 1,895 of 4,242 genomes that way — and a saturated genome carries no
information at all.

**Deletion is the wrong counterfactual for a point mutation.** Ablation asks whether the model still
predicts resistance with a gene removed. But ``gyrA``, ``parC`` and ``rpoB`` are in every genome and
differ only by a substitution, so the model never learned "``gyrA`` present → resistant"; at most it
learned "``gyrA`` with this sequence → resistant". A gradient reads the response to the protein's
*embedding*, which is where that sequence difference lives, and costs ``n_steps`` passes for the
whole genome rather than one per protein. Ablation remains available and is the right question for
an acquired gene such as ``blaKPC``, which genuinely is present or absent.

**Attribution is descriptive.** That a model attributes heavily to ``rpoB`` is not evidence it
learned causality until the signal is shown to survive stratification by lineage; a high-attribution
gene outside the catalogue is not thereby a lineage correlate either. Both directions are open
hypotheses — see ``PROJECT_STATE.md`` §6.
"""

"""Gene-level attribution of a deployed AMR fine-tune — captum ablation, catalogue join, statistics.

One protein token is ablated at a time and the shift in the model's output recorded, giving every
gene in every genome an attribution score. The question it answers is **which genes the model
reads**, so that a model predicting rifampicin resistance at 0.96 AUROC can be checked against
``rpoB``, and one predicting ciprofloxacin against ``gyrA`` / ``parC`` and the acquired ``qnr`` family.

Stages, each its own module and each runnable alone:

- :mod:`~bacpredict.engine.explain.select_attribution_cohort` — pick the genomes, through
  :func:`bacpredict.engine.splits.load_splits.load_splits` and no other route.
- :mod:`~bacpredict.engine.explain.gene_attribution` — the GPU attribution itself. Use
  ``--method ablation``; the two gradient methods are present but ⛔ **invalid on this
  architecture** (see below).
- :mod:`~bacpredict.engine.explain.annotate_attributions` — flat index → gene → catalogue.

**The stored quantity is the shift in the logit, plus each genome's base logit.** The probability-
scale score is then recoverable exactly as ``sigmoid(base) - sigmoid(base - delta)``, so both scales
come out of one run. Attributing the probability directly saturates on a confident model — the
published Bacformer run lost 1,895 of 4,242 genomes that way — and a saturated genome carries no
information at all.

**Deletion is an imperfect counterfactual for a point mutation, but the gradient methods wired in
here do not fix it.** Ablation asks whether the model still predicts resistance with a gene removed.
``gyrA``, ``parC`` and ``rpoB`` are in every genome and differ only by a substitution, so the model
never learned "``gyrA`` present → resistant"; at most "``gyrA`` with this sequence → resistant".
Ablation is nonetheless the right question for an acquired gene such as ``blaKPC``, and it is **not**
degenerate: ablating ``rpoB`` moved the logit ``+1.231`` and ``+2.382`` in two ``p.Ser450Leu``
genomes against ``+0.016`` in a wild-type one (job 41326249).

⛔ **A zero-baseline gradient measures nothing here.** The checkpoint's bundled
``modeling_bacformer.py`` runs ``nn.LayerNorm`` as the *first* operation on the raw protein
embeddings (``:642``, ``:666``), so ``f(alpha * x) == f(x)`` exactly above ``alpha ~ 1e-4``. The model
is degree-0 homogeneous in its input: ``input_x_gradient`` is identically zero by Euler's theorem
(measured ``-6.0e-08`` over a genome), and the IG integrand goes as ``1/alpha``. LayerNorm's
``eps`` keeps that integral finite but concentrates nearly all of it in ``alpha < ~5e-5``, below the
smallest Gauss-Legendre node at any affordable ``n_steps`` — so the answer swings with ``n_steps``
instead of converging, and more steps made it *worse*. The published
Bacformer repo is unaffected — its model is ``linear(x)`` *then* LayerNorm, which is not
scale-invariant. ⚠ So is the pip-installed ``bacformer`` package: it is a **different architecture**
from what these checkpoints run via ``trust_remote_code``, and testing against it does not describe
the deployed model.

A gradient path is valid only against a **non-degenerate baseline** (a cohort-mean protein — the
continuous form of a wild-type-substitution counterfactual) or on a **post-LayerNorm layer**
(``LayerIntegratedGradients`` at ``bacformer.embeddings.norm1``), and then only at ``n_steps >= 200``.
Neither variant is wired into this package yet.

**Attribution is descriptive.** That a model attributes heavily to ``rpoB`` is not evidence it
learned causality until the signal is shown to survive stratification by lineage; a high-attribution
gene outside the catalogue is not thereby a lineage correlate either. Both directions are open
hypotheses — see ``PROJECT_STATE.md`` §6.
"""

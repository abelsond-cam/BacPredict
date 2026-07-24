"""Transition re-export — the LR fit/score engine now lives in :mod:`bacpredict.engine.segment_amr_lr.fit_lr`.

Kept so the to-be-deleted read-out modules (``gene_lr/kfold_probe.py``, ``gene_lr/coding_amr_lr.py``,
``concat/concatenate_bacformer_genome_esm_protein_emb.py``) keep importing during the ``segment_amr_lr``
unification. Remove once those are folded in / deleted and every consumer imports from
``bacpredict.engine.segment_amr_lr.fit_lr``.
"""

from __future__ import annotations

from bacpredict.engine.segment_amr_lr.fit_lr import LOGREG_KW, fit_score_step

__all__ = ["LOGREG_KW", "fit_score_step"]

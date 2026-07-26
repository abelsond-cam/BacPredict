"""Shared primitive for catalogue-ceiling scoring (WHO / Kleborate / CARD).

The three catalogue modules (``apps.tb.tbprofiler_gene_lr``, ``apps.kleb.kleborate_determinant_lr``,
``apps.kleb.card_determinant_lr``) build a one-hot determinant frame from **different** sources with
**different** mechanism schemas and output columns — that part is genuinely catalogue-specific and
stays in each app. What they share is the scoring step: fit the determinant one-hot through the
canonical k-fold LR harness and return its aggregate. That primitive lives here so the three don't
each carry a copy.
"""
from __future__ import annotations

import pandas as pd

from bacpredict.engine.gene_lr.kfold_probe import FeatureSpec, run_kfold_probe


def score_onehot_frame(
    frame: pd.DataFrame,
    label_map: dict[str, int],
    seeds: tuple[int, ...],
    *,
    n_folds: int = 5,
    evaluate_seed: int = 1,
    evaluate_fraction: float = 0.20,
) -> dict | None:
    """k-fold AUROC/AUPRC aggregate (mean±sd …) for one binary determinant frame, or ``None``.

    Returns ``None`` for a degenerate frame (no columns, or all-zero — nothing to fit). Otherwise
    runs the frame as a single ``numeric`` feature (no standardisation — the values are 0/1) through
    :func:`bacpredict.engine.gene_lr.kfold_probe.run_kfold_probe` and returns the full aggregate dict.
    """
    if frame.shape[1] == 0 or int(frame.to_numpy().sum()) == 0:
        return None
    kf = run_kfold_probe(
        {"f": FeatureSpec(frame, kind="numeric", standardise=False)},
        label_map,
        n_folds=n_folds,
        seeds=seeds,
        evaluate_seed=evaluate_seed,
        evaluate_fraction=evaluate_fraction,
    )
    return kf["frames"]["f"]["aggregate"]

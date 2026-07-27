"""Shared primitive for catalogue-ceiling scoring (WHO / Kleborate / CARD).

The three catalogue modules (``apps.tb.tbprofiler_gene_lr``, ``apps.kleb.kleborate_determinant_lr``,
``apps.kleb.card_determinant_lr``) build a one-hot determinant frame from **different** sources with
**different** mechanism schemas and output columns — that part is genuinely catalogue-specific and stays in
each app. What they share is the scoring step: fit the determinant one-hot on the deployed model's **train**
split and score it on the **holdout**, so the catalogue ceiling lands on the *same* holdout as the FT-mean +
per-segment data rungs. That primitive lives here so the three don't each carry a copy.

Migrated onto the deployment split (:mod:`bacpredict.engine.splits.load_splits`): the retired version ran its
own whole-cohort k-fold (``run_kfold_probe``), so the RED ceiling sat on a *different* holdout than the data
rungs. Now every ladder rung — the four data configs **and** the ceiling — is fit-on-train / scored-on-the-one
holdout by the *same* estimator (:func:`bacpredict.engine.segment_amr_lr.fit_lr.fit_one_segment`), so they are
directly comparable. (``run_kfold_probe`` stays for the delta-significance probes that still want a CV spread.)
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from bacpredict.engine.segment_amr_lr.fit_lr import fit_one_segment


def score_onehot_frame(
    frame: pd.DataFrame,
    label_map: dict[str, int],
    train_ids: Sequence[str],
    holdout_ids: Sequence[str],
    *,
    n_folds: int = 5,
    seed: int = 1,
) -> dict | None:
    """Held-out AUROC/AUPRC for one binary determinant frame — fit on ``train``, scored on the deployment ``holdout``.

    The determinant one-hot is a full 0/1 matrix over the frame's genomes (a genome carrying no determinant is
    an all-zero row), so it is scored by the *same* estimator the ladder's data rungs use
    (:func:`bacpredict.engine.segment_amr_lr.fit_lr.fit_one_segment`): fit the LR on the ``train`` genomes,
    predict the held-out ``holdout`` genomes, and report the held-out AUROC/AUPRC. Because it reads the same
    ``(train, holdout)`` partition (via ``splits.load_splits``) the catalogue ceiling is directly comparable to
    the FT-mean + per-segment rungs — the leak-free spine covers the last rung.

    Returns ``None`` for a degenerate frame (no columns / all-zero), an empty holdout, or a single-class *train*
    split (``fit_one_segment`` cannot fit). The return is shaped like the retired k-fold aggregate —
    ``{"auroc": {"mean", "sd"}, "auprc": {...}, "n_eval", "n_train"}`` with ``sd = 0.0`` (a single holdout has no
    spread) — so callers reading ``agg["auroc"]["mean"]`` are unchanged.
    """
    if frame.shape[1] == 0 or int(frame.to_numpy().sum()) == 0:
        return None
    train_set, holdout_set = set(map(str, train_ids)), set(map(str, holdout_ids))
    f = frame.copy()
    f.index = f.index.astype(str)
    all_ids = [s for s in f.index if s in train_set or s in holdout_set]
    if not all_ids or not any(s in holdout_set for s in all_ids):
        return None
    x = f.loc[all_ids].to_numpy(dtype=np.float32)
    y = np.array([label_map[s] for s in all_ids], dtype=int)
    fit = fit_one_segment(all_ids, x, y, n_folds=n_folds, seed=seed, eval_ids=holdout_set)
    if not fit or fit.get("n_eval", 0) == 0:
        return None
    ev_ids = [s for s in all_ids if s in holdout_set]
    probs = np.array([fit["eval_prob"].get(s, np.nan) for s in ev_ids])
    yv = np.array([label_map[s] for s in ev_ids], dtype=int)
    auprc = (float(average_precision_score(yv, probs))
             if not np.isnan(probs).any() and 0 < int(yv.sum()) < len(yv) else float("nan"))
    return {
        "auroc": {"mean": float(fit["eval_auroc"]), "sd": 0.0},
        "auprc": {"mean": auprc, "sd": 0.0},
        "n_eval": int(fit["n_eval"]),
        "n_train": int(fit["n_train"]),
    }

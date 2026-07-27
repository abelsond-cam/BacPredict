"""Ceiling migration smoke — ``score_onehot_frame`` fits on ``train`` and scores on the deployment ``holdout``.

Pins the leak-free contract: the catalogue one-hot is fit on the train genomes and evaluated on exactly the
holdout genomes (validate genomes are neither trained on nor scored), the same partition the FT-mean +
per-segment data rungs use — so every ladder rung is comparable.
"""
from __future__ import annotations

import pandas as pd

from bacpredict.engine.ref_catalogues.base import score_onehot_frame


def _frame_and_split(n: int = 40):
    """One determinant present iff resistant (a perfect signal), with train/validate/holdout id lists."""
    samples = [f"g{i:03d}" for i in range(n)]
    resistant = [i % 2 == 0 for i in range(n)]  # alternating → every contiguous chunk keeps both classes
    frame = pd.DataFrame({"det": [1 if r else 0 for r in resistant]}, index=samples)
    label_map = {s: (1 if r else 0) for s, r in zip(samples, resistant, strict=True)}
    train_ids = samples[:28]
    holdout_ids = samples[32:]  # 8 genomes, both classes; samples[28:32] are validate (ignored)
    return frame, label_map, train_ids, holdout_ids


def test_score_onehot_frame_fits_train_scores_holdout():
    frame, label_map, train_ids, holdout_ids = _frame_and_split()
    agg = score_onehot_frame(frame, label_map, train_ids, holdout_ids)
    assert agg is not None
    assert agg["auroc"]["mean"] > 0.9            # perfect determinant signal, scored on the holdout
    assert agg["auroc"]["sd"] == 0.0             # a single holdout has no spread
    assert agg["n_eval"] == len(holdout_ids)     # scored on exactly the holdout genomes
    assert 0 < agg["n_train"] <= len(train_ids)  # fit on (a subset of) train only, never the holdout/validate


def test_score_onehot_frame_degenerate_returns_none():
    samples = [f"g{i}" for i in range(10)]
    label_map = {s: i % 2 for i, s in enumerate(samples)}
    train, holdout = samples[:6], samples[6:]
    zero = pd.DataFrame({"det": [0] * 10}, index=samples)
    assert score_onehot_frame(zero, label_map, train, holdout) is None          # all-zero frame → nothing to fit
    good = pd.DataFrame({"det": [i % 2 for i in range(10)]}, index=samples)
    assert score_onehot_frame(good, label_map, train, []) is None               # empty holdout → None

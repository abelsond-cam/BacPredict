"""Unit tests for the shared LR engine — fit-on-train / OOF-select / eval-on-holdout semantics."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")
import pandas as pd

from bacpredict.engine.segment_amr_lr.fit_lr import (
    fit_one_segment,
    fit_one_segment_imputed,
    fit_score_step,
)


def _separable(n: int = 60, seed: int = 0) -> tuple[list[str], np.ndarray, np.ndarray]:
    """A cleanly separable 2-class 4-D set → the LR should score well above chance."""
    rng = np.random.default_rng(seed)
    y = np.array([i % 2 for i in range(n)], dtype=int)
    x = rng.normal(size=(n, 4)).astype(np.float32) + y[:, None] * 2.5
    ids = [f"S{i:03d}" for i in range(n)]
    return ids, x, y


def test_fit_one_segment_oof_only(tmp_path) -> None:
    """No eval_ids → OOF AUROC on all ids, eval fields empty, oof_prob keyed by every id."""
    ids, x, y = _separable()
    fit = fit_one_segment(ids, x, y, n_folds=5, seed=1)
    assert fit is not None
    assert fit["auroc"] > 0.9
    assert set(fit["oof_prob"]) == set(ids)
    assert np.isnan(fit["eval_auroc"]) and fit["n_eval"] == 0


def test_fit_one_segment_eval_holdout_split() -> None:
    """With eval_ids: fit on non-eval, report a held-out eval_auroc on the eval genomes only."""
    ids, x, y = _separable(n=80)
    eval_ids = set(ids[:20])
    fit = fit_one_segment(ids, x, y, n_folds=5, seed=1, eval_ids=eval_ids)
    assert fit is not None
    assert set(fit["oof_prob"]).isdisjoint(eval_ids)          # OOF is on the fit set only
    assert set(fit["eval_prob"]) == eval_ids                  # eval probs cover exactly the holdout
    assert fit["n_eval"] == 20 and fit["eval_auroc"] > 0.9


def test_fit_one_segment_single_class_returns_none() -> None:
    """A single-class fit set has no resistance contrast → None (dropped from the ranking)."""
    ids = [f"S{i}" for i in range(20)]
    x = np.random.default_rng(0).normal(size=(20, 4)).astype(np.float32)
    assert fit_one_segment(ids, x, np.zeros(20, dtype=int), n_folds=5, seed=1) is None


def test_fit_one_segment_imputed_zero_fills_absent() -> None:
    """Absent genomes get a 0-vector over the full universe (presence/absence enters the design matrix)."""
    ids, x, y = _separable(n=60)
    present_ids, present_x = ids[:40], x[:40]          # only 40 carry the segment
    all_ids = ids
    y_all = y
    fit = fit_one_segment_imputed(present_ids, present_x, all_ids, y_all, dim=4, n_folds=5, seed=1)
    assert fit is not None and fit["n_train"] == 60    # fit over the full universe, absent → 0-vector


def test_fit_score_step_fits_train_scores_evaluate() -> None:
    """fit_score_step trains on TRAIN, reports metrics on EVALUATE, and a Youden op-point from VALIDATE."""
    ids, x, y = _separable(n=90)
    feat = pd.DataFrame(x, index=ids)
    label_map = dict(zip(ids, y.tolist(), strict=True))
    out = fit_score_step(
        feat, kind="numeric", standardise=True, label_map=label_map,
        train_ids=ids[:60], validate_ids=ids[60:75], evaluate_ids=ids[75:],
    )
    assert out["metrics"]["auroc"] > 0.9
    assert out["operating_point"]["selected_on"] == "validate"
    assert set(out["eval_probs"]) == set(ids[75:])

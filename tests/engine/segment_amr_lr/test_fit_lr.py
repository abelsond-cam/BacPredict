"""Unit tests for the shared LR engine — fit-on-train / OOF-select / eval-on-holdout semantics."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")
import pandas as pd

from bacpredict.engine.segment_amr_lr.fit_lr import (
    fit_one_segment,
    fit_one_segment_imputed,
    fit_per_segment,
    fit_score_step,
)

DIM = 6


def _separable(n: int = 60, seed: int = 0) -> tuple[list[str], np.ndarray, np.ndarray]:
    """A cleanly separable 2-class 4-D set → the LR should score well above chance."""
    rng = np.random.default_rng(seed)
    y = np.array([i % 2 for i in range(n)], dtype=int)
    x = rng.normal(size=(n, 4)).astype(np.float32) + y[:, None] * 2.5
    ids = [f"S{i:03d}" for i in range(n)]
    return ids, x, y


def _label_map(n_pos: int, n_neg: int) -> tuple[list[str], dict[str, int]]:
    ids = [f"R{i}" for i in range(n_pos)] + [f"S{i}" for i in range(n_neg)]
    return ids, {s: (1 if s.startswith("R") else 0) for s in ids}


def _separable_matrix(ids: list[str], label_map: dict[str, int], *, sep: float, seed: int) -> np.ndarray:
    """A design matrix whose class means are ``±sep`` apart in feature 0 (separable when ``sep`` is large)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(len(ids), DIM))
    x[:, 0] += np.array([sep if label_map[s] == 1 else -sep for s in ids])
    return x


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


# ---------------------------------------------------------------------------
# fit_per_segment — the per-segment orchestrator over fit_one_segment(_imputed)
# ---------------------------------------------------------------------------


def test_fit_per_segment_oof_is_leakage_safe() -> None:
    """A separable segment yields high OOF AUROC; OOF probs cover every train id; the full fit is stored."""
    ids, label_map = _label_map(12, 12)
    x = _separable_matrix(ids, label_map, sep=4.0, seed=0)

    fitted = fit_per_segment({"rpoB": (ids, x)}, label_map, n_folds=5, seed=1)

    assert "rpoB" in fitted
    f = fitted["rpoB"]
    assert f["auroc"] > 0.9  # separable -> strong, but a held-out estimate (not asserted ==1.0)
    assert set(f["oof_prob"]) == set(ids)  # every train genome got an out-of-fold value
    assert hasattr(f["clf"], "predict_proba") and hasattr(f["scaler"], "transform")


def test_fit_per_segment_threads_eval_ids() -> None:
    """``fit_per_segment(..., eval_ids=...)`` yields the held-out eval columns for every segment."""
    ids, label_map = _label_map(20, 20)
    x = _separable_matrix(ids, label_map, sep=4.0, seed=5)
    eval_ids = {ids[0], ids[20]}  # 1 pos + 1 neg

    fitted = fit_per_segment({"rpoB": (ids, x)}, label_map, n_folds=5, seed=1, eval_ids=eval_ids)

    f = fitted["rpoB"]
    assert f["n_eval"] == 2 and f["n_eval_pos"] == 1
    assert 0.0 <= f["eval_auroc"] <= 1.0
    assert set(f["oof_prob"]).isdisjoint(eval_ids)  # eval genomes never got an OOF (leakage-safe)


def test_fit_per_segment_drops_single_class() -> None:
    """A segment whose train labels are all one class has no resistance contrast and is dropped."""
    ids = [f"S{i}" for i in range(8)]
    label_map = dict.fromkeys(ids, 0)
    x = np.random.default_rng(0).normal(size=(8, DIM))
    assert fit_per_segment({"g": (ids, x)}, label_map, n_folds=5, seed=1) == {}


def test_fit_per_segment_impute_absent_recovers_presence_signal() -> None:
    """Zero-imputing absent genomes lets the LR use the presence/absence signal the drop-absent fit can't.

    The acquired-gene case: the segment is present *only* in the resistant genomes (presence == resistance).
    Drop-absent fits on present genomes only — all one class — so it is dropped. Zero-impute over the full
    universe makes the absent (susceptible) genomes 0-vectors and the present (resistant) ones real, which is
    perfectly separable → high AUROC. This is the acquired-gene fix.
    """
    ids, label_map = _label_map(10, 10)
    present_ids = [s for s in ids if label_map[s] == 1]  # carried only by the resistant genomes
    # Real embeddings sit well away from the origin, so a 0-vector is far from any real one — offset the
    # synthetic present vectors to reflect that (an origin-centred cloud would swallow the 0 rows).
    x_present = np.random.default_rng(0).normal(loc=5.0, size=(len(present_ids), DIM))
    matrices = {"acq": (present_ids, x_present)}

    assert fit_per_segment(matrices, label_map, n_folds=5, seed=1) == {}  # drop-absent: single-class

    fitted = fit_per_segment(matrices, label_map, n_folds=5, seed=1, all_ids=ids, impute_absent_zero=True)
    assert "acq" in fitted
    assert fitted["acq"]["auroc"] > 0.9  # presence/absence now recovered
    assert fitted["acq"]["n_train"] == len(ids)  # fit over the full read universe, not just present


def test_fit_one_segment_accepts_float16_storage() -> None:
    """float16 design-matrix storage still fits (upcast to float32) — the whole-cohort memory path."""
    ids, label_map = _label_map(12, 12)
    x = _separable_matrix(ids, label_map, sep=4.0, seed=1).astype(np.float16)
    y = np.array([label_map[s] for s in ids], dtype=int)

    f = fit_one_segment(ids, x, y, n_folds=5, seed=1)
    assert f is not None and f["auroc"] > 0.9

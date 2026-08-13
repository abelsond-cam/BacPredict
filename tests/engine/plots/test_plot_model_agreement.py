"""Tests for the two-model agreement scatter.

The number this figure exists to report is r², so the arithmetic is checked against numpy directly
rather than trusted. The other risk is quiet mis-alignment: two score files joined in row order
rather than by genome would produce a beautiful, meaningless cloud.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from bacpredict.engine.plots.plot_model_agreement import (
    agreement_stats,
    load_scores,
    logit,
    plot_agreement,
)


def _write_scores(path, ids, probs, y_true=None, split=None):
    kw = {"sample_ids": np.asarray(ids, dtype=np.str_), "y_prob": np.asarray(probs, dtype=float)}
    if y_true is not None:
        kw["y_true"] = np.asarray(y_true, dtype=int)
    if split is not None:
        kw["split"] = np.asarray(split, dtype=np.str_)
    np.savez(path, **kw)
    return path


def test_r2_matches_numpy_on_a_known_input():
    rng = np.random.default_rng(0)
    a = rng.uniform(0.02, 0.98, 500)
    b = np.clip(a + rng.normal(0, 0.05, 500), 0.01, 0.99)
    s = agreement_stats(a, b)
    expected_prob = float(np.corrcoef(a, b)[0, 1]) ** 2
    expected_logit = float(np.corrcoef(logit(a), logit(b))[0, 1]) ** 2
    assert s["r2_prob"] == pytest.approx(expected_prob)
    assert s["r2_logit"] == pytest.approx(expected_logit)
    assert s["n"] == 500


def test_identical_models_give_r2_of_one():
    a = np.linspace(0.05, 0.95, 50)
    s = agreement_stats(a, a.copy())
    assert s["r2_prob"] == pytest.approx(1.0)
    assert s["r2_logit"] == pytest.approx(1.0)
    assert s["spearman_rho"] == pytest.approx(1.0)


def test_alignment_is_by_genome_not_row_order(tmp_path):
    """The two files list the same genomes in different orders; a positional join would score noise."""
    a_ids = ["S1", "S2", "S3", "S4"]
    a_probs = [0.9, 0.8, 0.2, 0.1]
    b_ids = ["S4", "S3", "S2", "S1"]          # reversed
    b_probs = [0.1, 0.2, 0.8, 0.9]            # same values per genome

    pa = _write_scores(tmp_path / "a.npz", a_ids, a_probs, y_true=[1, 1, 0, 0])
    pb = _write_scores(tmp_path / "b.npz", b_ids, b_probs)
    out = tmp_path / "fig.png"
    stats_ = plot_agreement(pa, pb, out, label_a="A", label_b="B")

    assert out.is_file()
    # Aligned by Sample the models are identical -> r2 == 1. A row-order join would give ~ -1 corr.
    assert stats_["main"]["r2_logit"] == pytest.approx(1.0)
    assert stats_["main"]["spearman_rho"] == pytest.approx(1.0)


def test_restrict_split_selects_only_that_split(tmp_path):
    ids = [f"S{i}" for i in range(10)]
    probs = np.linspace(0.05, 0.95, 10)
    split = ["train"] * 6 + ["evaluate"] * 4
    pa = _write_scores(tmp_path / "a.npz", ids, probs, y_true=[0, 1] * 5, split=split)
    pb = _write_scores(tmp_path / "b.npz", ids, probs[::-1], split=split)

    stats_ = plot_agreement(pa, pb, tmp_path / "f.png", label_a="A", label_b="B",
                            restrict_split="evaluate")
    assert stats_["main"]["n"] == 4


def test_missing_sample_ids_is_refused(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(p, y_prob=np.array([0.1, 0.9]))
    with pytest.raises(ValueError, match="sample_ids"):
        load_scores(p)


def test_disjoint_genome_sets_fail_loudly(tmp_path):
    pa = _write_scores(tmp_path / "a.npz", ["A1", "A2"], [0.1, 0.9])
    pb = _write_scores(tmp_path / "b.npz", ["B1", "B2"], [0.1, 0.9])
    with pytest.raises(SystemExit, match="share no genomes"):
        plot_agreement(pa, pb, tmp_path / "f.png", label_a="A", label_b="B")


def test_inset_panel_renders_and_reports_its_own_stats(tmp_path):
    ids = [f"S{i}" for i in range(20)]
    probs = np.linspace(0.05, 0.95, 20)
    pa = _write_scores(tmp_path / "a.npz", ids, probs, y_true=[0, 1] * 10)
    pb = _write_scores(tmp_path / "b.npz", ids, probs)
    inset = pd.DataFrame({"a": [0.2, 0.5, 0.8], "b": [0.25, 0.45, 0.85]})

    out = tmp_path / "f.png"
    stats_ = plot_agreement(pa, pb, out, label_a="A", label_b="B", inset=inset)
    assert out.is_file()
    assert stats_["inset"]["n"] == 3
    json.dumps(stats_)  # must be serialisable for the sidecar

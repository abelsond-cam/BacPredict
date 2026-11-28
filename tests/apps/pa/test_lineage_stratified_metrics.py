"""The lineage read-outs must separate "knows the clone" from "knows the genome"."""

import math

import numpy as np
import pytest

from bacpredict.apps.pa.lineage_stratified_metrics import (
    _auc,
    stratified_auroc,
    within_stratum_mic,
)


def test_a_model_that_only_knows_the_lineage_scores_well_overall_and_chance_within():
    """The whole point of read-out (e): a perfect between-clone model is a coin flip inside a clone.

    Two STs, one mostly resistant and one mostly susceptible. The score is the ST's mean label plus noise
    that carries no information. Overall AUROC is high; conditional on ST there is nothing left.
    """
    rng = np.random.default_rng(0)
    st = np.array(["A"] * 200 + ["B"] * 200, dtype=object)
    y = np.concatenate([rng.random(200) < 0.9, rng.random(200) < 0.1]).astype(int)
    score = np.where(st == "A", 1.0, 0.0) + rng.normal(0, 0.01, 400)

    assert _auc(y, score) > 0.85                      # looks like a good model
    out = stratified_auroc(y, score, st)
    assert out["n_strata"] == 2
    assert 0.45 < out["auroc"] < 0.55                 # and knows nothing inside a lineage


def test_a_model_that_knows_the_genome_keeps_its_discrimination_within_a_lineage():
    """The contrast case: the score tracks the label itself, so stratifying does not remove it."""
    rng = np.random.default_rng(1)
    st = np.array(["A"] * 200 + ["B"] * 200, dtype=object)
    y = (rng.random(400) < 0.5).astype(int)
    score = y + rng.normal(0, 0.3, 400)
    assert stratified_auroc(y, score, st)["auroc"] > 0.9


def test_strata_are_pooled_by_discordant_pairs_not_averaged_over_strata():
    """A three-genome ST must not count as much as a three-hundred-genome one."""
    rng = np.random.default_rng(2)
    big_y = (rng.random(300) < 0.5).astype(int)
    big_s = big_y + rng.normal(0, 0.2, 300)           # a good model in the big stratum
    small_y = np.array([0, 1, 1])
    small_s = np.array([5.0, -5.0, -4.0])             # perfectly backwards in a tiny one
    y = np.concatenate([big_y, small_y])
    s = np.concatenate([big_s, small_s])
    st = np.array(["big"] * 300 + ["small"] * 3, dtype=object)
    out = stratified_auroc(y, s, st)
    assert out["auroc"] > 0.9                         # pair-weighted: the big stratum dominates
    assert out["auroc_unweighted"] < out["auroc"]     # the mean over strata is dragged down by n=3
    assert out["n_pairs"] == out["per_stratum"][0]["pairs"] + 2


def test_a_stratum_with_one_class_contributes_nothing_and_an_empty_result_is_nan():
    """An AUROC over a stratum with no contrast is undefined, which is not the same as 0.5."""
    y = np.array([1, 1, 1, 0])
    s = np.array([3.0, 2.0, 1.0, 0.0])
    st = np.array(["A", "A", "A", "B"], dtype=object)
    out = stratified_auroc(y, s, st)
    assert out["n_strata"] == 0 and math.isnan(out["auroc"])
    assert math.isnan(_auc(np.array([1, 1]), np.array([1.0, 2.0])))


def test_ties_count_as_half_not_as_a_win():
    """A saturated score can tie exactly; counting that as concordant would flatter the model."""
    assert _auc(np.array([1, 0]), np.array([1.0, 1.0])) == pytest.approx(0.5)


def test_within_stratum_slope_collapses_for_a_model_that_predicts_the_clone_mean():
    """Read-out (e) on the MIC scale: centring inside the ST is what exposes the collapse."""
    rng = np.random.default_rng(3)
    st = np.repeat(np.array(["A", "B", "C"], dtype=object), 60)
    offset = {"A": -3.0, "B": 0.0, "C": 3.0}
    k = np.array([offset[s] for s in st]) + rng.integers(0, 3, 180)   # true ladder point
    censor = np.array(["exact"] * 180, dtype=object)

    honest = within_stratum_mic(k - 0.5, k, censor, st)
    assert honest["slope_log2"] == pytest.approx(1.0) and honest["n_strata"] == 3

    clone_mean = np.array([offset[s] for s in st], dtype=float)       # knows the ST and nothing else
    collapsed = within_stratum_mic(clone_mean, k, censor, st)
    assert math.isnan(collapsed["slope_log2"]) or abs(collapsed["slope_log2"]) < 1e-9


def test_within_stratum_mic_ignores_censored_cells_and_single_cell_strata():
    """A censored cell has no MIC to regress on; a stratum of one has no within-variation."""
    st = np.array(["A", "A", "B"], dtype=object)
    censor = np.array(["exact", "right", "exact"], dtype=object)
    out = within_stratum_mic(np.array([0.5, 9.0, 2.5]), np.array([1.0, np.inf, 3.0]), censor, st)
    assert out["n_strata"] == 0 and math.isnan(out["slope_log2"])

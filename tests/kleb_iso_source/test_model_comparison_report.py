"""Tests for the invasion model-comparison report.

Every failure mode guarded here is silent. A Youden threshold computed off the wrong array still
returns a plausible number; a 2x2 whose cells no longer sum to n still renders; a top/bottom pair
that overlaps still produces a shortlist. The one that has already cost real work is the gate: the
leakage-free and selection-advantaged unitig cohorts differ by a directory suffix and hold an
identically named archive, so opening the wrong one is invisible until a number moves.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from kleb_iso_source.build_model_comparison_report import (
    ID_COL,
    POOLED_COL,
    UNITIG_COL,
    agreement_2x2,
    assert_gate,
    comparison_set,
    load_cohort_scores,
    split_auroc,
    top_bottom,
    youden_threshold,
)


def _write_scores(tmp_path, name, sample_ids, y_true, y_prob, split):
    p = tmp_path / name
    np.savez(p, sample_ids=np.asarray(sample_ids, dtype=np.str_), y_true=np.asarray(y_true),
             y_prob=np.asarray(y_prob), split=np.asarray(split, dtype=np.str_))
    return p


def _preds(n=6, pooled=None, unitig=None, sl=None):
    return pd.DataFrame({
        ID_COL: [f"S{i}" for i in range(n)],
        "LabID": [f"L{i}" for i in range(n)],
        "strain": [f"strain{i}" for i in range(n)],
        POOLED_COL: pooled if pooled is not None else np.linspace(0.1, 0.9, n),
        UNITIG_COL: unitig if unitig is not None else np.linspace(0.2, 0.8, n),
        "Sublineage": sl if sl is not None else ["SL258"] * n,
    })


# --- Youden ---------------------------------------------------------------------------------------


def test_youden_threshold_matches_hand_computed_value():
    """On a perfectly separable vector the cut sits at the lowest positive score."""
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    t = youden_threshold(y, p)
    assert t["threshold"] == pytest.approx(0.7)
    assert t["sensitivity"] == pytest.approx(1.0)
    assert t["specificity"] == pytest.approx(1.0)
    assert t["youden_j"] == pytest.approx(1.0)


def test_youden_threshold_on_overlapping_scores():
    """One misordered genome costs exactly one unit of sensitivity, and J reflects it."""
    y = np.array([0, 0, 1, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    t = youden_threshold(y, p)
    assert 0.0 < t["threshold"] <= 1.0
    assert t["youden_j"] == pytest.approx(t["sensitivity"] + t["specificity"] - 1.0)
    assert t["youden_j"] < 1.0


# --- the gate -------------------------------------------------------------------------------------


def test_gate_passes_on_the_recorded_auroc(tmp_path):
    """A matching AUROC returns the recomputed value rather than the expectation it was checked against."""
    path = _write_scores(tmp_path, "cohort_scores.npz", ["a", "b", "c", "d"], [0, 1, 0, 1],
                         [0.1, 0.9, 0.2, 0.8], ["evaluate"] * 4)
    df = load_cohort_scores(path, "pooled")
    got, n = split_auroc(df, "pooled")
    assert got == pytest.approx(1.0)
    assert n == 4
    assert assert_gate(df, "pooled", 1.0) == pytest.approx(1.0)


def test_gate_refuses_the_wrong_archive(tmp_path):
    """The whole point: a drifted AUROC is a wrong file, and must stop the run."""
    path = _write_scores(tmp_path, "unitig_cohort_scores.npz", ["a", "b", "c", "d"], [0, 1, 0, 1],
                         [0.9, 0.1, 0.8, 0.2], ["evaluate"] * 4)
    df = load_cohort_scores(path, "unitig")
    with pytest.raises(ValueError, match="GATE FAILED"):
        assert_gate(df, "unitig", 0.7655)


def test_gate_scores_only_the_evaluate_split(tmp_path):
    """Train rows are fitted-on; including them would inflate the gate and hide a mismatch."""
    path = _write_scores(tmp_path, "cohort_scores.npz", list("abcdef"), [0, 1, 0, 1, 1, 0],
                         [0.9, 0.1, 0.1, 0.9, 0.8, 0.2], ["train", "train", "evaluate",
                                                          "evaluate", "evaluate", "evaluate"])
    df = load_cohort_scores(path, "pooled")
    assert split_auroc(df, "pooled") == (pytest.approx(1.0), 4)
    assert split_auroc(df, "pooled", "train")[0] == pytest.approx(0.0)


def test_load_cohort_scores_rejects_an_archive_missing_a_key(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(p, sample_ids=np.array(["a"], dtype=np.str_), y_prob=np.array([0.5]))
    with pytest.raises(ValueError, match="y_true"):
        load_cohort_scores(p, "pooled")


# --- the 2x2 --------------------------------------------------------------------------------------


def test_agreement_cells_sum_to_n():
    rng = np.random.default_rng(0)
    p, u = rng.random(671), rng.random(671)
    cells = agreement_2x2(p, u, 0.5, 0.5)
    total = (cells["both_invasive"] + cells["both_faeces"]
             + cells["bacformer_invasive_unitig_faeces"] + cells["unitig_invasive_bacformer_faeces"])
    assert total == cells["n"] == 671
    assert cells["disagree"] == (cells["bacformer_invasive_unitig_faeces"]
                                 + cells["unitig_invasive_bacformer_faeces"])
    assert cells["concordance"] == pytest.approx((cells["both_invasive"] + cells["both_faeces"]) / 671)


def test_agreement_uses_each_models_own_threshold():
    """A compressed second model must not be forced through the first model's cut-point."""
    p = np.array([0.9, 0.8, 0.2, 0.1])
    u = np.array([0.55, 0.54, 0.45, 0.44])   # same ranking, far narrower scale
    shared = agreement_2x2(p, u, 0.5, 0.5)
    own = agreement_2x2(p, u, 0.5, np.median(u))
    assert own["disagree"] <= shared["disagree"]
    assert own["both_invasive"] == 2
    assert own["both_faeces"] == 2


def test_agreement_perfect_concordance_gives_kappa_one():
    p = np.array([0.9, 0.8, 0.2, 0.1])
    cells = agreement_2x2(p, p, 0.5, 0.5)
    assert cells["disagree"] == 0
    assert cells["cohens_kappa"] == pytest.approx(1.0)


# --- shortlists -----------------------------------------------------------------------------------


def test_top_bottom_slices_are_disjoint_and_ordered():
    df = _preds(n=40, pooled=np.linspace(0.01, 0.99, 40))
    out = top_bottom(df, k=10)
    top = out[out["shortlist"] == "top"]
    bottom = out[out["shortlist"] == "bottom"]
    assert len(top) == len(bottom) == 10
    assert set(top[ID_COL]).isdisjoint(set(bottom[ID_COL]))
    assert top[POOLED_COL].min() > bottom[POOLED_COL].max()
    assert top[POOLED_COL].is_monotonic_decreasing


def test_top_bottom_never_overlaps_when_the_group_is_small():
    """With 15 genomes a naive head(10)/tail(10) would list five of them twice."""
    df = _preds(n=15, pooled=np.linspace(0.1, 0.9, 15))
    out = top_bottom(df, k=10)
    assert len(out) == 14
    assert out[ID_COL].is_unique


def test_top_bottom_returns_empty_for_a_single_genome():
    out = top_bottom(_preds(n=1), k=10)
    assert out.empty
    assert "shortlist" in out.columns


# --- denominators ---------------------------------------------------------------------------------


def test_comparison_set_excludes_rather_than_imputes_a_missing_score():
    df = _preds(n=5)
    df.loc[0, UNITIG_COL] = np.nan
    df.loc[1, POOLED_COL] = np.nan
    comp = comparison_set(df)
    assert len(comp) == 3
    assert comp[POOLED_COL].notna().all()
    assert comp[UNITIG_COL].notna().all()
    assert not comp[ID_COL].isin(["S0", "S1"]).any()


def test_comparison_set_leaves_the_input_untouched():
    df = _preds(n=5)
    df.loc[0, UNITIG_COL] = np.nan
    comp = comparison_set(df)
    comp[POOLED_COL] = 0.0
    assert df[POOLED_COL].iloc[1] != 0.0


# --- thresholds file ------------------------------------------------------------------------------


def test_thresholds_json_round_trips(tmp_path):
    """The compare/shortlists commands read the threshold back by name — keep the contract explicit."""
    payload = {"schema_version": "1.0",
               "models": {"bacformer_pooled": {"threshold": 0.51}, "unitig": {"threshold": 0.49}}}
    p = tmp_path / "thresholds.json"
    p.write_text(json.dumps(payload))
    back = json.loads(p.read_text())
    assert back["models"]["bacformer_pooled"]["threshold"] == pytest.approx(0.51)
    assert back["models"]["unitig"]["threshold"] == pytest.approx(0.49)

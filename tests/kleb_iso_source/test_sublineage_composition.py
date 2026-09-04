"""Tests for the per-sublineage composition tables.

Every failure mode guarded here is silent. An `other` bucket that drops a group still renders; a
percentage averaged across sublineages instead of pooled still looks plausible; a scope filter that
selects the wrong splits still produces a full table. The one that would be worst is the join guard:
the split CSV and the scored archive are written by different jobs, and a mismatched pair would give
per-lineage numbers for a cohort that no model was ever fitted to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kleb_iso_source.sublineage_composition import (
    LABEL_COL,
    OTHER_LABEL,
    SL_COL,
    SPLIT_COL,
    compose,
    load_cohort,
)


def _cohort(sls, labels, splits, probs):
    return pd.DataFrame({
        "Sample": [f"S{i}" for i in range(len(sls))],
        SL_COL: sls, LABEL_COL: labels, SPLIT_COL: splits, "prob": probs,
    })


def _write_pair(tmp_path, samples, sls, labels, splits, probs):
    csv = tmp_path / "split.csv"
    pd.DataFrame({"Sample": samples, SL_COL: sls, LABEL_COL: labels, SPLIT_COL: splits}).to_csv(csv, index=False)
    npz = tmp_path / "cohort_scores.npz"
    np.savez(npz, sample_ids=np.asarray(samples, dtype=np.str_), y_prob=np.asarray(probs),
             y_true=np.asarray(labels), split=np.asarray(splits, dtype=np.str_))
    return csv, npz


# --- the join guard ---------------------------------------------------------------------------------


def test_load_cohort_joins_a_matching_pair(tmp_path):
    csv, npz = _write_pair(tmp_path, ["a", "b"], ["SL1", "SL2"], [1, 0], ["train", "evaluate"], [0.9, 0.1])
    out = load_cohort(csv, npz)
    assert len(out) == 2
    assert set(out.columns) >= {"Sample", SL_COL, LABEL_COL, SPLIT_COL, "prob"}


def test_load_cohort_refuses_a_partial_join(tmp_path):
    csv, npz = _write_pair(tmp_path, ["a", "b"], ["SL1", "SL2"], [1, 0], ["train", "evaluate"], [0.9, 0.1])
    pd.DataFrame({"Sample": ["a", "ghost"], SL_COL: ["SL1", "SL2"], LABEL_COL: [1, 0],
                  SPLIT_COL: ["train", "evaluate"]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="join is not total"):
        load_cohort(csv, npz)


def test_load_cohort_refuses_disagreeing_labels(tmp_path):
    """A row count match is not proof the two files describe the same cohort."""
    csv, npz = _write_pair(tmp_path, ["a", "b"], ["SL1", "SL2"], [1, 0], ["train", "evaluate"], [0.9, 0.1])
    pd.DataFrame({"Sample": ["a", "b"], SL_COL: ["SL1", "SL2"], LABEL_COL: [0, 1],   # flipped
                  SPLIT_COL: ["train", "evaluate"]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="y_true"):
        load_cohort(csv, npz)


# --- composition ------------------------------------------------------------------------------------


def test_other_aggregates_every_group_beyond_top_n_and_counts_reconcile():
    sls = ["SL1"] * 10 + ["SL2"] * 6 + ["SL3"] * 3 + ["SL4"] * 2 + ["SL5"] * 1
    n = len(sls)
    rows, totals = _composed(sls, top_n=2)
    assert list(rows[SL_COL]) == ["SL1", "SL2", OTHER_LABEL]
    other = rows[rows.is_other].iloc[0]
    assert other.n == 6 and other.n_groups == 3          # SL3+SL4+SL5
    assert int(rows.n.sum()) == n == totals["n"]
    assert totals["n_sublineages"] == 5 and totals["n_named"] == 2


def _composed(sls, top_n=2, labels=None, splits=None, probs=None, threshold=0.5, scope="all"):
    n = len(sls)
    labels = labels if labels is not None else [i % 2 for i in range(n)]
    splits = splits if splits is not None else ["train"] * n
    probs = probs if probs is not None else [0.9 if i % 2 else 0.1 for i in range(n)]
    return compose(_cohort(sls, labels, splits, probs), threshold, scope=scope, top_n=top_n)


def test_other_is_pinned_last_even_though_it_is_the_largest():
    """At 'all' scope other is 45% of the real cohort — sorting by n would put it first."""
    sls = ["SL1"] * 5 + ["SL2"] * 4 + [f"R{i}" for i in range(30)]
    rows, _ = _composed(sls, top_n=2)
    assert rows.iloc[-1][SL_COL] == OTHER_LABEL
    assert rows.iloc[-1].n > rows.iloc[0].n          # larger, yet last


def test_percentages_are_pooled_not_averaged_across_sublineages():
    """A 26-genome lineage must not weigh the same as a 2,416-genome one."""
    sls = ["SL1"] * 100 + ["SL2"] * 2
    labels = [1] * 50 + [0] * 50 + [1, 1]            # SL1 50%, SL2 100%
    rows, totals = _composed(sls, top_n=5, labels=labels, probs=[0.1] * 102)
    assert totals["pct_blood"] == pytest.approx(52 / 102 * 100)   # pooled
    assert totals["pct_blood"] != pytest.approx(75.0)             # not the mean of 50 and 100


def test_scope_selects_the_right_splits():
    sls = ["SL1"] * 6
    splits = ["train", "train", "train", "validate", "evaluate", "evaluate"]
    assert _composed(sls, splits=splits, scope="all")[1]["n"] == 6
    assert _composed(sls, splits=splits, scope="heldout")[1]["n"] == 3      # validate + evaluate
    assert _composed(sls, splits=splits, scope="evaluate")[1]["n"] == 2


def test_predicted_counts_follow_the_threshold():
    sls = ["SL1"] * 4
    probs = [0.2, 0.4, 0.6, 0.8]
    assert _composed(sls, probs=probs, threshold=0.5)[1]["n_pred_blood"] == 2
    assert _composed(sls, probs=probs, threshold=0.3)[1]["n_pred_blood"] == 3
    assert _composed(sls, probs=probs, threshold=0.9)[1]["n_pred_blood"] == 0


def test_global_offset_is_reported_so_per_lineage_gaps_are_read_against_it():
    """The Youden point under-calls the positive class globally; that is not a lineage finding."""
    sls = ["SL1"] * 10
    labels = [1] * 6 + [0] * 4                       # 60% blood
    probs = [0.9] * 3 + [0.1] * 7                    # 30% predicted blood
    _rows, totals = _composed(sls, labels=labels, probs=probs)
    assert totals["pct_blood"] == pytest.approx(60.0)
    assert totals["pct_pred_blood"] == pytest.approx(30.0)
    assert totals["global_offset_pp"] == pytest.approx(-30.0)


def test_a_sublineage_with_no_call_is_excluded_and_counted():
    rows, totals = _composed(["SL1", "SL1", "", "nan"], top_n=5)
    assert totals["n"] == 2
    assert totals["n_no_sublineage_call"] == 2
    assert OTHER_LABEL not in set(rows[SL_COL])


def test_no_other_row_when_every_group_fits_in_top_n():
    rows, _ = _composed(["SL1", "SL1", "SL2"], top_n=5)
    assert not rows.is_other.any()
    assert len(rows) == 2


def test_unknown_scope_is_refused():
    with pytest.raises(ValueError, match="unknown scope"):
        _composed(["SL1"], scope="nonsense")

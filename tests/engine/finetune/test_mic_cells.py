"""``resolve_mic_cells``: a MIC arm may drop cells from a split table, and may never move one between splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bacpredict.engine.finetune.finetune_amr import resolve_mic_cells, resolve_split_table

INF = float("inf")


@pytest.fixture
def cohort(tmp_path):
    """Twelve genomes: 6 train, 2 validate, 3 holdout, 1 holdout_st. Four of them have no measured MIC."""
    rows, intervals = [], []
    for i in range(12):
        split = ["train", "train", "train", "train", "train", "train",
                 "validate", "validate", "holdout", "holdout", "holdout", "holdout_st"][i]
        label = i % 2
        rows.append({"Sample": f"G{i}", "ast_label": label, "split": split})
        bare = i in (2, 7, 9, 11)
        intervals.append({
            "Sample": f"G{i}", "ast_label": label,
            "lo": 0.0 if label else -INF, "hi": INF if label else 0.0,
            "point": np.nan if bare else (1.0 if label else -1.0),
            "censor": "bare" if bare else "exact",
            "basis": "bare_call" if bare else "mic",
        })
    split_path = tmp_path / "meropenem_split.csv"
    pd.DataFrame(rows).to_csv(split_path, index=False)
    mic_path = tmp_path / "meropenem_mic_intervals.csv"
    pd.DataFrame(intervals).to_csv(mic_path, index=False)
    return resolve_split_table(split_path), mic_path


def test_all_cells_keeps_every_genome_including_the_bare_calls(cohort):
    """The censored arm reads a bare R/S call as an open interval, so it loses nothing."""
    splits, mic_path = cohort
    out, intervals, _, _ = resolve_mic_cells(splits, mic_path, "censored", "all")
    assert (len(out.train), len(out.validate), len(out.holdout)) == (6, 2, 3)
    assert len(intervals) == 12


def test_mic_only_drops_exactly_the_bare_calls(cohort):
    """Arm 0' and the least-squares arm see the same cells, which is what makes them comparable."""
    splits, mic_path = cohort
    out, _, _, _ = resolve_mic_cells(splits, mic_path, "censored", "mic_only")
    assert (len(out.train), len(out.validate), len(out.holdout)) == (5, 1, 2)
    assert out.extra["holdout_st"] == []            # G11 was a bare call


def test_no_genome_changes_split(cohort):
    """The holdout authority is ``engine.splits``; a MIC arm subsets its splits and never re-derives them."""
    splits, mic_path = cohort
    out, _, _, _ = resolve_mic_cells(splits, mic_path, "censored", "mic_only")
    for name in ("train", "validate", "holdout"):
        before, after = set(getattr(splits, name)), set(getattr(out, name))
        assert after <= before
    assert set(out.train) & set(out.holdout) == set()


def test_the_resistant_share_before_and_after_is_reported(cohort):
    """Dropping cells with no MIC is not a random thinning, so the shift it causes goes on the record."""
    splits, mic_path = cohort
    _, _, _, report = resolve_mic_cells(splits, mic_path, "censored", "mic_only")
    assert report["by_split"]["train"]["n_before"] == 6
    assert report["by_split"]["train"]["n_after"] == 5
    assert report["by_split"]["train"]["pct_resistant_before"] == pytest.approx(50.0)
    assert report["by_split"]["train"]["pct_resistant_after"] == pytest.approx(60.0)
    assert report["cells"] == "mic_only" and report["objective"] == "censored"


def test_least_squares_on_all_cells_is_refused(cohort):
    """It would have to invent a point MIC for every bare call — the bias the comparison is measuring."""
    splits, mic_path = cohort
    with pytest.raises(ValueError, match="mic_only"):
        resolve_mic_cells(splits, mic_path, "mse", "all")


def test_an_empty_split_after_filtering_is_an_error_not_a_silent_zero(cohort):
    """A drug whose holdout has no measured MIC cannot be scored on MIC, and must say so."""
    splits, mic_path = cohort
    table = pd.read_csv(mic_path)
    table.loc[table["Sample"].isin(["G8", "G10"]), "basis"] = "bare_call"   # G9 was bare already
    table.to_csv(mic_path, index=False)
    with pytest.raises(RuntimeError, match="no usable MIC cells"):
        resolve_mic_cells(splits, mic_path, "mse", "mic_only")

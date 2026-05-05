"""Tests for predict_kleb_by_bacformer.pp.split_utils."""

import pandas as pd
import pytest

from predict_kleb_by_bacformer.pp.split_utils import add_splits


def _make_df(n: int) -> pd.DataFrame:
    """Return a minimal DataFrame with n unique Sample IDs."""
    return pd.DataFrame({"Sample": [f"S{i:03d}" for i in range(n)], "label": [i % 2 for i in range(n)]})


def test_add_splits_proportions_10():
    """70/10/20 split over 10 samples yields 7/1/2."""
    df = _make_df(10)
    result = add_splits(df, seed=42)
    splits = result.groupby("Sample")["train_val_eval"].first()
    assert (splits == "train").sum() == 7
    assert (splits == "validate").sum() == 1
    assert (splits == "evaluate").sum() == 2


def test_add_splits_proportions_100():
    """70/10/20 split over 100 samples: train+val+eval == 100, train ≈ 70."""
    df = _make_df(100)
    result = add_splits(df, seed=1)
    splits = result.groupby("Sample")["train_val_eval"].first()
    n_train = (splits == "train").sum()
    n_val = (splits == "validate").sum()
    n_eval = (splits == "evaluate").sum()
    assert n_train + n_val + n_eval == 100
    assert n_train == 70
    assert n_val == 10
    assert n_eval == 20


def test_add_splits_no_overlap():
    """No sample appears in more than one split."""
    df = _make_df(50)
    result = add_splits(df, seed=7)
    for split in ("train", "validate", "evaluate"):
        ids_in_split = set(result[result["train_val_eval"] == split]["Sample"])
        for other in ("train", "validate", "evaluate"):
            if other == split:
                continue
            ids_other = set(result[result["train_val_eval"] == other]["Sample"])
            assert ids_in_split.isdisjoint(ids_other), f"Overlap between {split} and {other}"


def test_add_splits_reproducible():
    """Same seed always produces the same assignment."""
    df = _make_df(20)
    r1 = add_splits(df, seed=99)
    r2 = add_splits(df, seed=99)
    pd.testing.assert_frame_equal(r1.reset_index(drop=True), r2.reset_index(drop=True))


def test_add_splits_different_seeds():
    """Different seeds produce different assignments (for sufficiently large n)."""
    df = _make_df(50)
    r1 = add_splits(df, seed=1)
    r2 = add_splits(df, seed=2)
    assert not r1["train_val_eval"].equals(r2["train_val_eval"])


def test_add_splits_preserves_rows():
    """add_splits does not drop or duplicate rows."""
    df = _make_df(30)
    result = add_splits(df, seed=3)
    assert len(result) == len(df)


@pytest.mark.parametrize("n", [1, 2, 9, 10, 11, 100])
def test_add_splits_all_assigned(n):
    """Every row receives a valid split label regardless of dataset size."""
    df = _make_df(n)
    result = add_splits(df, seed=1)
    assert result["train_val_eval"].isin(["train", "validate", "evaluate"]).all()
    assert result["train_val_eval"].notna().all()

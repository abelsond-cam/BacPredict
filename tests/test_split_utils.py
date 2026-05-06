"""Tests for predict_kleb_by_bacformer.pp.split_utils."""

import pandas as pd
import pytest

from predict_kleb_by_bacformer.pp.split_utils import add_splits, generate_kfold_splits


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


# ── generate_kfold_splits ─────────────────────────────────────────────────────


def test_kfold_evaluate_no_overlap_with_train_val():
    """Evaluate set must be disjoint from every fold's train and validate sets."""
    df = _make_df(50)
    evaluate_ids, folds = generate_kfold_splits(df, n_folds=5, seed=1)
    for train_ids, val_ids in folds:
        assert evaluate_ids.isdisjoint(train_ids), "evaluate overlaps with train"
        assert evaluate_ids.isdisjoint(val_ids), "evaluate overlaps with validate"


def test_kfold_fold_train_val_disjoint():
    """Within each fold, train and validate sets must be disjoint."""
    df = _make_df(50)
    _, folds = generate_kfold_splits(df, n_folds=5, seed=1)
    for i, (train_ids, val_ids) in enumerate(folds):
        assert train_ids.isdisjoint(val_ids), f"Fold {i}: train and val overlap"


def test_kfold_covers_non_evaluate_samples():
    """Each non-evaluate sample appears in exactly one fold's validate set."""
    df = _make_df(50)
    evaluate_ids, folds = generate_kfold_splits(df, n_folds=5, seed=1)
    all_val = set()
    for _, val_ids in folds:
        all_val |= val_ids
    non_evaluate = set(df["Sample"]) - evaluate_ids
    assert all_val == non_evaluate


def test_kfold_fold_sizes_approximately_equal():
    """Fold validate-set sizes differ by at most 1 sample."""
    df = _make_df(53)
    _, folds = generate_kfold_splits(df, n_folds=5, seed=1)
    val_sizes = [len(val_ids) for _, val_ids in folds]
    assert max(val_sizes) - min(val_sizes) <= 1


def test_kfold_evaluate_stable_across_seeds():
    """evaluate_ids is identical when only seed changes (evaluate_seed controls it)."""
    df = _make_df(50)
    eval1, _ = generate_kfold_splits(df, n_folds=5, seed=1, evaluate_seed=42)
    eval2, _ = generate_kfold_splits(df, n_folds=5, seed=7, evaluate_seed=42)
    assert eval1 == eval2


def test_kfold_evaluate_changes_with_evaluate_seed():
    """evaluate_ids changes when evaluate_seed changes."""
    df = _make_df(50)
    eval1, _ = generate_kfold_splits(df, n_folds=5, seed=1, evaluate_seed=1)
    eval2, _ = generate_kfold_splits(df, n_folds=5, seed=1, evaluate_seed=2)
    assert eval1 != eval2


def test_kfold_reproducible():
    """Same arguments always produce the same result."""
    df = _make_df(40)
    eval1, folds1 = generate_kfold_splits(df, n_folds=4, seed=3, evaluate_seed=9)
    eval2, folds2 = generate_kfold_splits(df, n_folds=4, seed=3, evaluate_seed=9)
    assert eval1 == eval2
    for (t1, v1), (t2, v2) in zip(folds1, folds2):
        assert t1 == t2
        assert v1 == v2


def test_kfold_different_seeds_give_different_folds():
    """Different seed values produce different fold assignments."""
    df = _make_df(50)
    _, folds1 = generate_kfold_splits(df, n_folds=5, seed=1)
    _, folds2 = generate_kfold_splits(df, n_folds=5, seed=2)
    # At least one fold's validate set should differ
    any_diff = any(v1 != v2 for (_, v1), (_, v2) in zip(folds1, folds2))
    assert any_diff

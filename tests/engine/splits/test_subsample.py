"""Unit tests for the class-balanced train-set reducer (``…splits.subsample.subsample_balanced``)."""

from __future__ import annotations

from bacpredict.engine.splits.subsample import subsample_balanced


def _label_map(n_pos: int, n_neg: int) -> tuple[list[str], dict[str, int]]:
    ids = [f"R{i}" for i in range(n_pos)] + [f"S{i}" for i in range(n_neg)]
    return ids, {s: (1 if s.startswith("R") else 0) for s in ids}


def test_subsample_balanced_caps_balances_and_is_deterministic() -> None:
    """Subsample hits ~max_n with both classes represented, is a subset, and is seed-deterministic."""
    ids, label_map = _label_map(100, 100)
    picked = subsample_balanced(ids, label_map, max_n=40, seed=7)
    assert len(picked) == 40
    assert set(picked) <= set(ids)
    n_pos = sum(label_map[s] for s in picked)
    assert n_pos == 20 and (len(picked) - n_pos) == 20  # balanced halves
    assert picked == subsample_balanced(ids, label_map, max_n=40, seed=7)  # deterministic


def test_subsample_balanced_backfills_from_larger_class() -> None:
    """When one class is too small, the target is met by backfilling from the larger class."""
    ids, label_map = _label_map(5, 100)  # 5 positives only
    picked = subsample_balanced(ids, label_map, max_n=40, seed=1)
    assert len(picked) == 40
    assert sum(label_map[s] for s in picked) == 5  # all 5 positives kept, rest negatives


def test_subsample_balanced_none_or_large_returns_all() -> None:
    """``max_n`` None or ≥ len returns the input unchanged (no subsampling)."""
    ids, label_map = _label_map(10, 10)
    assert subsample_balanced(ids, label_map, max_n=None, seed=1) == ids
    assert subsample_balanced(ids, label_map, max_n=999, seed=1) == ids

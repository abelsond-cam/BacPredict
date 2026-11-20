"""Unit tests for ``load_splits`` — the ONE reader of the materialized split table."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandas")
import pandas as pd

from bacpredict.engine.splits.generate_kfold_splits import build_split_table
from bacpredict.engine.splits.load_splits import load_split_ids, load_splits


def _table_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame({
        "Sample": ["a", "b", "c", "d", "e"],
        "ast_label": [1, 0, 1, 0, 0.5],   # 'e' ambiguous → dropped
        "split": ["train", "train", "validate", "holdout", "holdout"],
    })
    path = tmp_path / "rifampin_split.csv"
    df.to_csv(path, index=False)
    return path


def test_load_splits_returns_clean_map_and_splits(tmp_path: Path) -> None:
    """0/1 rows only; each split is a subset of the label map; the ambiguous row is excluded everywhere."""
    label_map, train, validate, holdout = load_splits(_table_csv(tmp_path))
    assert label_map == {"a": 1, "b": 0, "c": 1, "d": 0}
    assert train == ["a", "b"] and validate == ["c"] and holdout == ["d"]  # 'e' dropped
    assert "e" not in label_map


def test_load_splits_rejects_a_table_missing_columns(tmp_path: Path) -> None:
    """A table without the Sample/ast_label/split columns is rejected (guards a wrong path)."""
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"Sample": ["a"], "value": [1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing column"):
        load_splits(bad)


def test_build_then_load_roundtrip(tmp_path: Path) -> None:
    """Materialize a table then read it back — the split partition survives the round-trip."""
    sheet = tmp_path / "binary_ast.csv"
    pd.DataFrame({"Sample": [f"S{i:03d}" for i in range(120)], "rifampin": [i % 2 for i in range(120)]}).to_csv(
        sheet, index=False)
    table_path = tmp_path / "rifampin_split.csv"
    build_split_table(sheet, "rifampin").to_csv(table_path, index=False)

    label_map, train, validate, holdout = load_splits(table_path)
    assert len(label_map) == 120
    assert set(train) | set(validate) | set(holdout) == set(label_map)
    assert not (set(train) & set(holdout)) and not (set(validate) & set(holdout))


def _table_with_holdout_st(tmp_path: Path) -> Path:
    df = pd.DataFrame({
        "Sample": ["a", "b", "c", "d", "e", "f", "g"],
        "species": "paeruginosa",
        "st_key": ["paeruginosa:1", "paeruginosa:1", "paeruginosa:2", "paeruginosa:3", "paeruginosa:235",
                   "paeruginosa:235", "paeruginosa:235"],
        "ast_label": ["1", "0", "1", "0", "1", "0", "0.5"],   # 'g' conflicting → never scored
        "split": ["train", "train", "validate", "holdout", "holdout_st", "holdout_st", "holdout_st"],
    })
    path = tmp_path / "ciprofloxacin_split.csv"
    df.to_csv(path, index=False)
    return path


def test_load_split_ids_reads_holdout_st_clean_rows_only(tmp_path: Path) -> None:
    """holdout_st comes back through load_split_ids, 0/1 rows only; load_splits' three lists exclude it."""
    path = _table_with_holdout_st(tmp_path)
    assert load_split_ids(path, "holdout_st") == ["e", "f"]
    label_map, train, validate, holdout = load_splits(path)
    assert train == ["a", "b"] and validate == ["c"] and holdout == ["d"]
    assert label_map == {"a": 1, "b": 0, "c": 1, "d": 0, "e": 1, "f": 0}  # labels for holdout_st too


def test_load_split_ids_equals_load_splits_for_the_three_splits(tmp_path: Path) -> None:
    """For train/validate/holdout the accessor returns exactly load_splits' lists (one read, one filter)."""
    for path in (_table_csv(tmp_path), _table_with_holdout_st(tmp_path)):
        _, train, validate, holdout = load_splits(path)
        assert [load_split_ids(path, s) for s in ("train", "validate", "holdout")] == [train, validate, holdout]


def test_load_split_ids_rejects_an_unknown_split(tmp_path: Path) -> None:
    """A misspelt split is an error, not an empty list."""
    with pytest.raises(ValueError, match="not one of"):
        load_split_ids(_table_with_holdout_st(tmp_path), "holdout-st")


def _two_species_table(tmp_path: Path) -> Path:
    df = pd.DataFrame({
        "Sample": ["e1", "e2", "e3", "p1", "p2", "p3", "p4"],
        "species": ["enterobacterales"] * 3 + ["paeruginosa"] * 4,
        "st_key": "x:1",
        "ast_label": ["1", "0", "1", "0", "1", "0.5", "1"],   # 'p3' conflicting → never scored
        "split": ["train", "holdout", "holdout", "train", "holdout", "holdout", "holdout_st"],
    })
    path = tmp_path / "meropenem_split.csv"
    df.to_csv(path, index=False)
    return path


def test_species_filter_keeps_only_that_species(tmp_path: Path) -> None:
    """species= restricts the map and every split to that species' clean rows; None is the pooled read."""
    path = _two_species_table(tmp_path)
    label_map, train, validate, holdout = load_splits(path, species="paeruginosa")
    assert label_map == {"p1": 0, "p2": 1, "p4": 1}
    assert train == ["p1"] and validate == [] and holdout == ["p2"]
    assert load_split_ids(path, "holdout_st", species="paeruginosa") == ["p4"]
    assert load_split_ids(path, "holdout", species="enterobacterales") == ["e2", "e3"]
    assert load_splits(path)[3] == ["e2", "e3", "p2"]  # unfiltered: pooled, exactly as before


def test_species_filter_is_refused_on_a_table_without_species_or_an_absent_species(tmp_path: Path) -> None:
    """A TB/Kp table has no species column; a misspelt species is an error, not an empty read."""
    for read in (lambda p, sp: load_splits(p, species=sp), lambda p, sp: load_split_ids(p, "holdout", species=sp)):
        with pytest.raises(ValueError, match="no species column"):
            read(_table_csv(tmp_path), "paeruginosa")
        with pytest.raises(ValueError, match="has no rows"):
            read(_two_species_table(tmp_path), "pseudomonas")


def test_species_none_is_the_unfiltered_read(tmp_path: Path) -> None:
    """species=None returns exactly what the call without it does, on single- and multi-species tables."""
    for path in (_table_csv(tmp_path), _table_with_holdout_st(tmp_path), _two_species_table(tmp_path)):
        assert load_splits(path, species=None) == load_splits(path)
        for s in ("train", "validate", "holdout", "holdout_st"):
            assert load_split_ids(path, s, species=None) == load_split_ids(path, s)

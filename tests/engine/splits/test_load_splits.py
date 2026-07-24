"""Unit tests for ``load_splits`` — the ONE reader of the materialized split table."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandas")
import pandas as pd

from bacpredict.engine.splits.generate_kfold_splits import build_split_table
from bacpredict.engine.splits.load_splits import load_splits


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

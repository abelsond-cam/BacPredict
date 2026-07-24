"""Unit tests for the materialized split table — the single source of sampling + labels.

Covers ``build_split_table`` (does it reproduce the deployed ``generate_kfold_splits`` fold-0 partition?
does an ambiguous label still get a split slot?) and ``verify_table_matches_deployed`` (does the one-time
migration check catch an id-set / count mismatch?).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandas")
import json

import pandas as pd

from bacpredict.engine.splits.generate_kfold_splits import (
    build_split_table,
    generate_kfold_splits,
    verify_table_matches_deployed,
)


def _sheet(tmp_path: Path, n: int = 200, drug: str = "rifampin") -> Path:
    """A minimal AST sheet: Sample + a 0/1 drug column (2 ambiguous 0.5 rows to exercise clean-filtering)."""
    rows = {"Sample": [f"S{i:03d}" for i in range(n)], drug: [(i % 2) for i in range(n)]}
    rows[drug][0] = 0.5  # ambiguous — kept in the partition, dropped at scoring
    rows[drug][1] = 0.5
    path = tmp_path / "binary_ast.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_build_split_table_reproduces_deployed_fold0(tmp_path: Path) -> None:
    """The table's holdout/validate/train must equal generate_kfold_splits(fold 0) over the labelled set."""
    sheet = _sheet(tmp_path)
    table = build_split_table(sheet, "rifampin")  # canonical fold-0/seed-1/evaluate_seed-1

    labeled = pd.read_csv(sheet)
    labeled = labeled[labeled["rifampin"].notna()]
    evaluate_set, folds = generate_kfold_splits(labeled, n_folds=5, seed=1, evaluate_seed=1)
    train_set, val_set = folds[0]

    got = {row.Sample: row.split for row in table.itertuples()}
    assert {s for s, v in got.items() if v == "holdout"} == {str(x) for x in evaluate_set}
    assert {s for s, v in got.items() if v == "validate"} == {str(x) for x in val_set}
    assert {s for s, v in got.items() if v == "train"} == {str(x) for x in train_set}


def test_build_split_table_partitions_are_disjoint_and_cover(tmp_path: Path) -> None:
    """Every labelled genome lands in exactly one split (no leakage across splits, none dropped)."""
    table = build_split_table(_sheet(tmp_path), "rifampin")
    assert set(table["split"]) == {"train", "validate", "holdout"}
    assert table["Sample"].is_unique
    assert len(table) == 200  # all labelled rows (incl. the 2 ambiguous) get a slot


def test_ambiguous_label_kept_in_table_but_flagged_non_binary(tmp_path: Path) -> None:
    """An ambiguous (0.5) label still occupies its deployed split slot (load_splits drops it at read time)."""
    table = build_split_table(_sheet(tmp_path), "rifampin")
    amb = table[table["ast_label"] == 0.5]
    assert len(amb) == 2
    assert set(amb["split"]).issubset({"train", "validate", "holdout"})


def test_verify_matches_deployed_ok_and_mismatch(tmp_path: Path) -> None:
    """verify_table_matches_deployed passes on a matching n_evaluate/id-set and fails on a mismatch."""
    table = build_split_table(_sheet(tmp_path), "rifampin")
    holdout = table.loc[table["split"] == "holdout", "Sample"].astype(str).tolist()

    run = tmp_path / "run"
    run.mkdir()
    (run / "results.json").write_text(json.dumps({"split": {"source": "kfold", "n_evaluate": len(holdout),
                                                             "holdout_ids": holdout}}))
    assert verify_table_matches_deployed(table, run)["ok"] is True

    (run / "results.json").write_text(json.dumps({"split": {"source": "kfold", "n_evaluate": len(holdout) + 3}}))
    bad = verify_table_matches_deployed(table, run)
    assert bad["ok"] is False and "n_evaluate" in bad["reason"]

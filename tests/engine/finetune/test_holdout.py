"""Tests for the canonical split resolver (bacpredict.engine.finetune.holdout).

The keystone of the train/test-leakage fix: an FT-derived feature must be scored on the SAME holdout the
deployed model was evaluated on. These pin that :func:`resolve_deployed_holdout` reproduces a run's own
recorded split (k-fold or CSV) and that :func:`resolve_clean_splits` routes through it when a checkpoint is
given (and falls back to the CSV single-split otherwise).
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from bacpredict.engine.finetune.holdout import (
    read_split_provenance,
    resolve_clean_splits,
    resolve_deployed_holdout,
    resolve_holdouts,
)


def _sheet(tmp_path, n=60):
    """A tiny AST sheet: Sample, a binary ``rifampin`` col (with a few 0.5 ambiguous), a CSV train_val_eval col."""
    rows = [
        {
            "Sample": f"s{i}",
            "rifampin": 0.5 if i % 20 == 0 else (i % 2),  # i in {0,20,40} are ambiguous → dropped by clean split
            "train_val_eval": "evaluate" if i % 5 == 0 else ("validate" if i % 5 == 1 else "train"),
        }
        for i in range(n)
    ]
    csv = tmp_path / "binary_ast_with_split.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def _write_results(run_dir, split):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(json.dumps({"split": split}))


def test_resolve_deployed_holdout_kfold_reproduces_resolve_holdouts(tmp_path):
    csv = _sheet(tmp_path)
    run_dir = tmp_path / "run"
    _write_results(run_dir, {"source": "kfold", "n_folds": 5, "fold": 0, "evaluate_seed": 1, "n_evaluate": 12})
    ev, val, _lm, source, n_expected = resolve_deployed_holdout(run_dir, csv, "rifampin")
    assert source == "kfold" and n_expected == 12
    # identical to calling resolve_holdouts with the same kfold params (the whole point: replay the split).
    ev2, val2, _lm2, src2 = resolve_holdouts(str(csv), "rifampin", n_folds=5, fold=0, seed=1, evaluate_seed=1)
    assert ev == ev2 and val == val2 and src2 == "kfold"


def test_resolve_deployed_holdout_csv_source(tmp_path):
    csv = _sheet(tmp_path)
    run_dir = tmp_path / "run"
    _write_results(run_dir, {"source": "csv", "n_folds": None, "fold": None, "evaluate_seed": None, "n_evaluate": 5})
    ev, _val, _lm, source, _n = resolve_deployed_holdout(run_dir, csv, "rifampin")
    assert source == "csv"
    df = pd.read_csv(csv)
    assert set(ev) == set(df[df["train_val_eval"] == "evaluate"]["Sample"].astype(str))


def test_resolve_deployed_holdout_finds_results_in_parent(tmp_path):
    """A ``checkpoint-*/`` subdir resolves the run-root results.json (via the parent search)."""
    csv = _sheet(tmp_path)
    run_dir = tmp_path / "run"
    _write_results(run_dir, {"source": "kfold", "n_folds": 5, "fold": 0, "evaluate_seed": 1, "n_evaluate": 12})
    ckpt = run_dir / "checkpoint-100"
    ckpt.mkdir()
    ev, *_ = resolve_deployed_holdout(ckpt, csv, "rifampin")
    assert ev  # resolved from the parent run dir


def test_resolve_clean_splits_checkpoint_uses_kfold_holdout(tmp_path):
    csv = _sheet(tmp_path)
    run_dir = tmp_path / "run"
    _write_results(run_dir, {"source": "kfold", "n_folds": 5, "fold": 0, "evaluate_seed": 1, "n_evaluate": 12})
    lm, train, val, evaluate, info = resolve_clean_splits(csv, "rifampin", checkpoint_dir=run_dir)
    assert info["source"] == "kfold" and info["n_evaluate_expected"] == 12
    assert set(lm.values()) <= {0, 1}  # clean labels only — the 0.5 ambiguous rows are dropped
    assert not (set(train) & set(evaluate)) and not (set(val) & set(evaluate))  # disjoint
    # the checkpoint-mode evaluate equals the k-fold evaluate (clean), NOT the CSV column.
    ev_kfold, _v, _l, _s = resolve_holdouts(str(csv), "rifampin", n_folds=5, fold=0, seed=1, evaluate_seed=1)
    assert set(evaluate) == {s for s in ev_kfold if s in lm}


def test_resolve_clean_splits_no_checkpoint_uses_csv(tmp_path):
    csv = _sheet(tmp_path)
    lm, _train, _val, evaluate, info = resolve_clean_splits(csv, "rifampin")  # no checkpoint → CSV single-split
    assert info["source"] == "csv" and info["n_evaluate_expected"] is None
    df = pd.read_csv(csv)
    assert set(evaluate) == {s for s in df[df["train_val_eval"] == "evaluate"]["Sample"].astype(str) if s in lm}


def test_read_split_provenance_missing_block_raises(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "results.json").write_text(json.dumps({"metrics": {"auroc": 0.9}}))  # no 'split'
    with pytest.raises(ValueError, match="no 'split' provenance"):
        read_split_provenance(run_dir)


def test_resolve_deployed_holdout_missing_json_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="results.json"):
        resolve_deployed_holdout(tmp_path / "nonexistent", tmp_path / "x.csv", "rifampin")

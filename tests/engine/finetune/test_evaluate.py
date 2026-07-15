"""Tests for the shared evaluator (bacpredict.engine.finetune.evaluate) — plotting + holdout reconstruction."""

import numpy as np
import pandas as pd
import pytest
import torch

from bacpredict.engine.finetune.evaluate import (
    collate_fn,
    plot_auroc_bar,
    plot_roc_pr_grid,
    resolve_checkpoint_dir,
    resolve_evaluate_ids,
    resolve_holdouts,
)
from bacpredict.engine.finetune.split_utils import generate_kfold_splits


def _scores(n=40, seed=0):
    rng = np.random.default_rng(seed)
    y_true = np.concatenate([np.zeros(n // 2, dtype=int), np.ones(n // 2, dtype=int)])
    y_prob = np.concatenate([rng.uniform(0, 0.4, n // 2), rng.uniform(0.6, 1.0, n // 2)])
    return y_true, y_prob


def test_plot_roc_pr_grid_writes_file(tmp_path):
    entries = [("ceftriaxone", *_scores(seed=1)), ("meropenem", *_scores(seed=2))]
    out = tmp_path / "sub" / "grid.png"
    plot_roc_pr_grid(entries, out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_auroc_bar_writes_file_and_accepts_4_tuples(tmp_path):
    # Mix 3-tuples and 4-tuples (with operating threshold) — bar chart ignores threshold.
    entries = [
        ("ceftriaxone", *_scores(seed=1)),
        ("meropenem", *_scores(seed=2), 0.4),
        ("colistin", *_scores(seed=3)),
    ]
    out = tmp_path / "bars.png"
    plot_auroc_bar(entries, out, ylim=(0.5, 1.05), title="Test panel")
    assert out.exists() and out.stat().st_size > 0


def test_plot_auroc_bar_with_colorbar(tmp_path):
    # colorbar_label set → bars coloured by prevalence + colorbar drawn.
    entries = [
        ("ceftriaxone", *_scores(seed=1)),
        ("meropenem", *_scores(seed=2)),
        ("colistin", *_scores(seed=3)),
    ]
    out = tmp_path / "bars_cb.png"
    plot_auroc_bar(entries, out, colorbar_label="resistance rate", cmap="YlOrRd")
    assert out.exists() and out.stat().st_size > 0


def test_plot_roc_pr_grid_with_threshold_and_label(tmp_path):
    # 4-tuples (with operating threshold) + a custom prevalence label must render.
    entries = [("ceftriaxone", *_scores(seed=1), 0.4), ("meropenem", *_scores(seed=2), 0.35)]
    out = tmp_path / "grid_thr.png"
    plot_roc_pr_grid(entries, out, prevalence_label="resistance rate")
    assert out.exists() and out.stat().st_size > 0


def test_resolve_holdouts_kfold_returns_disjoint_eval_and_val(tmp_path):
    df = pd.DataFrame({"Sample": [f"S{i:03d}" for i in range(60)], "ceftriaxone": [i % 2 for i in range(60)]})
    csv = tmp_path / "binary_ast_with_split.csv"
    df.to_csv(csv, index=False)

    ev, val, label_map, source = resolve_holdouts(str(csv), "ceftriaxone", n_folds=5, fold=0, seed=1, evaluate_seed=1)
    eval_set, folds = generate_kfold_splits(
        df.assign(Sample=df["Sample"].astype(str)), n_folds=5, seed=1, evaluate_seed=1
    )
    _, expected_val = folds[0]

    assert source == "kfold"
    assert set(ev) == eval_set
    assert set(val) == expected_val
    assert set(ev).isdisjoint(set(val))
    assert len(label_map) == 60


def test_resolve_holdouts_csv_mode_splits_eval_and_val(tmp_path):
    df = pd.DataFrame({
        "Sample": ["A", "B", "C", "D"],
        "train_val_eval": ["train", "evaluate", "validate", "evaluate"],
        "ceftriaxone": [0, 1, 0, 1],
    })
    csv = tmp_path / "sheet.csv"
    df.to_csv(csv, index=False)

    ev, val, _, source = resolve_holdouts(str(csv), "ceftriaxone", n_folds=None, fold=0, seed=1, evaluate_seed=1)
    assert source == "csv"
    assert set(ev) == {"B", "D"}
    assert set(val) == {"C"}


def test_resolve_evaluate_ids_kfold_matches_split_utils(tmp_path):
    df = pd.DataFrame({"Sample": [f"S{i:03d}" for i in range(60)], "ceftriaxone": [i % 2 for i in range(60)]})
    csv = tmp_path / "binary_ast_with_split.csv"
    df.to_csv(csv, index=False)

    ids, label_map, source = resolve_evaluate_ids(str(csv), "ceftriaxone", n_folds=5, seed=1, evaluate_seed=1)
    expected_set, _ = generate_kfold_splits(df.assign(Sample=df["Sample"].astype(str)), n_folds=5, seed=1, evaluate_seed=1)

    assert source == "kfold"
    assert set(ids) == expected_set
    assert len(label_map) == 60


def test_resolve_evaluate_ids_csv_mode(tmp_path):
    df = pd.DataFrame({
        "Sample": ["A", "B", "C", "D"],
        "train_val_eval": ["train", "evaluate", "evaluate", "validate"],
        "ceftriaxone": [0, 1, 0, 1],
    })
    csv = tmp_path / "sheet.csv"
    df.to_csv(csv, index=False)

    ids, _, source = resolve_evaluate_ids(str(csv), "ceftriaxone", n_folds=None, seed=1, evaluate_seed=1)
    assert source == "csv"
    assert set(ids) == {"B", "C"}


def test_resolve_checkpoint_dir_prefers_config_then_subdir(tmp_path):
    # Direct dir with config.json → returned as-is.
    direct = tmp_path / "direct"
    direct.mkdir()
    (direct / "config.json").write_text("{}")
    assert resolve_checkpoint_dir(direct) == direct

    # Single checkpoint-* subdir (save_total_limit=1) → returned regardless of step.
    run = tmp_path / "run"
    (run / "checkpoint-3250").mkdir(parents=True)
    (run / "checkpoint-3250" / "config.json").write_text("{}")
    assert resolve_checkpoint_dir(run) == run / "checkpoint-3250"


def test_resolve_checkpoint_dir_honours_best_model_checkpoint(tmp_path):
    """When several checkpoints are kept, trainer_state.best_model_checkpoint wins over highest step."""
    import json

    run = tmp_path / "run"
    for step in (500, 5000):
        (run / f"checkpoint-{step}").mkdir(parents=True)
        (run / f"checkpoint-{step}" / "config.json").write_text("{}")
    # best is the *lower* step; path is intentionally stale (another host) → match by basename.
    (run / "checkpoint-5000" / "trainer_state.json").write_text(
        json.dumps({"best_model_checkpoint": "/some/other/host/run/checkpoint-500"})
    )
    assert resolve_checkpoint_dir(run) == run / "checkpoint-500"


def test_resolve_checkpoint_dir_raises_when_none(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint_dir(empty)


def test_collate_fn_pads_to_max_length():
    s1 = {
        "protein_embeddings": torch.randn(1, 3, 8),
        "attention_mask": torch.ones(1, 3),
        "contig_ids": torch.zeros(1, 3, dtype=torch.long),
        "labels": torch.tensor(1.0),
    }
    s2 = {
        "protein_embeddings": torch.randn(1, 5, 8),
        "attention_mask": torch.ones(1, 5),
        "contig_ids": torch.zeros(1, 5, dtype=torch.long),
        "labels": torch.tensor(0.0),
    }
    batch = collate_fn([s1, s2])
    assert batch["protein_embeddings"].shape == (2, 5, 8)
    assert batch["attention_mask"].shape == (2, 5)
    assert batch["labels"].tolist() == [1.0, 0.0]

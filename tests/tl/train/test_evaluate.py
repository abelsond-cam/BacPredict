"""Tests for the shared evaluator (tl.train.evaluate) — plotting + holdout reconstruction."""

import numpy as np
import pandas as pd
import torch

from tl.train.evaluate import collate_fn, plot_roc_pr_grid, resolve_evaluate_ids
from tl.train.split_utils import generate_kfold_splits


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

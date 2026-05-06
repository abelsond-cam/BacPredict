"""Shared utilities for creating train/validate/evaluate splits over genomic sample sets."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_splits(df: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    """Add train_val_eval column with a 70/10/20 split over unique Sample IDs.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``Sample`` column of unique sample identifiers.
    seed : int
        Random seed controlling the shuffle.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with a ``train_val_eval`` column whose values are one of
        ``"train"``, ``"validate"``, or ``"evaluate"``.
    """
    rng = np.random.default_rng(seed)
    sample_ids = df["Sample"].unique()
    rng.shuffle(sample_ids)

    n_total = len(sample_ids)
    n_train = int(0.7 * n_total)
    n_val = int(0.1 * n_total)
    train_ids = set(sample_ids[:n_train])
    val_ids = set(sample_ids[n_train : n_train + n_val])

    def _assign(sample_id: str) -> str:
        if sample_id in train_ids:
            return "train"
        if sample_id in val_ids:
            return "validate"
        return "evaluate"

    out = df.copy()
    out["train_val_eval"] = out["Sample"].map(_assign)
    return out


def generate_kfold_splits(
    df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 1,
    evaluate_fraction: float = 0.20,
    evaluate_seed: int = 1,
) -> tuple[set[str], list[tuple[set[str], set[str]]]]:
    """Return a fixed evaluate holdout and k-fold train/validate splits.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``Sample`` column.
    n_folds : int
        Number of cross-validation folds applied to the non-evaluate samples.
    seed : int
        Controls the shuffle before splitting into folds. Change this to
        generate different fold assignments without touching the evaluate set.
    evaluate_fraction : float
        Fraction of unique samples reserved as the fixed holdout.
    evaluate_seed : int
        Controls the shuffle used to select the evaluate set. Changing
        ``seed`` alone does NOT affect the evaluate set.

    Returns
    -------
    evaluate_ids : set[str]
        Fixed holdout sample IDs (identical for any ``seed`` when
        ``evaluate_seed`` and ``evaluate_fraction`` are unchanged).
    folds : list[tuple[set[str], set[str]]]
        Length-``n_folds`` list of ``(train_ids, validate_ids)`` pairs.
        Fold *i* uses fold *i* as validation and the remaining folds as training.
    """
    sample_ids = np.array(df["Sample"].unique())

    # Fixed evaluate set — determined only by evaluate_seed
    eval_rng = np.random.default_rng(evaluate_seed)
    eval_order = sample_ids.copy()
    eval_rng.shuffle(eval_order)
    n_evaluate = max(1, int(evaluate_fraction * len(eval_order)))
    evaluate_ids: set[str] = set(eval_order[-n_evaluate:])
    remaining = eval_order[:-n_evaluate].copy()

    # K-fold on remaining — determined by seed
    fold_rng = np.random.default_rng(seed)
    fold_rng.shuffle(remaining)
    fold_arrays = np.array_split(remaining, n_folds)

    folds: list[tuple[set[str], set[str]]] = []
    for i in range(n_folds):
        val_ids: set[str] = set(fold_arrays[i])
        train_ids: set[str] = set(np.concatenate([fold_arrays[j] for j in range(n_folds) if j != i]))
        folds.append((train_ids, val_ids))

    return evaluate_ids, folds

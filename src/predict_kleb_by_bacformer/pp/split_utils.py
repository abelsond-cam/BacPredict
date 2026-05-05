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

"""The ONE reader of the materialized split table.

Everything downstream — the fine-tuned genome-mean, every ESM/baclm per-segment LR, the catalogue one-hot,
and the trainer — calls :func:`load_splits` on a drug's ``<drug>_split.csv`` and nothing else. There is no
CSV ``train_val_eval`` fallback and no per-module k-fold derivation, so a fine-tuned feature is provably
scored on the deployed model's own holdout.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("Sample", "ast_label", "split")


def load_splits(
    split_table_path: str | Path,
) -> tuple[dict[str, int], list[str], list[str], list[str]]:
    """Read a drug's split table → ``(ast_label_map, train_ids, validate_ids, holdout_ids)``.

    The table (``<drug>_split.csv``, written by
    :func:`bacpredict.engine.splits.generate_kfold_splits.build_split_table`) must carry ``Sample``,
    ``ast_label``, and ``split`` columns. Only clean 0/1 ``ast_label`` rows are kept (an ambiguous label still
    occupies its deployed split slot in the table but is never scored); duplicate ``Sample`` keeps the first.

    Returns
    -------
    ast_label_map : dict[str, int]
        ``Sample -> 0/1`` AST phenotype label.
    train_ids, validate_ids, holdout_ids : list[str]
        Sample IDs in each split (each a subset of ``ast_label_map``), preserving table order.
    """
    df = pd.read_csv(split_table_path, low_memory=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Split table {split_table_path} is missing column(s) {missing}; has {list(df.columns)[:20]}")
    df["Sample"] = df["Sample"].astype(str)

    clean = df[df["ast_label"].isin([0, 1])].drop_duplicates(subset="Sample", keep="first")
    ast_label_map = {row["Sample"]: int(row["ast_label"]) for _, row in clean.iterrows()}

    def _ids(split_value: str) -> list[str]:
        return [s for s in clean.loc[clean["split"] == split_value, "Sample"] if s in ast_label_map]

    train_ids, validate_ids, holdout_ids = _ids("train"), _ids("validate"), _ids("holdout")
    logger.info(
        "load_splits(%s): train=%d validate=%d holdout=%d",
        Path(split_table_path).name, len(train_ids), len(validate_ids), len(holdout_ids),
    )
    return ast_label_map, train_ids, validate_ids, holdout_ids

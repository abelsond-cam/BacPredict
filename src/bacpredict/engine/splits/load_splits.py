"""The ONE reader of the materialized split table.

Everything downstream — the fine-tuned genome-mean, every ESM/baclm per-segment LR, the catalogue one-hot,
and the trainer — calls :func:`load_splits` on a drug's ``<drug>_split.csv`` and nothing else. There is no
CSV ``train_val_eval`` fallback and no per-module k-fold derivation, so a fine-tuned feature is provably
scored on the deployed model's own holdout.

:func:`load_split_ids` is the same read, for one split by name. It exists for the split :func:`load_splits`
does not return: ``holdout_st``, the zero-shot held-out ST of the multi-species tables
(:mod:`bacpredict.engine.splits.build_amr_holdout_splits`). Both go through :func:`_read_clean_rows`, so they
cannot disagree about which rows count.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("Sample", "ast_label", "split")
# Every split a table may carry. TB/Kp tables use the first three; ``holdout_st`` (an exact-ST zero-shot
# holdout) is written only by build_amr_holdout_splits and returned only by load_split_ids.
SPLIT_VALUES = ("train", "validate", "holdout", "holdout_st")


def _read_clean_rows(split_table_path: str | Path) -> pd.DataFrame:
    """The table's scoreable rows: clean 0/1 ``ast_label``, first row per ``Sample``, table order kept."""
    df = pd.read_csv(split_table_path, low_memory=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Split table {split_table_path} is missing column(s) {missing}; has {list(df.columns)[:20]}")
    df["Sample"] = df["Sample"].astype(str)
    return df[df["ast_label"].isin([0, 1])].drop_duplicates(subset="Sample", keep="first")


def load_splits(
    split_table_path: str | Path,
) -> tuple[dict[str, int], list[str], list[str], list[str]]:
    """Read a drug's split table → ``(ast_label_map, train_ids, validate_ids, holdout_ids)``.

    The table (``<drug>_split.csv``, written by
    :func:`bacpredict.engine.splits.generate_kfold_splits.build_split_table` or
    :mod:`bacpredict.engine.splits.build_amr_holdout_splits`) must carry ``Sample``, ``ast_label``, and
    ``split`` columns. Only clean 0/1 ``ast_label`` rows are kept (an ambiguous label still occupies its
    deployed split slot in the table but is never scored); duplicate ``Sample`` keeps the first.

    ``holdout_st`` rows are in ``ast_label_map`` but in none of the three lists: read them with
    :func:`load_split_ids`. So never derive a split as "``ast_label_map`` minus another split".

    Returns
    -------
    ast_label_map : dict[str, int]
        ``Sample -> 0/1`` AST phenotype label, over every clean row of the table.
    train_ids, validate_ids, holdout_ids : list[str]
        Sample IDs in each split (each a subset of ``ast_label_map``), preserving table order.
    """
    clean = _read_clean_rows(split_table_path)
    ast_label_map = {row["Sample"]: int(row["ast_label"]) for _, row in clean.iterrows()}

    def _ids(split_value: str) -> list[str]:
        return [s for s in clean.loc[clean["split"] == split_value, "Sample"] if s in ast_label_map]

    train_ids, validate_ids, holdout_ids = _ids("train"), _ids("validate"), _ids("holdout")
    logger.info(
        "load_splits(%s): train=%d validate=%d holdout=%d",
        Path(split_table_path).name, len(train_ids), len(validate_ids), len(holdout_ids),
    )
    n_st = int((clean["split"] == "holdout_st").sum())
    if n_st:
        logger.info("load_splits(%s): also %d holdout_st rows (read them with load_split_ids)",
                    Path(split_table_path).name, n_st)
    return ast_label_map, train_ids, validate_ids, holdout_ids


def load_split_ids(split_table_path: str | Path, split_value: str) -> list[str]:
    """Sample IDs of one split (clean 0/1 rows only, table order) — the accessor for ``holdout_st``.

    Same read and the same row filter as :func:`load_splits`, so for ``train``/``validate``/``holdout`` it
    returns exactly the list :func:`load_splits` does. Their labels are in :func:`load_splits`'s
    ``ast_label_map``. ``split_value`` must be one of :data:`SPLIT_VALUES`: a misspelt split would otherwise
    read as an empty one.
    """
    if split_value not in SPLIT_VALUES:
        raise ValueError(f"split_value {split_value!r} is not one of {SPLIT_VALUES}")
    clean = _read_clean_rows(split_table_path)
    return clean.loc[clean["split"] == split_value, "Sample"].tolist()

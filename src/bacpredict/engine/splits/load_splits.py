"""The ONE reader of the materialized split table.

Everything downstream — the fine-tuned genome-mean, every ESM/baclm per-segment LR, the catalogue one-hot,
and the trainer — calls :func:`load_splits` on a drug's ``<drug>_split.csv`` and nothing else. There is no
CSV ``train_val_eval`` fallback and no per-module k-fold derivation, so a fine-tuned feature is provably
scored on the deployed model's own holdout.

:func:`load_split_ids` is the same read, for one split by name. It exists for the split :func:`load_splits`
does not return: ``holdout_st``, the zero-shot held-out ST of the multi-species tables
(:mod:`bacpredict.engine.splits.build_amr_holdout_splits`). Both go through :func:`_read_clean_rows`, so they
cannot disagree about which rows count.

Both take an optional ``species``. A cross-species table pools its species in every split, so scoring one
species (PsA, for the cross-species comparison) means asking the reader for it: ``species="paeruginosa"``
returns that species' rows of the table and nothing else. Never filter a table's ``species`` column yourself —
that is a second reader. ``species=None`` (the default) reads the whole table, exactly as before.
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


def _read_clean_rows(split_table_path: str | Path, species: str | None = None) -> pd.DataFrame:
    """The table's scoreable rows: clean 0/1 ``ast_label``, first row per ``Sample``, table order kept.

    With ``species``, only that species' rows of those (the filter comes after the dedup, so the result is always
    the unfiltered read restricted to the species). Refuses a species filter on a table with no ``species``
    column (TB/Kp tables) or a species with no rows in the table: a misspelt species would otherwise read as an
    empty one.
    """
    df = pd.read_csv(split_table_path, low_memory=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Split table {split_table_path} is missing column(s) {missing}; has {list(df.columns)[:20]}")
    df["Sample"] = df["Sample"].astype(str)
    clean = df[df["ast_label"].isin([0, 1])].drop_duplicates(subset="Sample", keep="first")
    if species is None:
        return clean
    if "species" not in df.columns:
        raise ValueError(f"species={species!r} asked of {split_table_path}, which has no species column "
                         "(a single-species table)")
    present = sorted(df["species"].dropna().astype(str).unique())
    if species not in present:
        raise ValueError(f"species {species!r} has no rows in {split_table_path}; it has {present}")
    return clean[clean["species"].astype(str) == species]


def load_splits(
    split_table_path: str | Path,
    *,
    species: str | None = None,
) -> tuple[dict[str, int], list[str], list[str], list[str]]:
    """Read a drug's split table → ``(ast_label_map, train_ids, validate_ids, holdout_ids)``.

    The table (``<drug>_split.csv``, written by
    :func:`bacpredict.engine.splits.generate_kfold_splits.build_split_table` or
    :mod:`bacpredict.engine.splits.build_amr_holdout_splits`) must carry ``Sample``, ``ast_label``, and
    ``split`` columns. Only clean 0/1 ``ast_label`` rows are kept (an ambiguous label still occupies its
    deployed split slot in the table but is never scored); duplicate ``Sample`` keeps the first.

    ``holdout_st`` rows are in ``ast_label_map`` but in none of the three lists: read them with
    :func:`load_split_ids`. So never derive a split as "``ast_label_map`` minus another split".

    Parameters
    ----------
    split_table_path : str or Path
        The drug's ``<drug>_split.csv``.
    species : str, optional
        Keep only this species' rows (the table's ``species`` column), for the map and all three lists. Refused on
        a table without a ``species`` column or with no rows of that species. ``None`` reads the whole table.

    Returns
    -------
    ast_label_map : dict[str, int]
        ``Sample -> 0/1`` AST phenotype label, over every clean row of the table (of ``species``, if given).
    train_ids, validate_ids, holdout_ids : list[str]
        Sample IDs in each split (each a subset of ``ast_label_map``), preserving table order.
    """
    clean = _read_clean_rows(split_table_path, species)
    ast_label_map = {row["Sample"]: int(row["ast_label"]) for _, row in clean.iterrows()}

    def _ids(split_value: str) -> list[str]:
        return [s for s in clean.loc[clean["split"] == split_value, "Sample"] if s in ast_label_map]

    train_ids, validate_ids, holdout_ids = _ids("train"), _ids("validate"), _ids("holdout")
    where = Path(split_table_path).name if species is None else f"{Path(split_table_path).name}, species={species}"
    logger.info("load_splits(%s): train=%d validate=%d holdout=%d",
                where, len(train_ids), len(validate_ids), len(holdout_ids))
    n_st = int((clean["split"] == "holdout_st").sum())
    if n_st:
        logger.info("load_splits(%s): also %d holdout_st rows (read them with load_split_ids)", where, n_st)
    return ast_label_map, train_ids, validate_ids, holdout_ids


def load_split_ids(split_table_path: str | Path, split_value: str, *, species: str | None = None) -> list[str]:
    """Sample IDs of one split (clean 0/1 rows only, table order) — the accessor for ``holdout_st``.

    Same read and the same row filter as :func:`load_splits`, so for ``train``/``validate``/``holdout`` it
    returns exactly the list :func:`load_splits` does, with the same ``species``. Their labels are in
    :func:`load_splits`'s ``ast_label_map``. ``split_value`` must be one of :data:`SPLIT_VALUES`: a misspelt split
    would otherwise read as an empty one. ``species`` keeps only that species' rows, as in :func:`load_splits`.
    """
    if split_value not in SPLIT_VALUES:
        raise ValueError(f"split_value {split_value!r} is not one of {SPLIT_VALUES}")
    clean = _read_clean_rows(split_table_path, species)
    return clean.loc[clean["split"] == split_value, "Sample"].tolist()

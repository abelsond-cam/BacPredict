r"""Build a ``Sample<TAB>cluster`` file for the within-lineage permutation null.

The permutation-null wrappers shuffle case/control labels *within* a lineage grouping so that the
phenotype↔lineage correlation (and hence the between-lineage population structure) is preserved
while genuine within-lineage signal is destroyed (see ``permute_phenotype_within_lineage.py``).
That shuffle needs one grouping column. This builder emits it at either resolution:

* ``--column "Sublineage"`` — the finest grouping (SL level, the strictest null); or
* ``--column "Clonal group"`` — the coarser clonal-group grouping (CG level; note the space).

It aligns the grouping to the phenotype's samples and writes a headerless ``Sample<TAB>value``
file in exactly the format ``permute_phenotype_within_lineage.py --clusters`` expects.

Resolution order for the grouping column:

1. If the cohort split CSV already carries ``--column``, read it straight from there.
2. Otherwise, if ``--metadata`` is given, read ``Sample`` + ``--column`` from the metadata table
   (``metadata_v2_all_samples_and_columns.tsv``) and join on ``Sample`` — the fallback for when
   the split CSV lacks the clonal-group column.

Missing/blank groups collapse to ``"unknown"``. A single resulting cluster is a hard warning: a
one-group file makes the within-lineage shuffle a *global* shuffle, which no longer preserves
structure — the null would be meaningless.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_UNKNOWN = "unknown"


def _read_group_source(split_csv: Path, column: str, metadata: Path | None) -> pd.DataFrame:
    """Return a ``Sample`` + ``column`` frame from the split CSV if present, else the metadata join.

    Parameters
    ----------
    split_csv : Path
        Cohort split CSV (leads with a ``Sample`` column).
    column : str
        Grouping column to resolve, e.g. ``"Sublineage"`` or ``"Clonal group"``.
    metadata : Path or None
        ``metadata_v2_all_samples_and_columns.tsv`` used as a fallback source when ``column`` is
        absent from the split CSV; ``None`` disables the fallback.

    Returns
    -------
    pandas.DataFrame
        Two columns, ``Sample`` and ``column`` (both as strings, not yet aligned to phenotype).
    """
    header = pd.read_csv(split_csv, nrows=0)
    if column in header.columns:
        df = pd.read_csv(split_csv, usecols=["Sample", column], low_memory=False)
        src = f"split CSV {split_csv}"
    elif metadata is not None:
        df = pd.read_csv(metadata, sep="\t", usecols=["Sample", column], low_memory=False)
        src = f"metadata {metadata}"
    else:
        raise SystemExit(
            f"column {column!r} not in split CSV {split_csv} and no --metadata fallback given "
            f"(split CSV columns: {list(header.columns)})"
        )
    print(f"read {column!r} for {len(df)} rows from {src}", file=sys.stderr)
    df["Sample"] = df["Sample"].astype(str)
    return df


def build_clusters(split_csv: Path, phenotype: Path, column: str, metadata: Path | None) -> pd.DataFrame:
    """Build the phenotype-aligned ``Sample`` / cluster frame for the within-lineage shuffle.

    Parameters
    ----------
    split_csv : Path
        Cohort split CSV carrying (or joinable to) the grouping ``column``.
    phenotype : Path
        pyseer ``phenotype.tsv``; its first column is the sample-id list to align to.
    column : str
        Grouping column, e.g. ``"Sublineage"`` (SL) or ``"Clonal group"`` (CG).
    metadata : Path or None
        Fallback metadata table when ``column`` is absent from the split CSV.

    Returns
    -------
    pandas.DataFrame
        Two columns, ``Sample`` and ``cluster``, restricted to the phenotype's samples.
    """
    samples = set(pd.read_csv(phenotype, sep="\t")["samples"].astype(str))
    df = _read_group_source(split_csv, column, metadata)
    df = df.drop_duplicates(subset=["Sample"])
    df = df[df["Sample"].isin(samples)]
    cluster = (
        df[column]
        .fillna(_UNKNOWN)
        .astype(str)
        .replace({"": _UNKNOWN, "nan": _UNKNOWN, "NaN": _UNKNOWN, "None": _UNKNOWN})
    )
    out = pd.DataFrame({"Sample": df["Sample"].to_numpy(), "cluster": cluster.to_numpy()})
    missing = len(samples) - len(out)
    if missing:
        print(f"WARNING: {missing} of {len(samples)} phenotype samples had no {column!r} value "
              f"(dropped from the cluster file — they fall to '_unassigned' in the shuffle)", file=sys.stderr)
    return out


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split-csv", type=Path, required=True, help="Cohort split CSV (leads with 'Sample').")
    p.add_argument("--phenotype", type=Path, required=True, help="pyseer phenotype.tsv (align target).")
    p.add_argument("--column", required=True, help="Grouping column, e.g. 'Sublineage' or 'Clonal group'.")
    p.add_argument("--metadata", type=Path, default=None,
                   help="metadata_v2 TSV fallback when --column is absent from the split CSV.")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    df = build_clusters(args.split_csv, args.phenotype, args.column, args.metadata)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, sep="\t", header=False, index=False)
    n_clusters = df["cluster"].nunique()
    print(f"wrote {args.out}: {len(df)} samples, {n_clusters} clusters (column {args.column!r})", file=sys.stderr)
    if n_clusters <= 1:
        print("WARNING: <=1 cluster — the within-lineage shuffle degenerates to a GLOBAL shuffle "
              "(structure no longer preserved). Check the grouping column.", file=sys.stderr)


if __name__ == "__main__":
    main()

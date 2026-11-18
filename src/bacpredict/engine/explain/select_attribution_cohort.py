r"""Choose which genomes to attribute, and pin the choice to a committed CSV.

Ablation costs one forward pass per protein, so the full holdout of a large cohort is not affordable
and a subsample is taken instead. The subsample is **stratified on phenotype and lineage**, which is
not cosmetic: the control that decides what the attribution means is a within-lineage repeat of the
resistant-vs-susceptible contrast, and that control is only possible if the lineages were sampled
deliberately here.

The holdout is resolved through :func:`bacpredict.engine.splits.load_splits.load_splits` over the
drug's materialised ``<drug>_split.csv``, and through nothing else. Re-deriving a holdout any other
way — in particular by reading a label CSV's ``train_val_eval`` column — is the 2026-07 read-out
leak, which inflated one drug's reported AUROC by more than a tenth.

Examples
--------
Pin 1,500 TB rifampin genomes, stratified on phenotype and main lineage::

    python -m bacpredict.engine.explain.select_attribution_cohort \\
        --organism tb --drug rifampin --n-genomes 1500 \\
        --lineage-csv "$B/train_tb_ast/tbprofiler/tbprofiler_lineage.csv" \\
        --out-csv attribution_cohort_rifampin.csv

Take every Kp ciprofloxacin holdout genome (no subsampling)::

    python -m bacpredict.engine.explain.select_attribution_cohort \\
        --organism kp --drug ciprofloxacin --out-csv attribution_cohort_ciprofloxacin.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bacpredict.engine.config import organism as resolve_organism
from bacpredict.engine.splits.load_splits import load_splits

logger = logging.getLogger(__name__)

UNKNOWN_LINEAGE = "unknown"


def default_split_table(organism_key: str, drug: str) -> Path:
    """Return the conventional ``<data root>/processed/<task>/splits/<drug>_split.csv``.

    Parameters
    ----------
    organism_key : str
        ``"tb"`` or ``"kp"``.
    drug : str
        Drug name as it appears in the split-table filename (e.g. ``"rifampin"``).

    Returns
    -------
    pathlib.Path
        Path to the drug's split table. Not checked for existence — the caller reports that.
    """
    return resolve_organism(organism_key).data_root() / "splits" / f"{drug}_split.csv"


def attach_lineage(frame: pd.DataFrame, lineage_csv: Path | None, lineage_column: str) -> pd.DataFrame:
    """Add a ``lineage`` column to ``frame``, defaulting to ``"unknown"`` where unavailable.

    Parameters
    ----------
    frame : pandas.DataFrame
        Must carry a ``Sample`` column.
    lineage_csv : pathlib.Path or None
        A CSV with ``Sample`` plus ``lineage_column`` (TB-Profiler's ``tbprofiler_lineage.csv``).
        When ``None``, every genome is assigned ``"unknown"`` and stratification falls back to
        phenotype alone.
    lineage_column : str
        Which column of ``lineage_csv`` to use (e.g. ``"main_lineage"``).

    Returns
    -------
    pandas.DataFrame
        ``frame`` with a ``lineage`` column added.

    Raises
    ------
    ValueError
        If ``lineage_csv`` lacks ``Sample`` or ``lineage_column``.
    """
    out = frame.copy()
    if lineage_csv is None:
        out["lineage"] = UNKNOWN_LINEAGE
        return out

    lineage = pd.read_csv(lineage_csv, low_memory=False)
    missing = [c for c in ("Sample", lineage_column) if c not in lineage.columns]
    if missing:
        raise ValueError(f"Lineage CSV {lineage_csv} is missing column(s) {missing}; has {list(lineage.columns)}")
    lineage["Sample"] = lineage["Sample"].astype(str)
    mapping = lineage.drop_duplicates(subset="Sample", keep="first").set_index("Sample")[lineage_column]
    out["lineage"] = out["Sample"].map(mapping).fillna(UNKNOWN_LINEAGE).astype(str)

    n_unknown = int((out["lineage"] == UNKNOWN_LINEAGE).sum())
    if n_unknown:
        logger.warning("%d of %d genomes have no %s and were binned as %r", n_unknown, len(out), lineage_column,
                       UNKNOWN_LINEAGE)
    return out


def allocate_by_largest_remainder(sizes: pd.Series, total: int) -> pd.Series:
    """Split ``total`` draws across strata in proportion to ``sizes``, by the largest-remainder rule.

    Proportional allocation keeps the subsample's phenotype and lineage composition close to the
    holdout's, so a statistic computed on it is not reweighted by the act of subsampling. Largest
    remainder is used rather than rounding so the parts sum to ``total`` exactly.

    Parameters
    ----------
    sizes : pandas.Series
        Stratum label → number of available genomes in that stratum.
    total : int
        Number of genomes to draw in all. Clamped to ``sizes.sum()``.

    Returns
    -------
    pandas.Series
        Stratum label → number to draw, summing to ``min(total, sizes.sum())`` and never exceeding
        the stratum's own size.
    """
    available = int(sizes.sum())
    total = min(int(total), available)
    if total <= 0 or available == 0:
        return pd.Series(0, index=sizes.index, dtype=int)

    exact = sizes.astype(float) * (total / available)
    floors = np.floor(exact).astype(int)
    # Cap at the stratum size before distributing the remainder, so a stratum can never be
    # over-drawn by the rounding step.
    floors = np.minimum(floors, sizes.astype(int))
    shortfall = total - int(floors.sum())

    if shortfall > 0:
        headroom = sizes.astype(int) - floors
        order = (exact - np.floor(exact)).where(headroom > 0, other=-1.0).sort_values(ascending=False)
        for label in order.index:
            if shortfall == 0:
                break
            if headroom[label] <= 0:
                continue
            floors[label] += 1
            shortfall -= 1
    return floors.astype(int)


def select_cohort(
    split_table: Path,
    *,
    n_genomes: int | None,
    lineage_csv: Path | None = None,
    lineage_column: str = "main_lineage",
    seed: int = 0,
) -> pd.DataFrame:
    """Resolve a drug's holdout and draw the stratified attribution cohort from it.

    Parameters
    ----------
    split_table : pathlib.Path
        The drug's ``<drug>_split.csv``.
    n_genomes : int or None
        How many genomes to draw. ``None`` (or a number at least the holdout size) takes the whole
        holdout.
    lineage_csv : pathlib.Path, optional
        Lineage table for stratification; see :func:`attach_lineage`.
    lineage_column : str, default "main_lineage"
        Column of ``lineage_csv`` to stratify on.
    seed : int, default 0
        Seed for the within-stratum draw. Recorded in the output so a cohort is reproducible.

    Returns
    -------
    pandas.DataFrame
        Columns ``Sample``, ``ast_label``, ``lineage``, ``stratum``, sorted by ``Sample``.

    Raises
    ------
    ValueError
        If the split table yields no holdout genomes.
    """
    ast_label_map, _train, _validate, holdout_ids = load_splits(split_table)
    if not holdout_ids:
        raise ValueError(f"{split_table} yielded an empty holdout; nothing to attribute.")

    frame = pd.DataFrame({"Sample": holdout_ids})
    frame["ast_label"] = frame["Sample"].map(ast_label_map).astype(int)
    frame = attach_lineage(frame, lineage_csv, lineage_column)
    frame["stratum"] = frame["ast_label"].astype(str) + "|" + frame["lineage"]

    if n_genomes is None or n_genomes >= len(frame):
        logger.info("Taking the whole holdout: %d genomes (requested %s)", len(frame), n_genomes)
        return frame.sort_values("Sample").reset_index(drop=True)

    sizes = frame.groupby("stratum").size()
    draws = allocate_by_largest_remainder(sizes, n_genomes)
    rng = np.random.default_rng(seed)
    picked = [
        group.iloc[rng.permutation(len(group))[: draws[label]]]
        for label, group in frame.groupby("stratum")
        if draws[label] > 0
    ]
    out = pd.concat(picked, ignore_index=True).sort_values("Sample").reset_index(drop=True)
    logger.info("Drew %d of %d holdout genomes across %d strata (seed=%d)", len(out), len(frame), int((draws > 0).sum()),
                seed)
    return out


def log_composition(frame: pd.DataFrame) -> None:
    """Log the cohort's phenotype and per-stratum composition, so a run's record is self-describing."""
    n_res = int((frame["ast_label"] == 1).sum())
    logger.info("Cohort: %d genomes — %d resistant, %d susceptible", len(frame), n_res, len(frame) - n_res)
    counts = frame.groupby(["lineage", "ast_label"]).size().unstack(fill_value=0)
    for lineage, row in counts.iterrows():
        logger.info("  %-18s R=%-6d S=%-6d", lineage, int(row.get(1, 0)), int(row.get(0, 0)))


def main() -> None:
    """CLI entry point — resolve a drug's holdout, draw the cohort, write the pinned CSV."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--organism", required=True, choices=("tb", "kp"), help="Organism key.")
    parser.add_argument("--drug", required=True, help="Drug name, as in <drug>_split.csv (e.g. rifampin).")
    parser.add_argument("--split-table", type=Path, default=None,
                        help="Override the split table path (default: <data root>/.../splits/<drug>_split.csv).")
    parser.add_argument("--n-genomes", type=int, default=None,
                        help="Subsample size. Omit to take the whole holdout.")
    parser.add_argument("--lineage-csv", type=Path, default=None,
                        help="CSV with Sample + a lineage column, for stratification.")
    parser.add_argument("--lineage-column", default="main_lineage", help="Lineage column to stratify on.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the within-stratum draw.")
    parser.add_argument("--out-csv", type=Path, required=True, help="Where to write the cohort table.")
    args = parser.parse_args()

    split_table = args.split_table or default_split_table(args.organism, args.drug)
    if not Path(split_table).exists():
        raise FileNotFoundError(f"Split table not found: {split_table}")

    frame = select_cohort(
        Path(split_table),
        n_genomes=args.n_genomes,
        lineage_csv=args.lineage_csv,
        lineage_column=args.lineage_column,
        seed=args.seed,
    )
    log_composition(frame)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_csv, index=False)
    logger.info("Wrote %d rows to %s", len(frame), args.out_csv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()

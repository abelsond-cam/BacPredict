r"""Each (genome, drug) MIC as the **interval** its latent log2 MIC lies in, in dilutions from the R breakpoint.

A reported MIC is never a point measurement. ``<= 2`` means "somewhere at or below 2"; ``> 8`` means "above 8";
and even a bare ``= 4`` means "growth at 2, none at 4", i.e. somewhere in ``(2, 4]`` — the assay reads a doubling
ladder, so an exact MIC is itself an interval one dilution wide. Imputing a point (``>= X`` -> ``X``, ``> X`` ->
one dilution up) throws that structure away and biases the fit toward the censoring bounds; carrying the interval
is the information the censored likelihood needs (:mod:`bacpredict.engine.finetune.censored_mic_loss`).

**The interval table** (agreed with David, 2026-11-26). ``X`` is the reported MIC and all bounds are log2 mg/L;
the written table shifts every bound by ``-log2(r_gt)``, so ``0`` is the EUCAST R breakpoint and the unit is one
doubling dilution. Intervals are half-open ``(lo, hi]``, matching "resistant iff MIC > R".

=================  ==========================================  ==================
reported           latent interval                             imputed point
=================  ==========================================  ==================
``= X``            ``(next doubling below X, X]``              ``X``
``> X``            ``(X, inf)``                                ``2X``
``>= X``           ``(next doubling below X, inf)``            ``X``
``<= X``           ``(-inf, X]``                               ``X``
``< X``            ``(-inf, X]``                               ``X/2``
``> X, < U``       ``(X, U]``                                  geometric mean
bare ``R``         ``(R, inf)``                                none
bare ``S``         ``(-inf, R]``                               none
=================  ==========================================  ==================

The *imputed point* column is **not** the estimator — it is David's long-standing hand rule, carried here only so
the MSE arm that uses it is built from the same table as the censored arm and the two differ in nothing else.

**Snapping.** A value within :data:`LADDER_TOL` of a power of two *is* that power of two: the stores hold
``16.001`` and ``4.001`` artefacts (two rows each in the PsA v2 tests), and ``log2(16.001) = 4.00009`` would
otherwise make ``= 16.001`` the interval ``(4, 5]`` — on the resistant side of an R = 16 breakpoint that the same
test is called susceptible by. Off-ladder values (0.19, 0.094, 640, …; 0.2-0.4 % of PsA rows) keep their real
``log2`` and take ``(floor, ceil]``.

**Collapsing a cell.** Several tests of one ``(Sample, drug)`` are **intersected** — ``lo = max(lo_i)``,
``hi = min(hi_i)`` — which is also what deduplicates the two ingests of the same assay (99 % of PsA repeat cells
are one AMR-GNN row and one CABBAGE v2 row of the same measurement, so the intersection is that measurement). When
the intersection is empty the tests genuinely disagree: the **hull** is taken instead and the cell is flagged
``hull``, never silently resolved to one of them.

**This module does not define a binary label.** It computes the interval's own call (R iff ``lo >= 0``, S iff
``hi <= 0``) only to *check* it against the split table, which stays the one authority (``engine.splits``). A cell
whose interval contradicts its split-table label is written to ``mic_interval_conflicts.csv`` and dropped from the
MIC arms, rather than overriding either.

Example
-------
::

    python -m bacpredict.engine.ast_labels.mic_intervals \
        --tests $AMR/processed/paeruginosa_cabbage_v2/ast_tests_called.parquet \
        --split-dir $AMR/processed/cross_species_psaonly_v2/splits \
        --drugs amikacin cefepime ceftazidime ciprofloxacin meropenem \
        --out-dir $AMR/processed/psaonly_mic/intervals
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bacpredict.engine.ast_labels.breakpoints import DEFAULT_BREAKPOINTS, load_breakpoints, resolve_r_gt
from bacpredict.engine.splits.load_splits import load_splits

logger = logging.getLogger(__name__)

#: ``|log2(x) - round(log2(x))| <= LADDER_TOL`` makes ``x`` that power of two. 0.1 in log2 is +-7 %, far wider
#: than the ``16.001`` artefacts (9e-5) and far narrower than the gap to the half-step ladder (0.5).
LADDER_TOL = 0.1

#: ``basis`` of a test called from an MIC against a breakpoint (``engine.ast_labels.ast_tests``).
MIC_BASIS = "mic"
#: ``basis`` of a test that kept the source's own R/S call (disk diffusion, a bare call).
BARE_BASIS = "source_call"

#: Censoring class of a collapsed cell.
CENSOR_EXACT, CENSOR_INTERVAL, CENSOR_LEFT, CENSOR_RIGHT, CENSOR_BARE = (
    "exact", "interval", "left", "right", "bare")

OUT_COLUMNS = (
    "Sample", "drug", "species", "lo", "hi", "point", "interval_label",
    "basis", "censor", "width", "n_tests", "n_mic_tests", "hull",
)


def snap_to_ladder(value: np.ndarray | pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """``log2`` of each MIC, snapped to the doubling ladder where it is within :data:`LADDER_TOL` of it.

    Returns
    -------
    l2 : numpy.ndarray
        ``log2(value)``, replaced by the exact integer where the value is on the ladder. Non-positive or missing
        values give ``nan``.
    on_ladder : numpy.ndarray
        Whether each value snapped.
    """
    v = np.asarray(value, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        l2 = np.where(v > 0, np.log2(np.where(v > 0, v, 1.0)), np.nan)
    k = np.round(l2)
    on = np.isfinite(l2) & (np.abs(l2 - k) <= LADDER_TOL)
    return np.where(on, k, l2), on


def intervals_from_reported_mic(
    sign: pd.Series | np.ndarray,
    value: pd.Series | np.ndarray,
    upper: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(lo, hi, point)`` in log2 mg/L for each MIC test, by the interval table in the module docstring.

    ``lo``/``hi`` bound the half-open interval ``(lo, hi]`` the latent log2 MIC lies in; ``point`` is David's
    hand-imputed value for the MSE arm. A sign outside
    :data:`bacpredict.engine.ast_labels.breakpoints.SIGNS`, or a non-positive value, gives ``nan`` throughout.
    """
    l2, on = snap_to_ladder(value)
    k = np.round(l2)
    with np.errstate(invalid="ignore"):
        step_lo = np.where(on, k - 1.0, np.floor(l2))
        step_hi = np.where(on, k, np.ceil(l2))
    l2u, _ = snap_to_ladder(upper)

    s = np.asarray(sign, dtype=object)
    neg_inf, pos_inf = np.full(l2.shape, -np.inf), np.full(l2.shape, np.inf)
    is_eq, is_gt, is_ge = s == "=", s == ">", s == ">="
    is_le, is_lt, is_both = s == "<=", s == "<", s == ">,<"

    lo = np.select([is_eq, is_gt, is_ge, is_le | is_lt, is_both],
                   [step_lo, l2, step_lo, neg_inf, l2], np.nan)
    hi = np.select([is_eq, is_gt, is_ge, is_le | is_lt, is_both],
                   [step_hi, pos_inf, pos_inf, l2, l2u], np.nan)
    point = np.select([is_eq, is_ge, is_gt, is_le, is_lt, is_both],
                      [l2, l2, l2 + 1.0, l2, l2 - 1.0, (l2 + l2u) / 2.0], np.nan)
    # a sign we recognise but whose value did not parse carries no interval at all
    bad = ~np.isfinite(l2)
    return np.where(bad, np.nan, lo), np.where(bad, np.nan, hi), np.where(bad, np.nan, point)


def ast_tests_to_intervals(tests: pd.DataFrame, breakpoints: pd.DataFrame) -> pd.DataFrame:
    """Per test: ``lo``, ``hi``, ``point`` in dilutions from the drug's R breakpoint, plus the row's class.

    MIC-based tests (``basis == "mic"``) get the interval table; bare calls (``basis == "source_call"`` with a
    0/1 ``y_test``) get the open interval their call implies — ``(0, inf)`` for R, ``(-inf, 0]`` for S. Every
    other row (no breakpoint, undecided censored MIC, no call) is dropped, and the counts logged.
    """
    t = tests.copy()
    t["Sample"] = t["Sample"].astype(str)
    basis = t["basis"].astype(str)
    r_gt = t["r_gt_mg_l"].to_numpy(dtype=float) if "r_gt_mg_l" in t.columns else resolve_r_gt(
        t["species"].astype(str), t["drug"].astype(str), breakpoints)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_r = np.where(r_gt > 0, np.log2(np.where(r_gt > 0, r_gt, 1.0)), np.nan)

    is_mic = (basis == MIC_BASIS).to_numpy()
    upper = t["mic_upper"] if "mic_upper" in t.columns else pd.Series(np.nan, index=t.index)
    lo, hi, point = intervals_from_reported_mic(t["mic_sign"], t["mic_value"], upper)
    lo, hi, point = lo - log_r, hi - log_r, point - log_r

    y_src = pd.to_numeric(t["y_test"], errors="coerce").to_numpy(dtype=float)
    is_bare = (basis == BARE_BASIS).to_numpy() & np.isin(y_src, (0.0, 1.0))
    lo = np.where(is_mic, lo, np.where(is_bare & (y_src == 1.0), 0.0, -np.inf))
    hi = np.where(is_mic, hi, np.where(is_bare & (y_src == 1.0), np.inf, 0.0))
    point = np.where(is_mic, point, np.nan)

    t["lo"], t["hi"], t["point"] = lo, hi, point
    t["is_mic"] = is_mic
    t["is_exact"] = is_mic & t["mic_sign"].astype(str).eq("=").to_numpy()
    # a usable row is an MIC test or a bare call whose BOTH bounds resolved (an infinity is a bound, not a gap)
    keep = (is_mic | is_bare) & ~(np.isnan(lo) | np.isnan(hi))
    dropped = pd.Series(np.asarray(basis)[~keep]).value_counts().to_dict()
    logger.info("tests: %d usable of %d (dropped by basis: %s)", int(keep.sum()), len(t), dropped)
    return t.loc[keep, ["Sample", "species", "drug", "lo", "hi", "point", "is_mic", "is_exact"]]


def collapse_to_cells(per_test: pd.DataFrame) -> pd.DataFrame:
    """Collapse the tests of each ``(Sample, drug)`` to one interval by **intersection**, hull on a conflict.

    ``lo = max(lo_i)``, ``hi = min(hi_i)``: every test must be satisfied, which is also the right answer when the
    "repeats" are two ingests of one assay. An empty intersection (``lo >= hi``) means the tests contradict each
    other; the hull ``(min lo_i, max hi_i]`` is taken and ``hull`` set, so the cell is still usable and still
    visible. ``point`` is the median of the tests that have one.
    """
    g = per_test.groupby(["Sample", "drug"], sort=True)
    cells = g.agg(
        species=("species", "first"),
        lo_int=("lo", "max"), hi_int=("hi", "min"),
        lo_hull=("lo", "min"), hi_hull=("hi", "max"),
        point=("point", "median"),
        n_tests=("lo", "size"), n_mic_tests=("is_mic", "sum"), n_exact=("is_exact", "sum"),
    ).reset_index()

    disjoint = (cells["lo_int"] >= cells["hi_int"]).to_numpy()
    lo = np.where(disjoint, cells["lo_hull"], cells["lo_int"])
    hi = np.where(disjoint, cells["hi_hull"], cells["hi_int"])
    cells["lo"], cells["hi"], cells["hull"] = lo, hi, disjoint

    n_mic, n_exact, n_tests = (cells[c].to_numpy() for c in ("n_mic_tests", "n_exact", "n_tests"))
    both_finite = np.isfinite(lo) & np.isfinite(hi)
    cells["basis"] = np.where(n_mic > 0, MIC_BASIS, "bare_call")
    cells["censor"] = np.select(
        [n_mic == 0, both_finite & (n_exact == n_tests), both_finite, np.isneginf(lo), np.isposinf(hi)],
        [CENSOR_BARE, CENSOR_EXACT, CENSOR_INTERVAL, CENSOR_LEFT, CENSOR_RIGHT], "unknown")
    cells["width"] = hi - lo
    cells["interval_label"] = np.where(lo >= 0, 1.0, np.where(hi <= 0, 0.0, np.nan))
    return cells[list(OUT_COLUMNS)]


def check_against_split_table(cells: pd.DataFrame, split_table: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(kept, conflicts)``: the cells of this drug's split table, minus those whose interval contradicts it.

    The split table is the one authority on both the holdout and the binary label, so this never rewrites either.
    A cell the interval calls R and the table calls S (or vice versa) is a cell whose interval is wrong, and
    training a MIC arm on it would feed the model a label its own likelihood contradicts: it is returned
    separately, to be excluded from every arm of the comparison.
    """
    label_map, *_ = load_splits(split_table)
    cells = cells[cells["Sample"].isin(label_map)].copy()
    cells["ast_label"] = cells["Sample"].map(label_map).astype(float)
    disagrees = cells["interval_label"].notna() & (cells["interval_label"] != cells["ast_label"])
    return cells.loc[~disagrees].reset_index(drop=True), cells.loc[disagrees].reset_index(drop=True)


def summarise(cells: pd.DataFrame) -> pd.DataFrame:
    """Per drug: how many cells of each censoring class, how many carry a point, how wide the intervals are."""
    rows = []
    for drug, g in cells.groupby("drug", sort=True):
        row: dict[str, object] = {"drug": drug, "n": len(g)}
        for c in (CENSOR_EXACT, CENSOR_INTERVAL, CENSOR_LEFT, CENSOR_RIGHT, CENSOR_BARE):
            row[f"n_{c}"] = int((g["censor"] == c).sum())
        mic = g[g["basis"] == MIC_BASIS]
        row["n_mic"] = len(mic)
        row["pct_censored"] = round(100.0 * float((mic["censor"] != CENSOR_EXACT).mean()), 1) if len(mic) else np.nan
        row["n_point"] = int(g["point"].notna().sum())
        row["n_hull"] = int(g["hull"].sum())
        row["pct_R"] = round(100.0 * float(g["ast_label"].mean()), 1) if "ast_label" in g else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    """CLI: per-drug censored-MIC interval tables from ``ast_tests_called.parquet`` + the split tables."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tests", required=True, help="ast_tests_called.parquet")
    p.add_argument("--breakpoints", default=str(DEFAULT_BREAKPOINTS), help="EUCAST breakpoint CSV")
    p.add_argument("--split-dir", required=True, help="directory of <drug>_split.csv (the cohort and the labels)")
    p.add_argument("--drugs", nargs="+", required=True)
    p.add_argument("--out-dir", required=True, help="<drug>_mic_intervals.csv + summary + conflicts")
    p.add_argument("--species", help="restrict the tests to this species column value")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tests = pd.read_parquet(args.tests)
    if args.species:
        tests = tests[tests["species"].astype(str) == args.species]
    tests = tests[tests["drug"].astype(str).isin(args.drugs)]
    per_test = ast_tests_to_intervals(tests, load_breakpoints(args.breakpoints))
    cells = collapse_to_cells(per_test)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kept_all, conflicts_all = [], []
    for drug in args.drugs:
        table = Path(args.split_dir) / f"{drug}_split.csv"
        if not table.is_file():
            raise SystemExit(f"no split table {table}: the cohort of a MIC arm is the split table's, not the tests'")
        kept, conflicts = check_against_split_table(cells[cells["drug"] == drug], table)
        kept.to_csv(out_dir / f"{drug}_mic_intervals.csv", index=False)
        logger.info("%s: %d cells (%d conflicts dropped) -> %s", drug, len(kept), len(conflicts),
                    out_dir / f"{drug}_mic_intervals.csv")
        kept_all.append(kept)
        conflicts_all.append(conflicts)

    kept = pd.concat(kept_all, ignore_index=True)
    conflicts = pd.concat(conflicts_all, ignore_index=True)
    conflicts.to_csv(out_dir / "mic_interval_conflicts.csv", index=False)
    summary = summarise(kept)
    summary.to_csv(out_dir / "mic_interval_summary.csv", index=False)
    print(f"\nconflicts (interval call vs split-table label): {len(conflicts)} of {len(kept) + len(conflicts)}")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

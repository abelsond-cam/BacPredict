"""How far each genome's MIC sits from the breakpoint that decided its label.

A binary AST label is a continuous measurement cut at one threshold, so a genome whose MIC sits *at* the
cut carries a label a single doubling dilution could flip. ISO 20776-2 essential agreement is ±1 dilution,
so that band is not an arbitrary choice — it is the set of label states measurement error can reverse.
This module computes, per ``(Sample, drug)``:

    d = log2(MIC / r_gt_mg_l)

and bands it, so saved predictions can be re-scored by distance from the cut
(:mod:`bacpredict.engine.finetune.stratified_metrics`).

**Why distance and not the EUCAST I category.** EUCAST v15 gives *P. aeruginosa* no susceptible category
at all for 7 of our 10 drugs — it prints ``S ≤ 0.001``, meaning the whole wild-type range is "susceptible,
increased exposure". For those drugs "I" is a tautology for "negative" and carries no information about
which labels are contestable: a cefepime MIC of 0.5 and one of 8 are both I and both labelled 0, but only
one is at the cut. The EUCAST zone is still emitted, as a descriptive cross-tab, never as the grouping.

⚠ **A censored MIC is an interval, not a point.** ``<= 2`` means "somewhere at or below 2". It is banded
only when the *whole* implied interval lies in one band, otherwise it is ``undetermined`` — never guessed,
never silently treated as its bound. A test with no MIC at all (the label came from the source's own call,
up to 55.8 % of a drug's holdout) is its own ``no_mic`` stratum, because folding it into ``far`` would
assert a distance nobody measured.

This is an app module, not engine: it reads the ``s_le_mg_l`` column that nothing else reads, and no other
organism needs it yet. It imports ``taxon_lineage`` from the engine but changes nothing there.

Example
-------
::

    python -m bacpredict.apps.pa.mic_distance_bands \\
        --tests $AMR/processed/paeruginosa/ast_tests_called.parquet \\
        --split-dir $AMR/processed/paeruginosa/splits \\
        --out-dir $AMR/processed/paeruginosa/mic_bands
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from bacpredict.engine.ast_labels.breakpoints import DEFAULT_BREAKPOINTS, load_breakpoints, taxon_lineage

logger = logging.getLogger(__name__)

#: Band edges in doubling dilutions from the cut. ``NEAR`` is ±1 — ISO 20776-2 essential agreement, so the
#: only band derived from a measurement property rather than chosen. ``MID`` at 3 is a convenience cut and
#: is labelled as such wherever it is reported.
NEAR_DILUTIONS = 1.0
MID_DILUTIONS = 3.0

#: A test is banded only from a measured MIC on the breakpoint scale.
MIC_BASIS = "mic"

UNDETERMINED = "undetermined"
NO_MIC = "no_mic"

OUT_COLUMNS = (
    "Sample", "drug", "d_min_abs", "d_median", "band", "band_signed", "eucast_zone",
    "n_mic_tests", "n_banded_tests", "band_disagreement", "near_legacy",
)


def band_of(d: float) -> tuple[str, str]:
    """A signed distance in dilutions → ``(band, band_signed)``.

    The partition is ``near`` ``(-1, 1]``, ``mid`` ``[-3, -1] ∪ (1, 3]``, ``far`` elsewhere. It is
    asymmetric about zero because the label rule is *resistant iff MIC > R*: ``d = 0`` is susceptible and
    any ``d`` above it is resistant, so the open end belongs on the susceptible side.

    Parameters
    ----------
    d : float
        ``log2(MIC / r_gt_mg_l)``. ``±inf`` is accepted and lands in ``far``.

    Returns
    -------
    band : str
        ``near`` / ``mid`` / ``far``.
    band_signed : str
        The same with an ``_S`` or ``_R`` suffix, by which side of the cut ``d`` falls.
    """
    side = "R" if d > 0 else "S"
    if -NEAR_DILUTIONS < d <= NEAR_DILUTIONS:
        band = "near"
    elif -MID_DILUTIONS <= d <= MID_DILUTIONS:
        band = "mid"
    else:
        band = "far"
    return band, f"{band}_{side}"


def interval_of(sign: str, value: float, upper: float, r_gt: float) -> tuple[float, float]:
    """The closed interval of ``d = log2(MIC / r_gt)`` a test's censored MIC is consistent with.

    Follows the censored-sign table in :mod:`bacpredict.engine.ast_labels.breakpoints`: ``<=``/``<`` bound
    the MIC from above only, ``>``/``>=`` from below only, and ``>,<`` from both. Nothing is assumed about
    a dilution series, so ``> 8`` against R = 16 spans everything above ``log2(8/16)``.

    Returns
    -------
    tuple[float, float]
        ``(lo, hi)``, either end possibly infinite. ``(nan, nan)`` when the test carries no usable MIC.
    """
    if not (np.isfinite(value) and np.isfinite(r_gt) and value > 0 and r_gt > 0):
        return float("nan"), float("nan")
    d = math.log2(value / r_gt)
    if sign in ("<=", "<"):
        return float("-inf"), d
    if sign in (">", ">="):
        return d, float("inf")
    if sign == ">,<":
        if not (np.isfinite(upper) and upper > 0):
            return d, float("inf")
        return d, math.log2(upper / r_gt)
    return d, d  # "=" and anything else parse_mic let through as exact


def band_of_interval(lo: float, hi: float) -> tuple[str, str]:
    """``(band, band_signed)`` when the whole interval lies in one band, else ``undetermined``.

    This is the rule that keeps a censored MIC from being counted as its own bound: ``<= 2`` against
    R = 8 is consistent with anything from ``far_S`` to ``mid_S``, so it is not banded at all, while
    ``<= 0.5`` against R = 8 (``d ≤ -4``) is unambiguously ``far_S``.
    """
    if math.isnan(lo) or math.isnan(hi):
        return UNDETERMINED, UNDETERMINED
    lo_band, lo_signed = band_of(lo)
    hi_band, hi_signed = band_of(hi)
    if lo_signed == hi_signed:
        return lo_band, lo_signed
    return UNDETERMINED, UNDETERMINED


def resolve_s_le(species: pd.Series, drug: pd.Series, breakpoints: pd.DataFrame) -> np.ndarray:
    """Per row, the S breakpoint (mg/L; susceptible iff MIC ≤ it) of the most specific matching row, else NaN.

    The same lineage walk ``engine.ast_labels.breakpoints.resolve_r_gt`` does, against the ``s_le_mg_l``
    column instead. ⚠ That column exists in the breakpoint table and is read by **no other code in the
    repository** — the binary label rule uses ``r_gt_mg_l`` alone, which is exactly why the Intermediate
    zone is invisible in every label artifact.
    """
    if "s_le_mg_l" not in breakpoints.columns:
        raise ValueError("breakpoint table has no s_le_mg_l column: the I zone cannot be resolved")
    lookup = {(r.taxon_rank, r.taxon, r.drug): float(r.s_le_mg_l) for r in breakpoints.itertuples()}
    pairs = pd.DataFrame({"species": species.to_numpy(), "drug": drug.to_numpy()})
    uniq = pairs.drop_duplicates().reset_index(drop=True)

    def _one(sp: str, dr: str) -> float:
        for rank, taxon in taxon_lineage(sp):
            if (rank, taxon, dr) in lookup:
                return lookup[(rank, taxon, dr)]
        return np.nan

    uniq["s_le"] = [_one(s, d) for s, d in zip(uniq.species, uniq.drug, strict=True)]
    return pairs.merge(uniq, on=["species", "drug"], how="left")["s_le"].to_numpy(dtype=float)


def eucast_zone_of(lo: float, hi: float, s_le: float, r_gt: float) -> str:
    """``S`` / ``I`` / ``R`` from the interval, or ``undetermined`` when it straddles a boundary.

    Zones on the ``d`` scale: ``S`` is ``d ≤ log2(s_le/r_gt)``, ``R`` is ``d > 0``, ``I`` between. A drug
    whose ``s_le`` equals its ``r_gt`` (amikacin, tobramycin) has no I zone and never returns ``I``.
    """
    if not np.isfinite(s_le) or not np.isfinite(r_gt) or s_le <= 0 or r_gt <= 0:
        return UNDETERMINED
    if math.isnan(lo) or math.isnan(hi):
        return UNDETERMINED
    cut = math.log2(s_le / r_gt)

    def _zone(d: float) -> str:
        if d > 0:
            return "R"
        return "S" if d <= cut else "I"

    return _zone(lo) if _zone(lo) == _zone(hi) else UNDETERMINED


def band_tests(tests: pd.DataFrame, breakpoints: pd.DataFrame) -> pd.DataFrame:
    """Add ``d_lo``, ``d_hi``, ``band``, ``band_signed``, ``eucast_zone`` and ``near_legacy`` per test.

    Only ``basis == "mic"`` rows are banded; everything else is ``no_mic``. ``near_legacy`` reproduces the
    project's earlier prose definition (an ``=`` MIC with ``R/2 ≤ MIC ≤ 2R``) so the new bands can be
    checked against the recorded ``near_breakpoint_by_drug.csv`` figures before anyone trusts them.
    """
    t = tests.copy()
    t["s_le_mg_l"] = resolve_s_le(t["species"].astype(str), t["drug"].astype(str), breakpoints)
    usable = t["basis"].astype(str).eq(MIC_BASIS).to_numpy()

    los, his, bands, signed, zones = [], [], [], [], []
    for use, sign, value, upper, r_gt, s_le in zip(
        usable, t["mic_sign"].astype(str), t["mic_value"], t.get("mic_upper", pd.Series(np.nan, index=t.index)),
        t["r_gt_mg_l"], t["s_le_mg_l"], strict=True,
    ):
        if not use:
            los.append(np.nan); his.append(np.nan)
            bands.append(NO_MIC); signed.append(NO_MIC); zones.append(NO_MIC)
            continue
        lo, hi = interval_of(sign, float(value), float(upper) if pd.notna(upper) else np.nan, float(r_gt))
        b, bs = band_of_interval(lo, hi)
        los.append(lo); his.append(hi); bands.append(b); signed.append(bs)
        zones.append(eucast_zone_of(lo, hi, float(s_le), float(r_gt)))

    t["d_lo"], t["d_hi"] = los, his
    t["band"], t["band_signed"], t["eucast_zone"] = bands, signed, zones
    # the superseded prose definition, kept only as a reproduction check
    exact = usable & t["mic_sign"].astype(str).eq("=").to_numpy()
    t["near_legacy"] = exact & (t["mic_value"] >= t["r_gt_mg_l"] / 2).to_numpy() & (
        t["mic_value"] <= 2 * t["r_gt_mg_l"]).to_numpy()
    return t


def aggregate_to_genome(banded: pd.DataFrame) -> pd.DataFrame:
    """Collapse banded tests to one row per ``(Sample, drug)``.

    The primary distance is ``min |d|`` over the genome's banded tests: if *any* test sat at the cut, the
    label was contestable. ``d_median`` is carried as the pre-registered sensitivity. Both are fixed here
    rather than chosen after seeing band sizes.

    A genome whose banded tests disagree gets ``band_disagreement = True`` and takes the band of its
    ``min |d|`` test. A genome with no banded test is ``no_mic`` if it had no MIC at all, else
    ``undetermined`` — the two are kept apart because they mean different things.
    """
    rows = []
    for (sample, drug), g in banded.groupby(["Sample", "drug"], sort=True):
        usable = g[g["band"].ne(NO_MIC) & g["band"].ne(UNDETERMINED) & g["d_lo"].notna()]
        n_mic = int(g["basis"].astype(str).eq(MIC_BASIS).sum())
        if usable.empty:
            band = NO_MIC if n_mic == 0 else UNDETERMINED
            zones = set(g["eucast_zone"]) - {NO_MIC, UNDETERMINED}
            rows.append((sample, drug, np.nan, np.nan, band, band, zones.pop() if len(zones) == 1 else band,
                         n_mic, 0, False, bool(g["near_legacy"].any())))
            continue
        # a banded test has lo == hi for an exact MIC; for a censored one the whole interval is in one band,
        # so the nearer end is the conservative representative
        d = usable.apply(lambda r: min(r["d_lo"], r["d_hi"], key=abs), axis=1)
        i = d.abs().idxmin()
        zones = set(usable["eucast_zone"]) - {UNDETERMINED}
        rows.append((
            sample, drug, float(d.loc[i]), float(d.median()),
            usable.loc[i, "band"], usable.loc[i, "band_signed"],
            zones.pop() if len(zones) == 1 else UNDETERMINED,
            n_mic, len(usable), bool(usable["band_signed"].nunique() > 1), bool(g["near_legacy"].any()),
        ))
    return pd.DataFrame(rows, columns=list(OUT_COLUMNS))


def summarise(genome_bands: pd.DataFrame) -> pd.DataFrame:
    """Per drug: band occupancy, EUCAST zone occupancy, and the legacy near-breakpoint share."""
    out = []
    for drug, g in genome_bands.groupby("drug", sort=True):
        row: dict[str, object] = {"drug": drug, "n": len(g)}
        for b in ("near", "mid", "far", NO_MIC, UNDETERMINED):
            row[f"n_{b}"] = int((g["band"] == b).sum())
        for z in ("S", "I", "R"):
            row[f"n_zone_{z}"] = int((g["eucast_zone"] == z).sum())
        row["pct_near"] = round(100.0 * row["n_near"] / len(g), 1)
        row["pct_zone_I"] = round(100.0 * row["n_zone_I"] / len(g), 1)
        row["pct_near_legacy"] = round(100.0 * float(g["near_legacy"].mean()), 1)
        row["n_band_disagreement"] = int(g["band_disagreement"].sum())
        out.append(row)
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> int:
    """CLI: per-drug MIC-band tables plus a summary, from ``ast_tests_called.parquet``."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tests", required=True, help="ast_tests_called.parquet")
    p.add_argument("--breakpoints", default=str(DEFAULT_BREAKPOINTS), help="EUCAST breakpoint CSV")
    p.add_argument("--split-dir", help="optional: restrict to genomes in these <drug>_split.csv tables")
    p.add_argument("--out-dir", required=True, help="per-drug <drug>_mic_bands.csv + mic_band_summary.csv")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tests = pd.read_parquet(args.tests)
    banded = band_tests(tests, load_breakpoints(args.breakpoints))
    genome_bands = aggregate_to_genome(banded)
    logger.info("banded %d tests into %d (Sample, drug) rows", len(banded), len(genome_bands))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for drug, g in genome_bands.groupby("drug", sort=True):
        g.to_csv(out_dir / f"{drug}_mic_bands.csv", index=False)
    summary = summarise(genome_bands)
    summary.to_csv(out_dir / "mic_band_summary.csv", index=False)
    logger.info("wrote %s", out_dir)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Binary AST labels re-derived from MIC against clinical breakpoints — organism-agnostic.

The breakpoint *values* are data, not code: one row per (taxon, drug) in a table beside this module
(:data:`DEFAULT_BREAKPOINTS`, EUCAST Clinical Breakpoint Tables v15.0, valid from 2025-01-01, transcribed
from the official ``v_15.0_Breakpoint_Tables.xlsx``). The code does two things only:

1. **Resolve** a genome's species to the most specific breakpoint row for a drug — species, then genus,
   then family, then order (:func:`resolve_r_gt`). A more specific row always wins, which is how
   *Salmonella* ciprofloxacin (R > 0.06) overrides the Enterobacterales row (R > 0.5).
2. **Apply the binary rule** (:func:`binary_call_from_mic`): **resistant iff MIC > the R breakpoint**.
   EUCAST's "susceptible, increased exposure" (I) is therefore *susceptible* — the rule AMR-GNN
   (Nguyen et al. 2026) used, under which a meropenem MIC of 8 in *P. aeruginosa* is S.

Censored MICs resolve only when the bound decides the call; otherwise the test is **undecided** (NaN):

=========  ===================================  ==================================
sign       resistant (1) when                   susceptible (0) when
=========  ===================================  ==================================
``=``      value > R                            value <= R
``<=``     never                                value <= R
``<``      never                                value <= R
``>``      value >= R                           never
``>=``     value > R                            never
``>,<``    lower >= R  (lower < MIC < upper)    upper <= R
=========  ===================================  ==================================

No dilution series is assumed, so e.g. ``>8`` against R > 16 stays undecided even though the next
doubling step would be 16. A species/drug pair with no breakpoint row gets NaN — the caller decides what
that means (see :mod:`bacpredict.engine.ast_labels.ast_tests`). Known gap, deliberately not encoded because
no portal genome needs it: EUCAST's aminoglycoside breakpoints do not apply to *Plesiomonas shigelloides*.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_BREAKPOINTS = Path(__file__).with_name("eucast_v15_breakpoints.csv")

# Taxonomy above genus for every genus the EBI AMR Portal (release 2026-07) holds that a breakpoint table
# can reach. Post-2016 Enterobacterales families (Adeleke et al.). A genus missing here can still match a
# genus- or species-rank row; it just cannot inherit a family/order row.
GENUS_TO_FAMILY: dict[str, str] = {
    "Escherichia": "Enterobacteriaceae", "Shigella": "Enterobacteriaceae", "Salmonella": "Enterobacteriaceae",
    "Klebsiella": "Enterobacteriaceae", "Enterobacter": "Enterobacteriaceae", "Citrobacter": "Enterobacteriaceae",
    "Raoultella": "Enterobacteriaceae", "Cronobacter": "Enterobacteriaceae", "Kluyvera": "Enterobacteriaceae",
    "Leclercia": "Enterobacteriaceae", "Kosakonia": "Enterobacteriaceae", "Pluralibacter": "Enterobacteriaceae",
    "Serratia": "Yersiniaceae", "Yersinia": "Yersiniaceae",
    "Proteus": "Morganellaceae", "Morganella": "Morganellaceae", "Providencia": "Morganellaceae",
    "Hafnia": "Hafniaceae", "Edwardsiella": "Hafniaceae", "Pantoea": "Erwiniaceae", "Erwinia": "Erwiniaceae",
    "Pseudomonas": "Pseudomonadaceae",
}
FAMILY_TO_ORDER: dict[str, str] = {
    "Enterobacteriaceae": "Enterobacterales", "Yersiniaceae": "Enterobacterales",
    "Morganellaceae": "Enterobacterales", "Hafniaceae": "Enterobacterales", "Erwiniaceae": "Enterobacterales",
    "Pseudomonadaceae": "Pseudomonadales",
}

# Units in which a measurement is an MIC on the breakpoint scale (mg/L == µg/mL).
MIC_UNITS = frozenset({"mg/l", "ug/ml", "µg/ml", "μg/ml"})
SIGNS = ("=", "<=", "<", ">", ">=", ">,<")
_SIGN_ALIASES = {"==": "="}


def load_breakpoints(path: str | Path = DEFAULT_BREAKPOINTS) -> pd.DataFrame:
    """Read a breakpoint table (``table, taxon_rank, taxon, drug, s_le_mg_l, r_gt_mg_l, eucast_row``).

    Raises if a (taxon_rank, taxon, drug) key repeats — a duplicate would make resolution order-dependent.
    """
    bp = pd.read_csv(path)
    need = {"taxon_rank", "taxon", "drug", "r_gt_mg_l"}
    if missing := need - set(bp.columns):
        raise ValueError(f"Breakpoint table {path} is missing column(s) {sorted(missing)}")
    if bad := set(bp.taxon_rank) - {"species", "genus", "family", "order"}:
        raise ValueError(f"Breakpoint table {path}: unknown taxon_rank {sorted(bad)}")
    dup = bp.duplicated(["taxon_rank", "taxon", "drug"], keep=False)
    if dup.any():
        raise ValueError(f"Breakpoint table {path}: duplicate keys\n{bp[dup]}")
    return bp


def taxon_lineage(species: str) -> list[tuple[str, str]]:
    """``"Salmonella enterica"`` → ``[(species, …), (genus, Salmonella), (family, …), (order, …)]``, most specific first."""
    genus = str(species).split()[0] if isinstance(species, str) and species.strip() else ""
    family = GENUS_TO_FAMILY.get(genus)
    order = FAMILY_TO_ORDER.get(family) if family else None
    out = [("species", str(species)), ("genus", genus)]
    if family:
        out.append(("family", family))
    if order:
        out.append(("order", order))
    return out


def resolve_r_gt(species: pd.Series, drug: pd.Series, breakpoints: pd.DataFrame) -> np.ndarray:
    """Per row, the R breakpoint (mg/L; resistant iff MIC > it) of the most specific matching row, else NaN.

    Vectorised over rows: the lineage walk runs once per distinct (species, drug) pair, not per test.
    """
    lookup = {(r.taxon_rank, r.taxon, r.drug): float(r.r_gt_mg_l) for r in breakpoints.itertuples()}
    pairs = pd.DataFrame({"species": species.to_numpy(), "drug": drug.to_numpy()})
    uniq = pairs.drop_duplicates().reset_index(drop=True)

    def _one(sp: str, dr: str) -> float:
        for rank, taxon in taxon_lineage(sp):
            if (rank, taxon, dr) in lookup:
                return lookup[(rank, taxon, dr)]
        return np.nan

    uniq["r_gt"] = [_one(s, d) for s, d in zip(uniq.species, uniq.drug, strict=True)]
    return pairs.merge(uniq, on=["species", "drug"], how="left")["r_gt"].to_numpy(dtype=float)


def parse_mic(measurement: pd.Series, sign: pd.Series | None = None) -> pd.DataFrame:
    """Parse MIC strings → ``DataFrame(mic_sign, mic_value, mic_upper)`` (vectorised).

    ``sign`` given (the EBI portal's ``measurement_sign`` column): the value is the bare number. ``sign``
    omitted: a leading operator is split off the string (``"<=0.5"``, ``">16"``, ``"8"``). Combination
    agents report the first component (``"16/4"`` → 16, the EUCAST convention — the inhibitor is fixed).
    Bracketed ``>,<`` takes ``"X,Y"`` meaning X < MIC < Y. Anything unparsable gives NaN value, NA sign.
    """
    s = measurement.astype("string").str.strip().str.replace(" ", "", regex=False)
    if sign is None:
        split = s.str.extract(r"^(>,<|<=|>=|==|=|<|>)?(.*)$")
        op, s = split[0].fillna("=").where(s.notna()), split[1].astype("string")
    else:
        op = sign.astype("string").str.strip()
    op = op.replace(_SIGN_ALIASES)

    bracket = op.eq(">,<").fillna(False)
    parts = s.str.split(",", n=1, expand=True).reindex(columns=[0, 1])
    first = s.str.split("/", n=1).str[0]
    value = pd.to_numeric(first.where(~bracket, parts[0]), errors="coerce")
    upper = pd.to_numeric(parts[1].where(bracket), errors="coerce")
    # a non-bracketed value containing a comma is ambiguous (decimal comma vs list) — refuse it
    value = value.where(~(~bracket & s.str.contains(",", regex=False).fillna(False)))
    value = value.where(~bracket | upper.notna())
    op = op.where(op.isin(SIGNS) & value.notna())
    value = value.where(op.notna())
    return pd.DataFrame({"mic_sign": op.astype("string"), "mic_value": value.astype(float),
                         "mic_upper": upper.where(op.eq(">,<").fillna(False)).astype(float)},
                        index=measurement.index)


def binary_call_from_mic(sign: pd.Series, value: pd.Series, upper: pd.Series, r_gt: np.ndarray) -> np.ndarray:
    """1.0 = resistant, 0.0 = susceptible, NaN = undecided (censoring straddles R) or no breakpoint/MIC.

    Resistant iff MIC > ``r_gt``; see the module docstring for the censored-sign table.
    """
    sg = sign.astype("string").fillna("").to_numpy(dtype=object)
    v = value.to_numpy(dtype=float)
    up = upper.to_numpy(dtype=float)
    r = np.asarray(r_gt, dtype=float)
    with np.errstate(invalid="ignore"):
        eq, le, lt = sg == "=", sg == "<=", sg == "<"
        gt, ge, br = sg == ">", sg == ">=", sg == ">,<"
        res = (eq & (v > r)) | (gt & (v >= r)) | (ge & (v > r)) | (br & (v >= r))
        sus = (eq & (v <= r)) | ((le | lt) & (v <= r)) | (br & (up <= r))
    out = np.full(len(v), np.nan)
    out[res] = 1.0
    out[sus] = 0.0
    out[np.isnan(v) | np.isnan(r)] = np.nan
    return out

"""One AST-test table, any source → per-test binary calls → one label per (genome, drug).

Every label source is first written in one long **test schema** (:data:`TEST_COLUMNS`, one row per
genome × drug × test). The EBI AMR Portal release is read by :func:`read_portal_tests`; any other source
(e.g. a paper's supplement) gets a small adapter in its app that writes the same columns, so the
labelling rule below is applied identically to every source.

**The labelling rule, per test** (:func:`call_tests`) — agreed for the multi-species AMR work:

* a test **with an MIC** (mg/L) is called from the MIC against the clinical breakpoint
  (:mod:`bacpredict.engine.ast_labels.breakpoints`, EUCAST v15.0, resistant iff MIC > R). A censored MIC
  that straddles the breakpoint is **undecided** — it is *not* rescued by the source's own call;
* a test **without an MIC** (disk diffusion, or a bare call) keeps the **source's own call**, with the
  canonical parser's semantics: ``resistant`` → 1, ``susceptible`` → 0, anything else (``intermediate``,
  ``non-susceptible``, ``susceptible-dose dependent``) → no call;
* an MIC for a species/drug pair with **no breakpoint row** gives no call (``basis = no_breakpoint``).
  Whether to fall back to the source's call there is an open method decision, so it is counted, not hidden.

**Per (genome, drug)** (:func:`aggregate_labels`) the called tests are **averaged**, exactly as the
canonical parser does (``parse_ebi_ast_to_binary``): a genome whose tests disagree gets a fractional
label, which ``load_splits`` never scores. The pivot written as ``binary_ast.csv`` keeps those fractions so
the downstream semantics are identical to the TB and Kp cohorts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bacpredict.engine.ast_labels.breakpoints import MIC_UNITS, binary_call_from_mic, parse_mic, resolve_r_gt

# The long test schema every source is converted to. Cohort columns (assembly / reads / study / ST) ride on
# every test row so a source needs only one file; build_binary_ast_labels collapses them per genome.
# ``mic_*`` carry the parsed measurement whatever its units — only rows whose ``mic_units`` is an MIC unit
# (mg/L) are treated as MICs; a disk zone (mm) is kept for audit and labelled by the source's call.
TEST_COLUMNS = [
    "Sample", "species", "drug", "source",
    "mic_sign", "mic_value", "mic_upper", "mic_units",
    "method", "platform", "ast_standard", "source_call", "database",
    "assembly_phenotype", "reads_accession", "study", "st_reported",
]
SOURCE_CALL_TO_BINARY = {"resistant": 1.0, "susceptible": 0.0}
_FLOAT_COLUMNS = ("mic_value", "mic_upper")


def coerce_test_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Order the columns as :data:`TEST_COLUMNS` and fix dtypes: the MIC numbers float, everything else string.

    Identifiers must stay strings end to end — a CSV round-trip would otherwise turn an ST of ``"235"`` into
    an int and a BioSample of all digits into a number.
    """
    if missing := [c for c in TEST_COLUMNS if c not in df.columns]:
        raise ValueError(f"Test table is missing column(s) {missing}")
    out = df[TEST_COLUMNS].copy()
    for c in TEST_COLUMNS:
        out[c] = out[c].astype(float) if c in _FLOAT_COLUMNS else out[c].astype("string")
    return out

_PORTAL_COLUMNS = [
    "BioSample_ID", "species", "antibiotic_name", "measurement", "measurement_sign", "measurement_units",
    "laboratory_typing_method", "platform", "ast_standard", "resistance_phenotype", "database",
    "assembly_ID", "SRA_accession", "AMR_associated_publications",
]


def read_portal_tests(
    phenotype_path: str | Path,
    species: list[str] | None = None,
    genera: list[str] | None = None,
    drugs: list[str] | None = None,
    source: str = "ebi_amr_portal",
) -> pd.DataFrame:
    """Read the EBI AMR Portal ``phenotype.parquet`` (unprefixed columns) into the test schema.

    Filters to genomes whose ``species`` is in ``species`` or whose genus (first word) is in ``genera``
    (either may be None; both None keeps every species), and to ``drugs`` when given. Rows with no
    BioSample are dropped — there is nothing to key a genome on.
    """
    p = pd.read_parquet(phenotype_path, columns=_PORTAL_COLUMNS)
    keep = pd.Series(species is None and genera is None, index=p.index)
    if species:
        keep |= p["species"].isin(species)
    if genera:
        keep |= p["species"].astype("string").str.split().str[0].isin(genera).fillna(False)
    if drugs:
        keep &= p["antibiotic_name"].isin(drugs)
    p = p[keep & p["BioSample_ID"].notna()].reset_index(drop=True)
    mic = parse_mic(p["measurement"], p["measurement_sign"])
    return pd.DataFrame({
        "Sample": p["BioSample_ID"].astype(str),
        "species": p["species"],
        "drug": p["antibiotic_name"],
        "source": source,
        "mic_sign": mic["mic_sign"],
        "mic_value": mic["mic_value"],
        "mic_upper": mic["mic_upper"],
        "mic_units": p["measurement_units"],
        "method": p["laboratory_typing_method"],
        "platform": p["platform"],
        "ast_standard": p["ast_standard"],
        "source_call": p["resistance_phenotype"].astype("string").str.strip().str.lower(),
        "database": p["database"],
        "assembly_phenotype": p["assembly_ID"],
        "reads_accession": p["SRA_accession"],
        "study": p["AMR_associated_publications"],
        "st_reported": pd.NA,
    }).pipe(coerce_test_schema)


def read_test_table(path: str | Path) -> pd.DataFrame:
    """Read an adapter-written test table (CSV or parquet) into the schema, identifiers kept as strings."""
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, dtype={c: str for c in TEST_COLUMNS if c not in _FLOAT_COLUMNS}, low_memory=False)
    return coerce_test_schema(df)


def call_tests(tests: pd.DataFrame, breakpoints: pd.DataFrame, unitless_mic_as_mg_l: bool = False) -> pd.DataFrame:
    """Add ``r_gt_mg_l``, ``y_test`` (1/0/NaN), ``basis`` and ``mic_units_missing`` to every test.

    ``basis`` is one of ``mic`` (called from the MIC), ``mic_undecided`` (censored MIC straddles R),
    ``no_breakpoint`` (MIC but no breakpoint row), ``source_call`` (no MIC; the source's R/S) or
    ``no_call`` (no MIC and the source's call is not R/S).

    ``mic_units_missing`` marks a parsed measurement with **no unit** that is not a disk-diffusion test —
    in the EBI portal these are MIC dilution steps (CDC AR Bank, CABBAGE E-tests) whose unit was never
    recorded. By default they are *not* treated as MICs (they fall to the source's own call, as a test
    without an MIC does); ``unitless_mic_as_mg_l=True`` calls them against the breakpoint as mg/L. Either
    way they are flagged, so the choice is visible in the counts.
    """
    t = tests.copy()
    t["r_gt_mg_l"] = resolve_r_gt(t["species"].astype(str), t["drug"].astype(str), breakpoints)
    units = t["mic_units"].astype("string").str.strip().str.lower()
    is_disk = t["method"].astype("string").str.lower().str.contains("disk", regex=False).fillna(False)
    t["mic_units_missing"] = (t["mic_value"].notna() & units.isna() & ~is_disk).to_numpy(dtype=bool)
    mic_unit = units.isin(MIC_UNITS).fillna(False).to_numpy(dtype=bool)
    if unitless_mic_as_mg_l:
        mic_unit = mic_unit | t["mic_units_missing"].to_numpy()
    has_mic = t["mic_value"].notna().to_numpy() & mic_unit
    y_mic = binary_call_from_mic(t["mic_sign"], t["mic_value"], t["mic_upper"], t["r_gt_mg_l"].to_numpy())
    y_src = t["source_call"].astype("string").str.strip().str.lower().map(SOURCE_CALL_TO_BINARY).astype(float)

    no_bp = has_mic & t["r_gt_mg_l"].isna().to_numpy()
    t["y_test"] = np.where(has_mic, y_mic, y_src.to_numpy())
    t["basis"] = np.select(
        [has_mic & ~no_bp & ~np.isnan(y_mic), has_mic & ~no_bp, no_bp, y_src.notna().to_numpy()],
        ["mic", "mic_undecided", "no_breakpoint", "source_call"],
        default="no_call",
    )
    return t


def aggregate_labels(called: pd.DataFrame) -> pd.DataFrame:
    """Per (Sample, drug): the mean of called tests plus how it was reached.

    Columns: ``y`` (mean; fractional = conflicting tests), ``n_tests``, ``n_called``, and one count per
    ``basis`` and of unitless measurements; ``label_from`` is ``mic`` / ``source_call`` / ``mixed`` / ``none``;
    ``conflict`` marks a fractional mean; ``sources`` lists which sources contributed any test.
    """
    g = called.groupby(["Sample", "drug"], sort=True)
    out = g["y_test"].agg(y="mean", n_called="count").join(g.size().rename("n_tests"))
    basis = pd.crosstab([called["Sample"], called["drug"]], called["basis"])
    for b in ("mic", "mic_undecided", "no_breakpoint", "source_call", "no_call"):
        out[f"n_{b}"] = basis[b].reindex(out.index).fillna(0).astype(int) if b in basis else 0
    out["label_from"] = np.select(
        [(out.n_mic > 0) & (out.n_source_call > 0), out.n_mic > 0, out.n_source_call > 0],
        ["mixed", "mic", "source_call"], default="none")
    out["n_mic_units_missing"] = g["mic_units_missing"].sum().astype(int) if "mic_units_missing" in called else 0
    out["conflict"] = (out["y"] > 0) & (out["y"] < 1)
    src = called[["Sample", "drug", "source"]].drop_duplicates().sort_values(["Sample", "drug", "source"])
    out["sources"] = src.groupby(["Sample", "drug"])["source"].agg(";".join)
    return out.reset_index()


def call_agreement(called: pd.DataFrame) -> pd.DataFrame:
    """Per (source, drug): how often the MIC-derived call matches the source's own R/S call.

    Restricted to tests that carry both. Against a source that already used the same standard (AMR-GNN:
    EUCAST v15) this is a check of the breakpoint table; against CLSI-called sources it counts the flips
    the re-derivation makes.
    """
    t = called[called["basis"].eq("mic")].copy()
    y_src = t["source_call"].astype("string").str.lower().map(SOURCE_CALL_TO_BINARY).astype(float)
    t = t.assign(y_src=y_src)[y_src.notna()]
    t = t.assign(agree=t.y_test == t.y_src, s2r=(t.y_src == 0) & (t.y_test == 1), r2s=(t.y_src == 1) & (t.y_test == 0))
    out = t.groupby(["source", "drug"]).agg(n_both=("y_test", "size"), agree=("agree", "sum"),
                                             src_S_to_R=("s2r", "sum"), src_R_to_S=("r2s", "sum"))
    return out.astype(int).reset_index()

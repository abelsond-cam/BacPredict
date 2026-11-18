r"""AMR-GNN supplement (Nguyen et al. 2026, Nat Commun) → the engine's AST-test schema.

Reads the tidied Supplementary Data 1 (``amr_gnn_long.csv``: run, study_id, study, ST, drug, MIC, RS — one
row per isolate × drug) and the isolate → BioSample map (``amr_gnn_isolate_biosample.csv``), and writes
one row per test in :data:`bacpredict.engine.ast_labels.ast_tests.TEST_COLUMNS`, keyed on the BioSample so
it merges with the EBI AMR Portal's own tests for the same isolates.

The paper's ``RS`` column is its own EUCAST v15 call; it becomes ``source_call`` and is used only for a
test with no MIC (none, in this release) and for the call-agreement check. Every MIC is in mg/L. The
paper's ``run`` column becomes ``reads_accession`` only where it is a run accession (SRR/ERR/DRR); for the
Kos et al. (2015) isolates it is the ``AZPAE…`` strain name, which the BioSample map already resolved. The
paper's ``ST`` is carried as ``st_reported`` for cross-checking — the cohort's STs come from our own
``mlst`` run on the assemblies, not from here.

Usage::

    uv run python -m bacpredict.apps.pa.amr_gnn_tests \
        --long $ROOT/raw/amr_gnn_nguyen2026/amr_gnn_long.csv \
        --biosample-map $ROOT/raw/amr_gnn_nguyen2026/amr_gnn_isolate_biosample.csv \
        --out $ROOT/processed/train_pa_ast/amr_gnn_tests.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bacpredict.engine.ast_labels.ast_tests import coerce_test_schema
from bacpredict.engine.ast_labels.breakpoints import parse_mic

SOURCE = "amr_gnn_nguyen2026"
SPECIES = "Pseudomonas aeruginosa"
RS_TO_CALL = {"R": "resistant", "S": "susceptible"}


def amr_gnn_tests(long: pd.DataFrame, biosample_map: pd.DataFrame) -> pd.DataFrame:
    """Convert the tidied supplement to the test schema; every isolate must map to exactly one BioSample."""
    m = biosample_map[["run", "sample_accession"]].drop_duplicates()
    if m["run"].duplicated().any():
        raise ValueError("isolate→BioSample map has a run mapped to more than one BioSample")
    df = long.merge(m, on="run", how="left", validate="many_to_one")
    if n := int(df["sample_accession"].isna().sum()):
        raise ValueError(f"{n} AMR-GNN tests have no BioSample — fix the map before labelling")
    mic = parse_mic(df["MIC"])
    if n := int(mic["mic_value"].isna().sum()):
        raise ValueError(f"{n} AMR-GNN MICs failed to parse: {df.loc[mic['mic_value'].isna(), 'MIC'].unique()[:10]}")
    out = pd.DataFrame({
        "Sample": df["sample_accession"].astype(str),
        "species": SPECIES,
        "drug": df["drug"].str.strip().str.lower(),
        "source": SOURCE,
        "mic_sign": mic["mic_sign"],
        "mic_value": mic["mic_value"],
        "mic_upper": mic["mic_upper"],
        "mic_units": "mg/L",
        "method": pd.NA,
        "platform": pd.NA,
        "ast_standard": "EUCAST",
        "source_call": df["RS"].str.strip().str.upper().map(RS_TO_CALL),
        "database": pd.NA,
        "assembly_phenotype": pd.NA,
        # the paper's "run" is a strain name (AZPAE…) for the Kos et al. isolates — only a real run is reads
        "reads_accession": df["run"].where(df["run"].str.match(r"^[SED]RR\d+$").fillna(False)),
        "study": df["study"],
        "st_reported": df["ST"].astype(str),
    }, index=df.index)
    return coerce_test_schema(out)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--long", type=Path, required=True, help="amr_gnn_long.csv")
    ap.add_argument("--biosample-map", type=Path, required=True, help="amr_gnn_isolate_biosample.csv")
    ap.add_argument("--out", type=Path, required=True, help="output test table (CSV)")
    args = ap.parse_args()
    out = amr_gnn_tests(pd.read_csv(args.long, dtype=str), pd.read_csv(args.biosample_map, dtype=str))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} tests on {out['Sample'].nunique()} BioSamples × {out['drug'].nunique()} drugs → {args.out}")


if __name__ == "__main__":
    main()

r"""CABBAGE v2 (``cabbage_v2.csv``) → a table in the EBI AMR Portal ``phenotype.parquet`` schema.

CABBAGE v2 is the AST source of the second multi-species experiment (``PROJECT_STATE.md`` §3.7). It is one CSV
with its own columns, not a portal release. Rather than give the label builder a second reader, this writes the
v2 tests in the portal's own column layout, so ``build_binary_ast_labels --phenotype <this>`` labels them with
**exactly the code path the 2026-07 experiment used** (:func:`~bacpredict.engine.ast_labels.ast_tests.read_portal_tests`
onward). Nothing in the engine changes; the two experiments differ only in their input table.

The mapping, column by column (v2 → portal):

=================================  =================================  ==============================================
v2                                 portal                             how
=================================  =================================  ==============================================
``BioSample_ID``                   ``BioSample_ID``                   stripped
``genus`` + ``species``            ``species``                        ``"<genus> <epithet>"``; the genus alone where
                                                                      the epithet is empty
``Antibiotic_name``                ``antibiotic_name``                lowercased, ``/`` → ``-`` (portal naming, e.g.
                                                                      ``piperacillin-tazobactam``)
``measurement_value``              ``measurement``                    as is
``measurement_sign``               ``measurement_sign``               **empty → ``=``** where there is a value
                                                                      (David, 2026-11-21)
``measurement_unit``               ``measurement_units``              as is (``mg/l``, ``mm``)
``laboratory_typing_method``       ``laboratory_typing_method``       as is
``platform``                       ``platform``                       as is
``testing_standard``               ``ast_standard``                   as is
``phenotype``                      ``resistance_phenotype``           as is (the builder lowercases it)
``database``                       ``database``                       ``|`` → ``;``, the portal's separator
``Assembly_ID``                    ``assembly_ID``                    as is
``SRA_run``                        ``SRA_accession``                  as is
``Pubmed_ID``                      ``AMR_associated_publications``    as is
=================================  =================================  ==============================================

Not carried: ``Updated_phenotype_CLSI`` / ``Updated_phenotype_EUCAST`` / ``used_ECOFF`` (CABBAGE's own
re-interpretations — every MIC is re-called from EUCAST v15 by the builder), and the isolate / host / date
metadata. ``database`` is kept so labels resting on the ``puffin`` source alone stay countable.

Usage (light: one CSV read, ~3.3M rows — a CSD3 login node is fine)::

    uv run python -m bacpredict.engine.ast_labels.cabbage_v2_phenotype \
        --csv $ROOT/raw/ebi_amr_portal/cabbage_v2_20261121/cabbage_v2.csv \
        --out $ROOT/processed/ebi_amr_portal/cabbage_v2_20261121/phenotype.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

V2_COLUMNS = [
    "BioSample_ID", "genus", "species", "Antibiotic_name", "measurement_sign", "measurement_value",
    "measurement_unit", "laboratory_typing_method", "platform", "testing_standard", "phenotype", "database",
    "Assembly_ID", "SRA_run", "Pubmed_ID",
]


def _clean(s: pd.Series) -> pd.Series:
    """Strip whitespace; an empty string becomes NA."""
    s = s.astype("string").str.strip()
    return s.mask(s.eq(""))


def cabbage_v2_as_portal(v2: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Map CABBAGE v2 rows to the portal ``phenotype.parquet`` columns; returns the table and its counts.

    Rows with no BioSample are dropped (nothing to key a genome on), as the portal reader would drop them.
    """
    if missing := [c for c in V2_COLUMNS if c not in v2.columns]:
        raise ValueError(f"CABBAGE v2 table is missing column(s) {missing}")
    v = v2[V2_COLUMNS].apply(_clean)
    n_in = len(v)
    v = v[v["BioSample_ID"].notna()]
    genus, epithet = v["genus"], v["species"]
    species = (genus + " " + epithet).fillna(genus)
    sign_filled = v["measurement_sign"].isna() & v["measurement_value"].notna()
    out = pd.DataFrame({
        "BioSample_ID": v["BioSample_ID"],
        "species": species,
        "antibiotic_name": v["Antibiotic_name"].str.lower().str.replace("/", "-", regex=False),
        "measurement": v["measurement_value"],
        "measurement_sign": v["measurement_sign"].mask(sign_filled, "="),
        "measurement_units": v["measurement_unit"],
        "laboratory_typing_method": v["laboratory_typing_method"],
        "platform": v["platform"],
        "ast_standard": v["testing_standard"],
        "resistance_phenotype": v["phenotype"],
        "database": v["database"].str.replace("|", ";", regex=False),
        "assembly_ID": v["Assembly_ID"],
        "SRA_accession": v["SRA_run"],
        "AMR_associated_publications": v["Pubmed_ID"],
    }).reset_index(drop=True)
    counts = {
        "n_rows_in": n_in,
        "n_rows_out": len(out),
        "n_dropped_no_biosample": n_in - len(out),
        "n_sign_filled_as_equals": int(sign_filled.sum()),
        "n_species_genus_only": int(epithet.isna().sum()),
    }
    return out, counts


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    """CLI entry point: write the portal-schema parquet and a ``.json`` of counts and the input md5 beside it."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, required=True, help="cabbage_v2.csv (unpacked)")
    ap.add_argument("--out", type=Path, required=True, help="output parquet in the portal phenotype schema")
    args = ap.parse_args()
    v2 = pd.read_csv(args.csv, dtype=str, usecols=V2_COLUMNS, keep_default_na=False)
    out, counts = cabbage_v2_as_portal(v2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    meta = {"input": {str(args.csv): _md5(args.csv)}, **counts}
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

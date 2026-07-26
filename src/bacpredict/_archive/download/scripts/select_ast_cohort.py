"""Split the all-species EBI AMR sheet into a single-species AST cohort CSV.

The Isambard EBI dump (`ebi_amr_records_all_species_20260630.csv`, all genera, ~1.19M
phenotype rows) is the source of both the download cohort and the AST labels. This script
filters it to one species (exact `phenotype-species` binomial) and writes the per-species
records CSV that the rest of the pipeline already expects:

  * the download planner `download_assemblies.py --metadata <out>` (intersects
    `phenotype-BioSample_ID` with the AllTheBacteria index), and
  * the AST label preprocessors (`preprocess_ebi_amr_records.py` / TB parsing) that pivot
    antibiotic x resistance_phenotype into the binary `binary_ast.csv`.

Every column is preserved (downstream code selects what it needs). Rows with a null
BioSample ID are dropped (nothing to download or key on). Prints a short cohort summary.

Example (on Isambard):
  python select_ast_cohort.py \
    --ebi-csv /projects/u6fp/david/raw/ebi_amr_records_all_species_20260630.csv \
    --species "Mycobacterium tuberculosis" \
    --out /projects/u6fp/david/raw/tb/ebi_tb_amr_records.csv
"""

import argparse
from pathlib import Path

import pandas as pd

BIOSAMPLE_COL = "phenotype-BioSample_ID"
SPECIES_COL = "phenotype-species"
ANTIBIOTIC_COL = "phenotype-antibiotic_name"
RESISTANCE_COL = "phenotype-resistance_phenotype"


def main() -> None:
    """Filter the EBI all-species sheet to one species and write the cohort CSV."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ebi-csv", required=True, help="all-species EBI AMR records CSV")
    ap.add_argument("--species", required=True, help='exact phenotype-species binomial, e.g. "Klebsiella pneumoniae"')
    ap.add_argument("--out", required=True, help="output per-species records CSV")
    args = ap.parse_args()

    df = pd.read_csv(args.ebi_csv, low_memory=False)
    if SPECIES_COL not in df.columns:
        raise SystemExit(f"{SPECIES_COL!r} not in {args.ebi_csv} columns")

    sub = df[df[SPECIES_COL] == args.species].copy()
    before = len(sub)
    sub = sub[sub[BIOSAMPLE_COL].notna()]
    dropped = before - len(sub)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)

    n_bs = sub[BIOSAMPLE_COL].nunique()
    n_drugs = sub[ANTIBIOTIC_COL].nunique() if ANTIBIOTIC_COL in sub.columns else "n/a"
    rvals = sub[RESISTANCE_COL].value_counts().to_dict() if RESISTANCE_COL in sub.columns else {}
    print(f"species              : {args.species}")
    print(f"phenotype rows        : {len(sub)}  (dropped {dropped} null-BioSample rows)")
    print(f"unique BioSample_ID   : {n_bs}")
    print(f"distinct antibiotics  : {n_drugs}")
    print(f"resistance_phenotype  : {rvals}")
    print(f"wrote                 : {out}")


if __name__ == "__main__":
    main()

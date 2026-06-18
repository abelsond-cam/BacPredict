#!/usr/bin/env python3
"""Build a minimal binary label CSV (``Sample``, ``<label_col>``, ``Sublineage``) for the GWAS.

The pyseer collation needs only a sample list + a 0/1 label + Sublineage — *not* the ESM
embeddings that the Bacformer ``prepare_…`` script crawls (which would wrongly drop any
sample lacking an embedding; a GWAS must keep every sample). This derives that minimal CSV
directly from the stratified sampler's ``stratified_selected_isolation_source_metadata.tsv``,
mapping the positive isolation source to 1 and the other to 0.

Reused across contrasts — only the tokens change, e.g.::

    build_binary_label_csv.py --stratified-tsv <cohort>/stratified_selected_isolation_source_metadata.tsv \
        --positive-substring respiratory --label-col respiratory_vs_faeces_label \
        --out-csv <cohort>/kpsc_human/binary_respiratory_vs_faeces_labels.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    """Parse args, map the positive isolation source to 1, write the label CSV."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stratified-tsv", type=Path, required=True,
                   help="Sampler cohort TSV (has Sample, isolation_source_category, Sublineage).")
    p.add_argument("--positive-substring", required=True,
                   help="Rows whose isolation_source_category contains this (case-insensitive) get label=1.")
    p.add_argument("--label-col", required=True,
                   help="Name of the 0/1 label column to write (e.g. respiratory_vs_faeces_label).")
    p.add_argument("--out-csv", type=Path, required=True, help="Output comma-separated label CSV.")
    args = p.parse_args()

    df = pd.read_csv(args.stratified_tsv, sep="\t", low_memory=False)
    for col in ("Sample", "isolation_source_category", "Sublineage"):
        if col not in df.columns:
            raise SystemExit(f"{args.stratified_tsv} has no '{col}' column (columns: {list(df.columns)})")
    df = df.drop_duplicates(subset=["Sample"])

    is_pos = df["isolation_source_category"].str.contains(args.positive_substring, case=False, na=False)
    out = pd.DataFrame(
        {
            "Sample": df["Sample"].astype(str),
            args.label_col: is_pos.astype(int),
            "Sublineage": df["Sublineage"],
        }
    )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    n_pos = int(out[args.label_col].sum())
    print(f"wrote {args.out_csv}: {len(out)} samples ({args.label_col}=1: {n_pos}, =0: {len(out) - n_pos})")


if __name__ == "__main__":
    main()

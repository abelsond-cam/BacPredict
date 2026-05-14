"""Find kpsc_final_list samples that are missing ESM embeddings.

Reads the curated metadata TSV (which already has `assembly_file` and `gff_file`
columns populated by `add_paths_gff_fna_to_metadata.py`), filters to
`kpsc_final_list == True`, and writes a CSV of samples whose
`{Sample}_esm_embeddings.pt` file is absent from the embeddings directory.

The output CSV is the input contract for the format-aware
`preprocess_assemblies_to_protein_sequences.py`: columns `Sample`,
`assembly_file`, `gff_file`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

METADATA_DEFAULT = Path(
    "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/final/metadata_final_curated_slimmed.tsv"
)
EMBEDDINGS_DIR_DEFAULT = Path(
    "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/klebsiella_esm_embeddings"
)
OUT_CSV_DEFAULT = Path(
    "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/missing_embeddings_kpsc.csv"
)


def find_missing(
    metadata_tsv: Path,
    embeddings_dir: Path,
) -> pd.DataFrame:
    """Return rows of kpsc_final_list samples that lack an embedding file."""
    df = pd.read_csv(metadata_tsv, sep="\t", low_memory=False)

    for col in ("Sample", "kpsc_final_list", "assembly_file", "gff_file"):
        if col not in df.columns:
            raise ValueError(
                f"Metadata TSV {metadata_tsv} is missing required column '{col}'. "
                "Run `add_paths_gff_fna_to_metadata.py` first."
            )

    kpsc = df[df["kpsc_final_list"].astype(bool)].copy()
    kpsc = kpsc.drop_duplicates(subset=["Sample"])

    # Single directory listing -> O(1) set lookup per sample (RDS stat-per-file is glacial).
    suffix = "_esm_embeddings.pt"
    existing = {
        name[: -len(suffix)]
        for name in os.listdir(embeddings_dir)
        if name.endswith(suffix)
    }
    missing = kpsc.loc[
        ~kpsc["Sample"].isin(existing), ["Sample", "assembly_file", "gff_file"]
    ].copy()
    return missing


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-tsv", type=Path, default=METADATA_DEFAULT)
    parser.add_argument("--embeddings-dir", type=Path, default=EMBEDDINGS_DIR_DEFAULT)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV_DEFAULT)
    args = parser.parse_args()

    print(f"Metadata: {args.metadata_tsv}")
    print(f"Embeddings dir: {args.embeddings_dir}")

    missing = find_missing(args.metadata_tsv, args.embeddings_dir)

    n_missing = len(missing)
    n_missing_with_paths = int(
        missing[["assembly_file", "gff_file"]].notna().all(axis=1).sum()
    )
    n_gcf = int(missing["Sample"].astype(str).str.startswith("GCF_").sum())
    print(f"kpsc_final_list samples missing embeddings: {n_missing}")
    print(f"  ...with both assembly_file and gff_file resolved: {n_missing_with_paths}")
    print(f"  ...GCF_*: {n_gcf}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    missing.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()

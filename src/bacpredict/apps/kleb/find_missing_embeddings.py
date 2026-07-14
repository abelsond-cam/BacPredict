"""Find kpsc_final_list samples that are missing ESM embeddings.

Reads the curated metadata TSV (which already has `sr_assembly_file` and `sr_gff_file`
columns populated by `add_paths_gff_fna_to_metadata.py`), filters to
`kpsc_final_list == True`, and writes a CSV of samples whose
`{Sample}_esm_embeddings.pt` file is absent from the embeddings directory.

The output CSV is the input contract for the format-aware
`preprocess_assemblies_to_protein_sequences.py`: columns `Sample`,
`sr_assembly_file`, `sr_gff_file`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from bacpredict.engine.config import KP, final_root, resolve_data_root


def _abs_path(value: str | float, root: Path) -> str | float:
    """Resolve data-root-relative paths to absolute; pass through NaN/absolute unchanged."""
    if not isinstance(value, str) or not value:
        return value
    p = Path(value)
    return str(p if p.is_absolute() else root / p)


def find_missing(
    metadata_tsv: Path,
    embeddings_dir: Path,
    data_mount_root: Path,
) -> pd.DataFrame:
    """Return rows of kpsc_final_list samples that lack an embedding file."""
    df = pd.read_csv(metadata_tsv, sep="\t", low_memory=False)

    for col in ("Sample", "kpsc_final_list", "sr_assembly_file", "sr_gff_file"):
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
        ~kpsc["Sample"].isin(existing), ["Sample", "sr_assembly_file", "sr_gff_file"]
    ].copy()
    missing["sr_assembly_file"] = missing["sr_assembly_file"].map(lambda v: _abs_path(v, data_mount_root))
    missing["sr_gff_file"] = missing["sr_gff_file"].map(lambda v: _abs_path(v, data_mount_root))
    return missing


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-tsv", type=Path, default=None,
                        help="Curated metadata TSV (default: <data-root>/final/"
                             "metadata_v2_all_samples_and_columns.tsv).")
    parser.add_argument("--embeddings-dir", type=Path, default=None,
                        help="ESM embedding store (default: <data-root>/processed/train_kleb_ast/esm).")
    parser.add_argument("--out-csv", type=Path, default=None,
                        help="Output CSV of missing samples (default: <data-root>/processed/"
                             "missing_embeddings_kpsc.csv).")
    parser.add_argument("--data-mount-root", type=Path, default=None,
                        help="Root that relative sr_assembly_file/sr_gff_file entries resolve against "
                             "(default: the parent of the resolved data root).")
    args = parser.parse_args()

    metadata_tsv = args.metadata_tsv or final_root() / "metadata_v2_all_samples_and_columns.tsv"
    embeddings_dir = args.embeddings_dir or KP.data_root() / "esm"
    out_csv = args.out_csv or resolve_data_root() / "processed" / "missing_embeddings_kpsc.csv"
    data_mount_root = args.data_mount_root or resolve_data_root().parent

    print(f"Metadata: {metadata_tsv}")
    print(f"Embeddings dir: {embeddings_dir}")

    missing = find_missing(metadata_tsv, embeddings_dir, data_mount_root)

    n_missing = len(missing)
    n_missing_with_paths = int(
        missing[["sr_assembly_file", "sr_gff_file"]].notna().all(axis=1).sum()
    )
    n_gcf = int(missing["Sample"].astype(str).str.startswith("GCF_").sum())
    print(f"kpsc_final_list samples missing embeddings: {n_missing}")
    print(f"  ...with both sr_assembly_file and sr_gff_file resolved: {n_missing_with_paths}")
    print(f"  ...GCF_*: {n_gcf}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    missing.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()

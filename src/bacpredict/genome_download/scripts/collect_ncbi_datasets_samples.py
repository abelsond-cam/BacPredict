#!/usr/bin/env python3
"""Collect NCBI GCF/GCA genome accessions from a metadata/missing-samples TSV.

This helper is intended to be used by Slurm scripts (e.g. download_ncbi_datasets.sh)
to generate batch files of accessions to pass to the NCBI `datasets` CLI.

By default, rows whose `Sample` already appears to correspond to a downloaded
NCBI GFF (based on existing .gff files in the NCBI GFF directory) are skipped.
Use --download-all to disable this filesystem-based skip-existing behavior.

Inputs
------
- --metadata: Path to TSV containing at least a `Sample` column, where values like
  GCF_... or GCA_... indicate NCBI genome accessions to download.
- --n: Number of accessions to include (10=test, -1=all). Default: -1.
- --batch-dir: If set, write batch_00, batch_01, ... files here.
- --batch-size: Number of accessions per batch file (default: 100).
- --output: If set (and --batch-dir is not), write a single accession list file.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


DEFAULT_METADATA_PATH = Path(
    "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/final/metadata_final_curated_slimmed.tsv"
)

NCBI_GFF_DIR = Path(
    "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/raw/ncbi_gff3"
)


def collect_accessions(
    metadata_path: Path,
    n: int = -1,
    download_all: bool = False,
) -> list[str]:
    """Return a list of NCBI genome accessions from the `Sample` column.

    Workflow:
    - Start from the full metadata file (default: curated metadata TSV).
    - Identify rows where bakta_gff3_downloaded is False / not truthy (not downloaded).
    - Optionally (default), skip rows whose Sample already appears to have a
      corresponding downloaded NCBI GFF based on existing .gff files in NCBI_GFF_DIR,
      unless --download-all is provided.
    - Report counts of first-3-character prefixes for Sample within this not-downloaded
      (and not-skip-existing) set (including GCF/GCA and any other prefixes).
    - For the actual accession list, restrict to the remaining rows where Sample starts
      with GCF_ or GCA_, and return those values.
    """
    df = pd.read_csv(metadata_path, sep="\t", low_memory=False)
    initial_count = len(df)

    if "Sample" not in df.columns:
        print("ERROR: 'Sample' column not found in metadata", file=sys.stderr)
        sys.exit(1)

    # Determine which rows are already downloaded according to bakta_gff3_downloaded
    downloaded_col = "bakta_gff3_downloaded"
    if downloaded_col in df.columns:
        downloaded_series = (
            df[downloaded_col]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )
    else:
        print(
            f"WARNING: '{downloaded_col}' column not found; treating all rows as not downloaded.",
            file=sys.stderr,
        )
        downloaded_series = pd.Series(False, index=df.index)

    not_downloaded = df[~downloaded_series].copy()
    not_downloaded_count = len(not_downloaded)

    print(
        f"In metadata: total rows={initial_count:,}, not-downloaded (bakta_gff3_downloaded=False)={not_downloaded_count:,}",
        file=sys.stderr,
    )

    if not_downloaded_count == 0:
        print("All rows appear to be marked as downloaded; nothing to do.", file=sys.stderr)
        return []

    # Optional filesystem-based skip-existing filter:
    # If download_all is False, drop rows whose Sample appears to correspond
    # to an already-downloaded NCBI GFF in NCBI_GFF_DIR, based on .gff stems.
    if not download_all:
        if NCBI_GFF_DIR.is_dir():
            gff_stems = {p.stem for p in NCBI_GFF_DIR.glob("*.gff")}
        else:
            print(
                f"WARNING: NCBI_GFF_DIR does not exist ({NCBI_GFF_DIR}); "
                "treating as if no .gff files are present.",
                file=sys.stderr,
            )
            gff_stems = set()

        if gff_stems:
            keep_mask = []
            nd_samples_for_mask = not_downloaded["Sample"].astype(str)
            skipped_due_to_gff = 0

            for sample in nd_samples_for_mask:
                if any(stem in sample for stem in gff_stems):
                    keep_mask.append(False)
                    skipped_due_to_gff += 1
                else:
                    keep_mask.append(True)

            not_downloaded = not_downloaded[keep_mask].copy()
            print(
                f"Skip-existing filter: skipped {skipped_due_to_gff:,} rows "
                "because a matching .gff file already exists in NCBI_GFF_DIR.",
                file=sys.stderr,
            )

        not_downloaded_count = len(not_downloaded)
        print(
            f"After skip-existing filter: remaining not-downloaded rows={not_downloaded_count:,}",
            file=sys.stderr,
        )

    nd_samples = not_downloaded["Sample"].astype(str)
    nd_prefix = nd_samples.str[:3]  # type: ignore[reportAttributeAccessIssue]

    # Counts for all prefixes among not-downloaded
    prefix_counts = nd_prefix.value_counts()
    total_gcf_nd = int((nd_prefix == "GCF").sum())
    total_gca_nd = int((nd_prefix == "GCA").sum())

    print(
        "Among not-downloaded rows, Sample prefix (first 3 chars) counts (top 10):",
        file=sys.stderr,
    )
    for prefix, count in prefix_counts.head(10).items():
        print(f"  {prefix or '<EMPTY>'}: {count}", file=sys.stderr)

    print(
        f"GCF among not-downloaded: {total_gcf_nd:,}; GCA among not-downloaded: {total_gca_nd:,}",
        file=sys.stderr,
    )

    # Accessions to download: not-downloaded rows where Sample starts with GCF_ or GCA_
    # Convert from full sample string (e.g. GCF_020526085.1_ASM2052608v1_genomic)
    # to a base accession that NCBI datasets understands (e.g. GCF_020526085.1).
    gcf_mask_nd = nd_samples.str.startswith("GCF_")  # type: ignore[reportAttributeAccessIssue]
    gca_mask_nd = nd_samples.str.startswith("GCA_")  # type: ignore[reportAttributeAccessIssue]
    mask_ncbi_nd = gcf_mask_nd | gca_mask_nd

    ncbi_samples = nd_samples[mask_ncbi_nd]

    def _to_base_accession(sample: str) -> str:
        parts = sample.split("_")
        # Expect patterns like GCF_<number>.<version>_..., so keep first two chunks
        if len(parts) >= 2:
            return "_".join(parts[:2])
        return sample

    base_accessions = [_to_base_accession(s) for s in ncbi_samples]
    # Deduplicate while preserving order
    seen: set[str] = set()
    accessions: list[str] = []
    for acc in base_accessions:
        if acc not in seen:
            seen.add(acc)
            accessions.append(acc)

    if n >= 0:
        accessions = accessions[:n]

    print(
        f"Selected {len(accessions):,} unique base accessions for download "
        f"from {len(ncbi_samples):,} not-downloaded GCF_/GCA_ samples",
        file=sys.stderr,
    )
    return accessions


def write_batches(
    accessions: list[str],
    batch_dir: Path | None,
    batch_size: int,
    output: Path | None,
) -> None:
    """Write accessions either to batch files or a single file."""
    if batch_dir is not None:
        batch_dir.mkdir(parents=True, exist_ok=True)
        num_batches = (len(accessions) + batch_size - 1) // batch_size if accessions else 0
        width = max(2, len(str(num_batches - 1))) if num_batches > 0 else 2

        for i in range(num_batches):
            start = i * batch_size
            chunk = accessions[start : start + batch_size]
            batch_path = batch_dir / f"batch_{i:0{width}d}"
            batch_path.write_text("\n".join(chunk) + ("\n" if chunk else ""))

        total = len(accessions)
        print(f"Wrote {num_batches} batch files to {batch_dir}", file=sys.stderr)
        print(f"TOTAL={total}", file=sys.stderr)
        print(f"NUM_BATCHES={num_batches}", file=sys.stderr)
    elif output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(accessions) + ("\n" if accessions else ""))
        print(f"Wrote {len(accessions)} accessions to {output}", file=sys.stderr)
    else:
        for acc in accessions:
            print(acc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect NCBI GCF/GCA genome accessions (for datasets CLI batching)."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help=(
            "Path to metadata TSV containing a 'Sample' column "
            f"(default: {DEFAULT_METADATA_PATH})"
        ),
    )
    parser.add_argument(
        "--n",
        type=int,
        default=-1,
        help="Number of accessions to include (10=test, -1=all). Default: -1",
    )
    parser.add_argument(
        "--download-all",
        action="store_true",
        help=(
            "If set, do not skip samples that appear to already have a "
            "downloaded NCBI GFF in the standard ncbi_gff3 directory."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file for accession IDs (default: stdout, unless --batch-dir is set).",
    )
    parser.add_argument(
        "--batch-dir",
        type=Path,
        default=None,
        help="If set, write batch_00, batch_01, ... here instead of a single file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Accessions per batch when --batch-dir is set (default: 100).",
    )

    args = parser.parse_args()

    accessions = collect_accessions(
        metadata_path=args.metadata,
        n=args.n,
        download_all=args.download_all,
    )
    write_batches(
        accessions=accessions,
        batch_dir=args.batch_dir,
        batch_size=args.batch_size,
        output=args.output,
    )


if __name__ == "__main__":
    main()


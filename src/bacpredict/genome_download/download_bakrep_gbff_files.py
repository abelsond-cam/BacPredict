#!/usr/bin/env python3
"""Download bakrep bakta-annotated files (GBFF or GFF3): collection, skip-existing, and report missing.

This script supports the bakrep download pipeline. It is typically called from
download_bakrep.sh, which handles the actual bakrep CLI downloads; this script
handles metadata filtering, sample collection, and post-download flag updates.

Pipeline overview
-----------------
1. collect: Filter metadata to obtain sample IDs to download.
   - Excludes samples where bakta_gbff_downloaded=True (or bakta_gff3_downloaded for gff3).
   - Keeps only sample_accession starting with "SAM" (BakRep accession format).
   - Limits to first N samples (use -1 for all).
   - Outputs sample IDs to batch files.

2. update-flags: ALWAYS runs after collection (if output-dir exists).
   - Scans for *.bakta.gbff.gz or *.bakta.gff3.gz depending on --filetype.
   - Updates bakta_gbff_downloaded or bakta_gff3_downloaded column in metadata.
   - Reports missing samples.

Variables / options
-------------------
- filetype (--filetype): gbff (default) or gff3. Controls file pattern and metadata column.
- n (--n): Number of samples to process in collect mode.
  10 = test run (default), -1 = all filtered samples.
- skip-existing (default: True): Exclude samples already marked as downloaded.
- batch-size: Controls samples per bakrep batch (default 100).
  Use 1 to retry failed samples individually.
- overwrite-existing: Re-download even if files exist (disables skip-existing).
- output-dir: Directory where downloaded files live (for automatic flag updates).
- missing-output: Optional file to write list of missing sample IDs.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

RDS_ROOT = Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw")
DEFAULT_METADATA_FILE = RDS_ROOT / "david" / "final" / "metadata_final_curated_slimmed.tsv"
DEFAULT_GBFF_DIR = RDS_ROOT / "david" / "raw" / "klebsiella_gbff"


def _downloaded_column(filetype: str) -> str:
    """Return metadata column name for the given filetype."""
    return f"bakta_{filetype}_downloaded"


def collect_sample_ids(
    metadata_path: Path,
    skip_existing: bool = True,
    n: int = 10,
    filetype: str = "gbff",
) -> list[str]:
    """Filter metadata to obtain sample IDs for download.

    Args:
        metadata_path: Path to metadata TSV.
        skip_existing: If True, exclude samples where downloaded flag=True.
        n: Number of samples to return. -1 = all filtered samples.
        filetype: gbff or gff3; determines which metadata column to check.

    Returns
    -------
        List of sample accession IDs.
    """
    df = pd.read_csv(metadata_path, sep="\t", low_memory=False)
    initial_count = len(df)
    df_all = df.copy()
    col = _downloaded_column(filetype)

    if "sample_accession" not in df.columns:
        print("ERROR: 'sample_accession' column not found in metadata", file=sys.stderr)
        sys.exit(1)

    # Summary of SAM vs non-SAM accessions in the full metadata
    sample_series = df_all["sample_accession"].astype(str)
    sam_mask = sample_series.str.startswith("SAM")
    num_sam = int(sam_mask.sum())
    num_non_sam = int((~sam_mask).sum())
    print(
        f"In metadata: total={initial_count:,}, SAM-prefixed={num_sam:,}, non-SAM={num_non_sam:,}",
        file=sys.stderr,
    )
    if num_non_sam > 0:
        non_sam = sample_series[~sam_mask]
        prefix_counts = non_sam.str[:3].value_counts().head(10)
        print("Top non-SAM prefixes (first 3 chars):", file=sys.stderr)
        for prefix, count in prefix_counts.items():
            print(f"  {prefix or '<EMPTY>'}: {count}", file=sys.stderr)

    # Filter 1: Exclude already-downloaded (if skip_existing)
    if skip_existing:
        if col in df.columns:
            df = df[
                ~df[col]
                .astype(str)
                .str.lower()
                .isin(["true", "1", "yes"])
            ]
            print(f"After {col} filter: {len(df):,} samples", file=sys.stderr)
        else:
            print(f"{col} column not found; skipping this filter", file=sys.stderr)
    else:
        print("Skip-existing disabled; including all samples", file=sys.stderr)

    # Filter 2: Only sample_accession starting with SAM
    df = df[df["sample_accession"].astype(str).str.startswith("SAM")]
    print(f"After SAM prefix filter: {len(df):,} samples", file=sys.stderr)

    sample_ids = df["sample_accession"].tolist()
    print(f"Filtered from {initial_count:,} to {len(sample_ids):,} samples", file=sys.stderr)

    if n >= 0:
        sample_ids = sample_ids[:n]
    return sample_ids


def collect_cmd(args: argparse.Namespace) -> None:
    """Collect subcommand: output sample IDs for download."""
    filetype = getattr(args, "filetype", "gbff")
    col = _downloaded_column(filetype)
    print("Filtering metadata with Python...", file=sys.stderr)
    if args.skip_existing:
        print(f"  - Excluding samples where {col} = True", file=sys.stderr)
    else:
        print("  - Including all samples (overwrite mode)", file=sys.stderr)
    print("  - Including only samples where sample_accession starts with 'SAM'", file=sys.stderr)

    sample_ids = collect_sample_ids(
        metadata_path=args.metadata,
        skip_existing=args.skip_existing,
        n=args.n,
        filetype=filetype,
    )

    if args.batch_dir is not None:
        # Write batch files directly (no intermediate file)
        args.batch_dir.mkdir(parents=True, exist_ok=True)
        batch_size = args.batch_size
        num_batches = (len(sample_ids) + batch_size - 1) // batch_size if sample_ids else 0
        width = max(2, len(str(num_batches - 1))) if num_batches > 0 else 2

        for i in range(num_batches):
            start = i * batch_size
            chunk = sample_ids[start : start + batch_size]
            batch_path = args.batch_dir / f"batch_{i:0{width}d}"
            batch_path.write_text("\n".join(chunk) + "\n" if chunk else "")

        total = len(sample_ids)
        print(f"Wrote {num_batches} batch files to {args.batch_dir}", file=sys.stderr)
        print(f"TOTAL={total}", file=sys.stderr)
        print(f"NUM_BATCHES={num_batches}", file=sys.stderr)
    elif args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(sample_ids) + "\n" if sample_ids else "")
        print(f"Wrote {len(sample_ids)} sample IDs to {args.output}", file=sys.stderr)
    else:
        for sid in sample_ids:
            print(sid)


def update_metadata_flags(
    metadata_path: Path,
    output_dir: Path,
    filetype: str,
    missing_output: Path | None = None,
) -> None:
    """Scan output dir, update metadata flags, and report missing samples.
    
    Args:
        metadata_path: Path to metadata TSV
        output_dir: Directory containing downloaded files
        filetype: gbff or gff3
        missing_output: Optional path to write missing sample IDs
    """
    col = _downloaded_column(filetype)
    pattern = f"*.bakta.{filetype}.gz"

    if not output_dir.exists():
        print(f"ERROR: Output directory not found: {output_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {output_dir} for {pattern} files...", file=sys.stderr)
    samples_with_files = collect_sample_accessions_from_files(output_dir, filetype)
    file_count = len(list(output_dir.rglob(pattern)))
    print(f"Found {file_count} {pattern} files", file=sys.stderr)
    print(f"Extracted {len(samples_with_files)} unique sample accessions from file paths", file=sys.stderr)

    df = pd.read_csv(metadata_path, sep="\t", low_memory=False)
    print(f"Loaded {len(df)} rows from metadata", file=sys.stderr)

    if "sample_accession" not in df.columns:
        print("ERROR: 'sample_accession' column not found in metadata", file=sys.stderr)
        sys.exit(1)

    # Expected samples (SAM-prefixed only, to match collect logic)
    expected = set(
        df[df["sample_accession"].astype(str).str.startswith("SAM")]["sample_accession"].astype(str)
    )
    missing = expected - samples_with_files

    # Update flag
    df[col] = df["sample_accession"].isin(samples_with_files)

    num_downloaded = int(df[col].sum())
    num_total = len(df)
    print("\nResults:")
    print(f"  Samples with bakta.{filetype}.gz: {num_downloaded:,} / {num_total:,} ({num_downloaded / num_total * 100:.1f}%)")
    print(f"  Samples without file: {num_total - num_downloaded:,}")

    if missing:
        print(f"  Missing sample IDs: {len(missing):,}")
        if missing_output:
            missing_output.parent.mkdir(parents=True, exist_ok=True)

            # Join missing IDs back to metadata and emit a TSV with selected columns
            missing_df = df[df["sample_accession"].astype(str).isin(missing)].copy()
            desired_cols = [
                "Sample",
                "is_kpsc",
                "kpsc_final_list",
                "is_refseq",
                "is_nctc",
                "sample_accession",
            ]
            existing_cols = [c for c in desired_cols if c in missing_df.columns]
            missing_cols = [c for c in desired_cols if c not in missing_df.columns]
            if missing_cols:
                print(
                    f"WARNING: The following columns were not found in metadata and will be omitted: {', '.join(missing_cols)}",
                    file=sys.stderr,
                )

            # If none of the desired columns exist, fall back to a single sample_accession column
            if not existing_cols:
                existing_cols = ["sample_accession"]

            missing_df[existing_cols].to_csv(missing_output, sep="\t", index=False)
            print(f"  Missing sample metadata written to: {missing_output}")

    df.to_csv(metadata_path, sep="\t", index=False)
    print(f"\nUpdated metadata: {metadata_path}")


def collect_sample_accessions_from_files(output_dir: Path, filetype: str = "gbff") -> set[str]:
    """Scan output dir for .bakta.<filetype>.gz and extract sample accessions."""
    pattern = f"*.bakta.{filetype}.gz"
    files = list(output_dir.rglob(pattern))
    return {f.parent.name for f in files}


def main() -> None:
    """Parse arguments and run collection + flag updates."""
    parser = argparse.ArgumentParser(
        description="Download bakrep GBFF/GFF3: collection and flag updates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA_FILE,
        help="Path to metadata TSV",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of samples (10=test, -1=all). Default: 10",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Include samples already marked as downloaded (overwrite mode)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file for sample IDs (default: stdout)",
    )
    parser.add_argument(
        "--batch-dir",
        type=Path,
        default=None,
        help="If set, write batch_00, batch_01, ... here instead of single file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Samples per batch when --batch-dir is set (default: 100)",
    )
    parser.add_argument(
        "--filetype",
        type=str,
        choices=["gbff", "gff3"],
        default="gbff",
        help="File format: gbff or gff3 (default: gbff)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_GBFF_DIR,
        help="Directory containing .bakta.gbff.gz or .bakta.gff3.gz files (for flag updates)",
    )
    parser.add_argument(
        "--missing-output",
        type=Path,
        default=None,
        help="Write missing sample IDs to this file",
    )

    args = parser.parse_args()
    args.skip_existing = not args.no_skip_existing

    # Determine if we should run collection
    run_collection = args.batch_dir is not None or args.output is not None
    
    # Run collection if requested
    if run_collection:
        collect_cmd(args)

    # Always update flags (if output-dir exists)
    if args.output_dir.exists():
        print("", file=sys.stderr)
        print("=" * 44, file=sys.stderr)
        print("Updating metadata flags...", file=sys.stderr)
        print("=" * 44, file=sys.stderr)
        update_metadata_flags(
            metadata_path=args.metadata,
            output_dir=args.output_dir,
            filetype=args.filetype,
            missing_output=args.missing_output,
        )
    else:
        print(f"Skipping flag update: {args.output_dir} does not exist", file=sys.stderr)


if __name__ == "__main__":
    main()

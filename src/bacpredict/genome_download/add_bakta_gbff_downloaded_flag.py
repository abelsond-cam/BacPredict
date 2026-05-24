"""Add bakta_gbff_downloaded flag to metadata based on file presence.

This script scans the klebsiella_gbff directory recursively for .bakta.gbff.gz files
and adds a boolean column to the metadata TSV indicating which samples have downloaded files.

Files are matched by sample_accession (extracted from parent directory name).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

RDS_ROOT = Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw")
GBFF_DIR = RDS_ROOT / "david" / "raw" / "klebsiella_gbff"


def collect_gbff_samples(gbff_dir: Path) -> set[str]:
    """Recursively find all .bakta.gbff.gz files and extract sample accessions.

    Args:
        gbff_dir: Directory to search for gbff files

    Returns
    -------
        Set of sample accessions that have .bakta.gbff.gz files
    """
    print(f"Scanning {gbff_dir} for *.bakta.gbff.gz files...")
    
    gbff_files = list(gbff_dir.rglob("*.bakta.gbff.gz"))
    print(f"Found {len(gbff_files)} .bakta.gbff.gz files")
    
    # Extract sample accessions from parent directory names
    # Assuming structure: gbff_dir/XXXX/SAMXXXXXXX/SAMXXXXXXX.bakta.gbff.gz
    sample_accessions = {filepath.parent.name for filepath in gbff_files}
    print(f"Extracted {len(sample_accessions)} unique sample accessions")
    
    return sample_accessions


def add_flag_to_metadata(
    metadata_path: Path,
    samples_with_files: set[str],
    output_path: Path | None = None,
    dry_run: bool = False,
) -> None:
    """Add bakta_gbff_downloaded flag to metadata TSV.

    Args:
        metadata_path: Path to input metadata TSV
        samples_with_files: Set of sample accessions that have gbff files
        output_path: Path to output TSV (defaults to overwriting input)
        dry_run: If True, only print statistics without writing output
    """
    print(f"\nLoading metadata from {metadata_path}...")
    metadata = pd.read_csv(metadata_path, sep="\t", low_memory=False)
    print(f"Loaded {len(metadata)} rows, {len(metadata.columns)} columns")
    
    # Check if sample_accession column exists
    if "sample_accession" not in metadata.columns:
        print("ERROR: 'sample_accession' column not found in metadata", file=sys.stderr)
        sys.exit(1)
    
    # Add or overwrite bakta_gbff_downloaded column
    metadata["bakta_gbff_downloaded"] = metadata["sample_accession"].isin(samples_with_files)
    
    # Print statistics
    num_downloaded = metadata["bakta_gbff_downloaded"].sum()
    num_total = len(metadata)
    print(f"\nResults:")
    print(f"  Samples with bakta.gbff.gz: {num_downloaded:,} / {num_total:,} ({num_downloaded/num_total*100:.1f}%)")
    print(f"  Samples without file: {num_total - num_downloaded:,}")
    
    if dry_run:
        print("\n--dry-run specified; not writing output")
        return
    
    # Determine output path
    if output_path is None:
        output_path = metadata_path
        print(f"\nOverwriting input file: {output_path}")
    else:
        print(f"\nWriting output to: {output_path}")
    
    # Write output
    metadata.to_csv(output_path, sep="\t", index=False)
    print("Done!")


def main():
    """Main entry point."""
    print("Starting add_bakta_gbff_downloaded_flag.py")
     
    parser = argparse.ArgumentParser(
        description="Add bakta_gbff_downloaded flag to metadata based on file presence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/final/metadata_final_curated_slimmed.tsv",
        help="Path to metadata TSV file",
    )
    parser.add_argument(
        "--gbff-dir",
        type=Path,
        default=GBFF_DIR,
        help=f"Directory to search for .bakta.gbff.gz files (default: {GBFF_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output TSV path (default: overwrite input)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts only, do not write output",
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.metadata.exists():
        print(f"ERROR: Metadata file not found: {args.metadata}", file=sys.stderr)
        sys.exit(1)
    
    if not args.gbff_dir.exists():
        print(f"ERROR: GBFF directory not found: {args.gbff_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Collect sample IDs with gbff files
    samples_with_files = collect_gbff_samples(args.gbff_dir)
    
    # Add flag to metadata
    add_flag_to_metadata(args.metadata, samples_with_files, args.output, args.dry_run)


if __name__ == "__main__":
    main()

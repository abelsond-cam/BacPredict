#!/usr/bin/env python3
"""
Filter ESMC embedding parquet files by Klebsiella metadata.

This script left-joins metadata with the genome-to-parquet mapping,
prints join and size-estimation statistics, and optionally filters
the raw ESMC parquet files to write only rows matching the metadata.
"""

import argparse
import os
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

# Primary dataset constants for size estimation
PRIMARY_TOTAL_GENOMES = 1_683_600
PRIMARY_TOTAL_TB = 14.7


def load_and_join(metadata_path: str, mapping_path: str) -> pd.DataFrame:
    """
    Load metadata and mapping files, then left join.

    Args:
        metadata_path: Path to metadata TSV file
        mapping_path: Path to mapping CSV file

    Returns
    -------
        DataFrame with metadata columns plus 'filename' from mapping
    """
    print(f"Loading metadata from {metadata_path}...")
    metadata = pd.read_csv(metadata_path, sep="\t")
    print(f"  Loaded {len(metadata)} metadata rows")

    print(f"Loading mapping from {mapping_path}...")
    mapping = pd.read_csv(mapping_path)
    print(f"  Loaded {len(mapping)} genome-to-file mappings")

    print("Performing left join on Sample = genome_id...")
    merged = metadata.merge(mapping[["genome_id", "filename"]], left_on="Sample", right_on="genome_id", how="left")
    # Drop the duplicate genome_id column from the right side
    merged = merged.drop(columns=["genome_id"])

    return merged


def print_join_statistics(merged_df: pd.DataFrame) -> None:
    """
    Print statistics about the join results.

    Args:
        merged_df: Result of left join with 'filename' column
    """
    rows_with_file = merged_df["filename"].notna().sum()
    unique_files = merged_df["filename"].dropna().nunique()

    print("\n=== Join Statistics ===")
    print(f"Rows in metadata with a matching file: {rows_with_file:,}")
    print(f"Unique filenames: {unique_files:,}")


def print_size_estimation(merged_df: pd.DataFrame) -> None:
    """
    Print estimated output size based on primary dataset totals.

    Args:
        merged_df: Result of left join with 'filename' column
    """
    n_samples_with_data = merged_df["filename"].notna().sum()
    fraction = n_samples_with_data / PRIMARY_TOTAL_GENOMES
    estimated_size_tb = PRIMARY_TOTAL_TB * fraction
    estimated_size_gib = estimated_size_tb * 1024

    print("\n=== Size Estimation ===")
    print(f"Samples with data in files: {n_samples_with_data:,}")
    print(f"Fraction of primary data: {fraction:.4%}")
    print(f"Estimated filtered size: {estimated_size_tb:.2f} TB ({estimated_size_gib:.1f} GiB)")

    if estimated_size_gib <= 100:
        print("✓ Estimated size fits within 100 GiB home quota")
    else:
        print(f"⚠ Estimated size exceeds 100 GiB home quota by {estimated_size_gib - 100:.1f} GiB")


def open_and_filter_genomes(merged_df: pd.DataFrame, data_dir: str, output_dir: str, max_files: int = -1) -> None:
    """
    Sequential filtering: open each unique parquet file, filter by metadata samples, and save.

    Args:
        merged_df: Result of left join with 'Sample' and 'filename' columns
        data_dir: Directory containing raw ESMC parquet files
        output_dir: Directory to write filtered parquet files
        max_files: Maximum number of files to process (-1 for all)
    """
    # Get rows with filenames and build filename -> samples mapping
    has_file = merged_df["filename"].notna()
    unique_filenames = merged_df.loc[has_file, "filename"].unique()

    if max_files > 0:
        unique_filenames = unique_filenames[:max_files]

    print(f"\nProcessing {len(unique_filenames)} unique parquet files sequentially...")

    os.makedirs(output_dir, exist_ok=True)

    for filename in tqdm(unique_filenames, desc="Filtering files", unit="file"):
        # Get samples for this file
        samples_for_file = merged_df.loc[merged_df["filename"] == filename, "Sample"].dropna().unique()
        samples_set = set(samples_for_file)

        # Open parquet file
        input_path = os.path.join(data_dir, filename)
        df = pd.read_parquet(input_path, engine="pyarrow")

        # Filter rows where genome_id is in our metadata samples
        df_filtered = df[df["genome_id"].isin(samples_set)]

        # Save filtered parquet
        output_path = os.path.join(output_dir, filename)
        df_filtered.to_parquet(output_path, engine="pyarrow", index=False)


def _process_one_file(args: tuple[str, set[str], str, str]) -> str:
    """
    Worker function to process one parquet file (for parallel execution).

    Args:
        args: Tuple of (filename, samples_set, data_dir, output_dir)

    Returns
    -------
        Filename that was processed
    """
    filename, samples_set, data_dir, output_dir = args

    # Open parquet file
    input_path = os.path.join(data_dir, filename)
    df = pd.read_parquet(input_path, engine="pyarrow")

    # Filter rows where genome_id is in our metadata samples
    df_filtered = df[df["genome_id"].isin(samples_set)]

    # Save filtered parquet
    output_path = os.path.join(output_dir, filename)
    df_filtered.to_parquet(output_path, engine="pyarrow", index=False)

    return filename


def filter_genomes_parallel(
    merged_df: pd.DataFrame, data_dir: str, output_dir: str, processes: int = 10, max_files: int = -1
) -> None:
    """
    Parallel filtering: process multiple parquet files concurrently.

    Args:
        merged_df: Result of left join with 'Sample' and 'filename' columns
        data_dir: Directory containing raw ESMC parquet files
        output_dir: Directory to write filtered parquet files
        processes: Number of parallel worker processes
        max_files: Maximum number of files to process (-1 for all)
    """
    # Get rows with filenames and build filename -> samples mapping
    has_file = merged_df["filename"].notna()
    unique_filenames = merged_df.loc[has_file, "filename"].unique()

    if max_files > 0:
        unique_filenames = unique_filenames[:max_files]

    # Build list of (filename, samples_set, data_dir, output_dir) tuples
    args_list = []
    for filename in unique_filenames:
        samples_for_file = merged_df.loc[merged_df["filename"] == filename, "Sample"].dropna().unique()
        samples_set = set(samples_for_file)
        args_list.append((filename, samples_set, data_dir, output_dir))

    print(f"\nProcessing {len(args_list)} unique parquet files with {processes} processes...")

    os.makedirs(output_dir, exist_ok=True)

    # Process files in parallel with progress bar
    with Pool(processes=processes) as pool:
        list(tqdm(pool.imap(_process_one_file, args_list), total=len(args_list), desc="Filtering files", unit="file"))

    print("Filtering complete!")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Filter ESMC embeddings by Klebsiella metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Only run load, join, and size estimation; do not filter or write output",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/metadata/metadata_v2_all_samples_and_columns.tsv",
        help="Path to metadata TSV file (default: data/metadata/metadata_v2_all_samples_and_columns.tsv)",
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default="data/metadata/genome_to_parquet_file_mapping.csv",
        help="Path to genome-to-parquet mapping CSV (default: data/metadata/genome_to_parquet_file_mapping.csv)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing the raw ESMC parquet files to filter.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write the KPSC-filtered parquet files.",
    )
    parser.add_argument(
        "--processes", type=int, default=10, help="Number of parallel processes; 0 or 1 for sequential (default: 10)"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=-1,
        help="Maximum number of parquet files to process; -1 for all (default: -1)",
    )

    args = parser.parse_args()

    # Load and join
    merged_df = load_and_join(args.metadata, args.mapping)

    # Print statistics
    print_join_statistics(merged_df)
    print_size_estimation(merged_df)

    # If estimate-only, exit here
    if args.estimate_only:
        print("\n--estimate-only specified; exiting without filtering.")
        return

    # Otherwise, proceed with filtering
    if args.processes <= 1:
        # Sequential processing
        open_and_filter_genomes(merged_df, args.data_dir, args.output_dir, args.max_files)
    else:
        # Parallel processing
        filter_genomes_parallel(merged_df, args.data_dir, args.output_dir, args.processes, args.max_files)

    print("\nDone!")


if __name__ == "__main__":
    main()

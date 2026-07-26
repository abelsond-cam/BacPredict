#!/usr/bin/env python
"""Command-line script to preprocess EBI AMR records.

This script loads EBI AMR records, converts resistance phenotypes to binary,
converts MIC values to log scale, filters antibiotics, creates metadata and
pivot tables, and generates visualizations. Works for any species in EBI format.

Usage:
    uv run python src/kleb_ast/preprocess_ebi_amr_records.py
    uv run python src/kleb_ast/preprocess_ebi_amr_records.py --input /path/to/ebi_amr.csv
"""

import argparse
import sys
from pathlib import Path

from bacpredict.engine.config import raw_root
from bacpredict.engine.labels.parse_ebi_ast_to_binary import parse_ebi_ast_to_binary as process_klebsiella_ast_data

DEFAULT_INPUT_LEAF = "klebsiella_ebi_amr_records_20260216.csv"


def main():
    """Main entry point for the EBI AMR preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="Preprocess EBI AMR records: convert phenotypes, create pivot tables, antibiogram."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to EBI AMR CSV file "
             f"(default: <data-root>/raw/{DEFAULT_INPUT_LEAF})",
    )
    parser.add_argument(
        "--min-antibiotic-count",
        type=int,
        default=1000,
        help="Minimum number of measurements to keep an antibiotic (default: 1000)",
    )
    args = parser.parse_args()

    input_file = args.input or raw_root() / DEFAULT_INPUT_LEAF

    print("=" * 80)
    print("EBI AMR RECORDS PREPROCESSING")
    print("=" * 80)
    print(f"\nInput file: {input_file}")
    print(f"Minimum antibiotic count: {args.min_antibiotic_count}")
    print("\n" + "=" * 80 + "\n")

    try:
        results = process_klebsiella_ast_data(
            input_file=input_file,
            min_antibiotic_count=args.min_antibiotic_count,
        )

        print("\n" + "=" * 80)
        print("PROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nOutput files created:")
        for key, path in results["paths"].items():
            print(f"  - {key}: {path}")
        print(f"\nTotal antibiotics kept: {len(results['kept_antibiotics'])}")
        print(f"Metadata samples: {len(results['metadata'])}")
        print(f"Binary AST shape: {results['binary_ast'].shape}")
        print(f"Regression log_mic shape: {results['regression_log_mic'].shape}")

    except FileNotFoundError as e:
        print("\n" + "=" * 80)
        print("ERROR: INPUT FILE NOT FOUND!")
        print("=" * 80)
        print(f"\nThe input file does not exist: {input_file}")
        print("\nPlease check that the file path is correct and the file exists.")
        print(f"\nError details: {e}")
        return 1

    except Exception as e:  # noqa: BLE001 (top-level CLI guard: report any failure and exit non-zero)
        print("\n" + "=" * 80)
        print("ERROR: PROCESSING FAILED!")
        print("=" * 80)
        print(f"\nError: {e}")
        print("\nFull traceback:")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

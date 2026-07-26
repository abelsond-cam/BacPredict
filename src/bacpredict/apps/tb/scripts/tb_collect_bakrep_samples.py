#!/usr/bin/env python3
"""Collect BioSamples from the TB AMR records CSV and prepare BakRep batches.

Standalone helper for run_download_bakrep.sh. Runs under the `bakrep_download`
micromamba env (pandas only - no project imports, no uv).

Two modes:

1. Collect (default): deduplicate BioSamples, filter to SAM* prefixes,
   skip those whose .bakta.<filetype>.gz already exists on disk, and
   write batch files for the bakrep CLI.

2. Verify (--verify): scan the output dir for downloaded .bakta.<filetype>.gz
   files and emit a missing-samples sidecar TSV listing BioSamples in the
   input CSV that have no matching file. Never mutates the input CSV.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

BIOSAMPLE_COL = "phenotype-BioSample_ID"


def _gff_pattern(filetype: str) -> str:
    return f"*.bakta.{filetype}.gz"


def _existing_biosamples(output_dir: Path, filetype: str) -> set[str]:
    """Return the set of BioSample IDs that already have a .bakta.<filetype>.gz on disk.

    BakRep writes one subdir per BioSample, so we key off the parent dir name.
    """
    if not output_dir.is_dir():
        return set()
    return {p.parent.name for p in output_dir.rglob(_gff_pattern(filetype))}


def _load_biosamples(metadata_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load the CSV and return the deduplicated SAM*-prefixed BioSample list.

    Returns the full DataFrame too (for downstream sidecar joins) plus the
    final list of unique BioSamples to process.
    """
    df = pd.read_csv(metadata_path, low_memory=False)
    initial_rows = len(df)

    if BIOSAMPLE_COL not in df.columns:
        print(f"ERROR: '{BIOSAMPLE_COL}' column not found in {metadata_path}", file=sys.stderr)
        sys.exit(1)

    biosamples = df[BIOSAMPLE_COL].astype(str).str.strip()
    sam_mask = biosamples.str.startswith("SAM")
    num_sam = int(sam_mask.sum())
    num_non_sam = int((~sam_mask).sum())

    print(
        f"Input CSV: {initial_rows:,} rows; SAM-prefixed BioSample rows: {num_sam:,}; "
        f"non-SAM rows: {num_non_sam:,}",
        file=sys.stderr,
    )
    if num_non_sam > 0:
        prefix_counts = biosamples[~sam_mask].str[:3].value_counts().head(10)
        print("Top non-SAM prefixes (first 3 chars):", file=sys.stderr)
        for prefix, count in prefix_counts.items():
            print(f"  {prefix or '<EMPTY>'}: {count}", file=sys.stderr)

    unique = (
        biosamples[sam_mask]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    print(f"Unique SAM* BioSamples: {len(unique):,}", file=sys.stderr)
    return df, unique


def collect_cmd(args: argparse.Namespace) -> None:
    """Generate batch files of BioSamples for the bakrep CLI."""
    _, biosamples = _load_biosamples(args.metadata)

    if args.skip_existing and args.output_dir is not None:
        existing = _existing_biosamples(args.output_dir, args.filetype)
        if existing:
            before = len(biosamples)
            biosamples = [b for b in biosamples if b not in existing]
            print(
                f"Skip-existing: dropped {before - len(biosamples):,} BioSamples already "
                f"downloaded ({len(existing):,} .bakta.{args.filetype}.gz files seen on disk)",
                file=sys.stderr,
            )
    elif not args.skip_existing:
        print("Skip-existing disabled; including all SAM* BioSamples", file=sys.stderr)

    if args.n >= 0:
        biosamples = biosamples[: args.n]
        print(f"Limiting to first {args.n} BioSamples", file=sys.stderr)

    if args.batch_dir is not None:
        args.batch_dir.mkdir(parents=True, exist_ok=True)
        batch_size = args.batch_size
        num_batches = (len(biosamples) + batch_size - 1) // batch_size if biosamples else 0
        width = max(2, len(str(max(num_batches - 1, 0))))

        for i in range(num_batches):
            chunk = biosamples[i * batch_size : (i + 1) * batch_size]
            (args.batch_dir / f"batch_{i:0{width}d}").write_text(
                "\n".join(chunk) + ("\n" if chunk else "")
            )

        print(f"Wrote {num_batches} batch files to {args.batch_dir}", file=sys.stderr)
        print(f"TOTAL={len(biosamples)}", file=sys.stderr)
        print(f"NUM_BATCHES={num_batches}", file=sys.stderr)
    elif args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(biosamples) + ("\n" if biosamples else ""))
        print(f"Wrote {len(biosamples)} BioSamples to {args.output}", file=sys.stderr)
    else:
        for b in biosamples:
            print(b)


def verify_cmd(args: argparse.Namespace) -> None:
    """Scan the output dir and write a missing-samples sidecar TSV."""
    if args.output_dir is None or not args.output_dir.exists():
        print(
            f"ERROR: --output-dir is required for --verify and must exist ({args.output_dir})",
            file=sys.stderr,
        )
        sys.exit(1)

    df, biosamples = _load_biosamples(args.metadata)
    expected = set(biosamples)
    existing = _existing_biosamples(args.output_dir, args.filetype)
    file_count = sum(1 for _ in args.output_dir.rglob(_gff_pattern(args.filetype)))

    missing = sorted(expected - existing)
    have = len(expected) - len(missing)

    print(f"Scanned {args.output_dir} for {_gff_pattern(args.filetype)}", file=sys.stderr)
    print(f"  files on disk: {file_count:,}", file=sys.stderr)
    print(f"  BioSamples with file: {have:,} / {len(expected):,} "
          f"({100 * have / max(1, len(expected)):.1f}%)", file=sys.stderr)
    print(f"  Missing: {len(missing):,}", file=sys.stderr)

    if args.missing_output is None:
        return

    args.missing_output.parent.mkdir(parents=True, exist_ok=True)
    if not missing:
        # Write a header-only file so downstream tools can rely on it existing.
        pd.DataFrame(columns=[BIOSAMPLE_COL]).to_csv(args.missing_output, sep="\t", index=False)
        print(f"No missing samples; wrote empty sidecar to {args.missing_output}", file=sys.stderr)
        return

    # Preserve original CSV rows for missing BioSamples (one row per BioSample).
    missing_df = (
        df[df[BIOSAMPLE_COL].astype(str).str.strip().isin(missing)]
        .drop_duplicates(subset=[BIOSAMPLE_COL])
        .copy()
    )
    missing_df.to_csv(args.missing_output, sep="\t", index=False)
    print(f"Wrote {len(missing_df):,} missing rows to {args.missing_output}", file=sys.stderr)


def main() -> None:
    """CLI entry point: collect BioSamples for BakRep GFF3 downloads (standalone, pandas only)."""
    parser = argparse.ArgumentParser(
        description="Collect BioSamples for BakRep GFF3 downloads (standalone, pandas only)."
    )
    parser.add_argument("--metadata", type=Path, required=True, help="Path to TB AMR records CSV")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="BakRep output directory (used for skip-existing and --verify)",
    )
    parser.add_argument(
        "--filetype",
        type=str,
        default="gff3",
        choices=["gff3", "gbff"],
        help="BakRep filetype (default: gff3)",
    )
    parser.add_argument(
        "--n", type=int, default=-1, help="Number of BioSamples (-1=all, default)"
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Disable filesystem-based skip-existing filter",
    )
    parser.add_argument("--output", type=Path, default=None, help="Write a single newline-delimited list here")
    parser.add_argument("--batch-dir", type=Path, default=None, help="Write batch_00, batch_01, ... here")
    parser.add_argument("--batch-size", type=int, default=100, help="BioSamples per batch")
    parser.add_argument("--verify", action="store_true", help="Verify downloads and emit missing-samples sidecar")
    parser.add_argument("--missing-output", type=Path, default=None, help="Path to missing-samples sidecar TSV")

    args = parser.parse_args()
    args.skip_existing = not args.no_skip_existing

    if args.verify:
        verify_cmd(args)
    else:
        collect_cmd(args)


if __name__ == "__main__":
    main()

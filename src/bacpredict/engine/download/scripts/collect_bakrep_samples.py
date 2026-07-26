#!/usr/bin/env python3
"""Minimal standalone script for BakRep collect and update-flags (pandas only, no bacpredict).

Used by download_bakrep.sh to avoid uv run. Run with: micromamba run -n bakrep_download python ...
Requires: pip install pandas (in bakrep_download env).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def _downloaded_column(filetype: str) -> str:
    return f"bakta_{filetype}_downloaded"


def collect_sample_ids(
    metadata_path: Path,
    skip_existing: bool = True,
    n: int = 10,
    filetype: str = "gbff",
) -> list[str]:
    """Return up to ``n`` sample IDs from the metadata TSV (skipping already-downloaded ``filetype`` if asked)."""
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

    df = df[df["sample_accession"].astype(str).str.startswith("SAM")]
    print(f"After SAM prefix filter: {len(df):,} samples", file=sys.stderr)

    sample_ids = df["sample_accession"].tolist()
    print(f"Filtered from {initial_count:,} to {len(sample_ids):,} samples", file=sys.stderr)

    if n >= 0:
        sample_ids = sample_ids[:n]
    return sample_ids


def collect_cmd(args: argparse.Namespace) -> None:
    """CLI subcommand: print the collected sample IDs (one per line) to stdout."""
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


def collect_sample_accessions_from_files(output_dir: Path, filetype: str = "gbff") -> set[str]:
    """Scan ``output_dir`` for downloaded ``*.bakta.<filetype>.gz`` files → the set of their sample accessions."""
    pattern = f"*.bakta.{filetype}.gz"
    files = list(output_dir.rglob(pattern))
    return {f.parent.name for f in files}


def update_metadata_flags(
    metadata_path: Path,
    output_dir: Path,
    filetype: str,
    missing_output: Path | None = None,
) -> None:
    """Refresh the ``bakta_<filetype>_downloaded`` metadata flag from disk; optionally write a missing-samples sidecar."""
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

    expected = set(
        df[df["sample_accession"].astype(str).str.startswith("SAM")]["sample_accession"].astype(str)
    )
    missing = expected - samples_with_files

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


def main() -> None:
    """CLI entry point: BakRep collect + update-flags (standalone, pandas only)."""
    parser = argparse.ArgumentParser(description="BakRep collect and update-flags (standalone, pandas only)")
    parser.add_argument("--metadata", type=Path, required=True, help="Path to metadata TSV")
    parser.add_argument("--n", type=int, default=10, help="Number of samples (10=test, -1=all)")
    parser.add_argument("--no-skip-existing", action="store_true", help="Overwrite mode")
    parser.add_argument("--output", type=Path, default=None, help="Output file for sample IDs")
    parser.add_argument("--batch-dir", type=Path, default=None, help="Write batch_00, batch_01, ... here")
    parser.add_argument("--batch-size", type=int, default=100, help="Samples per batch")
    parser.add_argument("--filetype", type=str, choices=["gbff", "gff3"], default="gbff")
    parser.add_argument("--output-dir", type=Path, default=None, help="Dir with .bakta.*.gz (for flag updates)")
    parser.add_argument("--missing-output", type=Path, default=None, help="Write missing sample IDs here")

    args = parser.parse_args()
    args.skip_existing = not args.no_skip_existing

    if args.batch_dir is not None or args.output is not None:
        collect_cmd(args)

    if args.output_dir is not None and args.output_dir.exists():
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
    elif args.output_dir is not None:
        print(f"Skipping flag update: {args.output_dir} does not exist", file=sys.stderr)


if __name__ == "__main__":
    main()

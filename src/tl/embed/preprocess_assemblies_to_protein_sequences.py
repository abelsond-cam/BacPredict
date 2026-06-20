"""Generate protein sequences from Klebsiella genome annotations (CPU-only).

Reads a CSV of samples + per-sample annotation paths (produced by
`find_missing_embeddings.py`) and writes one `{Sample}_protein_sequences.parquet`
per sample for downstream GPU embedding generation.

Per-sample extractor is chosen by file extension on the `sr_gff_file` column:
- `.gff` / `.gff3` / `.gff.gz` / `.gff3.gz`: parse CDS features, splice from
  the sibling FASTA (`sr_assembly_file`), translate with codon table 11.
- `.gbff` / `.gbff.gz`: fall back to bacformer's
  `preprocess_genome_assembly` (retained for completeness; new samples are
  expected to come in via GFF + FNA).
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd
from bacformer.pp import preprocess_genome_assembly
from tqdm import tqdm

from tl.embed.extract_proteins_from_gff_fna import (
    extract_proteins_from_gff_fna,
    is_gbff_path,
    is_gff_path,
)

RDS_ROOT = Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw")
PROTEIN_SEQUENCES_DIR = RDS_ROOT / "david" / "processed" / "klebsiella_protein_sequences"
INPUT_CSV_DEFAULT = RDS_ROOT / "david" / "processed" / "missing_embeddings_kpsc.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("protein_sequences_processing.log"),
    ],
)
logger = logging.getLogger(__name__)


def load_input_csv(input_csv: Path, limit: int | None = None) -> pd.DataFrame:
    """Load the (Sample, sr_assembly_file, sr_gff_file) input CSV produced upstream."""
    df = pd.read_csv(input_csv)
    required = {"Sample", "sr_assembly_file", "sr_gff_file"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input CSV {input_csv} is missing columns: {missing_cols}")
    df = df.dropna(subset=["Sample"]).copy()
    df["Sample"] = df["Sample"].astype(str)
    if limit is not None and limit > 0:
        df = df.head(limit)
    return df.reset_index(drop=True)


def check_output_exists(sample_id: str, output_dir: Path) -> bool:
    """Check if the protein-sequences parquet already exists for a sample."""
    return (output_dir / f"{sample_id}_protein_sequences.parquet").exists()


def save_to_parquet(data_dict: dict, output_path: Path) -> None:
    """Save a single-row dict as a parquet file."""
    df = pd.DataFrame([data_dict])
    df.to_parquet(output_path, engine="pyarrow", compression="snappy")


def _extract_genome_info(sample_id: str, sr_assembly_file: str, sr_gff_file: str,
                         *, keep_internal_stop: bool = False) -> dict:
    """Dispatch to the correct extractor based on the `sr_gff_file` extension."""
    if sr_gff_file and is_gff_path(sr_gff_file):
        if not sr_assembly_file:
            raise ValueError(
                f"{sample_id}: sr_gff_file is set but sr_assembly_file is empty; "
                "GFF+FNA extractor needs both."
            )
        return extract_proteins_from_gff_fna(
            sr_gff_file, sr_assembly_file, keep_internal_stop=keep_internal_stop
        )
    if sr_gff_file and is_gbff_path(sr_gff_file):
        return preprocess_genome_assembly(filepath=str(sr_gff_file))
    if sr_assembly_file and is_gbff_path(sr_assembly_file):
        return preprocess_genome_assembly(filepath=str(sr_assembly_file))
    raise ValueError(
        f"{sample_id}: no recognised annotation file. "
        f"sr_gff_file={sr_gff_file!r}, sr_assembly_file={sr_assembly_file!r}"
    )


def process_single_genome(args_tuple: tuple) -> tuple[str, bool, str, float]:
    """Worker: extract proteins for one sample and write its parquet.

    Returns
    -------
    tuple
        ``(sample_id, success, error_message, processing_time)``.
    """
    sample_id, sr_assembly_file, sr_gff_file, output_dir, skip_existing, keep_internal_stop = args_tuple
    start_time = time.time()

    try:
        if skip_existing and check_output_exists(sample_id, output_dir):
            return sample_id, True, "Already exists (skipped)", 0.0

        genome_info = _extract_genome_info(
            sample_id, sr_assembly_file, sr_gff_file, keep_internal_stop=keep_internal_stop
        )

        # Strip metadata-only keys that the downstream embedding step does not consume.
        for k in ("strain_name", "accession_name", "protein_name", "accession_id"):
            genome_info.pop(k, None)
        genome_info = {"sample_id": sample_id, **genome_info}

        protein_output_path = output_dir / f"{sample_id}_protein_sequences.parquet"
        save_to_parquet(genome_info, protein_output_path)
        elapsed = time.time() - start_time
        return sample_id, True, "", elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"Error processing {sample_id}: {e}"
        logger.error(error_msg)
        return sample_id, False, error_msg, elapsed


def main():
    """Main execution function."""
    logger.info("Script started")
    sys.stdout.flush()

    parser = argparse.ArgumentParser(
        description="Generate protein-sequence parquets from a CSV of (Sample, sr_assembly_file, sr_gff_file)."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=INPUT_CSV_DEFAULT,
        help="CSV with columns Sample, sr_assembly_file, sr_gff_file (default: missing_embeddings_kpsc.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROTEIN_SEQUENCES_DIR,
        help="Where to write {Sample}_protein_sequences.parquet files.",
    )
    parser.add_argument("--n", type=int, default=None, help="Limit number of samples (for testing)")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip samples whose parquet already exists",
    )
    parser.add_argument("--workers", type=int, default=76, help="Number of parallel workers")
    parser.add_argument(
        "--keep-internal-stop",
        action="store_true",
        help="Keep CDS with an internal stop codon (retains the '*'), to reproduce a historical "
        "protein order — e.g. regenerating Kp parquets that must align with pre-existing embeddings.",
    )

    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PHASE 1: Setup")
    logger.info("=" * 60)
    logger.info(f"Input CSV: {args.input_csv}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Workers requested: {args.workers}")
    sys.stdout.flush()

    logger.info("")
    logger.info("PHASE 2: Loading input CSV")
    sys.stdout.flush()
    df = load_input_csv(args.input_csv, limit=args.n)
    logger.info(f"Loaded {len(df)} rows from {args.input_csv}")

    if df.empty:
        logger.error(f"No rows in {args.input_csv}")
        sys.exit(1)

    logger.info("")
    logger.info("PHASE 3: Filtering (skip-existing check)")
    sys.stdout.flush()
    if args.skip_existing:
        before = len(df)
        df = df[~df["Sample"].apply(lambda s: check_output_exists(s, output_dir))]
        skipped = before - len(df)
        if skipped > 0:
            logger.info(f"Skipping {skipped} samples with existing parquets")

    if df.empty:
        logger.info("All samples already have protein-sequence parquets")
        return

    num_workers = min(args.workers, cpu_count(), len(df))

    logger.info("")
    logger.info("=" * 60)
    logger.info("PHASE 4: Parallel processing")
    logger.info("=" * 60)
    logger.info(f"Total samples to process: {len(df)}")
    logger.info(f"Worker processes: {num_workers}")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    sys.stdout.flush()

    process_args = [
        (
            row["Sample"],
            str(row["sr_assembly_file"]) if pd.notna(row["sr_assembly_file"]) else "",
            str(row["sr_gff_file"]) if pd.notna(row["sr_gff_file"]) else "",
            output_dir,
            args.skip_existing,
            args.keep_internal_stop,
        )
        for _, row in df.iterrows()
    ]

    overall_start_time = time.time()

    results = {"success": [], "failed": []}
    error_log = []
    genome_times = []

    with Pool(processes=num_workers) as pool:
        pbar = tqdm(
            total=len(df),
            desc="Processing genomes",
            unit="sample",
            file=sys.stdout,
            mininterval=5.0,
            dynamic_ncols=False,
            smoothing=0.1,
        )
        for sample_id, success, error_msg, elapsed in pool.imap_unordered(
            process_single_genome, process_args
        ):
            if success:
                results["success"].append(sample_id)
                if elapsed > 0:
                    genome_times.append(elapsed)
                if error_msg:
                    logger.debug(f"{sample_id}: {error_msg}")
                else:
                    logger.info(f"{sample_id}: SUCCESS ({elapsed:.1f}s)")
            else:
                results["failed"].append(sample_id)
                error_log.append({"sample_id": sample_id, "error": error_msg})
                logger.warning(f"{sample_id}: FAILED")

            pbar.update(1)

            completed = len(results["success"]) + len(results["failed"])
            if len(genome_times) > 0 and completed % 25 == 0:
                avg_time = sum(genome_times) / len(genome_times)
                remaining = len(df) - completed
                est_remaining = timedelta(seconds=int(avg_time * remaining / num_workers))
                logger.info(
                    f"Progress: {completed}/{len(df)} samples | "
                    f"Avg {avg_time:.1f}s/sample | "
                    f"Est. remaining: {est_remaining}"
                )
                sys.stdout.flush()

        pbar.close()

    overall_elapsed = time.time() - overall_start_time

    logger.info("")
    logger.info("=" * 60)
    logger.info("PHASE 5: Summary")
    logger.info("=" * 60)
    logger.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total time elapsed: {timedelta(seconds=int(overall_elapsed))}")
    logger.info(f"Successfully processed: {len(results['success'])} genomes")
    logger.info(f"Failed: {len(results['failed'])} genomes")

    if genome_times:
        avg_time = sum(genome_times) / len(genome_times)
        min_time = min(genome_times)
        max_time = max(genome_times)
        logger.info("Timing statistics (for newly processed genomes):")
        logger.info(f"  - Average: {avg_time:.1f}s per genome")
        logger.info(f"  - Min: {min_time:.1f}s")
        logger.info(f"  - Max: {max_time:.1f}s")
        logger.info(f"  - Total genomes processed: {len(genome_times)}")
        logger.info(f"  - Throughput: {len(genome_times) / overall_elapsed * 60:.1f} genomes/minute")

    if error_log:
        error_log_path = Path("protein_sequences_errors.log")
        with open(error_log_path, "w") as f:
            for entry in error_log:
                f.write(f"{entry['sample_id']}: {entry['error']}\n")
        logger.info(f"Error details saved to {error_log_path}")

    logger.info("=" * 60)
    sys.stdout.flush()


if __name__ == "__main__":
    main()

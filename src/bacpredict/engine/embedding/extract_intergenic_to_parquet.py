"""Extract intergenic (non-coding) DNA regions to per-sample parquet (CPU-only).

The DNA half of the baclm store. Mirrors
:mod:`bacpredict.engine.embedding.preprocess_assemblies_to_protein_sequences` (the protein half): a
``multiprocessing.Pool`` runs :func:`extract_intergenic_from_gff_fna` over every
``(Sample, sr_assembly_file, sr_gff_file)`` row of the embedding-input CSV and writes one
``{Sample}_intergenic.parquet`` per genome. The GPU baclm job then reads these parquets instead of
re-extracting on the GPU node — the same CPU/GPU two-stage split the ESM-C pipeline uses.

Each parquet is a single row with list-valued columns: ``noncoding_sequence`` (lowercase DNA strings
for the maximal non-CDS runs) + ``noncoding_seqid`` / ``noncoding_start`` / ``noncoding_end``, and the
per-RNA index ``rna_sequence`` / ``rna_gene_name`` / ``rna_type`` / ``rna_seqid`` / ``rna_start`` /
``rna_end`` (see :mod:`bacpredict.engine.embedding.extract_intergenic_from_gff_fna`). Coordinates are 1-based inclusive,
forward strand.
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from bacpredict.engine.embedding.extract_intergenic_from_gff_fna import extract_intergenic_from_gff_fna

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("intergenic_processing.log")],
)
logger = logging.getLogger(__name__)


def check_output_exists(sample_id: str, output_dir: Path) -> bool:
    """Check if the intergenic parquet already exists for a sample."""
    return (output_dir / f"{sample_id}_intergenic.parquet").exists()


def process_single_genome(args_tuple: tuple) -> tuple[str, bool, str, float]:
    """Worker: extract intergenic regions for one sample and write its parquet.

    Returns ``(sample_id, success, error_message, processing_time)``.
    """
    sample_id, sr_assembly_file, sr_gff_file, output_dir, min_len, skip_existing = args_tuple
    start_time = time.time()
    try:
        if skip_existing and check_output_exists(sample_id, output_dir):
            return sample_id, True, "Already exists (skipped)", 0.0
        ig = extract_intergenic_from_gff_fna(sr_gff_file, sr_assembly_file, min_len=min_len)
        row = {"sample_id": sample_id, **ig}
        pd.DataFrame([row]).to_parquet(
            output_dir / f"{sample_id}_intergenic.parquet", engine="pyarrow", compression="snappy"
        )
        return sample_id, True, "", time.time() - start_time
    except Exception as e:  # noqa: BLE001
        msg = f"Error processing {sample_id}: {e}"
        logger.error(msg)
        return sample_id, False, msg, time.time() - start_time


def main() -> None:
    """Extract intergenic-region parquets for every genome in the input CSV (parallel CPU)."""
    ap = argparse.ArgumentParser(description="Extract intergenic DNA regions to per-sample parquet.")
    ap.add_argument("--input-csv", type=Path, required=True, help="CSV: Sample, sr_assembly_file, sr_gff_file")
    ap.add_argument("--output-dir", type=Path, required=True, help="dir for {Sample}_intergenic.parquet")
    ap.add_argument("--n", type=int, default=None, help="limit number of samples (testing)")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--min-intergenic-len", type=int, default=30)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input_csv)
    required = {"Sample", "sr_assembly_file", "sr_gff_file"}
    if required - set(df.columns):
        raise ValueError(f"{args.input_csv} missing columns: {required - set(df.columns)}")
    df = df.dropna(subset=["Sample", "sr_assembly_file", "sr_gff_file"]).copy()
    df["Sample"] = df["Sample"].astype(str)
    if args.n:
        df = df.head(args.n)

    if args.skip_existing:
        before = len(df)
        df = df[~df["Sample"].apply(lambda s: check_output_exists(s, args.output_dir))]
        if before - len(df):
            logger.info(f"skip-existing: dropped {before - len(df)} already-done")
    if df.empty:
        logger.info("nothing to do")
        return

    num_workers = min(args.workers, cpu_count(), len(df))
    process_args = [
        (r["Sample"], str(r["sr_assembly_file"]), str(r["sr_gff_file"]),
         args.output_dir, args.min_intergenic_len, args.skip_existing)
        for _, r in df.iterrows()
    ]

    logger.info(f"START {datetime.now():%Y-%m-%d %H:%M:%S} | {len(df)} genomes | {num_workers} workers")
    t0 = time.time()
    n_ok = n_fail = 0
    times: list[float] = []
    with Pool(processes=num_workers) as pool:
        pbar = tqdm(total=len(df), desc="intergenic", unit="sample", file=sys.stdout, mininterval=5.0)
        for sample_id, ok, msg, elapsed in pool.imap_unordered(process_single_genome, process_args):
            if ok:
                n_ok += 1
                if elapsed > 0:
                    times.append(elapsed)
            else:
                n_fail += 1
                logger.warning(f"{sample_id}: FAILED — {msg}")
            pbar.update(1)
        pbar.close()

    logger.info(f"DONE {datetime.now():%Y-%m-%d %H:%M:%S} | ok={n_ok} fail={n_fail} "
                f"| {timedelta(seconds=int(time.time() - t0))}")
    if times:
        logger.info(f"per-genome: avg {sum(times) / len(times):.2f}s (min {min(times):.2f} max {max(times):.2f})")


if __name__ == "__main__":
    main()

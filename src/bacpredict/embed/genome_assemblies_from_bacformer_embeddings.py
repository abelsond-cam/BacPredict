"""Generate genome-level embeddings from BacFormer protein embeddings.

This script processes BacFormer protein embeddings to create genome-level embeddings:
1. Discovers all .pt files containing BacFormer embeddings (~65,000 files)
2. For each file:
   a. Loads the tensor (shape: [n_proteins, 960])
   b. Computes arithmetic mean across all protein embeddings
   c. Extracts sample_id from filename
3. Collects all genome embeddings into a single matrix
4. Saves as a parquet file with sample_id as index

The script uses multiprocessing for efficient parallel processing.
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("genome_embeddings_processing.log"),
    ],
)
logger = logging.getLogger(__name__)

# Constants
INPUT_DIR = Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/klebsiella_bacformer_embeddings")
OUTPUT_PATH = Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/klebsiella_genome_embeddings.pq")
EXPECTED_EMBEDDING_DIM = 960


def find_bacformer_embedding_files(input_dir: Path, limit: int | None = None) -> list[Path]:
    """Find all BacFormer embedding .pt files in the input directory.

    Args:
        input_dir: Directory to search for .pt files
        limit: Optional limit on number of files (takes first n if set)

    Returns
    -------
        List of paths to .pt files
    """
    logger.info(f"Finding all *_bacformer_embeddings.pt files in {input_dir}")
    pt_files = sorted(input_dir.glob("*_bacformer_embeddings.pt"))

    logger.info(f"Found {len(pt_files)} total files")

    if limit:
        pt_files = pt_files[:limit]
        logger.info(f"Processing first {limit} files")

    return pt_files


def process_embedding_file(filepath: Path) -> tuple[str, np.ndarray, int] | None:
    """Process a single BacFormer embedding file.

    Loads the tensor, flattens the nested structure (contigs -> proteins),
    computes the mean across all embeddings, and extracts sample_id.

    Args:
        filepath: Path to the .pt file containing BacFormer embeddings
                 Structure: Each tensor has shape [batch_size (1), n_proteins (usually 4000-6000), dimension (960)]

    Returns
    -------
        Tuple of (sample_id, genome_embedding, num_embeddings) or None if failed
    """
    try:
        # Load tensor from .pt file
        embeddings = torch.load(filepath, map_location="cpu", weights_only=False)
        # Squeeze out the batch dimension to get [n_proteins, 960]
        all_embeddings_tensor = embeddings.squeeze(0)
        # Get number of embeddings for verification
        # Shape should now be [n_proteins, 960]
        num_embeddings = all_embeddings_tensor.shape[0]

        # Compute arithmetic mean across all protein embeddings (dimension 0)
        genome_embedding = all_embeddings_tensor.mean(dim=0).numpy()

        # Parse sample_id from filename
        # Format: {sample_id}_bacformer_embeddings.pt
        sample_id = filepath.stem.replace("_bacformer_embeddings", "")

        return (sample_id, genome_embedding, num_embeddings)

    except Exception as e:
        logger.error(f"Failed to process {filepath}: {e}")
        return None


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate genome-level embeddings from BacFormer protein embeddings"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Limit number of files to process (for testing)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count() - 1,
        help=f"Number of parallel worker processes (default: {cpu_count() - 1})",
    )

    args = parser.parse_args()

    # Log configuration
    logger.info("=" * 80)
    logger.info(f"STARTING GENOME EMBEDDINGS GENERATION: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    logger.info(f"Input directory: {INPUT_DIR}")
    logger.info(f"Output path: {OUTPUT_PATH}")
    logger.info(f"Workers: {args.workers}")

    # Verify input directory exists
    if not INPUT_DIR.exists():
        logger.error(f"Input directory does not exist: {INPUT_DIR}")
        sys.exit(1)

    # Find all .pt files
    pt_files = find_bacformer_embedding_files(INPUT_DIR, limit=args.n)

    if not pt_files:
        logger.error(f"No *_bacformer_embeddings.pt files found in {INPUT_DIR}")
        sys.exit(1)

    # Create output directory if needed
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Process files in parallel
    logger.info("=" * 80)
    logger.info("Processing BacFormer embedding files...")
    logger.info("=" * 80)

    sample_ids = []
    embeddings = []
    num_embeddings_list = []
    failed_files = []

    overall_start_time = time.time()

    with Pool(processes=args.workers) as pool:
        for result in tqdm(
            pool.imap_unordered(process_embedding_file, pt_files),
            total=len(pt_files),
            desc="Processing files",
        ):
            if result is not None:
                sample_id, genome_embedding, num_embeddings = result
                sample_ids.append(sample_id)
                embeddings.append(genome_embedding)
                num_embeddings_list.append(num_embeddings)
            else:
                failed_files.append(result)

    overall_elapsed = time.time() - overall_start_time

    # Verification statistics
    logger.info("=" * 80)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total time elapsed: {timedelta(seconds=int(overall_elapsed))}")
    logger.info(f"Successfully processed: {len(sample_ids)} files")
    logger.info(f"Failed files: {len(failed_files)}")

    if num_embeddings_list:
        avg_embeddings = np.mean(num_embeddings_list)
        min_embeddings = np.min(num_embeddings_list)
        max_embeddings = np.max(num_embeddings_list)
        logger.info(f"Embeddings per file statistics:")
        logger.info(f"  - Average: {avg_embeddings:.1f}")
        logger.info(f"  - Min: {min_embeddings}")
        logger.info(f"  - Max: {max_embeddings}")

    if not sample_ids:
        logger.error("No files were successfully processed. Exiting.")
        sys.exit(1)

    # Create DataFrame with embeddings
    logger.info("=" * 80)
    logger.info("Creating DataFrame and saving to parquet...")
    logger.info("=" * 80)

    # Stack embeddings into a 2D array
    embeddings_array = np.stack(embeddings)

    # Create DataFrame with sample_ids as index
    df = pd.DataFrame(
        embeddings_array,
        index=sample_ids,
        columns=[f"dim_{i}" for i in range(EXPECTED_EMBEDDING_DIM)],
    )
    
    # Name the index so it's clear when loading
    df.index.name = "sample_id"

    logger.info(f"Final matrix shape: {df.shape}")
    logger.info(f"Index (sample_ids): {len(df.index)} entries")

    # Save to parquet
    df.to_parquet(OUTPUT_PATH, engine="pyarrow", compression="snappy", index=True)
    logger.info(f"Successfully saved to: {OUTPUT_PATH}")

    # Verify saved file
    file_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    logger.info(f"Output file size: {file_size_mb:.2f} MB")

    logger.info("=" * 80)
    logger.info("ALL DONE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

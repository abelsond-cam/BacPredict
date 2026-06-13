"""Generate ESM-C embeddings for genome assemblies (GPU).

Processes pre-generated protein sequences through ESM-C (ESM++) and saves the
Bacformer-input tensors as `{sample_id}_esm_embeddings.pt`. This is the format
downstream fine-tuning consumes (`LabelInjectingFileDataset`).

Pass `--bacformer-embeddings` to additionally run those tensors through the
Bacformer model and save its contextualised outputs to
`{sample_id}_bacformer_embeddings.pt`. ESM-C-only is the default (~99 % of
downstream uses). Models are loaded once and reused across genomes.
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import torch
from bacformer.pp import (
    compute_genome_protein_embeddings,
    load_plm,
    protein_embeddings_to_inputs,
)
from tqdm import tqdm
from transformers import AutoModel

RDS_ROOT = Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw")
PROTEIN_SEQUENCES_DIR = RDS_ROOT / "david" / "processed" / "klebsiella_protein_sequences"
ESM_EMBEDDINGS_DIR = RDS_ROOT / "david" / "processed" / "klebsiella_esm_embeddings"
BACFORMER_EMBEDDINGS_DIR = RDS_ROOT / "david" / "processed" / "klebsiella_bacformer_embeddings"

# The refreshed Bacformer complete-genomes model (not the MAG-trained one).
BACFORMER_MODEL_ID = "macwiatrak/bacformer-large-masked-complete-genomes"


def load_bacformer_model(device: str, dtype="auto") -> torch.nn.Module:
    """Load the frozen Bacformer complete-genomes model, on ``device``, in eval mode.

    Single source of truth for loading Bacformer — reused by this script's
    embedding pipeline and by :mod:`snp_embeddings.frozen_bacformer_rpob_vectors`.
    ``dtype="auto"`` lets HF pick the checkpoint dtype (works on CPU for Stage-A
    smoke); pass ``torch.bfloat16`` to force the GPU pipeline's historical dtype.
    """
    model = AutoModel.from_pretrained(BACFORMER_MODEL_ID, trust_remote_code=True, torch_dtype=dtype)
    return model.to(device).eval()


def bacformer_last_hidden_state(model: torch.nn.Module, inputs: dict) -> torch.Tensor:
    """Run a frozen Bacformer forward and return its ``last_hidden_state`` tensor.

    ``inputs`` is the Bacformer-input bundle (``protein_embeddings``,
    ``special_tokens_mask``/``attention_mask``, ``token_type_ids`` …) already on
    the model's device.
    """
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
    return outputs["last_hidden_state"]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bacformer_embeddings_processing.log"),
    ],
)
logger = logging.getLogger(__name__)


def find_protein_sequence_files(input_dir: Path, limit: int | None = None) -> list[Path]:
    """Find all protein sequence parquet files in the input directory.

    Args:
        input_dir: Directory to search for protein sequence parquet files
        limit: Optional limit on number of files (takes first n if set)

    Returns
    -------
        List of paths to protein sequence parquet files
    """
    logger.info(f"Finding all *_protein_sequences.parquet files in {input_dir}")
    protein_files = sorted(input_dir.glob("*_protein_sequences.parquet"))

    logger.info(f"Found {len(protein_files)} total files")

    if limit:
        protein_files = protein_files[:limit]
        logger.info(f"Processing first {limit} files")

    return protein_files


def extract_sample_id(filepath: Path) -> str:
    """Extract sample ID from protein sequence parquet filepath.

    Example: /path/to/SAMD00052611_protein_sequences.parquet -> SAMD00052611

    Args:
        filepath: Path to the protein sequence parquet file

    Returns
    -------
        Sample ID string
    """
    # The sample ID is the filename without _protein_sequences.parquet suffix
    return filepath.stem.replace("_protein_sequences", "")


def check_embeddings_exist(sample_id: str, esm_dir: Path, bacformer_dir: Path | None) -> bool:
    """Check if embeddings already exist as .pt files.

    Args:
        sample_id: Sample ID to check
        esm_dir: Directory containing ESM embeddings
        bacformer_dir: Directory containing Bacformer embeddings, or None to
            skip the Bacformer-side check (ESM-only mode)

    Returns
    -------
        True if the ESM `.pt` exists (and, when `bacformer_dir` is provided, the
        Bacformer `.pt` also exists); False otherwise
    """
    if not (esm_dir / f"{sample_id}_esm_embeddings.pt").exists():
        return False
    if bacformer_dir is None:
        return True
    return (bacformer_dir / f"{sample_id}_bacformer_embeddings.pt").exists()


def load_protein_sequences(protein_seq_path: Path) -> tuple[str, list]:
    """Load protein sequences from parquet file.

    Args:
        protein_seq_path: Path to the protein sequences parquet file

    Returns
    -------
        Tuple of (sample_id, protein_sequences)
    """
    df = pd.read_parquet(protein_seq_path)
    sample_id = df["sample_id"].iloc[0]
    protein_sequences = df["protein_sequence"].iloc[0]
    return sample_id, protein_sequences


def process_genome_from_protein_sequences(
    protein_seq_path: Path,
    esm_model: torch.nn.Module,
    esm_tokenizer,
    bacformer_model: torch.nn.Module | None,
    device: str,
    esm_dir: Path,
    bacformer_dir: Path | None,
    batch_size: int = 128,
    max_proteins: int = 6000,
    max_contigs: int = 1000,
) -> tuple[str, bool, str]:
    """Process a single genome from pre-generated protein sequences.

    Always saves the ESM-C embeddings (Bacformer-input tensor format). If
    `bacformer_model` and `bacformer_dir` are both provided, also runs the
    Bacformer forward pass and saves its contextualised output.

    Args:
        protein_seq_path: Path to the protein sequences parquet file
        esm_model: Loaded ESM model for protein embeddings
        esm_tokenizer: ESM tokenizer
        bacformer_model: Loaded Bacformer model, or None to skip the Bacformer stage
        device: Device to run inference on
        esm_dir: Directory to save ESM embeddings
        bacformer_dir: Directory to save Bacformer embeddings, or None to skip
        batch_size: Batch size for ESM embedding computation
        max_proteins: Maximum number of proteins per genome
        max_contigs: Maximum number of contigs per genome

    Returns
    -------
        Tuple of (sample_id, success, error_message)
    """
    sample_id = extract_sample_id(protein_seq_path)

    try:
        # Load protein sequences from parquet
        logger.debug(f"Processing {sample_id}: Loading protein sequences")
        _, protein_sequences = load_protein_sequences(protein_seq_path)

        # Generate protein embeddings using ESM model (GPU)
        logger.debug(f"Processing {sample_id}: Generating protein embeddings with ESM")
        protein_embeddings = compute_genome_protein_embeddings(
            model=esm_model,
            tokenizer=esm_tokenizer,
            protein_sequences=protein_sequences,
            model_type="esmc",
            batch_size=batch_size,
            max_prot_seq_len=1024,
            genome_pooling_method=None,
        )

        # Pack into Bacformer-input tensors (also the schema downstream training expects)
        logger.debug(f"Processing {sample_id}: Converting to Bacformer inputs")
        bacformer_inputs = protein_embeddings_to_inputs(
            protein_embeddings=protein_embeddings,
            max_n_proteins=max_proteins,
            max_n_contigs=max_contigs,
            bacformer_model_type="large",
        )

        # Move inputs to device
        bacformer_inputs = {k: v.to(device) for k, v in bacformer_inputs.items()}

        # Save ESM embeddings (bacformer_inputs) as .pt file
        esm_output_path = esm_dir / f"{sample_id}_esm_embeddings.pt"
        logger.debug(f"Processing {sample_id}: Saving ESM embeddings to {esm_output_path}")
        esm_embeddings_cpu = {k: v.float().cpu() for k, v in bacformer_inputs.items()}
        torch.save(esm_embeddings_cpu, esm_output_path)

        # Optional Bacformer stage
        if bacformer_model is not None and bacformer_dir is not None:
            logger.debug(f"Processing {sample_id}: Generating Bacformer embeddings")
            last_hidden_state = bacformer_last_hidden_state(bacformer_model, bacformer_inputs)
            bacformer_output_path = bacformer_dir / f"{sample_id}_bacformer_embeddings.pt"
            logger.debug(f"Processing {sample_id}: Saving Bacformer embeddings to {bacformer_output_path}")
            bacformer_embeddings_cpu = last_hidden_state.float().cpu()
            torch.save(bacformer_embeddings_cpu, bacformer_output_path)

        return sample_id, True, ""

    except Exception as e:
        error_msg = f"Error processing {sample_id}: {str(e)}"
        logger.error(error_msg)
        return sample_id, False, error_msg


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate ESM-C embeddings for genome assemblies "
        "(opt-in Bacformer outputs via --bacformer-embeddings)."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Limit number of genomes to process (for testing)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for ESM embedding computation (default: 128)",
    )
    parser.add_argument(
        "--max-proteins",
        type=int,
        default=6000,
        help="Maximum number of proteins per genome (default: 6000)",
    )
    parser.add_argument(
        "--max-contigs",
        type=int,
        default=1000,
        help="Maximum number of contigs per genome (default: 1000)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip genomes that have already been processed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for inference (default: cuda:0)",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=None,
        help="Start index for processing files (for array jobs)",
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="End index for processing files (for array jobs)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROTEIN_SEQUENCES_DIR,
        help="Directory of *_protein_sequences.parquet files "
        "(default: klebsiella_protein_sequences)",
    )
    parser.add_argument(
        "--esm-dir",
        type=Path,
        default=ESM_EMBEDDINGS_DIR,
        help="Output directory for ESM embeddings (default: klebsiella_esm_embeddings)",
    )
    parser.add_argument(
        "--bacformer-dir",
        type=Path,
        default=BACFORMER_EMBEDDINGS_DIR,
        help="Output directory for Bacformer embeddings (only used with "
        "--bacformer-embeddings; default: klebsiella_bacformer_embeddings)",
    )
    parser.add_argument(
        "--bacformer-embeddings",
        action="store_true",
        help="Also generate Bacformer contextualised embeddings alongside the "
        "ESM-C embeddings (off by default).",
    )

    args = parser.parse_args()

    # Setup input directory path
    input_dir = args.input_dir
    logger.info(f"Input directory: {input_dir}")

    # Setup output directories
    esm_dir = args.esm_dir
    esm_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"ESM embeddings directory: {esm_dir}")

    bacformer_dir: Path | None = args.bacformer_dir if args.bacformer_embeddings else None
    if bacformer_dir is not None:
        bacformer_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Bacformer embeddings directory: {bacformer_dir}")
    else:
        logger.info("Bacformer stage disabled (pass --bacformer-embeddings to enable)")

    # Find all protein sequence parquet files (takes first n if --n is set)
    protein_files = find_protein_sequence_files(input_dir, limit=args.n)

    if not protein_files:
        logger.error(f"No protein sequence parquet files found in {input_dir}")
        logger.error("Please run generate_protein_sequences.py first to generate protein sequences")
        sys.exit(1)

    # Apply array job slicing BEFORE --skip-existing. Each task's slice indexes
    # the full sorted parquet list, so its responsibility set is fixed regardless
    # of when the task is scheduled or how many `.pt` files siblings have written
    # in the meantime. Slicing after --skip-existing introduced a race where
    # late-starting tasks indexed a shrunken list and left gaps no task covered.
    if args.start_idx is not None and args.end_idx is not None:
        total_files = len(protein_files)
        protein_files = protein_files[args.start_idx:args.end_idx]
        logger.info(f"Array job slice: processing files {args.start_idx} to {args.end_idx} of {total_files}")
        logger.info(f"Files in this slice: {len(protein_files)}")

    # Filter out files that already have embeddings if requested (within slice)
    if args.skip_existing:
        original_count = len(protein_files)
        protein_files = [
            f for f in protein_files
            if not check_embeddings_exist(extract_sample_id(f), esm_dir, bacformer_dir)
        ]
        skipped = original_count - len(protein_files)
        if skipped > 0:
            logger.info(f"Skipping {skipped} genomes that already have embeddings")

    if not protein_files:
        logger.info("Nothing to do in this slice (all already have embeddings)")
        return

    # Setup device
    device = args.device
    logger.info(f"Using device: {device}")

    # Load ESM model (protein language model) - ONCE
    logger.info("Loading ESM model (ESM++)...")
    esm_model, esm_tokenizer = load_plm(
        model_path="Synthyra/ESMplusplus_small",
        model_type="esmc",
    )
    logger.info("ESM model loaded")

    # Load Bacformer model - ONCE (only when --bacformer-embeddings is set)
    bacformer_model: torch.nn.Module | None = None
    if args.bacformer_embeddings:
        logger.info("Loading Bacformer model...")
        bacformer_model = load_bacformer_model(device, dtype=torch.bfloat16)
        logger.info("Bacformer model loaded")

    # Process genomes
    results = {"success": [], "failed": []}
    error_log = []
    genome_times = []

    # Record start time
    overall_start_time = time.time()
    logger.info("=" * 80)
    logger.info(f"STARTING PROCESSING: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total genomes to process: {len(protein_files)}")
    logger.info("=" * 80)

    for idx, protein_file in enumerate(tqdm(protein_files, desc="Processing genomes"), 1):
        genome_start = time.time()

        sample_id, success, error_msg = process_genome_from_protein_sequences(
            protein_file,
            esm_model,
            esm_tokenizer,
            bacformer_model,
            device,
            esm_dir,
            bacformer_dir,
            batch_size=args.batch_size,
            max_proteins=args.max_proteins,
            max_contigs=args.max_contigs,
        )

        genome_elapsed = time.time() - genome_start
        genome_times.append(genome_elapsed)

        if success:
            results["success"].append(sample_id)
            logger.info(f"[{idx}/{len(protein_files)}] {sample_id}: SUCCESS ({genome_elapsed:.1f}s)")
        else:
            results["failed"].append(sample_id)
            error_log.append({"sample_id": sample_id, "error": error_msg})
            logger.warning(f"[{idx}/{len(protein_files)}] {sample_id}: FAILED ({genome_elapsed:.1f}s)")

        # Log timing estimates every 10 genomes
        if idx % 10 == 0 and genome_times:
            avg_time = sum(genome_times) / len(genome_times)
            remaining = len(protein_files) - idx
            est_remaining = timedelta(seconds=int(avg_time * remaining))
            logger.info(f"Average time per genome: {avg_time:.1f}s, Estimated time remaining: {est_remaining}")

    # Calculate total elapsed time
    overall_elapsed = time.time() - overall_start_time

    # Summary
    logger.info("=" * 80)
    logger.info(f"PROCESSING COMPLETE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    logger.info(f"Total time elapsed: {timedelta(seconds=int(overall_elapsed))}")
    logger.info(f"Successfully processed: {len(results['success'])} genomes")
    logger.info(f"Failed: {len(results['failed'])} genomes")

    if genome_times:
        avg_time = sum(genome_times) / len(genome_times)
        min_time = min(genome_times)
        max_time = max(genome_times)
        logger.info(f"Timing statistics:")
        logger.info(f"  - Average: {avg_time:.1f}s per genome")
        logger.info(f"  - Min: {min_time:.1f}s")
        logger.info(f"  - Max: {max_time:.1f}s")
        logger.info(f"  - Total genomes processed: {len(genome_times)}")

    if error_log:
        error_log_path = Path("bacformer_embeddings_errors.log")
        with open(error_log_path, "w") as f:
            for entry in error_log:
                f.write(f"{entry['sample_id']}: {entry['error']}\n")
        logger.info(f"Error details saved to {error_log_path}")

    logger.info("=" * 80)


if __name__ == "__main__":
    main()

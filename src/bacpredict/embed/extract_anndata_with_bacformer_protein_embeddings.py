"""Extract AnnData objects from Bacformer protein embeddings.

This script loads Bacformer protein embeddings for a selected set of genomes
(filtered by clonal group and randomly sampled) and creates an AnnData object
with gene-level embeddings and metadata for downstream analysis.

Workflow:
1. Load metadata and filter by clonal group
2. Randomly sample n genomes from the filtered set
3. Load PyTorch embedding files for each selected genome
4. Explode embeddings from genome-level to gene-level
5. Merge with selected metadata columns
6. Create AnnData object with embeddings as X and metadata in obs
7. Save to .h5ad format for efficient loading in analysis notebooks

Memory Management:
- Intermediate objects are explicitly deleted after AnnData creation
- Garbage collection is forced to free memory immediately
- Suitable for running on compute nodes with memory constraints
"""

import argparse
import gc
import logging
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

RDS_ROOT = Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw")
METADATA_FILE = RDS_ROOT / "david" / "final" / "metadata_final_curated_slimmed.tsv"
BACFORMER_EMBEDDINGS_DIR = RDS_ROOT / "david" / "processed" / "klebsiella_bacformer_embeddings"
ANNDATA_DIR = RDS_ROOT / "david" / "processed" / "klebsiella_anndata"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("extract_anndata.log"),
    ],
)
logger = logging.getLogger(__name__)


def load_metadata(metadata_file: Path) -> pd.DataFrame:
    """Load metadata from TSV file.

    Args:
        metadata_file: Path to metadata TSV file

    Returns
    -------
        DataFrame with metadata
    """
    logger.info(f"Loading metadata from {metadata_file}")
    metadata = pd.read_csv(metadata_file, sep="\t", low_memory=False)
    logger.info(f"Metadata shape: {metadata.shape}")
    return metadata


def filter_and_sample_genomes(
    metadata: pd.DataFrame,
    clonal_group: str,
    n_samples: int,
    seed: int,
) -> list[str]:
    """Filter metadata by clonal group and randomly sample genomes.

    Args:
        metadata: Full metadata DataFrame
        clonal_group: Clonal group to filter by (e.g., 'CG258')
        n_samples: Number of samples to randomly select
        seed: Random seed for reproducibility

    Returns
    -------
        List of sample IDs
    """
    # Filter by clonal group
    logger.info(f"Filtering metadata by clonal group: {clonal_group}")
    filtered = metadata[metadata["Clonal group"] == clonal_group]
    logger.info(f"Found {len(filtered)} genomes in {clonal_group}")

    if len(filtered) == 0:
        raise ValueError(f"No genomes found for clonal group: {clonal_group}")

    if len(filtered) < n_samples:
        logger.warning(
            f"Requested {n_samples} samples but only {len(filtered)} available. "
            f"Using all {len(filtered)} samples."
        )
        n_samples = len(filtered)

    # Get sample IDs
    all_samples = filtered["Sample"].tolist()

    # Random sample
    logger.info(f"Randomly sampling {n_samples} genomes (seed={seed})")
    import random

    random.seed(seed)
    sampled_genomes = random.sample(all_samples, n_samples)

    logger.info(f"Selected samples (first 5): {sampled_genomes[:5]}")
    return sampled_genomes


def load_embeddings_for_samples(
    sample_ids: list[str],
    embeddings_dir: Path,
) -> pd.DataFrame:
    """Load Bacformer embeddings for selected samples.

    Args:
        sample_ids: List of sample IDs to load
        embeddings_dir: Directory containing embedding .pt files

    Returns
    -------
        DataFrame with embeddings indexed by sample ID
    """
    logger.info(f"Loading embeddings from {embeddings_dir}")
    sample_embeddings = pd.DataFrame()
    count_existing = 0
    missing_samples = []

    for sample in tqdm(sample_ids, desc="Loading embeddings"):
        sample_file = f"{sample}_bacformer_embeddings.pt"
        sample_file_path = embeddings_dir / sample_file

        if sample_file_path.exists():
            # Read the embeddings file (PyTorch format)
            embeddings = torch.load(sample_file_path)
            # Remove batch dimension
            embeddings = embeddings.squeeze(0)
            # Convert to pandas dataframe
            embeddings_df = pd.DataFrame(
                {"embeddings": [embeddings.numpy()]}, index=[sample]
            )
            # Append to collection
            sample_embeddings = pd.concat([sample_embeddings, embeddings_df])
            count_existing += 1
        else:
            logger.warning(f"Embeddings file for {sample} does not exist")
            missing_samples.append(sample)

    logger.info(f"Successfully loaded embeddings for {count_existing}/{len(sample_ids)} samples")
    if missing_samples:
        logger.warning(f"Missing embeddings for {len(missing_samples)} samples: {missing_samples[:5]}...")

    if count_existing == 0:
        raise ValueError("No embeddings found for any of the selected samples")

    return sample_embeddings


def explode_embeddings_to_gene_level(
    sample_embeddings: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Explode sample-level embeddings to gene-level embeddings.

    Each sample has a matrix of shape (n_genes, embedding_dim).
    This function stacks all samples into a single matrix and tracks
    which sample each gene belongs to.

    Args:
        sample_embeddings: DataFrame with 'embeddings' column containing
                          numpy arrays of shape (n_genes, embedding_dim)

    Returns
    -------
        Tuple of (embeddings_matrix, sample_ids_list)
        - embeddings_matrix: numpy array of shape (total_genes, embedding_dim)
        - sample_ids_list: list of sample IDs matching each row
    """
    logger.info("Exploding embeddings to gene-level")
    all_embeddings_list = []
    sample_ids_list = []

    for sample_id, row in sample_embeddings.iterrows():
        embedding_matrix = row["embeddings"]  # Shape: (n_genes, embedding_dim)
        all_embeddings_list.append(embedding_matrix)
        # Create sample_id array matching number of genes for this sample
        sample_ids_list.extend([sample_id] * len(embedding_matrix))

    # Stack all embeddings into single array
    embeddings_exploded = np.vstack(all_embeddings_list)

    logger.info(f"Exploded embeddings shape: {embeddings_exploded.shape}")
    logger.info(f"Total genes: {len(sample_ids_list)}")
    logger.info(f"Unique samples: {len(set(sample_ids_list))}")

    return embeddings_exploded, sample_ids_list


def create_anndata_with_metadata(
    embeddings: np.ndarray,
    sample_ids: list[str],
    metadata: pd.DataFrame,
    selected_samples: list[str],
    clonal_group: str,
    metadata_cols: list[str],
) -> ad.AnnData:
    """Create AnnData object with embeddings and metadata.

    Args:
        embeddings: Gene-level embeddings matrix
        sample_ids: List of sample IDs for each gene
        metadata: Full metadata DataFrame
        selected_samples: List of samples that were selected
        clonal_group: Clonal group that was filtered for
        metadata_cols: Metadata columns to include in obs

    Returns
    -------
        AnnData object with embeddings as X and metadata in obs
    """
    logger.info("Creating AnnData object")

    # Create obs DataFrame with sample_id
    obs_df = pd.DataFrame({"sample_id": sample_ids})

    # Filter metadata for the clonal group and set Sample as index
    cg_metadata = metadata.loc[metadata["Clonal group"] == clonal_group, :]
    sample_metadata = cg_metadata.set_index("Sample")

    # Join the selected metadata columns to obs_df
    obs_with_metadata = obs_df.merge(
        sample_metadata[metadata_cols],
        left_on="sample_id",
        right_index=True,
        how="left",
    )

    logger.info(f"obs_with_metadata shape: {obs_with_metadata.shape}")
    logger.info(f"obs_with_metadata columns: {obs_with_metadata.columns.tolist()}")

    # Create AnnData with gene-level embeddings and selected metadata
    adata = ad.AnnData(X=embeddings, obs=obs_with_metadata)

    # Store sample-level metadata only for selected samples in adata.uns for reference
    # Filter to only the samples that were actually loaded (in case some were missing)
    selected_sample_ids = obs_with_metadata["sample_id"].unique().tolist()
    sample_metadata_filtered = sample_metadata.loc[
        sample_metadata.index.isin(selected_sample_ids)
    ].copy()
    
    # Convert object columns to strings to avoid h5py serialization issues
    for col in sample_metadata_filtered.columns:
        if sample_metadata_filtered[col].dtype == "object":
            sample_metadata_filtered[col] = sample_metadata_filtered[col].astype(str)
    
    adata.uns["sample_metadata"] = sample_metadata_filtered

    logger.info(f"AnnData X shape: {adata.X.shape}")
    logger.info(f"AnnData obs shape: {adata.obs.shape}")
    logger.info(f"AnnData obs columns: {adata.obs.columns.tolist()}")
    logger.info(
        f"Sample metadata stored in adata.uns['sample_metadata']: {adata.uns['sample_metadata'].shape}"
    )

    # Check for missing metadata
    logger.info("Missing metadata check:")
    for col in metadata_cols:
        n_missing = adata.obs[col].isna().sum()
        logger.info(f"  {col}: {n_missing} missing values")

    return adata


def cleanup_memory(*objects):
    """Delete objects and force garbage collection.

    Args:
        *objects: Variable number of object names as strings
    """
    logger.info("Cleaning up memory")
    for obj in objects:
        if obj in globals():
            del globals()[obj]
    gc.collect()
    logger.info("Memory cleanup complete")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Extract AnnData from Bacformer protein embeddings"
    )
    parser.add_argument(
        "--clonal-group",
        type=str,
        required=True,
        help="Clonal group to filter by (e.g., CG258)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=20,
        help="Number of samples to randomly select (default: 20)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename for .h5ad file (default: auto-generated based on clonal group and n)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--metadata-cols",
        type=str,
        nargs="+",
        default=["Clonal group", "Sublineage", "K_locus", "K_type"],
        help="Metadata columns to include in obs (default: Clonal group, Sublineage, K_locus, K_type)",
    )

    args = parser.parse_args()

    # Setup paths
    metadata_file = METADATA_FILE
    embeddings_dir = BACFORMER_EMBEDDINGS_DIR
    output_dir = ANNDATA_DIR

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Generate output filename if not provided
    if args.output is None:
        # Sanitize clonal group name for filename
        cg_sanitized = args.clonal_group.replace(" ", "_").lower()
        output_filename = f"{cg_sanitized}_n{args.n_samples}_seed{args.seed}_bacformer_anndata.h5ad"
    else:
        output_filename = args.output

    output_path = output_dir / output_filename
    logger.info(f"Output file: {output_path}")

    # Check if output already exists
    if output_path.exists():
        logger.warning(f"Output file already exists: {output_path}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != "y":
            logger.info("Exiting without overwriting")
            sys.exit(0)

    # Load metadata
    metadata = load_metadata(metadata_file)

    # Filter and sample genomes
    selected_samples = filter_and_sample_genomes(
        metadata,
        args.clonal_group,
        args.n_samples,
        args.seed,
    )

    # Load embeddings for selected samples
    sample_embeddings = load_embeddings_for_samples(
        selected_samples,
        embeddings_dir,
    )

    # Explode to gene-level
    embeddings_exploded, sample_ids_list = explode_embeddings_to_gene_level(
        sample_embeddings
    )

    # Create AnnData object
    adata = create_anndata_with_metadata(
        embeddings_exploded,
        sample_ids_list,
        metadata,
        selected_samples,
        args.clonal_group,
        args.metadata_cols,
    )

    # Store processing parameters in uns
    adata.uns["processing_params"] = {
        "clonal_group": args.clonal_group,
        "n_samples_requested": args.n_samples,
        "n_samples_actual": len(selected_samples),
        "seed": args.seed,
        "metadata_cols": args.metadata_cols,
    }

    # Clean up memory before saving
    logger.info("Cleaning up intermediate objects")
    del sample_embeddings
    del embeddings_exploded
    del sample_ids_list
    gc.collect()
    logger.info("Memory cleanup complete")

    # Save to .h5ad format
    logger.info(f"Saving AnnData to {output_path}")
    adata.write_h5ad(output_path)
    logger.info(f"Successfully saved AnnData object")

    # Summary
    logger.info("=" * 80)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Clonal group: {args.clonal_group}")
    logger.info(f"Samples: {adata.uns['processing_params']['n_samples_actual']}")
    logger.info(f"Total genes (observations): {adata.n_obs}")
    logger.info(f"Embedding dimensions: {adata.n_vars}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"File size: {output_path.stat().st_size / 1024**2:.2f} MB")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

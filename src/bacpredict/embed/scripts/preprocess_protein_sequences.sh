#!/bin/bash
#SBATCH --job-name=protein_seqs_klebsiella
#SBATCH --output=protein_seqs_%A.out
#SBATCH --error=protein_seqs_%A.err
#SBATCH --time=00:20:00
#SBATCH --partition=icelake
#SBATCH --account=FLOTO-SL2-CPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=76
#SBATCH --mem=200G

# Script to run protein-sequence extraction on HPC with CPU parallelization.
# The python script reads a CSV of (Sample, assembly_file, gff_file) — generate it
# first with `find_missing_embeddings.py`.
#
# Usage:
#   sbatch slurm_scripts/preprocess_protein_sequences.sh --n 5                    # dry-run on 5 samples
#   sbatch slurm_scripts/preprocess_protein_sequences.sh --skip-existing          # resume
#   sbatch slurm_scripts/preprocess_protein_sequences.sh --input-csv /path/to/missing.csv

# Force Python unbuffered output for real-time logging
export PYTHONUNBUFFERED=1
# Use work directory for UV cache to avoid disk space issues in home directory
export UV_CACHE_DIR=/home/dca36/rds/hpc-work/.uv_cache

# Ensure uv is in PATH (adjust this path to where uv is installed)
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

# Change to project directory
cd /home/dca36/workspace/BacPredict

echo "=========================================="
echo "Protein Sequence Extraction"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: ${SLURM_MEM_PER_NODE}M"
echo "Start time: $(date)"
echo "Arguments: $@"
echo "=========================================="

echo "Using uv: $(which uv)"
# Run the Python script with all passed arguments ($@) passed as arguments to the script
uv run python src/bacpredict/embed/preprocess_assemblies_to_protein_sequences.py "$@"

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="

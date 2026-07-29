#!/bin/bash
#SBATCH --job-name=protein_seqs_klebsiella
#SBATCH --output=/rds/user/dca36/hpc-work/logs/%x-%j.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/%x-%j.out
#SBATCH --time=00:20:00
#SBATCH --partition=icelake-himem
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=76
#SBATCH --mem=200G
# CSD3/UoHPC variant (when it returns): --partition=icelake --account=FLOTO-SL2-CPU,
#   logs → protein_seqs_%A.out/.err (repo-relative).

# Script to run protein-sequence extraction on HPC with CPU parallelization.
# The python script reads a CSV of (Sample, sr_assembly_file, sr_gff_file) — generate it
# first with `find_missing_embeddings.py`.
#
# Usage:
#   sbatch slurm_scripts/preprocess_protein_sequences.sh --n 5                    # dry-run on 5 samples
#   sbatch slurm_scripts/preprocess_protein_sequences.sh --skip-existing          # resume
#   sbatch slurm_scripts/preprocess_protein_sequences.sh --input-csv /path/to/missing.csv

set -uo pipefail
# Data root + env — cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
: "${BACPREDICT_DATA_ROOT:="$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/bac_ast_prediction"}"
D="$BACPREDICT_DATA_ROOT"
PY="$HOME/workspace/BacPredict/.venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"

# Force Python unbuffered output for real-time logging
export PYTHONUNBUFFERED=1

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

# Run the Python script with all passed arguments ($@) passed as arguments to the script
"$PY" -m bacpredict.engine.embedding.preprocess_assemblies_to_protein_sequences "$@"

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="

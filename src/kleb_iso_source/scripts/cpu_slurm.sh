#!/bin/bash
#SBATCH --job-name=prep_isol_training_data
#SBATCH --output=prep_isol_training_data_%j.out     
#SBATCH --error=prep_isol_training_data_%j.err
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --time=01:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU  

cd /home/dca36/workspace/BacPredict

# Force Python unbuffered output for real-time logging
export PYTHONUNBUFFERED=1

python_script="src/kleb_iso_source/prepare_esmc_embeddings_and_labels_to_finetune_isolation_source.py"
isolation_sources="urine catheter"

echo "========================================================================"
echo "Starting to prepare training data for isolation source prediction"
echo "Isolation sources: $isolation_sources"
echo "Python script: $python_script"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "========================================================================"
echo ""

# ESM embeddings
# uv run python src/kleb_ast/prepare_esmc_embeddings_and_labels_to_finetune_amr.py --skip-existing
#uv run python src/kleb_ast/add_paths_gff_fna_to_metadata.py

uv run python $python_script --isolation-sources $isolation_sources

echo ""
echo "========================================================================"
echo "Processing complete!"
echo "========================================================================"

# Run with: sbatch cpu_slurm.sh
# Check on progress with: squeue -u dca36
# To cancel: scancel jobid


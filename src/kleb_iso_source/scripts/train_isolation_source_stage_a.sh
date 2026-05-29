#!/bin/bash
# Stage A smoke test for isolation-source fine-tuning (n=10).
# Runs on a GPU node — login-node CPU cannot host Bacformer-large + RDS embedding I/O at this scale.
# Identical to train_isolation_source.sh except: no array, no k-fold, short walltime, --n-samples 10.

#SBATCH --job-name=stage_a_blood_faeces
#SBATCH --output=stage_a_blood_faeces_%j.out
#SBATCH --error=stage_a_blood_faeces_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --open-mode=append

cd /home/dca36/workspace/BacPredict

python_script="src/kleb_iso_source/train_isolation_source.py"
isolation_sources="blood faeces"
processed_base_dir="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_iso_source"
cohort_dir="${processed_base_dir}/blood_faeces/all_samples"
sheet_path="${cohort_dir}/binary_blood_vs_faeces_with_split.csv"
embeddings_dir="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/klebsiella_esm_embeddings"
output_dir="${cohort_dir}/smoke"   # absolute → used verbatim (transient; keeps models/ clean)
model_name_or_path="macwiatrak/bacformer-large-masked-complete-genomes"

module purge
module load cuda/12.4
module load cudnn/8.9_cuda-12.4

export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=info

echo "========================================================================"
echo "Stage A smoke test — Bacformer fine-tune on isolation-source pair"
echo "Isolation sources: $isolation_sources"
echo "Sheet:             $sheet_path"
echo "Model:             $model_name_or_path"
echo "n_samples:         10 (forced smoke-test mode, 100 epochs train=val)"
echo "Job ID:            $SLURM_JOB_ID"
echo "Node:              $SLURMD_NODENAME, GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================================================"

uv run python "$python_script" \
  --isolation-sources $isolation_sources \
  --processed-base-dir "$processed_base_dir" \
  --sheet-path "$sheet_path" \
  --embeddings-dir "$embeddings_dir" \
  --output-dir "$output_dir" \
  --model-name-or-path "$model_name_or_path" \
  --n-samples 10 \
  --num-workers 4 \
  --batch-size 1 \
  --grad-accumulation-steps 8 \
  --logging-steps 1

echo ""
echo "Stage A finished — inspect .out / .err for the 100-epoch loss curve."

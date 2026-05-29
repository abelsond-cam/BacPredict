#!/bin/bash
# Stage C (pooled-threads cohort): full-data fine-tune for blood-vs-faeces on
# the 2:1 country cap WITHOUT thread-segregation (AMR/Surv/NA pooled). Recovers
# the ~4k samples lost to thread segregation while keeping the country control.
# Cohort: sampled_country_2_1_all/kpsc_human (~14.2k after KPSC+Sublineage filter).
# Identical to stage_c_stratified.sh except the cohort_dir.

#SBATCH --job-name=stage_c_pooled_blood_faeces
#SBATCH --output=stage_c_pooled_blood_faeces_%j.out
#SBATCH --error=stage_c_pooled_blood_faeces_%j.err
#SBATCH --time=36:00:00
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=250G
#SBATCH --open-mode=append

cd /home/dca36/workspace/BacPredict

python_script="src/kleb_iso_source/train_isolation_source.py"
isolation_sources="blood faeces"
processed_base_dir="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_iso_source"
cohort_dir="${processed_base_dir}/blood_faeces/sampled_country_2_1_all/kpsc_human"
sheet_path="${cohort_dir}/binary_blood_vs_faeces_with_split.csv"
embeddings_dir="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/klebsiella_esm_embeddings"
output_dir="${cohort_dir}/models"   # absolute → used verbatim
model_name_or_path="macwiatrak/bacformer-large-masked-complete-genomes"

# Single-split, single-seed (Stage C, not k-fold). train_val_eval column used directly.
seed=1

warmup_proportion=0.1
lr=0.00015
eval_steps=700              # cohort ~14.2k (train ~9.9k); batch=1 × grad_accum=8 → ~1240 steps/epoch,
                           # so eval_steps=700 ≈ every ~0.55 epoch (between strict ~0.5 and all-sample ~0.4)
max_steps=100000           # generous cap; early-stopping decides actual stop
early_stopping_patience=30

module purge
module load cuda/12.4
module load cudnn/8.9_cuda-12.4

export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=info

echo "========================================================================"
echo "Stage C (pooled-threads 2:1 country cohort) — isolation source (blood vs faeces)"
echo "Tokens:            $isolation_sources"
echo "Sheet:             $sheet_path"
echo "Model:             $model_name_or_path"
echo "Single fold/seed:  seed=$seed"
echo "Output:            ${output_dir}"
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
  --lr "$lr" \
  --warmup-proportion "$warmup_proportion" \
  --batch-size 1 \
  --grad-accumulation-steps 8 \
  --num-workers 15 \
  --eval-steps "$eval_steps" \
  --max-steps "$max_steps" \
  --early-stopping-patience "$early_stopping_patience" \
  --seed "$seed"
status=$?

echo ""
if [ "$status" -ne 0 ]; then
  echo "Stage C (pooled cohort) FAILED with exit code $status — inspect the .err log."
  exit "$status"
fi
echo "Stage C (pooled cohort) finished — checkpoint in ${output_dir}."
echo "Compare AUROC against the stratified (KPSC-clean) and mixed-species (0.752) baselines."

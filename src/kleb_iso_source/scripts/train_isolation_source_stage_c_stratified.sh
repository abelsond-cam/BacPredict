#!/bin/bash
# Stage C (strict country-controlled cohort): full-data fine-tune for blood-vs-faeces,
# single fold × single seed, on the max-2:1-per-country cohort.
# Confounding test vs the all-sample run: if the all-sample ~0.82 AUROC was mostly the
# model using country as a label shortcut, this strict cohort should collapse toward the
# 0.55–0.62 baseline. Identical to train_isolation_source_stage_c.sh except the cohort paths.

#SBATCH --job-name=stage_c_strat_blood_faeces
#SBATCH --output=stage_c_strat_blood_faeces_%j.out
#SBATCH --error=stage_c_strat_blood_faeces_%j.err
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
processed_base_dir="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_on_sr_mags"
sheet_path="${processed_base_dir}/training_blood_faeces/stratified_country_cohort/binary_blood_vs_faeces_with_split.csv"
embeddings_dir="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/klebsiella_esm_embeddings"
output_dir="stratified_country_cohort/stage_c_full"
model_name_or_path="macwiatrak/bacformer-large-masked-complete-genomes"

# Single-split, single-seed (Stage C, not k-fold). train_val_eval column used directly.
seed=1

warmup_proportion=0.1
lr=0.00015
eval_steps=500              # cohort ~10.5k (train ~7.4k); batch=1 × grad_accum=8 → ~920 steps/epoch,
                           # so eval_steps=500 ≈ every ~0.5 epoch (cohort is smaller than all-sample)
max_steps=100000           # generous cap; early-stopping decides actual stop
early_stopping_patience=30

module purge
module load cuda/12.4
module load cudnn/8.9_cuda-12.4

export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=info

echo "========================================================================"
echo "Stage C (strict 2:1 country cohort) — isolation source (blood vs faeces)"
echo "Tokens:            $isolation_sources"
echo "Sheet:             $sheet_path"
echo "Model:             $model_name_or_path"
echo "Single fold/seed:  seed=$seed"
echo "Output:            ${processed_base_dir}/training_blood_faeces/${output_dir}"
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
  # Propagate the real failure so SLURM marks the job FAILED instead of COMPLETED.
  # (Without this, the script's trailing echoes exit 0 and mask a dead training run.)
  echo "Stage C (strict cohort) FAILED with exit code $status — inspect the .err log."
  exit "$status"
fi
echo "Stage C (strict cohort) finished — checkpoint in ${processed_base_dir}/training_blood_faeces/${output_dir}."
echo "Compare AUROC against both the 0.55–0.62 baseline and the all-sample run."

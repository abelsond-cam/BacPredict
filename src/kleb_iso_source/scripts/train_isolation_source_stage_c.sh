#!/bin/bash
# Stage C: full-data fine-tune for blood-vs-faeces, single fold × single seed.
# Per src/kleb_iso_source/CLAUDE.md milestone 4 and root §0.2: "1 × 1, GPU HPC SLURM, ~36 h".
# K-fold × multi-seed is reserved for publication; do not enable it here.

#SBATCH --job-name=stage_c_blood_faeces
#SBATCH --output=stage_c_blood_faeces_%j.out
#SBATCH --error=stage_c_blood_faeces_%j.err
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
cohort_dir="${processed_base_dir}/blood_faeces/all_samples/kpsc_human"
sheet_path="${cohort_dir}/binary_blood_vs_faeces_with_split.csv"
embeddings_dir="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/klebsiella_esm_embeddings"
output_dir="${cohort_dir}/models"   # absolute → used verbatim
model_name_or_path="macwiatrak/bacformer-large-masked-complete-genomes"

# Single-split, single-seed (Stage C, not k-fold). The train_val_eval column in the CSV
# is used directly; no --n-folds. Seed only governs trainer RNG (data shuffling, dropout, etc.).
seed=1

warmup_proportion=0.1
lr=0.00015
eval_steps=1000             # train set ~18k samples; with batch=1 × grad_accum=8 → ~2.3k optimizer
                            # steps per epoch, so eval_steps=1000 ≈ every ~0.4 epoch
max_steps=100000            # generous cap; early-stopping decides actual stop
# SUPERSEDED by train_isolation_source_cohort.sh (this copy also hardcodes output_dir).
# Patience was 30: the 2026-08-11 bf16 runs peaked at steps 31,500/30,500/15,000 and were still
# 9/15/23 non-improving evals short of firing when the 36 h wall killed them. See the cohort script.
early_stopping_patience=8

module purge
module load cuda/12.4
module load cudnn/8.9_cuda-12.4

export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=info

echo "========================================================================"
echo "Stage C full-data fine-tune — isolation source (blood vs faeces)"
echo "Tokens:            $isolation_sources"
echo "Sheet:             $sheet_path"
echo "Model:             $model_name_or_path"
echo "Single fold/seed:  seed=$seed (k-fold is parked until publication, per root §0.2)"
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

echo ""
echo "Stage C finished — checkpoint in ${output_dir}."
echo "Compare AUROC against the 0.55–0.62 baseline noted in src/kleb_iso_source/CLAUDE.md."

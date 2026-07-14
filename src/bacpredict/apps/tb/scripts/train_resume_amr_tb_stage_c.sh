#!/bin/bash
#SBATCH --job-name=tb_resume_stagec
#SBATCH --output=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
#SBATCH --error=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
#SBATCH --time=36:00:00
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=250G
#SBATCH --open-mode=append
# CSD3/UoHPC variant (when it returns): --partition=ampere --account=FLOTO-SL2-GPU,
#   logs → relative or ~/rds/hpc-work/logs/, and `module load cuda/12.4 cudnn/8.9_cuda-12.4`.

# Resume Stage C training from the most recent saved checkpoint of a previous
# (TIMEOUT'd) run. Preserves optimizer state, LR schedule, global step counter,
# best-metric tracking, and the early-stopping counter via HF Trainer's resume
# mechanism — so this is a true continuation, not a warm start.
#
# Drug is $1 (default rifampin). The existing run dir is discovered by glob —
# uses the most recent jobid for that drug (lexicographic tail). Inside, the
# latest checkpoint-N (highest N) is picked as the resume point.
#
# Submit (after the eval-panel job has snapshotted the current best):
#   sbatch --dependency=afterany:<eval_panel_jobid> \
#     --job-name=tb_<drug>_resume_stagec \
#     src/bacpredict/apps/tb/scripts/train_resume_amr_tb_stage_c.sh <drug>

set -uo pipefail

# Data root — one env var, cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
: "${BACPREDICT_DATA_ROOT:="$SCRATCHDIR"}"
D="$BACPREDICT_DATA_ROOT"
PY="$SCRATCHDIR/envs/bacpredict-gpu-venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"

drug=${1:-rifampin}
species=mycobacterium_tuberculosis
warmup_proportion=0.1
lr=0.00015
eval_steps=250
model_name_or_path="macwiatrak/bacformer-large-masked-complete-genomes"

BASE=$D/processed/train_tb_ast
CKPT_ROOT=$BASE/checkpoints

# Find the most recent run dir for this drug, and the latest checkpoint inside it.
RUN_DIR=$(ls -d "$CKPT_ROOT"/${species}_${drug}_stage_c_* 2>/dev/null | tail -n 1)
if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
    echo "ERROR: no run dir found matching ${species}_${drug}_stage_c_*" >&2
    exit 1
fi
LATEST=$(ls -d "$RUN_DIR"/checkpoint-* 2>/dev/null \
    | awk -F'checkpoint-' '{print $NF, $0}' \
    | sort -k1,1n \
    | tail -n 1 \
    | cut -d' ' -f2-)
if [ -z "$LATEST" ] || [ ! -d "$LATEST" ]; then
    echo "ERROR: no checkpoint-* dir found under $RUN_DIR" >&2
    exit 1
fi

export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=info

echo "TB AMR Stage C RESUME (Bacformer finetuning)"
echo "drug:         $drug"
echo "run dir:      $RUN_DIR"
echo "resume from:  $LATEST"
echo "Job ID:       $SLURM_JOB_ID  Node: $SLURMD_NODENAME  GPU: $CUDA_VISIBLE_DEVICES"

"$PY" -m bacpredict.engine.finetune.finetune_amr --task tb_ast \
    --embeddings-dir $BASE/esm \
    --ast-sheet-path $BASE/binary_ast_with_split.csv \
    --lr $lr \
    --model-name-or-path $model_name_or_path \
    --warmup-proportion $warmup_proportion \
    --drug ${drug} \
    --num-workers 15 \
    --grad-accumulation-steps 8 \
    --batch-size 1 \
    --eval-steps $eval_steps \
    --max-steps 100000 \
    --early-stopping-patience 30 \
    --output-dir "$RUN_DIR" \
    --resume-from-checkpoint "$LATEST"

echo "End of script — check the .out / .err logs."

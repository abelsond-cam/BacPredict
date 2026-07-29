#!/bin/bash
#SBATCH --job-name=embeddings_array_tb
#SBATCH --output=/rds/user/dca36/hpc-work/logs/%x-%A_%a.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/%x-%A_%a.out
#SBATCH --time=04:00:00
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --array=0-63
# CSD3/UoHPC variant (when it returns): --partition=ampere --account=FLOTO-SL2-GPU,
#   logs → relative or ~/rds/hpc-work/logs/, and `module load cuda/12.4 cudnn/8.9_cuda-12.4`.

# TB ESM-C embedding generation (GPU array job).
#
# Mirrors run_embeddings_array.sh but points at the TB processed dirs via the
# --input-dir/--esm-dir CLI args (organism-agnostic; Klebsiella behaviour is
# unchanged because those args default to the klebsiella_* constants when
# omitted). ESM-C only by default; add --bacformer-embeddings (and
# --bacformer-dir "${TB_BACFORMER_DIR}") to also produce Bacformer outputs.
#
# Sizing vs the 36h GPU per-job wall-clock limit: ~38k genomes * ~15s ~= 158
# GPU-h (ESM+Bacformer; ESM-only is a touch faster). 64 array tasks -> ~600
# genomes/task -> ~2.5h compute; --time=04:00:00 leaves large headroom.
# --skip-existing filters before index slicing so a maintenance kill/requeue
# resumes for free.
#
# Usage:
#   sbatch src/bacpredict/apps/tb/scripts/run_embeddings_array_tb.sh
#   sbatch --array=0-127 --time=02:00:00 src/bacpredict/apps/tb/scripts/run_embeddings_array_tb.sh

set -uo pipefail

# Data root — one env var, cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
: "${BACPREDICT_DATA_ROOT:="$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/bac_ast_prediction"}"
D="$BACPREDICT_DATA_ROOT"
PY="$HOME/workspace/BacPredict/.venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"

# Caches (HF_HOME/UV_CACHE_DIR/TMPDIR/...) come from ~/.bashrc on Isambard — do not hardcode here.
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=info

TB_INPUT_DIR="$D/processed/train_tb_ast/protein_sequences"
TB_ESM_DIR="$D/processed/train_tb_ast/esm"
# TB_BACFORMER_DIR="$D/processed/train_tb_ast/bacformer"  # uncomment + pass --bacformer-embeddings if needed

echo "=========================================="
echo "TB Embedding Generation (Array Job; ESM-C default)"
echo "Job ID: $SLURM_JOB_ID  Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME  GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=========================================="

# Chunk by the TOTAL parquet count (stable across tasks), not the unprocessed
# count. The unprocessed count is racy — late-starting tasks would see a
# shrunken list and slice into the wrong index space, leaving gaps. Each task
# now slices [start..end) of the full sorted parquet list; `--skip-existing`
# inside the Python script then filters within the slice.
read TOTAL_FILES UNPROCESSED <<< "$("$PY" -c "
from pathlib import Path
inp = Path('${TB_INPUT_DIR}')
esm = Path('${TB_ESM_DIR}')
files = sorted(inp.glob('*_protein_sequences.parquet'))
todo = [
    f for f in files
    if not (esm / f'{f.stem.replace(\"_protein_sequences\", \"\")}_esm_embeddings.pt').exists()
]
print(len(files), len(todo))
")"
echo "Total parquet files: $TOTAL_FILES  (unprocessed: $UNPROCESSED)"

# Derive chunk size from the array task count (set by Slurm, overridable via
# sbatch --array=...); fall back to 64 if unset.
NTASKS=${SLURM_ARRAY_TASK_COUNT:-64}
CHUNK_SIZE=$((TOTAL_FILES / NTASKS + 1))
START_IDX=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
END_IDX=$(((SLURM_ARRAY_TASK_ID + 1) * CHUNK_SIZE))
if [ $END_IDX -gt $TOTAL_FILES ]; then END_IDX=$TOTAL_FILES; fi
echo "Tasks: $NTASKS  Chunk: $CHUNK_SIZE  Indices: $START_IDX..$END_IDX"
echo "=========================================="

"$PY" -m bacpredict.engine.embedding.generate_embeddings \
    --input-dir "$TB_INPUT_DIR" \
    --esm-dir "$TB_ESM_DIR" \
    --skip-existing \
    --start-idx $START_IDX \
    --end-idx $END_IDX

echo "=========================================="
echo "End time: $(date)  Array Task $SLURM_ARRAY_TASK_ID completed"
echo "=========================================="

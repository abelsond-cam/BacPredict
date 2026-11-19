#!/usr/bin/env bash
# SLURM GPU array: ESM-C embeddings for one slice of a protein-sequence directory (generate_embeddings).
#
# Organism-agnostic and driven by --export; the older run_embeddings_array*.sh are per-organism copies.
#   INPUT_DIR     directory of <Sample>_protein_sequences.parquet
#   ESM_DIR       output directory for <Sample>_esm_embeddings.pt
#   MAX_PROTEINS  per-genome protein cap. No default on purpose: generate_embeddings' own default (6,000)
#                 silently truncates a P. aeruginosa genome (~6,400 CDS); the trainer's cap is 9,000.
#   SAVE_DTYPE    float32 | bfloat16. No default on purpose: a store must not mix dtypes, so a top-up of an
#                 existing store has to name the dtype that store already holds.
#   NTASKS        the FULL array size the slices are cut for (16 for --array=0-15). Explicit, not
#                 SLURM_ARRAY_TASK_COUNT: re-running two failed tasks with --array=3,7 must redo slices 3 and 7
#                 of 16, not cut the list in two and leave both tasks an empty range.
#   N             optional: only the first N files (sorted) — the smoke test
#
# Each task takes [start, end) of the FULL sorted parquet list, and --skip-existing then filters within that
# slice, so a requeued task resumes. The parquet directory must be complete before submission (the drivers
# check it), or tasks would count different totals and their slices would not tile. Embeddings are written
# atomically, so an existing file is a whole one. Per-genome failures do not fail the task: check completeness
# against the cohort afterwards, never by exit code.
#
#   sbatch --array=0-15 --export=ALL,INPUT_DIR=$T/protein_sequences,ESM_DIR=$T/esm,MAX_PROTEINS=9000,\
#   SAVE_DTYPE=bfloat16,NTASKS=16 src/bacpredict/engine/embedding/scripts/esm_embeddings_array.sh
#SBATCH --job-name=esm-embeddings
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/rds/user/dca36/hpc-work/logs/%x_%A_%a.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/%x_%A_%a.err
set -euo pipefail

: "${INPUT_DIR:?}" "${ESM_DIR:?}" "${MAX_PROTEINS:?}" "${SAVE_DTYPE:?}" "${NTASKS:?full array size}"
REPO=${REPO:-$HOME/workspace/BacPredict}
export PYTHONUNBUFFERED=1

TOTAL=$(find "$INPUT_DIR" -maxdepth 1 -name '*_protein_sequences.parquet' | wc -l)
if [[ -n "${N:-}" && "$N" -lt "$TOTAL" ]]; then TOTAL=$N; fi
NT=$NTASKS
TASK=${SLURM_ARRAY_TASK_ID:-0}
(( TASK < NT )) || { echo "task $TASK is outside the NTASKS=$NT slices"; exit 1; }
CHUNK=$(( (TOTAL + NT - 1) / NT ))
START=$(( TASK * CHUNK ))
END=$(( START + CHUNK < TOTAL ? START + CHUNK : TOTAL ))

# generate_embeddings writes its error log to the working directory: keep it out of the repo and the store
RUN_DIR=/rds/user/dca36/hpc-work/logs/esm_embeddings_${SLURM_ARRAY_JOB_ID:-local}_${TASK}
mkdir -p "$RUN_DIR" "$ESM_DIR"
cd "$RUN_DIR"
echo "[$(date -Is)] host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none} task=$TASK/$NT files=[$START,$END) of $TOTAL" \
     "max_proteins=$MAX_PROTEINS dtype=$SAVE_DTYPE repo=$REPO@$(git -C "$REPO" rev-parse --short HEAD)"
if (( START >= END )); then echo "empty slice"; exit 0; fi

uv run --project "$REPO" --no-sync python -m bacpredict.engine.embedding.generate_embeddings \
    --input-dir "$INPUT_DIR" --esm-dir "$ESM_DIR" --max-proteins "$MAX_PROTEINS" --save-dtype "$SAVE_DTYPE" \
    --skip-existing --start-idx "$START" --end-idx "$END" ${N:+--n "$N"}
echo "[$(date -Is)] done"

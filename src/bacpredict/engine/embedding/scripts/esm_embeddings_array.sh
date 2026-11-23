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
#   ESM_DTYPE     optional: bfloat16 (default, every store so far) | float32 — the dtype ESM-C runs in. float32 is
#                 the precision check; give it its own ESM_DIR and SAVE_DTYPE=float32
#   N             optional: only the first N files (sorted) — the smoke test
#   SAMPLES       optional, an absolute path: embed these samples only (one per non-blank line, sorted and unique;
#                 engine/embedding/sample_list.py). The work list is then <INPUT_DIR>/<Sample>_protein_sequences.parquet
#                 in list order, not the directory's listing: the protein store is flat and shared across species,
#                 so a listing would embed every species at once and shift its slices whenever any species wrote a
#                 parquet mid-array. A task exits before loading the model if a parquet of its slice is missing.
#   SAMPLES_MD5   optional: refuse to run (exit 3) if SAMPLES' md5 is not this — the list a pending task would
#                 slice must be the one its siblings sliced. Also passed on as generate_embeddings --samples-md5,
#                 which checks it against the very bytes it parses (the list could change between the two reads)
#   REPO          the checkout to run (default ~/workspace/BacPredict). For a long array, a detached worktree at
#                 the submitting SHA with .venv symlinked to the shared env; its src/ goes first on PYTHONPATH.
#   EXPECT_SHA    optional: refuse to run (exit 3) if REPO's HEAD is not it
#
# Each task takes [start, end) of the FULL sorted parquet list (or of SAMPLES), and --skip-existing then filters
# within that slice, so a requeued task resumes. Without SAMPLES the parquet directory must be complete before
# submission (the drivers check it), or tasks would count different totals and their slices would not tile.
# Embeddings are written atomically, so an existing file is a whole one. Per-genome failures do not fail the task:
# check completeness against the cohort afterwards, never by exit code. Never run two arrays into one ESM_DIR over
# the same samples at once: their writers share a sample's temporary .part name.
#
#   sbatch --array=0-15 --export=ALL,INPUT_DIR=$T/protein_sequences,ESM_DIR=$T/esm,MAX_PROTEINS=9000,\
#   SAVE_DTYPE=bfloat16,NTASKS=16 src/bacpredict/engine/embedding/scripts/esm_embeddings_array.sh
#   (add SAMPLES=$T/esm_samples.txt,SAMPLES_MD5=$(md5sum <$T/esm_samples.txt | cut -d' ' -f1) for a shared store)
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

# The job inherits the submitting shell's environment (sbatch --export=ALL). A cclake login node (login-p-*) loads
# spack modules whose LD_LIBRARY_PATH points at cascadelake-built libraries; on ampere's AMD CPUs `import torch` then
# dies with "Fatal Python error: Illegal instruction" and exit 1, printing nothing unless -X faulthandler is on
# (measured 2026-11-20, jobs 41520932 and 41525227; a clean environment imports torch 2.10.0+cu128 fine). The uv env
# is self-contained (torch ships its own CUDA libraries), so drop the loader paths.
unset LD_LIBRARY_PATH LIBRARY_PATH LD_PRELOAD

: "${INPUT_DIR:?}" "${ESM_DIR:?}" "${MAX_PROTEINS:?}" "${SAVE_DTYPE:?}" "${NTASKS:?full array size}"
REPO=${REPO:-$HOME/workspace/BacPredict}
export PYTHONUNBUFFERED=1
# $REPO's own code, even when $REPO is a worktree whose .venv is the shared env (its editable install points at the
# main checkout): PYTHONPATH precedes site-packages
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
head=$(git -C "$REPO" rev-parse HEAD)
if [[ -n "${EXPECT_SHA:-}" && "$head" != "$EXPECT_SHA"* ]]; then
    echo "the checkout moved: HEAD ${head:0:7}, this array was submitted at $EXPECT_SHA"; exit 3
fi

if [[ -n "${SAMPLES:-}" ]]; then
    [[ $SAMPLES == /* && -r $SAMPLES ]] || { echo "SAMPLES must be a readable absolute path: $SAMPLES"; exit 4; }
    SAMPLES_SUM=$(md5sum < "$SAMPLES" | cut -d' ' -f1)
    if [[ -n "${SAMPLES_MD5:-}" && "$SAMPLES_SUM" != "$SAMPLES_MD5" ]]; then
        echo "SAMPLES changed since submit: md5 $SAMPLES_SUM, submitted with $SAMPLES_MD5 ($SAMPLES)"; exit 3
    fi
    # awk, not grep -c (which exits 1 on a count of 0): the non-blank lines, exactly as sample_list.py reads them
    TOTAL=$(awk '{ gsub(/[[:space:]]/, "") } length($0) { n++ } END { print n + 0 }' "$SAMPLES")
elif [[ -n "${SAMPLES_MD5:-}" ]]; then
    echo "SAMPLES_MD5 is set but SAMPLES is not"; exit 4
else
    TOTAL=$(find "$INPUT_DIR" -maxdepth 1 -name '*_protein_sequences.parquet' | wc -l)
fi
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
     "max_proteins=$MAX_PROTEINS dtype=$SAVE_DTYPE esm_dtype=${ESM_DTYPE:-bfloat16} repo=$REPO@${head:0:7}" \
     "${SAMPLES:+samples=$SAMPLES md5=$SAMPLES_SUM}"
if (( START >= END )); then echo "empty slice"; exit 0; fi

uv run --project "$REPO" --no-sync python -X faulthandler -m bacpredict.engine.embedding.generate_embeddings \
    --input-dir "$INPUT_DIR" --esm-dir "$ESM_DIR" --max-proteins "$MAX_PROTEINS" --save-dtype "$SAVE_DTYPE" \
    --esm-dtype "${ESM_DTYPE:-bfloat16}" \
    --skip-existing --start-idx "$START" --end-idx "$END" ${N:+--n "$N"} ${SAMPLES:+--samples-file "$SAMPLES"} \
    ${SAMPLES_MD5:+--samples-md5 "$SAMPLES_MD5"}
echo "[$(date -Is)] done"

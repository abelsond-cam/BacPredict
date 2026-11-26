#!/usr/bin/env bash
# SLURM GPU array: score one named split of one drug's split table with an already-trained checkpoint, keeping
# the LOGITS (engine/finetune/score_split). One array task per drug.
#
# Why it exists: a finished run writes per-sample scores for its holdouts and nothing for `validate`, which is
# the split a post-hoc calibration must be fitted on. Without this, an arm trained before validate scores were
# saved can only be compared untempered against one that was tempered, which credits the estimator with the
# scaling rather than measuring it.
#
#   DRUGS            colon- or comma-separated; array index i scores DRUGS[i]. Use colons inside --export,
#                    which itself splits on commas
#   SPLIT_DIR        directory of <drug>_split.csv (engine/splits)
#   CHECKPOINT_ROOT  the trained arm: task i reads <CHECKPOINT_ROOT>/<drug> (a run dir; the resolver finds its
#                    best checkpoint-<step> itself)
#   EMBEDDINGS_DIR   directory of <Sample>_esm_embeddings.pt
#   OUT_DIR          <OUT_DIR>/<drug>_validate_scores.npz. NEVER inside the scored run: this adds a file to
#                    nobody else's results directory
#   SPLIT            which split to score (default validate)
#   SPECIES          optional: restrict a cross-species table to one species' rows
#   REPO / EXPECT_SHA / LOG_ROOT   as engine/finetune/scripts/finetune_amr_array.sh
#
# Inference only — no gradients, no optimizer, nothing written into the scored run. A ~1,200-genome drug takes
# a few minutes, so the wall is tight on purpose: a short GPU job's only lever on a congested ampere queue is
# its wall (the cores and memory of a 1-GPU job are normalised to ~32 / ~250 GB whatever is requested).
#SBATCH --job-name=score-split
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --time=00:40:00
#SBATCH --output=/rds/user/dca36/hpc-work/logs/%x_%A_%a.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/%x_%A_%a.err
set -euo pipefail

# a cclake login node's spack LD_LIBRARY_PATH kills `import torch` on ampere's AMD CPUs (finetune_amr_array.sh)
unset LD_LIBRARY_PATH LIBRARY_PATH LD_PRELOAD
if [[ -d /local && -w /local ]]; then export TMPDIR=/local; fi

: "${DRUGS:?one per array index}" "${SPLIT_DIR:?}" "${CHECKPOINT_ROOT:?}" "${EMBEDDINGS_DIR:?}" "${OUT_DIR:?}"
REPO=${REPO:-$HOME/workspace/BacPredict}
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
head=$(git -C "$REPO" rev-parse HEAD)
if [[ -n "${EXPECT_SHA:-}" && "$head" != "$EXPECT_SHA"* ]]; then
    echo "the checkout moved: HEAD ${head:0:7}, this array was submitted at $EXPECT_SHA"; exit 3
fi
if [[ -n "${EXPECT_SHA:-}" ]]; then
    dirty=$(git -C "$REPO" status --porcelain --untracked-files=no)
    [[ -z "$dirty" ]] || { echo "$REPO is at $EXPECT_SHA but modified, so what would run is at no SHA:"; \
                           echo "$dirty" | head -20; exit 3; }
fi

IFS=,: read -r -a drugs <<<"$DRUGS"
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
(( TASK_ID < ${#drugs[@]} )) || { echo "array index $TASK_ID has no drug in DRUGS=$DRUGS"; exit 1; }
drug=${drugs[$TASK_ID]}
split=${SPLIT:-validate}
table=$SPLIT_DIR/${drug}_split.csv
run=$CHECKPOINT_ROOT/$drug
out=$OUT_DIR/${drug}_${split}_scores.npz
[[ -s $table ]] || { echo "no split table $table"; exit 4; }
[[ -d $run ]] || { echo "no run dir $run"; exit 4; }
[[ ! -e $out ]] || { echo "$out exists: move it aside first"; exit 5; }

RUN_DIR=${LOG_ROOT:-/rds/user/dca36/hpc-work/logs}/score_split_${SLURM_ARRAY_JOB_ID:-local}_${TASK_ID}
mkdir -p "$RUN_DIR" "$OUT_DIR"
cd "$RUN_DIR"
echo "[$(date -Is)] host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none} drug=$drug split=$split run=$run" \
     "table_md5=$(md5sum < "$table" | cut -d' ' -f1) out=$out repo=$REPO@${head:0:7}"
nvidia-smi -L || true

uv run --project "$REPO" --no-sync python -X faulthandler -m bacpredict.engine.finetune.score_split \
    --checkpoint "$run" --split-table "$table" --split "$split" --embeddings-dir "$EMBEDDINGS_DIR" \
    --num-workers "${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}" ${SPECIES:+--species "$SPECIES"} --out "$out"
echo "[$(date -Is)] done"

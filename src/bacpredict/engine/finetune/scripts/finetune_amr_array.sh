#!/usr/bin/env bash
# SLURM GPU array: one Bacformer AMR fine-tune per array task, from a split table (finetune_amr --split-table).
#
# Organism-agnostic and driven by --export (the PsA driver is apps/pa/scripts/train_pa_models.sh). Task i trains the
# i-th drug of DRUGS on <SPLIT_DIR>/<drug>_split.csv and writes <OUT_ROOT>/<drug>/: checkpoints, results.json (the
# holdout and every extra holdout), eval_scores.npz and eval_scores_<split>.npz. One model per drug.
#   TASK            the --task label recorded in results.json: pa_ast, or a cross-species label such as pa_entero_ast
#   DRUGS           colon- or comma-separated; array index i trains DRUGS[i]. Use colons inside sbatch --export,
#                   which itself splits on commas
#   SPLIT_DIR       directory of <drug>_split.csv (engine/splits/build_amr_holdout_splits) and split_manifest.json
#   SPLIT_MANIFEST_MD5  optional: refuse to run (exit 3) if split_manifest.json's md5 is not this. The tables a pending
#                   task reads must be the ones its siblings read: a rebuilt split would make the drugs incomparable
#   EMBEDDINGS_DIR  directory of <Sample>_esm_embeddings.pt (bf16 and fp32 files may mix; the dataset upcasts)
#   OUT_ROOT        models go to <OUT_ROOT>/<drug>. A task refuses (exit 5) if <OUT_ROOT>/<drug>/results.json exists
#   N_SAMPLES       optional: 10 is the Stage A/B smoke (a balanced 10 genomes, train = validate = holdout)
#   EXTRA_HOLDOUTS  splits scored beside the holdout (default holdout_st; set it empty for the holdout only)
#   MODEL LR EVAL_STEPS MAX_EPOCHS PATIENCE THRESHOLD SEED   optional; defaults are the deployed TB/Kp settings
#   NUM_WORKERS     DataLoader workers (default: the allocation, SLURM_CPUS_PER_TASK)
#   REPO            the checkout to run: a detached worktree at the submitting SHA, .venv symlinked to the shared
#                   env (default ~/workspace/BacPredict); its src/ goes first on PYTHONPATH
#   EXPECT_SHA      optional: refuse to run (exit 3) if REPO's HEAD is not it
#   LOG_ROOT        the per-task working directory's parent (default /rds/user/dca36/hpc-work/logs)
#
# Precision is bf16 master weights (CLAUDE.md §0.2; never dtype="auto"). The wall is 35 h: a ~2,500-genome drug runs
# ~16 min per epoch at ~3 s per step (batch 1 × grad-accum 8), so the 40-epoch cap is ~11 h of training plus the
# validation passes; the QOS gpu1 cap of FLOTO-SL2-GPU is 36 h. A killed run resumes with
# --resume-from-checkpoint <OUT_ROOT>/<drug>/checkpoint-<step> (finetune_amr), not by resubmitting this array.
#
#   sbatch --array=0-9 --export=ALL,TASK=pa_ast,DRUGS=ciprofloxacin:ceftazidime:...,SPLIT_DIR=$P/splits,EMBEDDINGS_DIR=$AMR/processed/esm,\
#   OUT_ROOT=$P/models,REPO=$WT,EXPECT_SHA=$(git -C $WT rev-parse HEAD) src/bacpredict/engine/finetune/scripts/finetune_amr_array.sh
#SBATCH --job-name=finetune-amr
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --time=35:00:00
#SBATCH --output=/rds/user/dca36/hpc-work/logs/%x_%A_%a.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/%x_%A_%a.err
set -euo pipefail

# The job inherits the submitting shell's environment (sbatch --export=ALL). A cclake login node loads spack modules
# whose LD_LIBRARY_PATH points at cascadelake-built libraries; on ampere's AMD CPUs `import torch` then dies with
# "Fatal Python error: Illegal instruction", exit 1, and nothing on stderr unless -X faulthandler is on (jobs 41520932,
# 41525227; engine/embedding/scripts/esm_embeddings_array.sh). The uv env is self-contained: drop the loader paths.
unset LD_LIBRARY_PATH LIBRARY_PATH LD_PRELOAD

: "${TASK:?}" "${DRUGS:?one per array index}" "${SPLIT_DIR:?}" "${EMBEDDINGS_DIR:?}" "${OUT_ROOT:?}"
REPO=${REPO:-$HOME/workspace/BacPredict}
export PYTHONUNBUFFERED=1 TRANSFORMERS_VERBOSITY=info
# $REPO's own code, even when $REPO is a worktree whose .venv is the shared env (its editable install points at the
# main checkout): PYTHONPATH precedes site-packages
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
head=$(git -C "$REPO" rev-parse HEAD)
if [[ -n "${EXPECT_SHA:-}" && "$head" != "$EXPECT_SHA"* ]]; then
    echo "the checkout moved: HEAD ${head:0:7}, this array was submitted at $EXPECT_SHA"; exit 3
fi
if [[ -n "${SPLIT_MANIFEST_MD5:-}" ]]; then
    now=$(md5sum < "$SPLIT_DIR/split_manifest.json" | cut -d' ' -f1)
    [[ $now == "$SPLIT_MANIFEST_MD5" ]] || {
        echo "split tables rebuilt since submit: split_manifest.json md5 $now, submitted with $SPLIT_MANIFEST_MD5"; exit 3; }
fi

IFS=,: read -r -a drugs <<<"$DRUGS"
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
(( TASK_ID < ${#drugs[@]} )) || { echo "array index $TASK_ID has no drug in DRUGS=$DRUGS"; exit 1; }
drug=${drugs[$TASK_ID]}
table=$SPLIT_DIR/${drug}_split.csv
[[ -s $table ]] || { echo "no split table $table"; exit 4; }
out=$OUT_ROOT/$drug
[[ ! -e $out/results.json ]] || { echo "$out/results.json exists: move the old run aside first"; exit 5; }
workers=${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}
# extra holdouts: unset → holdout_st; set but empty → none (the flag with no value)
read -r -a extra <<<"${EXTRA_HOLDOUTS-holdout_st}"

# finetune_amr writes nothing to its working directory that should live in the repo: run from a logs dir
RUN_DIR=${LOG_ROOT:-/rds/user/dca36/hpc-work/logs}/finetune_amr_${SLURM_ARRAY_JOB_ID:-local}_${TASK_ID}
mkdir -p "$RUN_DIR" "$OUT_ROOT"
cd "$RUN_DIR"
echo "[$(date -Is)] host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none} task=$TASK drug=$drug table=$table" \
     "md5=$(md5sum < "$table" | cut -d' ' -f1) out=$out workers=$workers repo=$REPO@${head:0:7}" \
     "${N_SAMPLES:+n_samples=$N_SAMPLES} extra_holdouts=${extra[*]+${extra[*]}}"
nvidia-smi -L || true

uv run --project "$REPO" --no-sync python -X faulthandler -m bacpredict.engine.finetune.finetune_amr \
    --task "$TASK" --drug "$drug" --split-table "$table" --embeddings-dir "$EMBEDDINGS_DIR" --output-dir "$out" \
    --model-name-or-path "${MODEL:-macwiatrak/bacformer-large-masked-complete-genomes}" --precision bf16 \
    --lr "${LR:-0.00015}" --batch-size 1 --grad-accumulation-steps 8 --warmup-proportion 0.1 \
    --eval-steps "${EVAL_STEPS:-250}" --max-steps 100000 --max-epochs "${MAX_EPOCHS:-40}" \
    --early-stopping-patience "${PATIENCE:-15}" --early-stopping-threshold "${THRESHOLD:-0.005}" \
    --seed "${SEED:-1}" --num-workers "$workers" ${N_SAMPLES:+--n-samples "$N_SAMPLES"} \
    --extra-holdouts ${extra[@]+"${extra[@]}"}
echo "[$(date -Is)] done"

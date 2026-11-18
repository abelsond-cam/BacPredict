#!/bin/bash
# GPU — gene-level ablation attribution of a deployed AMR fine-tune (engine/explain).
#
# Ablates one protein at a time and records the shift in the model's logit, giving every gene in
# every genome an attribution score. Cost is one forward pass per protein, so a TB genome (~4,100
# proteins) is ~4,100 forwards and a Kp genome (~5,400) rather more.
#
# Runs as a SHARD of an array job; each task takes a contiguous slice of the cohort CSV and writes
# one parquet per genome, skipping genomes already done — so a re-submit resumes rather than repeats.
#
# Usage — Stage A smoke, ONE genome, and never on a login node (a CPU smoke here silently
# produces nothing useful):
#   LIMIT=1 sbatch --time=00:40:00 src/bacpredict/engine/scripts/run_gene_ablation.sh
#
# Usage — the pilot, 200 genomes over 10 shards:
#   N_SHARDS=10 sbatch --array=0-9 --time=04:00:00 src/bacpredict/engine/scripts/run_gene_ablation.sh
#
# Knobs (all env vars, all with defaults below): ORGANISM DRUG COHORT_CSV CHECKPOINT OUT_DIR
#   N_SHARDS LIMIT DTYPE PERTURBATIONS_PER_EVAL.
#
#SBATCH --job-name=gene-ablation
# %j is the task's own job id, unique for a plain job AND for each array task — %A_%a would
# render a non-array job as <id>_4294967294, which is what the first smoke produced.
#SBATCH --output=/rds/user/dca36/hpc-work/logs/%x-%j.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/%x-%j.out
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
# mem=32G: the model is ~0.7 GB in bf16 on the GPU and one genome's tensors are ~15 MB on the host,
# so this is already generous; on CSD3's ampere, memory is bought with cores (8 GB/core).
# --time is deliberately overridden per submission: a smoke is minutes, a pilot shard is hours.

set -uo pipefail

: "${BACPREDICT_DATA_ROOT:="$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/bac_ast_prediction"}"
D="$BACPREDICT_DATA_ROOT"
PY="$HOME/workspace/BacPredict/.venv/bin/python"
export PYTHONUNBUFFERED=1

: "${ORGANISM:=tb}"
: "${DRUG:=rifampin}"
: "${DTYPE:=bf16}"
: "${PERTURBATIONS_PER_EVAL:=1}"

if [[ "$ORGANISM" == "tb" ]]; then
    TASK_DIR="$D/processed/train_tb_ast"
    : "${CHECKPOINT:=$TASK_DIR/checkpoints/mycobacterium_tuberculosis_${DRUG}_lr_0.00015_finetuned_fold00_seed1}"
else
    TASK_DIR="$D/processed/train_kleb_ast"
    : "${CHECKPOINT:=$TASK_DIR/models/finetune/klebsiella_pneumoniae_${DRUG}_lr_0.00015_finetuned_fold00_seed1}"
fi
: "${COHORT_CSV:=$TASK_DIR/attribution/attribution_cohort_${DRUG}.csv}"
: "${OUT_DIR:=$TASK_DIR/attribution/${DRUG}}"

for required in "$COHORT_CSV" "$CHECKPOINT"; do
    if [[ ! -e "$required" ]]; then
        echo "ERROR: missing $required" >&2
        echo "       Run engine.explain.select_attribution_cohort first, and check the checkpoint path." >&2
        exit 1
    fi
done
mkdir -p "$OUT_DIR"

ARGS=(--organism "$ORGANISM" --cohort-csv "$COHORT_CSV" --checkpoint "$CHECKPOINT"
      --out-dir "$OUT_DIR" --dtype "$DTYPE" --perturbations-per-eval "$PERTURBATIONS_PER_EVAL")
# Shard only inside an array; a bare submission does the whole cohort.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && -n "${N_SHARDS:-}" ]]; then
    ARGS+=(--shard-index "$SLURM_ARRAY_TASK_ID" --n-shards "$N_SHARDS")
fi
if [[ -n "${LIMIT:-}" ]]; then
    ARGS+=(--limit "$LIMIT")
fi

echo "========================================================================"
echo "Gene-ablation attribution — $ORGANISM / $DRUG"
echo "Checkpoint:  $CHECKPOINT"
echo "Cohort:      $COHORT_CSV"
echo "Output:      $OUT_DIR"
echo "dtype:       $DTYPE   perturbations/eval: $PERTURBATIONS_PER_EVAL"
echo "Shard:       ${SLURM_ARRAY_TASK_ID:-none} of ${N_SHARDS:-1}   limit: ${LIMIT:-none}"
echo "Job:         ${SLURM_JOB_ID:-local}"
echo "========================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$PY" -m bacpredict.engine.explain.gene_ablation_attribution "${ARGS[@]}"
status=$?
echo "Exit status: $status"
exit $status

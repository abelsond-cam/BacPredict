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
#   LIMIT=1 sbatch --time=00:40:00 src/bacpredict/engine/scripts/run_gene_attribution.sh
#
# Usage — the pilot, 200 genomes over 10 shards:
#   N_SHARDS=10 sbatch --array=0-9 --time=04:00:00 src/bacpredict/engine/scripts/run_gene_attribution.sh
#
# Knobs (all env vars, all with defaults below): ORGANISM DRUG COHORT_CSV CHECKPOINT OUT_DIR
#   N_SHARDS LIMIT DTYPE METHOD PERTURBATIONS_PER_EVAL N_STEPS INTERNAL_BATCH_SIZE ESM_DIR.
#
# DTYPE defaults to fp32, NOT the deployed bf16. Measured on job 41289828: in bf16 a Kp genome
# returns 13 distinct attribution values with 96.2% of genes exactly zero, because the base logit
# sits near 7.3 where the bf16 ULP is ~0.016. fp32 returns 4,903 distinct values and no zeros.
#
# METHOD defaults to integrated_gradients. Ablation deletes a gene, which is the wrong counterfactual
# for a determinant present in every genome and differing only by a point mutation (gyrA, parC, rpoB).
# It is also ~50x cheaper: n_steps forward-backward passes for the whole genome, against one forward
# per protein.
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
: "${DTYPE:=fp32}"
: "${METHOD:=integrated_gradients}"
: "${PERTURBATIONS_PER_EVAL:=1}"
: "${N_STEPS:=50}"
: "${INTERNAL_BATCH_SIZE:=4}"

if [[ "$ORGANISM" == "tb" ]]; then
    TASK_DIR="$D/processed/train_tb_ast"
    : "${CHECKPOINT:=$TASK_DIR/checkpoints/mycobacterium_tuberculosis_${DRUG}_lr_0.00015_finetuned_fold00_seed1}"
else
    TASK_DIR="$D/processed/train_kleb_ast"
    : "${CHECKPOINT:=$TASK_DIR/models/finetune/klebsiella_pneumoniae_${DRUG}_lr_0.00015_finetuned_fold00_seed1}"
fi
: "${COHORT_CSV:=$TASK_DIR/attribution/attribution_cohort_${DRUG}.csv}"
: "${OUT_DIR:=$TASK_DIR/attribution/${DRUG}_${METHOD}_${DTYPE}}"

for required in "$COHORT_CSV" "$CHECKPOINT"; do
    if [[ ! -e "$required" ]]; then
        echo "ERROR: missing $required" >&2
        echo "       Run engine.explain.select_attribution_cohort first, and check the checkpoint path." >&2
        exit 1
    fi
done
mkdir -p "$OUT_DIR"

ARGS=(--organism "$ORGANISM" --cohort-csv "$COHORT_CSV" --checkpoint "$CHECKPOINT"
      --out-dir "$OUT_DIR" --dtype "$DTYPE" --method "$METHOD"
      --perturbations-per-eval "$PERTURBATIONS_PER_EVAL"
      --n-steps "$N_STEPS" --internal-batch-size "$INTERNAL_BATCH_SIZE")
# TB's embedding store was moved to the RDV archive tier, so a TB run points at whatever subset has
# been restored. ESM_DIR overrides the store location; unset, the organism's default is used.
if [[ -n "${ESM_DIR:-}" ]]; then
    ARGS+=(--esm-dir "$ESM_DIR")
fi

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
echo "method:      $METHOD   dtype: $DTYPE"
echo "knobs:       perturbations/eval=$PERTURBATIONS_PER_EVAL n_steps=$N_STEPS ibs=$INTERNAL_BATCH_SIZE"
echo "Shard:       ${SLURM_ARRAY_TASK_ID:-none} of ${N_SHARDS:-1}   limit: ${LIMIT:-none}"
echo "Job:         ${SLURM_JOB_ID:-local}"
echo "========================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$PY" -m bacpredict.engine.explain.gene_attribution "${ARGS[@]}"
status=$?
echo "Exit status: $status"
exit $status

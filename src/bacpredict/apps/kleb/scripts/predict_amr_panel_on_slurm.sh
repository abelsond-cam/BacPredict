#!/bin/bash
#SBATCH --job-name=kleb_amr_predict
#SBATCH --output=/scratch/u6fp/dca36.u6fp/logs/%x-%A_%a.out
#SBATCH --error=/scratch/u6fp/dca36.u6fp/logs/%x-%A_%a.out
#SBATCH --time=04:00:00
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-21
#SBATCH --open-mode=append
# CSD3/UoHPC variant (when it returns): --partition=ampere --account=FLOTO-SL2-GPU,
#   logs → relative or ~/rds/hpc-work/logs/, and `module load cuda/12.4 cudnn/8.9_cuda-12.4`.
#
# Score every kpsc_final_list sample with the fine-tuned Bacformer head for one
# drug per array task. Per-drug parquets land at:
#
#   $BASE/predictions_for_metadata/<drug>.parquet
#
# The downstream BacHGT merge step joins these into the v2 metadata table.
#
# Submit:   sbatch src/bacpredict/apps/kleb/scripts/predict_amr_panel_on_slurm.sh
# Re-run one drug only (e.g. cipro = index 3):
#           sbatch --array=3 src/bacpredict/apps/kleb/scripts/predict_amr_panel_on_slurm.sh

set -uo pipefail

# Data root — one env var, cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
: "${BACPREDICT_DATA_ROOT:="$SCRATCHDIR"}"
D="$BACPREDICT_DATA_ROOT"
PY="$SCRATCHDIR/envs/bacpredict-gpu-venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

BASE=$D/processed/train_kleb_ast
FT=$BASE/models/finetune
EMB=$D/processed/train_kleb_ast/esm
METADATA=$D/final/metadata_v2_all_samples_and_columns.tsv
OUT_DIR=$BASE/predictions_for_metadata
mkdir -p "$OUT_DIR"

# Panel order matches src/bacpredict/apps/kleb/scripts/eval_panel_on_slurm.sh.
DRUGS=(
  gentamicin
  ceftazidime
  meropenem
  ciprofloxacin
  trimethoprim-sulfamethoxazole
  amikacin
  ceftriaxone
  piperacillin-tazobactam
  cefoxitin
  aztreonam
  cefazolin
  tobramycin
  cefepime
  imipenem
  levofloxacin
  cefotaxime
  cefuroxime
  ampicillin-sulbactam
  ertapenem
  tetracycline
  azithromycin
  colistin
)

d=${DRUGS[$SLURM_ARRAY_TASK_ID]}
CK=$FT/klebsiella_pneumoniae_${d}_lr_0.00015_finetuned_fold00_seed1
OUT=$OUT_DIR/${d}.parquet

echo "=== array task $SLURM_ARRAY_TASK_ID / drug $d ==="
echo "checkpoint: $CK"
echo "output:     $OUT"

if [ ! -d "$CK" ]; then
  echo "NO CHECKPOINT (skipping): $d"
  exit 0
fi

# Idempotent: skip drugs whose parquet already exists (set FORCE_RECOMPUTE=1 to override).
if [ -f "$OUT" ] && [ "${FORCE_RECOMPUTE:-0}" != "1" ]; then
  echo "ALREADY DONE (skip; set FORCE_RECOMPUTE=1 to re-run): $d"
  exit 0
fi

# NUM_WORKERS env (default 8) controls the DataLoader worker count.
#   Past pitfall: at deployment scale (~73k items) the file_system sharing
#   strategy + many workers can exhaust mmap on GPU nodes. workers=0 is
#   always safe but slow. workers=8 with cpus-per-task=8 is the production
#   default; failures should surface fast (within minutes) — see fail-fast
#   wrapper below.
#
# Canary support: if N_SAMPLES is set, predict only that many samples (quick
# end-to-end smoke). Submit canary with:
#   sbatch --array=0 --time=00:30:00 --export=ALL,N_SAMPLES=10 <this script>
# Submit production with:
#   sbatch --time=12:00:00 --array=0-21 <this script>          # 8 workers default
#   sbatch --time=12:00:00 --array=0-21 --export=ALL,NUM_WORKERS=4 <this script>
NUM_WORKERS=${NUM_WORKERS:-8}
N_SAMPLES_ARG=()
if [ -n "${N_SAMPLES:-}" ]; then
  N_SAMPLES_ARG=(--n-samples "$N_SAMPLES")
  echo "CANARY mode: N_SAMPLES=$N_SAMPLES"
fi
echo "Using NUM_WORKERS=$NUM_WORKERS"

if ! "$PY" -m bacpredict.apps.kleb.predict_amr_for_metadata \
  --drug "$d" \
  --checkpoint "$CK" \
  --metadata-tsv "$METADATA" \
  --embeddings-dir "$EMB" \
  --out "$OUT" \
  --batch-size 1 \
  --num-workers "$NUM_WORKERS" \
  "${N_SAMPLES_ARG[@]}"; then
  echo "PREDICT_FAILED: $d (see .err log)"
  exit 1
fi

echo "PREDICT_DONE: $d"

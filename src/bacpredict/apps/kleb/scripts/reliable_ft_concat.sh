#!/bin/bash
# Phase 2b (CPU): reliable-label ESM-vs-FT per-gene head-to-head + FT-mean ⊕ best-gene concat, one drug
# per array task. Reads the Phase-2b FT token cache (ft_amr_cache/<drug>/) + the ESM store; no forward
# pass. Run AFTER the matching cache_ft_amr_proteins GPU task for the drug has finished.
#
#     sbatch --array=11 src/bacpredict/apps/kleb/scripts/reliable_ft_concat.sh                 # smoke (TMP-SMX)
#     sbatch --dependency=afterok:<gpu_jid> --array=0-21 src/bacpredict/apps/kleb/scripts/reliable_ft_concat.sh
#
#SBATCH --job-name=kleb_reliable_ft_concat
#SBATCH --output=/scratch/u6fp/dca36.u6fp/logs/%x-%A_%a.out
#SBATCH --error=/scratch/u6fp/dca36.u6fp/logs/%x-%A_%a.out
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
# CSD3/UoHPC variant (when it returns): --partition=icelake-himem --account=FLOTO-PROJECT-K-SL2-CPU,
#   logs → relative or ~/rds/hpc-work/logs/.

set -uo pipefail

# Data root — one env var, cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
: "${BACPREDICT_DATA_ROOT:="$SCRATCHDIR"}"
D="$BACPREDICT_DATA_ROOT"
PY="$SCRATCHDIR/envs/bacpredict-gpu-venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

DRUGS=(cefotaxime ertapenem ampicillin-sulbactam ceftriaxone cefuroxime ciprofloxacin ceftazidime \
       gentamicin cefazolin imipenem meropenem trimethoprim-sulfamethoxazole tobramycin amikacin \
       levofloxacin piperacillin-tazobactam cefoxitin tetracycline aztreonam cefepime azithromycin colistin)
DRUG=${DRUGS[$SLURM_ARRAY_TASK_ID]}
if [[ -z "$DRUG" ]]; then echo "ERROR: no drug for array index $SLURM_ARRAY_TASK_ID" >&2; exit 1; fi

FTC=$D/processed/train_kleb_ast/pangena_predict/ft_amr_cache/$DRUG
FRC=$D/processed/train_kleb_ast/pangena_predict/frozen_amr_cache/$DRUG
OUT=$D/processed/train_kleb_ast/pangena_predict/reliable_ft_concat/$DRUG
mkdir -p "$OUT"
if [[ ! -f "$FTC/ft_genome_mean_${DRUG}.npz" ]]; then echo "ERROR: FT cache missing: $FTC" >&2; exit 1; fi
[[ -d "$FRC/frozen_amr_emb" ]] || echo "WARN: frozen cache missing ($FRC) — frozen per-gene LR skipped"
echo "=== Kp reliable ESM-vs-FT(+frozen) + concat — drug=$DRUG (task $SLURM_ARRAY_TASK_ID) ==="

"$PY" -m bacpredict.apps.kleb.reliable_ft_concat \
    --drug "$DRUG" \
    --ast-sheet-path "$D/processed/train_kleb_ast/binary_ast_with_split.csv" \
    --ft-cache-dir "$FTC" \
    --frozen-cache-dir "$FRC" \
    --esm-store-dir "$D/processed/train_kleb_ast/esm" \
    --parquet-dir "$D/processed/train_kleb_ast/protein_sequences" \
    --sidecar-dir "$D/processed/train_kleb_ast/amr_annotation" \
    --out-dir "$OUT" --grain family --n-folds 5 --seed 1

echo "Kp reliable ESM-vs-FT(+frozen) + concat ($DRUG) finished — $OUT"

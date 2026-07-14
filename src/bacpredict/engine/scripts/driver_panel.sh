#!/usr/bin/env bash
# Per-drug DRIVER PANEL (one-hot | baclm | ESM | Bacformer) for every driving mutation of a species.
# Reads the per-drug driver CSVs committed under docs/visualisations/<prefix>_<drug>/ and scores the
# coding drivers' baclm + ESM (+ Bacformer, if --bacformer-npz points at a sweep) vs the drug label,
# leaving non-coding / rRNA rows blank (one-hot only). CPU-only (NO --gres; --mem required).
#
#   sbatch --export=ALL,TASK=tb   -J driver-panel-tb   src/bacpredict/engine/scripts/driver_panel.sh
#   # optional: BACFORMER_NPZ=/path/to/panel_tokens.npz to fill the Bacformer column
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
#SBATCH --error=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
set -uo pipefail
: "${SCRATCHDIR:?}" "${TASK:=tb}"
S="$SCRATCHDIR"
PY="$S/envs/bacpredict-gpu-venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"
export MPLBACKEND=Agg
AMR_ARG=""
case "$TASK" in
  tb)   SPECIES=tb; FOLDER=tb; CSVPREFIX=tbprofiler_gene_lr; CSVSUFFIX=""
        VIS="$HOME/BacPredict/src/bacpredict/docs/visualisations" ;;
  kleb) SPECIES=kp; FOLDER=kp; CSVPREFIX=card_determinant_lr; CSVSUFFIX="_family"
        VIS="$HOME/BacPredict/src/bacpredict/apps/kleb/docs/visualisations/amr_per_abx"
        # CARD sidecars locate acquired genes Bakta misses; falls back to Bakta names if unpopulated.
        AMR_ARG="--amr-sidecar-dir $S/processed/train_kleb_ast/amr_annotation" ;;
  *) echo "unknown TASK=$TASK (want tb|kleb)"; exit 1 ;;
esac
OUT="$S/processed/train_${TASK}_ast/pangena_predict/driver_panel"
NPZ_ARG=""; [ -n "${BACFORMER_NPZ:-}" ] && NPZ_ARG="--bacformer-npz ${BACFORMER_NPZ}"

echo "=== driver panel: species=$SPECIES csv=$VIS/${FOLDER}_* npz=${BACFORMER_NPZ:-none} ==="
"$PY" "$HOME/BacPredict/src/bacpredict/engine/plots/driver_panel.py" \
  --species "$SPECIES" \
  --csv-dir "$VIS" --folder-prefix "$FOLDER" --csv-prefix "$CSVPREFIX" --csv-suffix "$CSVSUFFIX" \
  --n-folds 5 --seeds 1,2,3 --pool-workers "${SLURM_CPUS_PER_TASK:-8}" \
  $NPZ_ARG $AMR_ARG \
  --output "$OUT"
echo "driver panel -> $OUT"

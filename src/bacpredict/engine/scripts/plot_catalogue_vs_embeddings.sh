#!/usr/bin/env bash
# Per-drug DRIVER PANEL (one-hot | baclm | ESM | Bacformer) for every driving mutation of a species.
# Reads the per-drug driver CSVs committed under visualisations/<organism>/<drug>/ and scores the
# coding drivers' baclm + ESM (+ Bacformer, if --bacformer-npz points at a sweep) vs the drug label,
# leaving non-coding / rRNA rows blank (one-hot only). CPU-only (NO --gres; --mem required).
#
#   sbatch --export=ALL,TASK=tb   -J driver-panel-tb   src/bacpredict/engine/scripts/driver_panel.sh
#   # optional: BACFORMER_NPZ=/path/to/panel_tokens.npz to fill the Bacformer column
#SBATCH --partition=icelake-himem
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/rds/user/dca36/hpc-work/logs/%x-%j.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/%x-%j.out
set -uo pipefail
: "${SCRATCHDIR:?}" "${TASK:=tb}"
S="$SCRATCHDIR"
PY="$S/envs/bacpredict-gpu-venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"
export MPLBACKEND=Agg
AMR_ARG=""
case "$TASK" in
  tb)   SPECIES=tb; FOLDER=tb; CSVPREFIX=tbprofiler_gene_lr; CSVSUFFIX=""
        VIS="$HOME/BacPredict/src/bacpredict/visualisations/tb" ;;
  kleb) SPECIES=kp; FOLDER=kp; CSVPREFIX=card_determinant_lr; CSVSUFFIX="_family"
        VIS="$HOME/BacPredict/src/bacpredict/visualisations/kp"
        # CARD sidecars locate acquired genes Bakta misses; falls back to Bakta names if unpopulated.
        AMR_ARG="--amr-sidecar-dir $S/processed/train_kleb_ast/amr_annotation" ;;
  *) echo "unknown TASK=$TASK (want tb|kleb)"; exit 1 ;;
esac
OUT="$S/processed/train_${TASK}_ast/pangena_predict/driver_panel"
NPZ_ARG=""; [ -n "${BACFORMER_NPZ:-}" ] && NPZ_ARG="--bacformer-npz ${BACFORMER_NPZ}"

echo "=== driver panel: species=$SPECIES csv=$VIS/<drug> npz=${BACFORMER_NPZ:-none} ==="
"$PY" -m bacpredict.engine.plots.plot_catalogue_vs_embeddings \
  --species "$SPECIES" \
  --csv-dir "$VIS" --csv-prefix "$CSVPREFIX" --csv-suffix "$CSVSUFFIX" \
  --n-folds 5 --seeds 1,2,3 --pool-workers "${SLURM_CPUS_PER_TASK:-8}" \
  $NPZ_ARG $AMR_ARG \
  --output "$OUT"
echo "driver panel -> $OUT"

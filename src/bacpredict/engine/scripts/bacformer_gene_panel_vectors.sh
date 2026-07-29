#!/usr/bin/env bash
# GPU — Bacformer gene-token sweep for the driver panel: one forward pass per genome, extract every
# coding driver gene's contextualised token (genes derived from the per-drug driver CSVs). Output NPZ
# feeds driver_panel.py's Bacformer column. Runs in the lean gpu venv (has bacformer + transformers).
#
#   sbatch --export=ALL,TASK=tb   -J bacformer-panel-tb   src/bacpredict/engine/scripts/bacformer_gene_panel_vectors.sh
#   sbatch --export=ALL,TASK=kleb -J bacformer-panel-kleb src/bacpredict/engine/scripts/bacformer_gene_panel_vectors.sh
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=18:00:00
#SBATCH --output=/rds/user/dca36/hpc-work/logs/%x-%j.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/%x-%j.out
set -uo pipefail
: "${SCRATCHDIR:?}" "${TASK:=tb}"
S="$SCRATCHDIR"
PY="$S/envs/bacpredict-gpu-venv/bin/python"
export HF_HOME="$S/cache/hf" TORCH_HOME="$S/cache/torch"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
AMR_ARG=""
case "$TASK" in
  tb)   DIR=train_tb_ast;   FOLDER=tb; CSVPREFIX=tbprofiler_gene_lr; CSVSUFFIX=""
        VIS="$HOME/BacPredict/src/bacpredict/visualisations/tb" ;;
  kleb) DIR=train_kleb_ast; FOLDER=kp; CSVPREFIX=card_determinant_lr; CSVSUFFIX="_family"
        VIS="$HOME/BacPredict/src/bacpredict/visualisations/kp"
        AMR_ARG="--amr-sidecar-dir $S/processed/train_kleb_ast/amr_annotation" ;;
  *) echo "unknown TASK=$TASK (want tb|kleb)"; exit 1 ;;
esac
PROC="$S/processed/$DIR"
OUT="$PROC/pangena_predict/driver_panel/bacformer_panel_tokens_${FOLDER}.npz"

echo "=== bacformer gene-panel sweep: task=$TASK ==="
"$PY" -m bacpredict.engine.concat.bacformer_gene_panel_vectors \
  --ast-sheet-path "$PROC/binary_ast_with_split.csv" \
  --parquet-dir "$PROC/protein_sequences" \
  --esm-store-dir "$PROC/esm" \
  --csv-dir "$VIS" --csv-prefix "$CSVPREFIX" --csv-suffix "$CSVSUFFIX" \
  $AMR_ARG \
  --pool-workers "${SLURM_CPUS_PER_TASK:-8}" \
  --output-npz "$OUT"
echo "bacformer panel tokens -> $OUT"

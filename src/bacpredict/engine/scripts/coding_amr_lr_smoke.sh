#!/usr/bin/env bash
# Stage-A smoke for the baclm-vs-ESM coding probe (Stage 2a). CPU-only, tiny (n=10) — safe to run
# inline on the Isambard login node. Confirms the pipeline runs end-to-end on real data before the
# full CPU panel job: gene→flat-index bridge, ESM + baclm readers, and the k-fold harness.
#
# Usage:  ssh <cluster> 'cd $HOME/BacPredict && setup/.../  ...'  or directly:
#   SCRATCHDIR=/scratch/u6fp/dca36.u6fp bash src/bacpredict/engine/scripts/coding_amr_lr_smoke.sh
set -uo pipefail
: "${SCRATCHDIR:?}"
S="$SCRATCHDIR"
PY="$S/envs/bacpredict-gpu-venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"
OUT="$S/processed/train_tb_ast/pangena_predict/coding_amr_lr/smoke_rpob_rifampin.json"

"$PY" "$HOME/BacPredict/src/bacpredict/engine/gene_lr/coding_amr_lr.py" \
  --species tb --gene rpoB --drug rifampin --n 10 --n-folds 3 --seeds 1 \
  --output "$OUT"
echo "smoke JSON -> $OUT"

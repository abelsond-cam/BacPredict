#!/bin/bash
# One-time setup of the ISOLATED DefensePredictor environment.
#
# DefensePredictor pins torch >=2.5.1,<2.6 + fair-esm + lightgbm, which conflicts with the
# main BacPredict env (Bacformer/transformers). So it lives in its own uv venv at
# src/dp_short_read/.venv-dp and is invoked with .venv-dp/bin/python (never `uv run`).
#
# Run on the LOGIN NODE (needs internet for the PyPI install + weight downloads).
# Weights (5 LightGBM folds + ESM2-150M + contact-regression) download once into the package
# dir; re-running detects them and skips. Safe to re-run.
#
# Usage:  bash src/dp_short_read/scripts/setup_dp_env.sh
set -euo pipefail

cd /home/dca36/workspace/BacPredict
VENV="src/dp_short_read/.venv-dp"

echo "=== Creating isolated DP venv at ${VENV} (python 3.11) ==="
uv venv --python 3.11 "${VENV}"

echo "=== Installing defense_predictor + convert-script deps (gffutils, biopython) ==="
# gffutils + biopython are needed by Panaroo's convert_bakta_to_prokka_gff.py, which we call
# to fuse Bakta GFF + assembly into the combined GFF DefensePredictor consumes.
uv pip install --python "${VENV}/bin/python" defense_predictor gffutils biopython

echo "=== Downloading DefensePredictor model weights (idempotent) ==="
"${VENV}/bin/defense_predictor_download"

echo ""
echo "=== Sanity check ==="
"${VENV}/bin/python" -c "import defense_predictor, gffutils, Bio, torch; \
print('defense_predictor OK | torch', torch.__version__, '| cuda', torch.cuda.is_available())"

echo ""
echo "Done. Run DefensePredictor with: ${VENV}/bin/python src/dp_short_read/run_defense_predictor.py ..."

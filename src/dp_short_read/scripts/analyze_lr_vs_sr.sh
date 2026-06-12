#!/bin/bash
# Matched-protein LR-vs-SR recovery analysis over the full DefensePredictor sweep.
# Translates the cached combined GFFs (~2,580 genomes), matches proteins by exact AA sequence,
# and computes recovery of LR defensive calls in the SR arm. CPU-only, parallel across pairs —
# an icelake job (no GPU). Reads/writes under processed/defence_predictor/full/.
#
# Submit:  sbatch src/dp_short_read/scripts/analyze_lr_vs_sr.sh

#SBATCH --job-name=dp_recovery
#SBATCH --output=dp_recovery_%j.out
#SBATCH --error=dp_recovery_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=icelake
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --open-mode=append

set -euo pipefail
cd /home/dca36/workspace/BacPredict
export PYTHONUNBUFFERED=1

PROJECT_K="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw"
FULL="${PROJECT_K}/david/processed/defence_predictor/full"
PY="src/dp_short_read/.venv-dp/bin/python"

"${PY}" src/dp_short_read/analyze_lr_vs_sr.py \
    --manifest "${FULL}/dp_manifest_full.tsv" \
    --results-dir "${FULL}" \
    --out-dir "${FULL}/analysis" \
    --workers 76

echo "Done — see ${FULL}/analysis/lr_vs_sr_recovery_{per_pair.tsv,summary.json}"

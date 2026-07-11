#!/bin/bash
# Concatenate the ~6.4k {Sample}_amr.parquet sidecars into one amr_calls_all.parquet store
# (kleb_ast.build_amr_calls_store). I/O-bound (~17 min wall, low CPU) — past the login ceiling, so submit it.
# Run once after the sidecar array; downstream modules (card_determinant_lr, ladders) then read it in seconds.
#
#     sbatch src/kleb_ast/scripts/build_amr_calls_store.sh
#
#SBATCH --job-name=kleb_amr_calls_store
#SBATCH --output=kleb_amr_calls_store_%j.out
#SBATCH --error=kleb_amr_calls_store_%j.err
#SBATCH --partition=icelake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --open-mode=append

set -euo pipefail
cd /home/dca36/workspace/BacPredict
export PYTHONUNBUFFERED=1

echo "=== build combined AMR-calls store ==="
uv run python -m kleb_ast.build_amr_calls_store
echo "done -> amr_annotation/amr_calls_all.parquet"

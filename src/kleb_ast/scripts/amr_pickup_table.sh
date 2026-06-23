#!/bin/bash
# CARD vs Kleborate vs Bakta AMR-gene pickup table (kleb_ast.amr_pickup_table).
#
# I/O-bound: concatenates ~6.4k {Sample}_amr.parquet sidecars off RDS (~17 min wall, low CPU), so it runs
# past the login-node watchdog ceiling — submit it as a job rather than running it on the login node.
#
#     sbatch src/kleb_ast/scripts/amr_pickup_table.sh
#
#SBATCH --job-name=kleb_amr_pickup_table
#SBATCH --output=kleb_amr_pickup_table_%j.out
#SBATCH --error=kleb_amr_pickup_table_%j.err
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

echo "=== CARD vs Kleborate vs Bakta pickup table ==="
uv run python -m kleb_ast.amr_pickup_table
echo "pickup table finished -> src/kleb_ast/docs/visualisations/amr_annotation/card_vs_kleborate_vs_bakta_pickup.{csv,md}"

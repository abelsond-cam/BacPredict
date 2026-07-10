#!/bin/bash
# Step 3b geometry probe — Stage-A smoke (CPU, in-silico RRDR panel, windowed LLR).
#
# Labels-free: builds in-silico WT->mutant rpoB pairs from the bundled H37Rv
# reference (no RDS data needed), runs ESM-C per-residue / per-layer geometry, and
# emits the d_site/d_window/d_pool/d_max/d_cls table + a windowed masked-LM LLR
# profile + two plots + JSON. Asserts WT codon identity and that ESM++ returns
# populated hidden_states. Expect d_site >> d_pool and the causal residue as the
# single LLR outlier.
#
# Small enough for an interactive box, but it loads the ESM-C small model (>128 MB),
# so it runs as a modest CPU sbatch rather than on the login node. Add --full-profile
# (all ~1,178 residues) only on a GPU.
#
# Usage:  sbatch src/pangena_predict/scripts/smoke_geometry_probe.sh
#
#SBATCH --job-name=geom_probe_smoke
#SBATCH --output=geom_probe_smoke_%j.out
#SBATCH --error=geom_probe_smoke_%j.err
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --open-mode=append

cd /home/dca36/workspace/BacPredict

export PYTHONUNBUFFERED=1

OUT_DIR=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_tb_ast/pangena_predict/geometry_probe_smoke
OUT_JSON=$OUT_DIR/geometry_probe_${SLURM_JOB_ID:-local}.json
mkdir -p "$OUT_DIR"

echo "========================================================================"
echo "Step 3b geometry probe smoke (CPU, in-silico RRDR panel)"
echo "Output JSON: $OUT_JSON"
echo "Plots dir:   $OUT_DIR"
echo "========================================================================"

uv run python src/pangena_predict/geometry_probe.py \
    --output-json "$OUT_JSON" \
    --output-dir "$OUT_DIR" \
    --device cpu \
    --window-k 5

echo "Geometry probe smoke finished — JSON + plots in $OUT_DIR"

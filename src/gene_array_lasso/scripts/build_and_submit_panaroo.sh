#!/bin/bash
# Step A — build the per-drug Panaroo sample TSV and submit the BacHGT Panaroo run.
#
# This is a LOGIN-NODE submit helper (not itself an sbatch job). It:
#   1. runs build_panaroo_sample_tsv.py (BacPredict) → <drug>.tsv on RDS (SR-only, kpsc forced),
#   2. submits the BacHGT runner (slurm_scripts/panaroo_run_strain.sh) with --sample-metadata-file,
#      overriding partition→icelake-himem and time→36h (generous; Panaroo on ~1.4k–2.4k genomes).
#
# The BacHGT runner cd's into ~/workspace/BacHGT, uses the `panaroo` micromamba env + the panaroo fork
# sibling, writes the GPA to <outdir>/<drug>/ (run label = TSV basename). Account stays the project_k
# SL2-CPU one set inside that script.
#
# Usage:  bash src/gene_array_lasso/scripts/build_and_submit_panaroo.sh <drug>
#   e.g.  bash src/gene_array_lasso/scripts/build_and_submit_panaroo.sh imipenem
set -euo pipefail

DRUG="${1:?usage: build_and_submit_panaroo.sh <drug>}"

BACPREDICT=/home/dca36/workspace/BacPredict
BACHGT=/home/dca36/workspace/BacHGT
RDS=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david
TSV_DIR="$RDS/processed/gene_array_lasso/panaroo_input_tsv"
OUTDIR="$RDS/processed/gene_array_lasso/panaroo"
LOGDIR="$BACPREDICT/src/gene_array_lasso/scripts/logs"
RUNNER="$BACHGT/src/bac_panaroo/slurm_scripts/panaroo_run_strain.sh"

mkdir -p "$LOGDIR" "$OUTDIR"

echo "=== Step A: build TSV for $DRUG ==="
cd "$BACPREDICT"
uv run python src/gene_array_lasso/build_panaroo_sample_tsv.py --drug "$DRUG" --out-dir "$TSV_DIR"

TSV="$TSV_DIR/$DRUG.tsv"
if [[ ! -f "$TSV" ]]; then echo "ERROR: TSV not written: $TSV" >&2; exit 1; fi

echo "=== Step A: submit Panaroo for $DRUG ==="
if [[ ! -f "$RUNNER" ]]; then echo "ERROR: BacHGT runner missing: $RUNNER" >&2; exit 1; fi

sbatch \
  --job-name="panaroo_gal_${DRUG}" \
  --partition=icelake-himem \
  --time=36:00:00 \
  --output="$LOGDIR/panaroo_gal_${DRUG}_%j.out" \
  --error="$LOGDIR/panaroo_gal_${DRUG}_%j.err" \
  "$RUNNER" \
  --sample-metadata-file "$TSV" \
  --outdir "$OUTDIR"

echo "Submitted. GPA will land in $OUTDIR/$DRUG/  (gene_presence_absence.csv, panaroo_genomes.tsv, …)"

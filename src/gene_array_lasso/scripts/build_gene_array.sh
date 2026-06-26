#!/bin/bash
# Step B+C — build the block-sparse frozen-ESM-C array for one drug from its completed Panaroo run.
#
# Heavy I/O (loads one ESM .pt per sample) + a large sparse matrix (~tens of GB at >1%), so this is a
# CPU himem job, not login-node work. Output -> gene_arrays/<drug>_p<pct>/ on RDS.
#
# Usage:  sbatch src/gene_array_lasso/scripts/build_gene_array.sh <drug> <min_prevalence> [panaroo_subdir]
#   e.g.  sbatch src/gene_array_lasso/scripts/build_gene_array.sh tetracycline 0.01
#         sbatch src/gene_array_lasso/scripts/build_gene_array.sh imipenem 0.01 imipenem_norefind
#
#SBATCH --job-name=gal_build_array
#SBATCH --partition=icelake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=350G
#SBATCH --time=24:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --output=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/gene_array_lasso/logs/gal_build_array_%j.out
#SBATCH --error=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/gene_array_lasso/logs/gal_build_array_%j.err
set -euo pipefail

DRUG="${1:?usage: build_gene_array.sh <drug> <min_prevalence> [panaroo_subdir]}"
MINPREV="${2:?give min_prevalence, e.g. 0.01}"
PSUBDIR="${3:-$DRUG}"   # which Panaroo run dir to read (default panaroo/<drug>; e.g. <drug>_norefind)

cd /home/dca36/workspace/BacPredict
export PYTHONUNBUFFERED=1
GAL=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/gene_array_lasso
PCT=$(python3 -c "print(f'{$MINPREV*100:g}')")   # 0.05 -> 5, 0.01 -> 1
OUT="$GAL/gene_arrays/${DRUG}_p${PCT}"

echo "=== build gene array: drug=$DRUG min_prev=$MINPREV (>$PCT%) panaroo=$PSUBDIR -> $OUT ==="
uv run python src/gene_array_lasso/build_gene_embedding_array.py \
  --drug "$DRUG" \
  --panaroo-dir "$GAL/panaroo/$PSUBDIR" \
  --splits-csv "$GAL/panaroo_input_tsv/${DRUG}_splits.csv" \
  --min-prevalence "$MINPREV" \
  --out-dir "$OUT"
echo "Done -> $OUT"

#!/bin/bash
# Experiment-4 unmasked-surprisal scan — STEP 1: genome manifest (CPU, no GPU).
#
# Genotype a pool of the canonical RIF split once and write manifest.csv of ~500 resistant
# rpoB-mutant + ~500 WT genomes (sample, role, rpoB flat index, genotype). The GPU array
# (STEP 2) shards over this manifest. Heavy I/O (reads ~pool-size protein parquets + aligns
# rpoB), so it runs as a CPU sbatch — not the login node (the prepare-script lesson).
#
# Usage:  sbatch src/snp_embeddings/scripts/unmasked_surprisal_manifest.sh
#
#SBATCH --job-name=usurp_manifest
#SBATCH --output=usurp_manifest_%j.out
#SBATCH --error=usurp_manifest_%j.err
#SBATCH --partition=icelake-himem
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

cd /home/dca36/workspace/BacPredict
export PYTHONUNBUFFERED=1

RDS=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_tb_ast
SHEET=$RDS/binary_ast_with_split.csv
PARQUET_DIR=$RDS/tb_protein_sequences
OUT_DIR=$RDS/snp_embeddings/unmasked_surprisal_scan
MANIFEST=$OUT_DIR/manifest.csv
mkdir -p "$OUT_DIR"

echo "Manifest → $MANIFEST  (pool 4000 → 500 resistant + 500 WT)"
uv run python src/snp_embeddings/unmasked_surprisal_scan.py \
    --mode manifest \
    --ast-sheet-path "$SHEET" \
    --parquet-dir "$PARQUET_DIR" \
    --manifest-csv "$MANIFEST" \
    --drug rifampin \
    --n-resistant 500 \
    --n-wt 500 \
    --pool-size 4000

echo "Manifest done: $MANIFEST"

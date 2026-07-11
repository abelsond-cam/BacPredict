#!/bin/bash
#SBATCH --job-name=build_per_gene_lr_panel_store
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=icelake-himem
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=08:00:00

# Build the per-gene logistic-regression probability panel store on the SAME 1000-genome
# manifest split as the surprisal panel, so the att_head run is directly comparable to the
# 0.9768 baseline. NO GPU — sklearn LRs over the existing ESM-C embeddings.
#
# For each core gene (single-copy in >95% of train genomes) it fits an out-of-fold LR on the
# train split, records each protein's predicted resistance probability, and writes two drop-in
# panel stores: filtered/ (out-of-fold train AUROC > 0.8) and unfiltered/ (all core genes).
# Leakage check to read in build_summary.json: rpoB out-of-fold AUROC ~0.95-0.97, NOT 1.0.
#
#   sbatch src/bacpredict/engine/scripts/build_per_gene_lr_panel_store.sh
#
# In-memory footprint ~ n_train x n_core x 960 floats (~8 GB for the 1000-genome manifest);
# the full ~38k cohort would need gene-batching (see the module docstring).

cd /home/dca36/workspace/BacPredict
git pull --ff-only || true
export PYTHONUNBUFFERED=1
# Pin BLAS to 1 thread/process so the joblib workers (one per-gene LR each) don't oversubscribe;
# --n-jobs tracks the SLURM core allocation below, so `sbatch --cpus-per-task=N` scales cleanly
# (32 schedules far faster than a whole node). The ~21k LR fits (3,500 genes x 6) run in a few
# minutes; the wall is dominated by the sequential .pt I/O passes (assembly + panel, ~12 min).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

RDS=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_tb_ast
SHEET=$RDS/tb_surprisal_panel/tb_rif_1000_split.csv   # the 700/100/200 manifest split
PARQUET=$RDS/tb_protein_sequences
EMB=$RDS/tb_esm_embeddings
OUT=$RDS/tb_per_gene_lr_panel

echo "Per-gene LR store build — sheet=$SHEET  out=$OUT"
mkdir -p "$OUT"

uv run python src/bacpredict/engine/gene_lr/build_per_gene_lr_store.py \
    --split-csv "$SHEET" \
    --drug rifampin \
    --parquet-dir "$PARQUET" \
    --esm-store-dir "$EMB" \
    --out-dir "$OUT" \
    --min-prevalence 0.95 \
    --auroc-filter 0.8 \
    --n-folds 5 \
    --seed 1 \
    --n-jobs "${SLURM_CPUS_PER_TASK:-32}"

echo "=== per_gene_lr_build_summary.json ==="
cat "$OUT/per_gene_lr_build_summary.json"
echo
echo "=== top of gene_lr_auroc.csv (per-gene out-of-fold AUROC) ==="
head -n 15 "$OUT/gene_lr_auroc.csv"
echo
echo "=== store dirs ==="
echo "filtered:   $(ls "$OUT/filtered" | grep -c _panel.npz) panels"
echo "unfiltered: $(ls "$OUT/unfiltered" | grep -c _panel.npz) panels"
cat "$OUT/filtered/panel_standardization.json"

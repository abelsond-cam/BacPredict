#!/bin/bash
#SBATCH --job-name=build_per_gene_lr_panel_store
#SBATCH --output=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
#SBATCH --error=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=08:00:00
# CSD3/UoHPC variant (when it returns): --partition=icelake-himem
#   --account=FLOTO-PROJECT-K-SL2-CPU, logs → %x_%j.out/.err (repo-relative).

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

set -uo pipefail
# Data root + env — cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
: "${BACPREDICT_DATA_ROOT:="$SCRATCHDIR"}"
D="$BACPREDICT_DATA_ROOT"
PY="$SCRATCHDIR/envs/bacpredict-gpu-venv/bin/python"
export PYTHONPATH="${BACPREDICT_REPO:-$SCRATCHDIR/worktrees/consolidate}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
# Pin BLAS to 1 thread/process so the joblib workers (one per-gene LR each) don't oversubscribe;
# --n-jobs tracks the SLURM core allocation below, so `sbatch --cpus-per-task=N` scales cleanly
# (32 schedules far faster than a whole node). The ~21k LR fits (3,500 genes x 6) run in a few
# minutes; the wall is dominated by the sequential .pt I/O passes (assembly + panel, ~12 min).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

RDS=$D/processed/train_tb_ast
# per_segment_lr reads a <Sample, ast_label, split{train,validate,holdout}> table (splits.load_splits);
# the legacy tb_rif_1000 manifest is a DIFFERENT schema — regenerate/convert it to that layout before running.
SPLIT_TABLE="${SPLIT_TABLE:-$RDS/tb_surprisal_panel/tb_rif_1000_split.csv}"   # 1000-genome panel manifest
PARQUET=$RDS/protein_sequences
EMB=$RDS/esm
OUT=$RDS/tb_per_gene_lr_panel

echo "Per-gene LR panel store build — split=$SPLIT_TABLE  out=$OUT"
mkdir -p "$OUT"

"$PY" -m bacpredict.engine.segment_amr_lr.per_segment_lr \
    --segment-type coding \
    --write-panels \
    --split-table "$SPLIT_TABLE" \
    --drug rifampin \
    --parquet-dir "$PARQUET" \
    --embed-dir "$EMB" \
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

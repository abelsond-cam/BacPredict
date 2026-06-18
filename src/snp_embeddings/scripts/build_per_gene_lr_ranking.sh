#!/bin/bash
# Per-gene LR ranking — wide gene×drug table of "does this gene's own ESM-C vector predict resistance?"
#
# The substrate for top-k-gene concat (Workstream A): for every gene present in ≥10% of genomes
# (core + accessory, not just core), fit a stand-alone out-of-fold LogisticRegression on that gene's
# 960-d ESM-C protein vector → the drug label, and rank genes by their out-of-fold train AUROC. The
# top genes are the causal-gene candidates we concat onto the Bacformer mean (rpoB should top the
# rifampin ranking — the sanity check). Writes the wide `per_gene_lr_<drug>.csv`
# (gene_name, annotation, prevalence, lr_auroc_<drug>, n_train, n_pos, kept_filtered).
#
# This is NOT the panel-store build (build_per_gene_lr_panel_store.sh, the att_head input on the 1000
# manifest): here the universe is the FULL cohort split (binary_ast_with_split.csv), fitting is over a
# random class-balanced 2000-genome subsample (--max-train-genomes; full-cohort fit is I/O-heavy and a
# population correction is deferred), and NO per-sample panels are written (--write-panels omitted).
#
# Usage:  sbatch src/snp_embeddings/scripts/build_per_gene_lr_ranking.sh
#
#SBATCH --job-name=per_gene_lr_ranking
#SBATCH --output=per_gene_lr_ranking_%j.out
#SBATCH --error=per_gene_lr_ranking_%j.err
#SBATCH --partition=icelake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --account=FLOTO-SL2-CPU
#SBATCH --open-mode=append
# CPU-only (sklearn LRs over precomputed ESM-C vectors). At the 2000-genome subsample the in-memory
# footprint is ~30 GB (2000 genomes × ~4k proteins × 960 floats), so one 128 GB job fits all genes —
# no gene-sharding array needed at this scale. icelake-himem, 24 h budget (never under-call walltime).

cd /home/dca36/workspace/BacPredict

export PYTHONUNBUFFERED=1
# Pin BLAS to 1 thread/process so the joblib per-gene-LR workers don't oversubscribe; --n-jobs tracks
# the SLURM core allocation so `sbatch --cpus-per-task=N` scales cleanly.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

RDS=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_tb_ast
SHEET=$RDS/binary_ast_with_split.csv            # the FULL cohort split (not the 1000-genome manifest)
PARQUET=$RDS/tb_protein_sequences
EMB=$RDS/tb_esm_embeddings
OUT=$RDS/snp_embeddings/per_gene_lr_ranking

echo "========================================================================"
echo "Per-gene LR ranking — wide gene×drug table (rifampin)"
echo "Sheet:   $SHEET"
echo "Out dir: $OUT  (subsample 2000 train, min-prevalence 0.10, no panels)"
echo "Job ID:  $SLURM_JOB_ID"
echo "========================================================================"
mkdir -p "$OUT"

uv run python src/snp_embeddings/build_per_gene_lr_store.py \
    --split-csv "$SHEET" \
    --drug rifampin \
    --parquet-dir "$PARQUET" \
    --esm-store-dir "$EMB" \
    --out-dir "$OUT" \
    --min-prevalence 0.10 \
    --auroc-filter 0.8 \
    --n-folds 5 \
    --seed 1 \
    --max-train-genomes 2000 \
    --sample-seed 1 \
    --n-jobs "${SLURM_CPUS_PER_TASK:-32}"

echo "=== per_gene_lr_build_summary.json ==="
cat "$OUT/per_gene_lr_build_summary.json"
echo
echo "=== top of the wide ranking table (per_gene_lr_rifampin.csv) ==="
head -n 20 "$OUT/per_gene_lr_rifampin.csv"
echo
echo "Sanity: rpoB should sit at/near the top of the rifampin ranking."
echo "Per-gene LR ranking finished — table at $OUT/per_gene_lr_rifampin.csv"

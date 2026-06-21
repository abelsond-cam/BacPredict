#!/bin/bash
# Kp per-gene ESM-C LR ranking — "does this gene's own ESM-C vector predict resistance?"
#
# The Kp port of src/snp_embeddings/scripts/build_per_gene_lr_ranking.sh (same module,
# snp_embeddings.build_per_gene_lr_store; only the paths + drug list change). For every gene present in
# >=10% of genomes (core + accessory) it fits a stand-alone out-of-fold LogisticRegression on that gene's
# 960-d ESM-C protein vector -> the drug label, and ranks genes by their out-of-fold train AUROC. The top
# gene per drug is the causal-gene candidate the concat probe then concatenates onto the Bacformer mean.
#
# Drugs (one per array task) = the chromosomal/intrinsic regime where the per-gene story matters: the
# three where Bacformer beats the Kleborate ceiling (azithromycin, colistin, tetracycline) + ciprofloxacin
# as the chromosomal positive control (expect gyrA/parC to top its ranking — the catalogue already wins
# there, so the ranking should recover the same gene). Writes the wide per_gene_lr_<drug>.csv
# (gene_name, annotation, prevalence, lr_auroc_<drug>, n_train, n_pos, kept_filtered).
#
# Usage:  sbatch src/kleb_ast/scripts/build_per_gene_lr_ranking.sh
#
#SBATCH --job-name=kleb_per_gene_lr
#SBATCH --output=kleb_per_gene_lr_%A_%a.out
#SBATCH --error=kleb_per_gene_lr_%A_%a.err
#SBATCH --array=0-3
#SBATCH --partition=icelake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --open-mode=append
# CPU-only (sklearn LRs over precomputed ESM-C vectors). At the 2000-genome subsample the in-memory
# footprint is ~20 GB, so one 128 GB task fits all genes — no gene-sharding needed. The wall-time is
# I/O-bound (~20 min of sequential per-genome reads) + a fast parallel fit phase, so ~30 min/task; 16
# cores is ample (the fits are trivial, the reads are sequential). 6 h is a generous ceiling. Uses the
# project_k SL2-CPU account (personal FLOTO-SL2-CPU is nearly exhausted; project_k has ample budget).

cd /home/dca36/workspace/BacPredict

export PYTHONUNBUFFERED=1
# Pin BLAS to 1 thread/process so the joblib per-gene-LR workers don't oversubscribe.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

DRUGS=(azithromycin colistin tetracycline ciprofloxacin)
DRUG=${DRUGS[$SLURM_ARRAY_TASK_ID]}
if [[ -z "$DRUG" ]]; then
    echo "ERROR: no drug for array index $SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi

D=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed
SHEET=$D/train_kleb_ast/binary_ast_with_split.csv   # the FULL cohort split (all drug columns)
PARQUET=$D/klebsiella_protein_sequences
EMB=$D/klebsiella_esm_embeddings
# Per-drug subdir: the module also writes non-drug-specific files (build_summary, gene_lr_auroc,
# gene_prevalence), so concurrent array tasks must not share an out-dir or they race on those.
OUT=$D/train_kleb_ast/snp_embeddings/per_gene_lr_ranking/$DRUG

echo "========================================================================"
echo "Kp per-gene LR ranking — drug=$DRUG (array task $SLURM_ARRAY_TASK_ID)"
echo "Sheet:   $SHEET"
echo "Out dir: $OUT  (subsample 2000 train, min-prevalence 0.10, no panels)"
echo "Job ID:  ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "========================================================================"
mkdir -p "$OUT"

uv run python src/snp_embeddings/build_per_gene_lr_store.py \
    --split-csv "$SHEET" \
    --drug "$DRUG" \
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

echo "=== top of the wide ranking table (per_gene_lr_${DRUG}.csv) ==="
head -n 12 "$OUT/per_gene_lr_${DRUG}.csv"
echo "Kp per-gene LR ranking ($DRUG) finished — table at $OUT/per_gene_lr_${DRUG}.csv"

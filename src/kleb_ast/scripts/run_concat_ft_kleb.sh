#!/bin/bash
# Kp FT-concat probe (CPU) — FT Bacformer genome-mean (+) ESM-C top-gene vector -> LR (the deployable
# read-out). Runs AFTER cache_ft_bacformer_gene_embeddings.sh has saved the per-drug FT genome-mean NPZ;
# this step is CPU-only (loads that NPZ via --bacformer-vectors, marked --mean-is-finetuned).
#
# Leakage discipline: the FT backbone saw the TRAIN labels, so --kfold-on-eval-holdout restricts the
# k-fold to the canonical evaluate split (FT-unseen genomes) for an honest estimate. The injected gene is
# the unsupervised top-AUROC gene (--gene-from-ranking, imputed). Writes concat_ft_<drug>_<jobid>.json ->
# the FT-mean+ESM ladder rung (build_kleb_ladder --ft-concat-dir).
#
# Usage:  sbatch src/kleb_ast/scripts/run_concat_ft_kleb.sh
#
#SBATCH --job-name=kleb_concat_ft
#SBATCH --output=kleb_concat_ft_%A_%a.out
#SBATCH --error=kleb_concat_ft_%A_%a.err
#SBATCH --array=0-6
#SBATCH --partition=icelake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --open-mode=append
# CPU-only: the FT mean is loaded from the cached NPZ (not recomputed); only the pooled ESM-C gene reads
# touch disk. project_k SL2-CPU account.

cd /home/dca36/workspace/BacPredict
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

DRUGS=(ciprofloxacin levofloxacin colistin tetracycline azithromycin meropenem gentamicin)
DRUG=${DRUGS[$SLURM_ARRAY_TASK_ID]}
if [[ -z "$DRUG" ]]; then echo "ERROR: no drug for array index $SLURM_ARRAY_TASK_ID" >&2; exit 1; fi

D=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed
SHEET=$D/train_kleb_ast/binary_ast_with_split.csv
PARQUET=$D/klebsiella_protein_sequences
EMB=$D/klebsiella_esm_embeddings
FTNPZ=$D/train_kleb_ast/snp_embeddings/ft_bacformer_cache/$DRUG/ft_genome_mean_${DRUG}.npz
RANK=$D/train_kleb_ast/snp_embeddings/per_gene_lr_ranking_imputed/$DRUG/per_gene_lr_${DRUG}.csv
OUT=$D/train_kleb_ast/snp_embeddings/concat_ft/$DRUG
mkdir -p "$OUT"
if [[ ! -f "$FTNPZ" ]]; then echo "ERROR: FT mean NPZ missing: $FTNPZ (run cache_ft_bacformer_gene_embeddings.sh first)" >&2; exit 1; fi
if [[ ! -f "$RANK" ]]; then echo "ERROR: ranking CSV missing: $RANK" >&2; exit 1; fi

echo "=== Kp FT-concat (CPU) — drug=$DRUG (task $SLURM_ARRAY_TASK_ID) ==="
echo "ftnpz=$FTNPZ"; echo "rank=$RANK"

uv run python src/snp_embeddings/concatenate_bacformer_genome_esm_protein_emb.py \
    --ast-sheet-path "$SHEET" \
    --parquet-dir "$PARQUET" \
    --esm-store-dir "$EMB" \
    --output-json "$OUT/concat_ft_${DRUG}_${SLURM_ARRAY_JOB_ID}.json" \
    --drug "$DRUG" \
    --gene-from-ranking "$RANK" \
    --bacformer-vectors "$FTNPZ" --mean-is-finetuned \
    --qc-log "$OUT/gene_presence_qc_${DRUG}.log" \
    --pool-workers "${SLURM_CPUS_PER_TASK:-16}" \
    --kfold 5 --kfold-on-eval-holdout --seeds 1 2 3 --evaluate-seed 1 --evaluate-fraction 0.20

echo "Kp FT-concat ($DRUG) finished — JSON in $OUT"

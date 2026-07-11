#!/bin/bash
# Frozen concat ladder — k=5 × m=3 significance over the CACHED frozen-Bacformer mean (CPU, no GPU).
#
# The frozen base Bacformer never saw any AST label, so its genome-mean is a label-blind feature: we
# cache it once (the GPU forward, job 30632514) and re-split + re-fit the logistic regression for every
# (fold, seed) honestly — no leakage, no GPU. This is where the small top-of-ladder deltas live, so it
# is the run that earns the error bars (concat 0.975 vs ESM-gene 0.971 vs frozen-mean 0.788):
#   esm_gene_only / bacformer_mean_only / concat_esm_gene_plus_mean — each reported mean ± sd over
#   5 folds × 3 seeds, plus PAIRED per-run AUROC deltas (does concat reliably beat ESM-gene alone?).
# The single canonical-split headline (the 0.975) is also emitted alongside.
#
# (For the FINE-TUNED mean this cached-NPZ k-fold would be LEAKY — the backbone trained on the genomes'
#  labels; see run_concat_ft_mean.sh. Use it only on the frozen mean, as here.)
#
# Usage:  sbatch src/bacpredict/engine/scripts/run_concat_kfold_frozen.sh
#
#SBATCH --job-name=concat_kfold_frozen
#SBATCH --output=concat_kfold_frozen_%j.out
#SBATCH --error=concat_kfold_frozen_%j.err
#SBATCH --partition=icelake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --account=FLOTO-SL2-CPU
#SBATCH --open-mode=append
# CPU-only (cached mean): genotype ~38k parquets + mmap ESM-gene reads (parallel over the cores) + 3
# frames × (1 single-split + 15 k-fold) logistic-regression fits. icelake-himem, 24 h budget — never
# under-call walltime (charged on time used, not requested).

cd /home/dca36/workspace/BacPredict

export PYTHONUNBUFFERED=1

RDS=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_tb_ast
SHEET=$RDS/binary_ast_with_split.csv
PARQUET_DIR=$RDS/tb_protein_sequences
ESM_STORE_DIR=$RDS/tb_esm_embeddings
OUT_DIR=$RDS/pangena_predict/concat_rpob_mean
OUT_JSON=$OUT_DIR/concat_frozen_kfold_${SLURM_JOB_ID}.json
QC_LOG=$OUT_DIR/gene_presence_qc_${SLURM_JOB_ID}.log
BAC_NPZ=$OUT_DIR/frozen_bacformer_vectors_30632514.npz   # cached frozen rpoB-token + genome-mean (job 30632514)

if [[ ! -f "$BAC_NPZ" ]]; then
    echo "ERROR: cached frozen-mean NPZ not found at $BAC_NPZ" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "========================================================================"
echo "Frozen concat ladder — k=5 × m=3 significance (CPU, cached mean)"
echo "Cached NPZ:  $BAC_NPZ"
echo "Output JSON: $OUT_JSON"
echo "Job ID:      $SLURM_JOB_ID"
echo "========================================================================"

uv run python src/bacpredict/engine/concat/concatenate_bacformer_genome_esm_protein_emb.py \
    --ast-sheet-path "$SHEET" \
    --parquet-dir "$PARQUET_DIR" \
    --esm-store-dir "$ESM_STORE_DIR" \
    --output-json "$OUT_JSON" \
    --qc-log "$QC_LOG" \
    --drug rifampin \
    --device cpu \
    --bacformer-vectors "$BAC_NPZ" \
    --kfold 5 --seeds 1 2 3 \
    --pool-workers "${SLURM_CPUS_PER_TASK:-32}"

echo "Frozen concat k-fold finished — JSON at $OUT_JSON"

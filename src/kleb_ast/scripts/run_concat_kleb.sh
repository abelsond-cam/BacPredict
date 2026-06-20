#!/bin/bash
# Kp concat probe — ESM-C top-gene vector (auto-picked from the per-gene ranking) (+) Bacformer
# genome-mean -> logistic regression, scored on the canonical eval fold + a k-fold x m-seed harness.
#
# The Kp port of src/snp_embeddings/scripts/run_concat_kfold_frozen.sh (same module,
# snp_embeddings.concatenate_bacformer_genome_esm_protein_emb). CPU-only: the Bacformer genome-mean is
# loaded from the cached frozen NPZ (bacformer_frozen_genome_mean.npz, 6838 x 960) via --bacformer-vectors,
# and --gene-from-ranking reads the top out-of-fold-AUROC gene from each drug's per_gene_lr_<drug>.csv.
# One array task per drug, the same four as the ranking. Writes concat_frozen_<drug>_<jobid>.json (with a
# "kfold" block of per-frame mean +/- sd + paired deltas) — the substrate for the ladder.
#
# Prereqs: the per-gene ranking job (build_per_gene_lr_ranking.sh) must have written per_gene_lr_<drug>.csv,
# and the genome-mean NPZ must exist. Usage:  sbatch src/kleb_ast/scripts/run_concat_kleb.sh
#
#SBATCH --job-name=kleb_concat
#SBATCH --output=kleb_concat_%A_%a.out
#SBATCH --error=kleb_concat_%A_%a.err
#SBATCH --array=0-3
#SBATCH --partition=icelake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --account=FLOTO-SL2-CPU
#SBATCH --open-mode=append
# CPU-only (LR over precomputed vectors; the Bacformer mean is loaded from the NPZ, not recomputed). The
# only real I/O is the pooled ESM-C gene reads over the cohort (--pool-workers). 16 cores / 128 GB is
# ample; 12 h budget is generous for a sub-hour job.

cd /home/dca36/workspace/BacPredict
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

DRUGS=(azithromycin colistin tetracycline ciprofloxacin)
DRUG=${DRUGS[$SLURM_ARRAY_TASK_ID]}
if [[ -z "$DRUG" ]]; then
    echo "ERROR: no drug for array index $SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi

D=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed
SHEET=$D/train_kleb_ast/binary_ast_with_split.csv
PARQUET=$D/klebsiella_protein_sequences
EMB=$D/klebsiella_esm_embeddings
NPZ=$D/train_kleb_ast/bacformer_frozen_genome_mean.npz
RANK=$D/train_kleb_ast/snp_embeddings/per_gene_lr_ranking/$DRUG/per_gene_lr_${DRUG}.csv
OUT=$D/train_kleb_ast/snp_embeddings/concat/$DRUG
mkdir -p "$OUT"

if [[ ! -f "$RANK" ]]; then
    echo "ERROR: ranking CSV missing: $RANK (run build_per_gene_lr_ranking.sh first)" >&2
    exit 1
fi

echo "========================================================================"
echo "Kp concat probe — drug=$DRUG (array task $SLURM_ARRAY_TASK_ID)"
echo "Ranking: $RANK   NPZ: $NPZ"
echo "Out:     $OUT/concat_frozen_${DRUG}_${SLURM_ARRAY_JOB_ID}.json"
echo "========================================================================"

uv run python src/snp_embeddings/concatenate_bacformer_genome_esm_protein_emb.py \
    --ast-sheet-path "$SHEET" \
    --parquet-dir "$PARQUET" \
    --esm-store-dir "$EMB" \
    --output-json "$OUT/concat_frozen_${DRUG}_${SLURM_ARRAY_JOB_ID}.json" \
    --drug "$DRUG" \
    --gene-from-ranking "$RANK" \
    --bacformer-vectors "$NPZ" \
    --qc-log "$OUT/gene_presence_qc_${DRUG}.log" \
    --pool-workers "${SLURM_CPUS_PER_TASK:-16}" \
    --kfold 5 --seeds 1 2 3 --evaluate-seed 1 --evaluate-fraction 0.20

echo "Kp concat probe ($DRUG) finished — JSON in $OUT"

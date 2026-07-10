#!/bin/bash
# SNP-vs-ESM linear probes — where does the rpoB / rifampicin signal get lost?
#
# Default phase (this script): the two CPU steps —
#   Step 1  onehot_rrdr       one-hot RRDR codon genotype (the SNP ceiling)
#   Step 2  pooled_esmc_rpob  frozen ESM-C mean-pooled rpoB 960-vector
# The head-line read-out is AUROC(Step 1) - AUROC(Step 2) on the common evaluate
# set: the information ESM-C's residue->protein mean throws away. Pure CPU — it
# genotypes ~37k rpoB sequences from the protein parquets and mmap-reads one rpoB
# row out of each .pt, so it runs as a CPU sbatch job, NOT on the login node.
#
# Steps 3a (masked_marginal_llr) and 2b (bacformer_rpob_token) need a model
# forward — run them as the GPU variant at the bottom, once the CPU gap is in hand.
#
# Usage:  sbatch src/pangena_predict/scripts/run_snp_vs_esm_prediction.sh
#
#SBATCH --job-name=snp_vs_esm
#SBATCH --output=snp_vs_esm_%j.out
#SBATCH --error=snp_vs_esm_%j.err
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
# Memory is bounded to a few GB (subset-column parquet reads + periodic pyarrow
# pool release in rpob_genotype.py; mmap one-row .pt reads in the predictor).
# 240 G is generous headroom (fits a standard 256 G icelake node) per the
# never-under-call rule. Genotyping is sequential (~0.3 s/parquet -> ~4-5 h over
# ~37k); 12 h budget. --pool-workers parallelises the .pt reads across the cores.
#SBATCH --open-mode=append

cd /home/dca36/workspace/BacPredict

export PYTHONUNBUFFERED=1

# --- Data paths (TB AST cohort; the deployed model's canonical split) ----------
# binary_ast_with_split.csv is the SAME 70/10/20 holdout tb_ast/train_amr.py used,
# so the probe AUROCs sit in one table with Bacformer's deployed ~0.9. Sample-ID
# column is 'phenotype-BioSample_ID' (SAMEA... = parquet/.pt stems); the probe's
# resolve_holdouts() auto-detects it.
RDS=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_tb_ast
SHEET=$RDS/binary_ast_with_split.csv
PARQUET_DIR=$RDS/tb_protein_sequences
ESM_STORE_DIR=$RDS/tb_esm_embeddings
OUT_DIR=$RDS/pangena_predict
OUT_JSON=$OUT_DIR/snp_vs_esm_${SLURM_JOB_ID}.json
QC_LOG=$OUT_DIR/rpob_copy_qc_${SLURM_JOB_ID}.log

# Optional: the deployed Bacformer rifampin eval_results.json. If set and present,
# the probe asserts its split source/n_evaluate match and records the reference
# AUROC in the head-line. Leave empty to skip the reference block.
REF_JSON=""

mkdir -p "$OUT_DIR"

echo "========================================================================"
echo "SNP-vs-ESM probes (Steps 1 + 2; CPU)"
echo "Split sheet: $SHEET"
echo "Parquets:    $PARQUET_DIR"
echo "ESM store:   $ESM_STORE_DIR"
echo "Output JSON: $OUT_JSON"
echo "QC log:      $QC_LOG"
echo "Job ID:      $SLURM_JOB_ID"
echo "========================================================================"

REF_ARG=()
if [[ -n "$REF_JSON" && -f "$REF_JSON" ]]; then
    REF_ARG=(--reference-results-json "$REF_JSON")
    echo "Reference Bacformer results: $REF_JSON"
fi

uv run python src/pangena_predict/snp_vs_esm_prediction.py \
    --ast-sheet-path "$SHEET" \
    --parquet-dir "$PARQUET_DIR" \
    --esm-store-dir "$ESM_STORE_DIR" \
    --output-json "$OUT_JSON" \
    --qc-log "$QC_LOG" \
    --drug rifampin \
    --steps onehot_rrdr pooled_esmc_rpob \
    --pool-workers "${SLURM_CPUS_PER_TASK:-8}" \
    "${REF_ARG[@]}"

echo "SNP-vs-ESM probes finished — JSON at $OUT_JSON"

# --- GPU variant (Step 3a masked-marginal LLR + Step 2b Bacformer token) -------
# Step 3a re-runs ESM-C as a masked LM; Step 2b needs a precomputed Bacformer
# rpoB-token NPZ (build it first with bacformer_genome_vectors.py). Switch
# the directives to:
#   #SBATCH --partition=ampere
#   #SBATCH --account=FLOTO-SL2-GPU
#   #SBATCH --gres=gpu:1
#   #SBATCH --cpus-per-task=8
#   #SBATCH --mem=128G
#   #SBATCH --time=08:00:00
# and run (BAC_NPZ = output of bacformer_genome_vectors.py):
#
#   module load cuda/12.4 cudnn/8.9_cuda-12.4
#   uv run python src/pangena_predict/snp_vs_esm_prediction.py \
#       --ast-sheet-path "$SHEET" --parquet-dir "$PARQUET_DIR" \
#       --esm-store-dir "$ESM_STORE_DIR" --output-json "$OUT_JSON" \
#       --qc-log "$QC_LOG" --drug rifampin \
#       --steps onehot_rrdr pooled_esmc_rpob masked_marginal_llr bacformer_rpob_token \
#       --device cuda:0 --masked-marginal-codons panel \
#       --bacformer-vectors "$BAC_NPZ" "${REF_ARG[@]}"

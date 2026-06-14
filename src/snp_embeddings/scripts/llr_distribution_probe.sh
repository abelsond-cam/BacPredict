#!/bin/bash
# Phase 0 surprisal diagnostic (experiment 4) — proxy proof (0A) + per-protein flag (0B).
#
#   0A  whole-gene masked-vs-unmasked proxy proof on N=100 resistant isolates:
#       for each isolate's rpoB, score masked (ablation) AND unmasked (cheap) surprisal
#       at EVERY residue; report per-isolate Pearson/Spearman, the across-isolate scatter
#       at the mutated residue, % where the SNP is the top unmasked anomaly, and the
#       distinct-genotype count (so 'n=100' is honest). Publication-grade.
#   0B  per-protein 'a SNP is here' flag across ALL ~4,000 proteins of a handful of
#       genomes (3 resistant w/ distinct codons + 3 susceptible): a LIST of candidate
#       per-protein statistics (max-surprisal, hotspot-z, max-p99, top1-top2, ...) to a
#       parquet sidecar, plus where mutated rpoB ranks among the ~4,000 and what else
#       gets flagged.
#
# Read-only — no training, no embedding store; runs from the protein parquets + the
# pinned ESM-C MLM. GPU (forwards ESM-C as a masked/clm LM). The module preamble
# (purge -> cuda/12.4 -> cudnn) is the lesson from the 6-second crash.
#
# Usage:  sbatch src/snp_embeddings/scripts/llr_distribution_probe.sh 0a   # n=100 proxy proof
#         sbatch src/snp_embeddings/scripts/llr_distribution_probe.sh 0b   # cross-4000 flag
#
#SBATCH --job-name=llr_phase0
#SBATCH --output=llr_phase0_%j.out
#SBATCH --error=llr_phase0_%j.err
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --open-mode=append

cd /home/dca36/workspace/BacPredict

module purge
module load cuda/12.4
module load cudnn/8.9_cuda-12.4

export PYTHONUNBUFFERED=1

PHASE=${1:-0a}

# --- Data paths (TB AST cohort; the deployed model's canonical split) ----------
RDS=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_tb_ast
SHEET=$RDS/binary_ast_with_split.csv
PARQUET_DIR=$RDS/tb_protein_sequences
OUT_DIR=$RDS/snp_embeddings/llr_distribution_probe
OUT_JSON=$OUT_DIR/llr_phase0_${PHASE}_${SLURM_JOB_ID:-local}.json
mkdir -p "$OUT_DIR"

# Per-phase configuration.
if [[ "$PHASE" == "0a" ]]; then
    # n=100 proof: whole-gene masked, natural mutation distribution (no codon dedup).
    PHASE_ARGS=(--phase 0a --masked-scope gene --n-resistant 100 --n-wt 0
                --no-diverse-hotspots --pool-size 600)
elif [[ "$PHASE" == "0b" ]]; then
    # cross-4000 flag: 3 distinct-codon resistant + 3 susceptible, all proteins.
    PHASE_ARGS=(--phase 0b --n-resistant 3 --n-wt 3 --diverse-hotspots --pool-size 600)
else
    echo "Unknown phase '$PHASE' (expected 0a or 0b)"; exit 1
fi

echo "========================================================================"
echo "Phase 0 surprisal diagnostic — phase $PHASE"
echo "Split sheet: $SHEET"
echo "Parquets:    $PARQUET_DIR"
echo "Output JSON: $OUT_JSON"
echo "Args:        ${PHASE_ARGS[*]}"
echo "Job ID:      $SLURM_JOB_ID"
echo "========================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

uv run python src/snp_embeddings/llr_distribution_probe.py \
    --ast-sheet-path "$SHEET" \
    --parquet-dir "$PARQUET_DIR" \
    --output-json "$OUT_JSON" \
    --output-dir "$OUT_DIR" \
    --drug rifampin \
    --device cuda:0 \
    "${PHASE_ARGS[@]}"

echo "Phase 0 ($PHASE) finished — JSON + sidecars + plots in $OUT_DIR"

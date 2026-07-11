#!/bin/bash
# A.1.i — FT-mean concat: ESM-C rpoB (960) ⊕ *fine-tuned* Bacformer genome-mean (960) → LR, full eval.
#
# The done concat probe (run_concat_rpob_mean.sh) used the FROZEN base Bacformer mean (ladder 0.788)
# and still hit 0.975. A.1.i swaps in the genome-mean from the FINE-TUNED 0.905 mean-pool checkpoint
# (29776879): does fine-tuning the backbone add anything on top of ESM-rpoB? Three steps, one common
# evaluate set:
#   esm_rpob_only              frozen ESM-C mean-pooled rpoB 960-vector        (ladder ~0.971)
#   bacformer_mean_only        FINE-TUNED Bacformer genome-mean 960-vector     (expect ~0.905)
#   concat_esm_rpob_plus_mean  the two concatenated (1,920-d)                  the test
# The two ablations are the harness sanity check (mean_only must reproduce ~0.905 now, not 0.788).
# GPU: one fine-tuned-Bacformer forward per genome to build the mean (ESM-C inputs are precomputed —
# NOT re-embedded), ~0.3 s/genome → ~2-3 GPU-h over the genotyped cohort. Caches the FT rpoB-token +
# mean NPZ so the k-fold significance pass can rerun on CPU (see the tail comment).
#
# Usage:  sbatch src/bacpredict/engine/scripts/run_concat_ft_mean.sh
#
#SBATCH --job-name=concat_ft_mean
#SBATCH --output=concat_ft_mean_%j.out
#SBATCH --error=concat_ft_mean_%j.err
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --open-mode=append
# ~2-3 GPU-h estimated (genotype ~30k parquets + one FT-Bacformer forward each); 24 h budget — never
# under-call walltime (charged on time used, not requested). --pool-workers parallelises the ESM-C
# rpoB .pt reads across the 8 cores.

cd /home/dca36/workspace/BacPredict

export PYTHONUNBUFFERED=1
module purge
module load cuda/12.4
module load cudnn/8.9_cuda-12.4

RDS=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_tb_ast
SHEET=$RDS/binary_ast_with_split.csv
PARQUET_DIR=$RDS/tb_protein_sequences
ESM_STORE_DIR=$RDS/tb_esm_embeddings
OUT_DIR=$RDS/pangena_predict/concat_ft_mean
OUT_JSON=$OUT_DIR/concat_ft_mean_${SLURM_JOB_ID}.json
QC_LOG=$OUT_DIR/rpob_copy_qc_${SLURM_JOB_ID}.log
SAVE_NPZ=$OUT_DIR/finetuned_bacformer_vectors_${SLURM_JOB_ID}.npz   # cache FT rpoB-token + mean for the CPU k-fold rerun

# Deployed RIF mean-pool checkpoint (job 29776879, ~0.905). resolve_checkpoint_dir picks the best
# checkpoint-*/ subdir; glob the run dir so we need not hardcode the species prefix.
CKPT=$(ls -d "$RDS"/checkpoints/*rifampin_stage_c_29776879* 2>/dev/null | head -1)
if [[ -z "$CKPT" ]]; then
    echo "ERROR: could not find the rifampin_stage_c_29776879 checkpoint under $RDS/checkpoints/" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "========================================================================"
echo "A.1.i FT-mean concat (ESM-rpoB ⊕ fine-tuned Bacformer-mean → LR)"
echo "Checkpoint:  $CKPT"
echo "Split sheet: $SHEET"
echo "Output JSON: $OUT_JSON"
echo "FT NPZ cache: $SAVE_NPZ"
echo "Job ID:      $SLURM_JOB_ID"
echo "========================================================================"

uv run python src/bacpredict/engine/concat/concatenate_bacformer_genome_esm_protein_emb.py \
    --ast-sheet-path "$SHEET" \
    --parquet-dir "$PARQUET_DIR" \
    --esm-store-dir "$ESM_STORE_DIR" \
    --output-json "$OUT_JSON" \
    --qc-log "$QC_LOG" \
    --drug rifampin \
    --device cuda:0 \
    --pool-workers "${SLURM_CPUS_PER_TASK:-8}" \
    --bacformer-checkpoint "$CKPT" \
    --save-bacformer-vectors "$SAVE_NPZ"

echo "A.1.i FT-mean concat finished — JSON at $OUT_JSON"
echo
echo "This is the k=1, m=1 number on the CANONICAL evaluate holdout (the genomes the FT backbone was"
echo "held out from) — directly comparable to the deployed 0.905 and the frozen-concat 0.975."
echo
echo "Do NOT k-fold this cached FT NPZ across the whole cohort: the backbone was fine-tuned on the"
echo "original TRAIN labels, so re-splitting would put FT-training genomes into the new evaluate fold"
echo "(representation leakage → optimistic). Honest FT k-fold needs re-fine-tuning the backbone per"
echo "fold (GPU per fold). The cheap cached-NPZ k-fold is valid only for the FROZEN concat, where the"
echo "base model never saw any AST label (see run_concat_rpob_mean.sh --kfold)."

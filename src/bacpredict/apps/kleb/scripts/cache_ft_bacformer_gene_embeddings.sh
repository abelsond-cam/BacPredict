#!/bin/bash
# Kp FT Bacformer cache (GPU) — per drug, one fine-tuned forward saves the FT genome-mean AND the
# per-gene FT contextualised embeddings for the top-N (AUROC>0.6) genes of the ESM screen.
#
# This is the single expensive GPU pass. Downstream is CPU:
#   - FT-mean ⊕ ESM ladder rung  -> run_concat_ft_kleb.sh (loads ft_genome_mean_<drug>.npz)
#   - future multi-gene Bacformer concat -> gene_emb/<gene>.npz (top-gene FT tokens, carriers only)
#
# Usage:  sbatch src/kleb_ast/scripts/cache_ft_bacformer_gene_embeddings.sh
#
#SBATCH --job-name=kleb_ft_cache
#SBATCH --output=kleb_ft_cache_%A_%a.out
#SBATCH --error=kleb_ft_cache_%A_%a.err
#SBATCH --array=0-21
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --open-mode=append
# SL2-GPU (personal account; SL3-GPU never schedules). The FT forward over each drug's labelled genomes
# is the cost (~0.5 s/genome on GPU); --cpus-per-task=8 keeps the DataLoader feeding the GPU.

cd /home/dca36/workspace/BacPredict
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8

# All 22 drugs (each has a deployed FT checkpoint). The FT cache is useful per drug for the future
# multi-gene Bacformer concat, so cache the whole panel.
DRUGS=(cefotaxime ertapenem ampicillin-sulbactam ceftriaxone cefuroxime ciprofloxacin ceftazidime \
       gentamicin cefazolin imipenem meropenem trimethoprim-sulfamethoxazole tobramycin amikacin \
       levofloxacin piperacillin-tazobactam cefoxitin tetracycline aztreonam cefepime azithromycin colistin)
DRUG=${DRUGS[$SLURM_ARRAY_TASK_ID]}
if [[ -z "$DRUG" ]]; then echo "ERROR: no drug for array index $SLURM_ARRAY_TASK_ID" >&2; exit 1; fi

D=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed
CKPT=$D/train_kleb_ast/models/finetune/klebsiella_pneumoniae_${DRUG}_lr_0.00015_finetuned_fold00_seed1
RANK=$D/train_kleb_ast/pangena_predict/per_gene_lr_ranking_imputed/$DRUG/per_gene_lr_${DRUG}.csv
OUT=$D/train_kleb_ast/pangena_predict/ft_bacformer_cache/$DRUG
mkdir -p "$OUT"
if [[ ! -d "$CKPT" ]]; then echo "ERROR: FT checkpoint missing: $CKPT" >&2; exit 1; fi
if [[ ! -f "$RANK" ]]; then echo "ERROR: ranking CSV missing: $RANK" >&2; exit 1; fi

echo "=== Kp FT Bacformer cache — drug=$DRUG (task $SLURM_ARRAY_TASK_ID) ==="
echo "ckpt=$CKPT"; echo "rank=$RANK"; echo "out=$OUT"

uv run python src/kleb_ast/cache_ft_bacformer_gene_embeddings.py \
    --ast-sheet-path "$D/train_kleb_ast/binary_ast_with_split.csv" \
    --drug "$DRUG" \
    --parquet-dir "$D/klebsiella_protein_sequences" \
    --esm-store-dir "$D/klebsiella_esm_embeddings" \
    --bacformer-checkpoint "$CKPT" \
    --ranking-csv "$RANK" \
    --out-dir "$OUT" \
    --auroc-threshold 0.6 --top-n 50 --device cuda:0 --eval-only

echo "Kp FT Bacformer cache ($DRUG) finished — $OUT"

#!/bin/bash
# Kp FT Bacformer cache (GPU) — per drug, one fine-tuned forward saves the FT genome-mean AND the
# per-gene FT contextualised embeddings for the top-N (AUROC>0.6) genes of the ESM screen.
#
# This is the single expensive GPU pass. Downstream is CPU:
#   - FT-mean ⊕ ESM ladder rung  -> run_concat_ft_kleb.sh (loads ft_genome_mean_<drug>.npz)
#   - future multi-gene Bacformer concat -> gene_emb/<gene>.npz (top-gene FT tokens, carriers only)
#
# Usage:  sbatch src/bacpredict/apps/kleb/scripts/cache_ft_bacformer_gene_embeddings.sh
#
#SBATCH --job-name=kleb_ft_cache
#SBATCH --output=/scratch/u6fp/dca36.u6fp/logs/%x-%A_%a.out
#SBATCH --error=/scratch/u6fp/dca36.u6fp/logs/%x-%A_%a.out
#SBATCH --array=0-21
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --open-mode=append
# CSD3/UoHPC variant (when it returns): --partition=ampere --account=FLOTO-SL2-GPU,
#   logs → a project-tier logs dir, and `module load cuda/12.4 cudnn/8.9_cuda-12.4`.
# The FT forward over each drug's labelled genomes is the cost (~0.5 s/genome on GPU);
# --cpus-per-task=8 keeps the DataLoader feeding the GPU.

set -uo pipefail

# Data root — one env var, cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
: "${BACPREDICT_DATA_ROOT:="$SCRATCHDIR"}"
D="$BACPREDICT_DATA_ROOT"
PY="$SCRATCHDIR/envs/bacpredict-gpu-venv/bin/python"
# bacpredict is NOT pip-installed in the gpu-venv and $HOME/BacPredict may be on another agent's branch;
# point PYTHONPATH at this branch's worktree via BACPREDICT_REPO (memory isambard-ft-fanout-run-mechanics).
export PYTHONPATH="${BACPREDICT_REPO:-$HOME/BacPredict}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8

# All 22 drugs (each has a deployed FT checkpoint). The FT cache is useful per drug for the future
# multi-gene Bacformer concat, so cache the whole panel.
DRUGS=(cefotaxime ertapenem ampicillin-sulbactam ceftriaxone cefuroxime ciprofloxacin ceftazidime \
       gentamicin cefazolin imipenem meropenem trimethoprim-sulfamethoxazole tobramycin amikacin \
       levofloxacin piperacillin-tazobactam cefoxitin tetracycline aztreonam cefepime azithromycin colistin)
DRUG=${DRUGS[$SLURM_ARRAY_TASK_ID]}
if [[ -z "$DRUG" ]]; then echo "ERROR: no drug for array index $SLURM_ARRAY_TASK_ID" >&2; exit 1; fi

CKPT=$D/processed/train_kleb_ast/models/finetune/klebsiella_pneumoniae_${DRUG}_lr_0.00015_finetuned_fold00_seed1
RANK=$D/processed/train_kleb_ast/pangena_predict/per_gene_lr_ranking_imputed/$DRUG/per_gene_lr_${DRUG}.csv
OUT=$D/processed/train_kleb_ast/pangena_predict/ft_bacformer_cache/$DRUG
mkdir -p "$OUT"
if [[ ! -d "$CKPT" ]]; then echo "ERROR: FT checkpoint missing: $CKPT" >&2; exit 1; fi
if [[ ! -f "$RANK" ]]; then echo "ERROR: ranking CSV missing: $RANK" >&2; exit 1; fi

echo "=== Kp FT Bacformer cache — drug=$DRUG (task $SLURM_ARRAY_TASK_ID) ==="
echo "ckpt=$CKPT"; echo "rank=$RANK"; echo "out=$OUT"

"$PY" -m bacpredict.engine.concat.cache_bacformer_gene_embeddings \
    --ast-sheet-path "$D/processed/train_kleb_ast/binary_ast_with_split.csv" \
    --drug "$DRUG" \
    --parquet-dir "$D/processed/train_kleb_ast/protein_sequences" \
    --esm-store-dir "$D/processed/train_kleb_ast/esm" \
    --bacformer-checkpoint "$CKPT" \
    --ranking-csv "$RANK" \
    --out-dir "$OUT" \
    --auroc-threshold 0.6 --top-n 50 --device cuda:0 --eval-only

echo "Kp FT Bacformer cache ($DRUG) finished — $OUT"

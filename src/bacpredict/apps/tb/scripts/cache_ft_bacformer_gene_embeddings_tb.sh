#!/bin/bash
# TB FT Bacformer cache (GPU) — per drug, one fine-tuned forward saves the FT genome-mean AND the
# per-gene FT contextualised embeddings for the top-N (AUROC>0.6) genes of the per-gene screen.
#
# The TB sibling of apps/kleb/scripts/cache_ft_bacformer_gene_embeddings.sh. TB FTs land one drug at a
# time (rifampin done; the headroom drugs as their fan-out completes), so this is DRUG-overridable and
# single-job rather than a fixed drug array — fire one per drug as its checkpoint appears:
#
#   WT=/scratch/u6fp/dca36.u6fp/worktrees/concat
#   sbatch --job-name=tb_ft_cache_rifampin --export=ALL,BACPREDICT_REPO=$WT,DRUG=rifampin \
#     $WT/src/bacpredict/apps/tb/scripts/cache_ft_bacformer_gene_embeddings_tb.sh
#
# Downstream is CPU: the FT-mean ⊕ ESM/baclm ladder rung loads ft_genome_mean_<drug>.npz.
#
#SBATCH --job-name=tb_ft_cache
#SBATCH --output=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
#SBATCH --error=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
#SBATCH --time=08:00:00
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --open-mode=append
# CSD3/UoHPC variant (when it returns): --partition=ampere --account=FLOTO-SL2-GPU,
#   logs → a project-tier logs dir, and `module load cuda/12.4 cudnn/8.9_cuda-12.4`.
# The FT forward over each drug's evaluate-holdout is the cost (~0.5 s/genome on GPU);
# --cpus-per-task=8 keeps the DataLoader feeding the GPU. --eval-only keeps it to the FT-unseen split.

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

species=mycobacterium_tuberculosis
drug=${DRUG:-rifampin}  # US spelling (rifampin); override per drug as its FT lands via --export=ALL,...,DRUG=<drug>

CKPT=$D/processed/train_tb_ast/checkpoints/${species}_${drug}_lr_0.00015_finetuned_fold00_seed1
RANK=$D/processed/train_tb_ast/pangena_predict/per_gene_lr_ranking_baclm/${drug}/per_gene_lr_${drug}.csv
OUT=$D/processed/train_tb_ast/pangena_predict/ft_bacformer_cache/${drug}
mkdir -p "$OUT"
if [[ ! -d "$CKPT" ]]; then echo "ERROR: FT checkpoint missing: $CKPT" >&2; exit 1; fi
if [[ ! -f "$RANK" ]]; then echo "ERROR: ranking CSV missing: $RANK" >&2; exit 1; fi

echo "=== TB FT Bacformer cache — drug=$drug ==="
echo "ckpt=$CKPT"; echo "rank=$RANK"; echo "out=$OUT"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  GPU: $CUDA_VISIBLE_DEVICES"

"$PY" -m bacpredict.engine.concat.cache_bacformer_gene_embeddings \
    --ast-sheet-path "$D/processed/train_tb_ast/binary_ast_with_split.csv" \
    --drug "$drug" \
    --parquet-dir "$D/processed/train_tb_ast/protein_sequences" \
    --esm-store-dir "$D/processed/train_tb_ast/esm" \
    --bacformer-checkpoint "$CKPT" \
    --ranking-csv "$RANK" \
    --out-dir "$OUT" \
    --auroc-threshold 0.6 --top-n 50 --device cuda:0 --eval-only

echo "TB FT Bacformer cache ($drug) finished — $OUT"

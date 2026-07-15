#!/bin/bash
# Kp concat-panel: genome-mean ⊕ top-k FT/ESM gene panels — CPU array, one drug per task.
#
# For each drug: mean_only, mean⊕top-k FT genes (by ft_lr_auroc), mean⊕top-k ESM genes (by esm_lr_auroc),
# k in {1,3,5,10}, scored over the eval holdout with the same zero-imputed out-of-fold k-fold LR as the
# per-gene comparison -> concat_panel_<drug>.csv. No forward pass: FT mean + FT gene vectors from the
# cache, ESM gene vectors from the store. Needs esm_vs_ft_per_gene_<drug>.csv (the panel rankings), so
# submit this AFTER the ESM-vs-FT array (or with --dependency=afterok:<that_jobid>).
#
# Usage:  sbatch [--dependency=afterok:<JOBID>] concat_gene_panel_kleb.sh
#
#SBATCH --job-name=kleb_concat_panel
#SBATCH --output=/scratch/u6fp/dca36.u6fp/logs/%x-%A_%a.out
#SBATCH --error=/scratch/u6fp/dca36.u6fp/logs/%x-%A_%a.out
#SBATCH --array=0-21
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --open-mode=append
# CSD3/UoHPC variant (when it returns): --partition=icelake-himem --account=FLOTO-PROJECT-K-SL2-CPU,
#   logs → a project-tier logs dir (e.g. ~/rds/hpc-work/logs/%x-%A_%a.out).

set -uo pipefail

# Data root — one env var, cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
: "${BACPREDICT_DATA_ROOT:="$SCRATCHDIR"}"
D="$BACPREDICT_DATA_ROOT"
PY="$SCRATCHDIR/envs/bacpredict-gpu-venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

DRUGS=(cefotaxime ertapenem ampicillin-sulbactam ceftriaxone cefuroxime ciprofloxacin ceftazidime \
       gentamicin cefazolin imipenem meropenem trimethoprim-sulfamethoxazole tobramycin amikacin \
       levofloxacin piperacillin-tazobactam cefoxitin tetracycline aztreonam cefepime azithromycin colistin)
DRUG=${DRUGS[$SLURM_ARRAY_TASK_ID]}
if [[ -z "$DRUG" ]]; then echo "ERROR: no drug for array index $SLURM_ARRAY_TASK_ID" >&2; exit 1; fi

FTC=$D/processed/train_kleb_ast/pangena_predict/ft_bacformer_cache/$DRUG
CMP=$D/processed/train_kleb_ast/pangena_predict/esm_vs_ft_per_gene/$DRUG/esm_vs_ft_per_gene_${DRUG}.csv
OUT=$D/processed/train_kleb_ast/pangena_predict/concat_gene_panel/$DRUG
mkdir -p "$OUT"
if [[ ! -f "$CMP" ]]; then echo "ERROR: comparison CSV missing (run ESM-vs-FT array first): $CMP" >&2; exit 1; fi
if [[ ! -f "$FTC/ft_genome_mean_${DRUG}.npz" ]]; then echo "ERROR: FT mean npz missing: $FTC" >&2; exit 1; fi

echo "=== Kp concat-panel — drug=$DRUG (task $SLURM_ARRAY_TASK_ID) ==="

"$PY" -m bacpredict.engine.concat.concat_gene_panel \
    --ast-sheet-path "$D/processed/train_kleb_ast/binary_ast_with_split.csv" \
    --drug "$DRUG" \
    --parquet-dir "$D/processed/train_kleb_ast/protein_sequences" \
    --esm-store-dir "$D/processed/train_kleb_ast/esm" \
    --ft-cache-dir "$FTC" \
    --comparison-csv "$CMP" \
    --out-dir "$OUT" --panel-sizes 1 3 5 10 --n-folds 5 --seed 1

echo "Kp concat-panel ($DRUG) finished — $OUT"

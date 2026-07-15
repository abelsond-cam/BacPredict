#!/bin/bash
# Per-IGR LR ranking — wide IGR×drug table of "does this intergenic region's baclm embedding predict
# resistance?" The non-coding sibling of build_per_gene_lr_ranking.sh (same out-of-fold LR machinery,
# reused from build_per_gene_lr_store.fit_per_gene).
#
# For every core intergenic region — named by its ordered 5'->3' flanking-gene pair left_gene->right_gene
# (both flanks consistently-named gene= symbols), single-copy in >10% of genomes — fit a stand-alone
# out-of-fold LogisticRegression on the region's 960-d baclm non-coding embedding -> the drug label, and
# rank pairs by out-of-fold train AUROC. The top pair is the best-IGR block for the 3-way concat
# (bacformerFT-mean + baclm-best-gene + baclm-best-IGR). Writes the wide per_igr_lr_<drug>.csv
# (igr_pair, left_gene, right_gene, prevalence, lr_auroc_<drug>, n_train, n_pos, kept_filtered).
#
# SPECIES (env, default tb) selects tb|kp; store paths + input_csv (Sample->GFF) resolve from the shared
# organism config. Runs as a SLURM ARRAY, one drug per task. Each task assembles its own random
# class-balanced 2000-genome subsample over the FULL cohort split, fits the per-IGR LRs, writes the ranking.
#
# Usage:  sbatch --array=0 src/bacpredict/engine/scripts/build_per_igr_lr_ranking.sh          # TB rifampin
#         SPECIES=kp sbatch --export=ALL,SPECIES=kp --array=5 \
#             src/bacpredict/engine/scripts/build_per_igr_lr_ranking.sh                        # Kp ciprofloxacin
#
#SBATCH --job-name=per_igr_lr_ranking
#SBATCH --output=/scratch/u6fp/dca36.u6fp/logs/%x-%A_%a.out
#SBATCH --error=/scratch/u6fp/dca36.u6fp/logs/%x-%A_%a.out
#SBATCH --array=0-9
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
# CPU-only (GFF parse + sklearn LRs over precomputed baclm vectors). The GFF+.pt read sweep is serial
# (torch.load can't be forked on aarch64 before the process-parallel fit); the per-IGR fits fan out over
# --n-jobs. 24 h is a generous ceiling (never under-call walltime).

set -uo pipefail
# Data root + env — cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
: "${BACPREDICT_DATA_ROOT:="$SCRATCHDIR"}"
D="$BACPREDICT_DATA_ROOT"
PY="$SCRATCHDIR/envs/bacpredict-gpu-venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"

export PYTHONUNBUFFERED=1
# Pin BLAS to 1 thread/process so the joblib per-IGR-LR workers don't oversubscribe.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

SPECIES=${SPECIES:-tb}
if [[ "$SPECIES" == "kp" ]]; then
    TASK=kleb_ast
    DRUGS=(cefotaxime ertapenem ampicillin-sulbactam ceftriaxone cefuroxime ciprofloxacin ceftazidime \
           gentamicin cefazolin imipenem meropenem trimethoprim-sulfamethoxazole tobramycin amikacin \
           levofloxacin piperacillin-tazobactam cefoxitin tetracycline aztreonam cefepime azithromycin colistin)
else
    TASK=tb_ast
    DRUGS=(rifampin isoniazid ethambutol pyrazinamide moxifloxacin levofloxacin streptomycin ethionamide rifabutin kanamycin)
fi
DRUG=${DRUGS[$SLURM_ARRAY_TASK_ID]}
if [[ -z "$DRUG" ]]; then
    echo "ERROR: no drug for array index $SLURM_ARRAY_TASK_ID (species=$SPECIES)" >&2
    exit 1
fi

# Per-drug + per-species subdir (the module also writes non-drug-specific files, so concurrent array
# tasks must not share an out-dir).
OUT=$D/processed/train_${TASK}/pangena_predict/per_igr_lr_ranking/$DRUG

echo "========================================================================"
echo "Per-IGR LR ranking — species=$SPECIES drug=$DRUG (array task $SLURM_ARRAY_TASK_ID)"
echo "Out dir: $OUT  (subsample 2000 train, min-prevalence 0.10, boundary-tol 3)"
echo "Job ID:  ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "========================================================================"
# Idempotent: skip drugs whose ranking already exists (lets a full array backfill the panel).
if [[ -f "$OUT/per_igr_lr_${DRUG}.csv" ]]; then
    echo "Ranking already exists for $DRUG — skipping."
    exit 0
fi
mkdir -p "$OUT"

"$PY" -m bacpredict.engine.gene_lr.build_per_igr_lr_store \
    --species "$SPECIES" \
    --drug "$DRUG" \
    --out-dir "$OUT" \
    --min-prevalence 0.10 \
    --auroc-filter 0.8 \
    --n-folds 5 \
    --seed 1 \
    --max-train-genomes 2000 \
    --sample-seed 1 \
    --boundary-tol 3 \
    --n-jobs "${SLURM_CPUS_PER_TASK:-32}"

echo "=== top of the wide IGR ranking table (per_igr_lr_${DRUG}.csv) ==="
head -n 12 "$OUT/per_igr_lr_${DRUG}.csv"
echo "Per-IGR LR ranking ($SPECIES $DRUG) finished — table at $OUT/per_igr_lr_${DRUG}.csv"

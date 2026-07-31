#!/bin/bash
# Per-unit (named non-CDS body) LR ranking — the rRNA/element screen. Third sibling of
# build_per_igr_lr_ranking.sh (flank-pair key) and build_upstream_region_lr_ranking.sh (upstream:<gene>
# key); this one keys the baclm re-embed feature_* channel by <feature_type>:<feature_name> (rrna:rrs,
# rrna:rrl, regulatory_region:*, crispr:*), mean-pooling multi-copy bodies to one row per genome. It is
# the screen that tests rrs->streptomycin/kanamycin, rrl->azithromycin — determinants the synteny keys
# structurally cannot see (an rRNA gene is carved out of the whole-IGR run and never named as a flank).
#
# Named bodies live ONLY in the 2d re-embed store, so BACLM_DIR defaults to .../baclm_reembed (the legacy
# baclm/ store has no feature_* keys — the module would find 0 units there). No GFF/input-csv (units
# self-identify from the store), so this is lighter than the flank/upstream launchers.
#
# Usage:  sbatch --array=0-9 src/bacpredict/engine/scripts/build_per_unit_lr_ranking.sh              # TB, all 10
#         SPECIES=kp sbatch --export=ALL,SPECIES=kp --array=0-21 \
#             src/bacpredict/engine/scripts/build_per_unit_lr_ranking.sh                              # Kp, all 22
#   Absence modes (FEATURE, own out-dir each — needs the concat worktree until consolidate is advanced):
#         sbatch --export=ALL,REPO=$SCRATCHDIR/worktrees/concat,FEATURE=imputed --array=6,9 \
#             src/bacpredict/engine/scripts/build_per_unit_lr_ranking.sh                              # TB strep+kan zero-imputed
#         (FEATURE=presence for the one-hot lineage control.)
#   Held-out-test ("real numbers", full cohort — fit train+validate, score the evaluate split):
#         sbatch --export=ALL,EVAL=1,SUFFIX=_eval,MAX_TRAIN=,STORE_DTYPE=float16 --mem=400G --array=0-9 \
#             src/bacpredict/engine/scripts/build_per_unit_lr_ranking.sh                              # TB eval
#
#SBATCH --job-name=per_unit_lr_ranking
#SBATCH --output=/rds/user/dca36/hpc-work/logs/%x-%A_%a.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/%x-%A_%a.out
#SBATCH --array=0-9
#SBATCH --partition=icelake-himem
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
# CPU-only (.pt reads + sklearn LRs over precomputed baclm vectors). The named-body vocab is tiny (a
# handful of RNA/CRISPR/regulatory units), so this is fast; 24 h is a generous ceiling (never under-call).

set -uo pipefail
: "${BACPREDICT_DATA_ROOT:="$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/bac_ast_prediction"}"
D="$BACPREDICT_DATA_ROOT"
REPO="${REPO:-$HOME/workspace/BacPredict}"
PY="$HOME/workspace/BacPredict/.venv/bin/python"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
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

SUFFIX="${SUFFIX:-}"                              # output-subdir suffix, e.g. _full
MAX_TRAIN="${MAX_TRAIN-2000}"                    # "" -> full cohort
STORE_DTYPE="${STORE_DTYPE:-float32}"            # float16 -> whole-cohort memory
UNIT_TYPES="${UNIT_TYPES:-}"                     # e.g. "rrna ncrna" to restrict; default all named bodies
SPLITS=$D/processed/train_${TASK}/splits         # per-drug <drug>_split.csv tables (generate_kfold_splits)
# Named bodies only exist in the re-embed store; override BACLM_DIR only to point elsewhere.
BACLM_DIR="${BACLM_DIR:-$D/processed/train_${TASK}/baclm_reembed}"

# FEATURE (env, default embedding) selects the absence variant, each writing its OWN ranking base dir so
# the carrier file is never overwritten. Prevalence bands differ from the synteny siblings ON PURPOSE:
# rRNA determinants (rrs/rrl) are near-UBIQUITOUS with a point mutation, so the embedding + zero-imputed
# screens must KEEP prevalence 1.0 units (no 0.99 ceiling) — the mutation, not carriage, carries the
# signal. Only the presence one-hot caps at 0.99 (an all-ones column for a ubiquitous unit is chance).
EVAL="${EVAL:-0}"                                # vestigial: the holdout now comes from the split table; only the echo below reads it
FEATURE=${FEATURE:-embedding}
case "$FEATURE" in
    presence)
        RANK_BASE=per_unit_presence_lr_ranking
        TABLE=per_unit_presence_lr_${DRUG}.csv
        FEATURE_ARGS=(--feature presence --min-prevalence 0.01 --max-prevalence 0.99)
        ;;
    imputed)
        RANK_BASE=per_unit_lr_ranking_imputed
        TABLE=per_unit_lr_${DRUG}.csv
        FEATURE_ARGS=(--impute-absent-zero --min-prevalence 0.01 --max-prevalence 1.0)
        ;;
    *)
        RANK_BASE=per_unit_lr_ranking
        TABLE=per_unit_lr_${DRUG}.csv
        FEATURE_ARGS=(--min-prevalence 0.0 --max-prevalence 1.0)
        ;;
esac
OUT=$D/processed/train_${TASK}/pangena_predict/${RANK_BASE}${SUFFIX}/$DRUG

echo "========================================================================"
echo "Per-unit LR ranking — species=$SPECIES drug=$DRUG feature=$FEATURE (array task $SLURM_ARRAY_TASK_ID)"
echo "Out dir: $OUT  (eval=$EVAL max-train=${MAX_TRAIN:-full}, <type>:<name> body key, mean-pooled copies)"
echo "Store:   $BACLM_DIR  (re-embed feature_* channel)"
echo "Job ID:  ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "========================================================================"
if [[ -f "$OUT/$TABLE" ]]; then
    echo "Ranking already exists for $DRUG ($FEATURE$SUFFIX) — skipping."
    exit 0
fi
mkdir -p "$OUT"

MAXT_ARG=""; [[ -n "$MAX_TRAIN" ]] && MAXT_ARG="--max-train-genomes $MAX_TRAIN"
UNIT_ARG=""; [[ -n "$UNIT_TYPES" ]] && UNIT_ARG="--unit-types ${UNIT_TYPES//,/ }"
"$PY" -m bacpredict.engine.segment_amr_lr.per_segment_lr \
    --segment-type unit \
    --species "$SPECIES" \
    --split-table "$SPLITS/${DRUG}_split.csv" \
    --drug "$DRUG" \
    --baclm-dir "$BACLM_DIR" \
    --out-dir "$OUT" \
    "${FEATURE_ARGS[@]}" \
    $UNIT_ARG \
    --auroc-filter 0.8 \
    --n-folds 5 \
    --seed 1 \
    $MAXT_ARG \
    --sample-seed 1 \
    --store-dtype "$STORE_DTYPE" \
    --n-jobs "${SLURM_CPUS_PER_TASK:-32}"

echo "=== top of the per-unit ranking table ($TABLE) ==="
head -n 12 "$OUT/$TABLE"
echo "Per-unit LR ranking ($SPECIES $DRUG $FEATURE) finished — table at $OUT/$TABLE"

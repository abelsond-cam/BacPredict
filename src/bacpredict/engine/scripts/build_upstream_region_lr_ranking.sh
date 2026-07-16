#!/bin/bash
# Per-gene UPSTREAM-REGION LR ranking — the synteny-anchored non-coding screen. Sibling of
# build_per_igr_lr_ranking.sh, but each baclm non-coding region is named by the gene it sits immediately
# 5' of (upstream:<gene>) rather than by its two flanking genes. This recovers regulatory regions the
# flank-pair scheme drops when the far neighbour is an unnamed CDS — e.g. the mabA-inhA operon promoter
# (ethionamide/isoniazid -15), 5' of fabG1 next to an unnamed AbiEi antitoxin, which never appears in the
# per-IGR ranking despite being embedded. Proven on the stale store (no GPU): eth upstream:fabg1 0.80
# (catalogue inhA-promoter 0.826), inh 0.62 (cat 0.646).
#
# For every named gene, single-copy in the genome, take the region abutting its 5' end, fit a stand-alone
# out-of-fold LogisticRegression on its 960-d baclm embedding -> the drug label, rank by AUROC. Writes the
# wide per_upstream_lr_<drug>.csv (upstream_gene, gene, prevalence, lr_auroc_<drug>, n_train, n_pos,
# kept_filtered). Runs on the current (stale) store today; re-run on the re-embed for the whole_igr-vs-
# fragment comparison. SLURM ARRAY, one drug per task; store paths resolve from --species.
#
# Usage:  sbatch --array=0-9 src/bacpredict/engine/scripts/build_upstream_region_lr_ranking.sh          # TB, all 10
#         SPECIES=kp sbatch --export=ALL,SPECIES=kp --array=0-21 \
#             src/bacpredict/engine/scripts/build_upstream_region_lr_ranking.sh                          # Kp, all 22
#   Held-out-test ("real numbers", full cohort — fit train+validate, score the evaluate split):
#         sbatch --export=ALL,EVAL=1,SUFFIX=_eval,MAX_TRAIN= --mem=256G --array=0-9 \
#             src/bacpredict/engine/scripts/build_upstream_region_lr_ranking.sh                          # TB eval
#         (SPECIES=kp + --array=0-21 for Kp.)  BACLM_DIR=<...>/baclm_reembed for the re-embed store.
#
#SBATCH --job-name=upstream_lr_ranking
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
# (torch.load can't be forked on aarch64 before the process-parallel fit); the per-anchor fits fan out over
# --n-jobs. 24 h is a generous ceiling (never under-call walltime).

set -uo pipefail
# Data root + env — cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
: "${BACPREDICT_DATA_ROOT:="$SCRATCHDIR"}"
D="$BACPREDICT_DATA_ROOT"
# Code checkout: default to the Isambard worktree on the consolidated branch ($HOME/BacPredict is a
# stale `dev` checkout without the bacpredict package). Override with REPO=... for another checkout.
REPO="${REPO:-$SCRATCHDIR/worktrees/consolidate}"
PY="$SCRATCHDIR/envs/bacpredict-gpu-venv/bin/python"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

export PYTHONUNBUFFERED=1
# Pin BLAS to 1 thread/process so the joblib per-anchor-LR workers don't oversubscribe.
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

# Held-out-test ("real numbers") mode — EVAL=1 fits each anchor's LR on train+validate and reports
# eval_auroc_<drug> on the untouched evaluate split (vs the OOF-only 2000-subsample default). MAX_TRAIN=""
# drops the cap (full cohort). SUFFIX namespaces the output dir so the eval rankings sit beside the OOF ones
# (upstream_lr_ranking_eval/). BACLM_DIR overrides the store (e.g. …/baclm_reembed for the re-embed pass).
EVAL="${EVAL:-0}"                                 # 1 → --eval-holdout
SUFFIX="${SUFFIX:-}"                              # output-subdir suffix, e.g. _eval
MAX_TRAIN="${MAX_TRAIN:-2000}"                    # "" → full cohort
# Per-drug + per-species subdir (the module also writes non-drug-specific files, so concurrent array tasks
# must not share an out-dir).
OUT=$D/processed/train_${TASK}/pangena_predict/upstream_lr_ranking${SUFFIX}/$DRUG
TABLE=per_upstream_lr_${DRUG}.csv

echo "========================================================================"
echo "Upstream-region LR ranking — species=$SPECIES drug=$DRUG (array task $SLURM_ARRAY_TASK_ID)"
echo "Out dir: $OUT  (eval=$EVAL max-train=${MAX_TRAIN:-full}, upstream:<gene> anchoring, boundary-tol 3)"
echo "Store:   ${BACLM_DIR:-<store_paths default>}"
echo "Job ID:  ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "========================================================================"
# Idempotent: skip drugs whose ranking already exists (lets a full array backfill the panel).
if [[ -f "$OUT/$TABLE" ]]; then
    echo "Ranking already exists for $DRUG ($SUFFIX) — skipping."
    exit 0
fi
mkdir -p "$OUT"

EVAL_ARG=""; [[ "$EVAL" == 1 ]] && EVAL_ARG="--eval-holdout"
MAXT_ARG=""; [[ -n "$MAX_TRAIN" ]] && MAXT_ARG="--max-train-genomes $MAX_TRAIN"
BACLM_ARG=""; [[ -n "${BACLM_DIR:-}" ]] && BACLM_ARG="--baclm-dir $BACLM_DIR"
"$PY" -m bacpredict.engine.gene_lr.build_upstream_region_lr_store \
    --species "$SPECIES" \
    --drug "$DRUG" \
    --out-dir "$OUT" \
    --min-prevalence 0.10 \
    --auroc-filter 0.8 \
    --n-folds 5 \
    --seed 1 \
    $MAXT_ARG \
    --sample-seed 1 \
    --boundary-tol 3 \
    $BACLM_ARG \
    $EVAL_ARG \
    --n-jobs "${SLURM_CPUS_PER_TASK:-32}"

echo "=== top of the wide upstream ranking table ($TABLE) ==="
head -n 12 "$OUT/$TABLE"
echo "Upstream-region LR ranking ($SPECIES $DRUG) finished — table at $OUT/$TABLE"

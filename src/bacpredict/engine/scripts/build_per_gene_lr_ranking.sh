#!/bin/bash
# Per-gene LR ranking — wide gene×drug table of "does this gene's own embedding predict resistance?"
#
# The substrate for top-k-gene concat (Workstream A): for every gene present in ≥10% of genomes
# (core + accessory, not just core), fit a stand-alone out-of-fold LogisticRegression on that gene's
# 960-d protein vector → the drug label, and rank genes by their out-of-fold train AUROC. The
# top gene per drug is the causal-gene candidate we concat onto the Bacformer mean (rpoB topped the
# rifampin ranking at 0.962; auto-discovery should land each drug on its own causal gene — katG/inh,
# embB/emb, pncA/pza, gyrA/fluoroquinolone, rpsL/str). Writes the wide `per_gene_lr_<drug>.csv`
# (gene_name, annotation, prevalence, lr_auroc_<drug>, n_train, n_pos, kept_filtered).
#
# EMBEDDING_STORE (env, default esm) selects the store: `esm` (ESM-C) or `baclm` (baclm coding
# channel). Same discovery+ranking; only the reader/suffix differs (build_per_gene_lr_store
# --embedding-store). The baclm ranking is Phase 2's source for the single best baclm gene block.
# Output is namespaced by store so esm/baclm rankings never collide.
#
# Runs as a SLURM ARRAY, one drug per task — the 10 TB drugs that have a fine-tuned stage_c checkpoint
# (so each drug's concat refinement can reuse its own FT mean later). Each task is independent: assembles
# its own random class-balanced 2000-genome subsample (--max-train-genomes; full-cohort fit is I/O-heavy,
# population correction deferred) over the FULL cohort split, fits ~2k per-gene LRs, NO per-sample panels.
#
# Usage:  sbatch src/bacpredict/engine/scripts/build_per_gene_lr_ranking.sh                 # TB ESM (default)
#         EMBEDDING_STORE=baclm sbatch --export=ALL,EMBEDDING_STORE=baclm --array=0-9 \
#             src/bacpredict/engine/scripts/build_per_gene_lr_ranking.sh                     # TB baclm
#         EMBEDDING_STORE=baclm SPECIES=kp sbatch --export=ALL,EMBEDDING_STORE=baclm,SPECIES=kp --array=0-21 \
#             src/bacpredict/engine/scripts/build_per_gene_lr_ranking.sh                     # Kp baclm (22 drugs)
#   Imputed (--impute-absent-zero) — the ranking the concat gene rung SELECTS from (build_amr_ladder
#   hard-fails without it); writes per_gene_lr_ranking_imputed_<store>/:
#         EMBEDDING_STORE=baclm FEATURE=imputed sbatch --export=ALL,EMBEDDING_STORE=baclm,FEATURE=imputed \
#             --array=0-9 src/bacpredict/engine/scripts/build_per_gene_lr_ranking.sh         # TB baclm imputed
#         (SPECIES=kp + --array=0-21 for Kp.)
#   Held-out-test ("real numbers", full cohort — override --mem/--cpus for the ~34k-genome float16 fit):
#         sbatch --export=ALL,EMBEDDING_STORE=baclm,EVAL=1,SUFFIX=_eval,MAX_TRAIN=,STORE_DTYPE=float16 \
#             --mem=400G --cpus-per-task=96 --array=0-9 \
#             src/bacpredict/engine/scripts/build_per_gene_lr_ranking.sh                     # TB baclm eval
#         (SPECIES=kp + --array=0-21 for Kp.)
#
#SBATCH --job-name=per_gene_lr_ranking
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
# CSD3/UoHPC variant (when it returns): --partition=icelake-himem --account=FLOTO-SL2-CPU,
#   logs → per_gene_lr_ranking_%A_%a.out/.err (repo-relative).
# CPU-only (sklearn LRs over precomputed ESM-C vectors). At the 2000-genome subsample the in-memory
# footprint is ~30 GB, so one 128 GB task fits all genes — no gene-sharding needed. icelake-himem,
# 24 h budget (never under-call walltime). 10 tasks run in parallel → ~12 min wall.

set -uo pipefail
# Data root + env — cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
: "${BACPREDICT_DATA_ROOT:="$SCRATCHDIR"}"
D="$BACPREDICT_DATA_ROOT"
PY="$SCRATCHDIR/envs/bacpredict-gpu-venv/bin/python"
# Code checkout: default to the Isambard worktree on the consolidated branch ($HOME/BacPredict is a stale
# `dev` checkout without the bacpredict package). Override with REPO=... for another checkout.
REPO="${REPO:-$SCRATCHDIR/worktrees/consolidate}"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

export PYTHONUNBUFFERED=1
# Pin BLAS to 1 thread/process so the joblib per-gene-LR workers don't oversubscribe.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# SPECIES (env, default tb) selects tb|kp — same drug lists + store layout as build_per_igr_lr_ranking.sh.
# One drug per array index: TB has 10 (array 0-9), Kp has 22 (array 0-21).
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

STORE=${EMBEDDING_STORE:-esm}                    # esm (default) | baclm
RDS=$D/processed/train_${TASK}
SHEET=$RDS/binary_ast_with_split.csv            # the FULL cohort split (all drug columns)
PARQUET=$RDS/protein_sequences
EMB=$RDS/$STORE                                  # $RDS/esm or $RDS/baclm — same flat parquet order
# Held-out-test ("real numbers") mode — EVAL=1 fits each gene's LR on train+validate and reports
# eval_auroc_<drug> on the untouched evaluate split (vs the OOF-only 2000-subsample default). It also
# drops the train cap (full cohort) and stores the design matrices in float16 so the ~34k-genome
# collection fits one node (the LR still fits in float32). SUFFIX namespaces the output dir so the eval
# rankings sit beside — not on top of — the OOF ones (per_gene_lr_ranking_baclm_eval/).
EVAL="${EVAL:-0}"                                 # 1 → --eval-holdout (train+val fit, evaluate test)
SUFFIX="${SUFFIX:-}"                              # output-subdir suffix, e.g. _eval
MAX_TRAIN="${MAX_TRAIN-2000}"                    # "" → full cohort (no subsample cap)
STORE_DTYPE="${STORE_DTYPE:-float32}"            # float16 → whole-cohort memory
# FEATURE (env, default embedding) picks how non-carriers are handled. The concat's gene rung is
# zero-imputed at the head, so its selection ranking MUST be the imputed one — build_amr_ladder now
# HARD-FAILS on a carrier-only gene ranking (selection≠usage). `imputed` passes --impute-absent-zero
# (fit each gene over ALL read genomes, a 0-vector for non-carriers) and tags the output dir
# `per_gene_lr_ranking_imputed_<store>/` so it sits beside the carrier-only one; `embedding` (default)
# is the carrier-only drop-absent ranking (still used for the causal plot's carrier-only panel).
FEATURE="${FEATURE:-embedding}"                  # embedding (carrier-only) | imputed (--impute-absent-zero)
IMPUTE_ARG=""; MODE_TAG=""
if [[ "$FEATURE" == "imputed" ]]; then
    IMPUTE_ARG="--impute-absent-zero"; MODE_TAG="imputed_"
fi
# Per-drug + per-store subdir: the module also writes non-drug-specific files (build_summary,
# gene_lr_auroc, gene_prevalence), so concurrent array tasks must not share an out-dir (they'd race
# on those), and esm vs baclm (and carrier vs imputed vs eval) rankings must not overwrite each other.
OUT=$RDS/pangena_predict/per_gene_lr_ranking_${MODE_TAG}${STORE}${SUFFIX}/$DRUG

echo "========================================================================"
echo "Per-gene LR ranking — species=$SPECIES drug=$DRUG store=$STORE (array task $SLURM_ARRAY_TASK_ID)"
echo "Sheet:   $SHEET"
echo "Emb:     $EMB"
echo "Out dir: $OUT  (feature=$FEATURE eval=$EVAL max-train=${MAX_TRAIN:-full} dtype=$STORE_DTYPE, min-prevalence 0.10, no panels)"
echo "Job ID:  ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "========================================================================"
# Idempotent: skip drugs whose ranking already exists (lets a full array backfill the panel).
if [[ -f "$OUT/per_gene_lr_${DRUG}.csv" ]]; then
    echo "Ranking already exists for $DRUG ($SPECIES $STORE$SUFFIX) — skipping."
    exit 0
fi
mkdir -p "$OUT"

EVAL_ARG=""; [[ "$EVAL" == 1 ]] && EVAL_ARG="--eval-holdout"
MAXT_ARG=""; [[ -n "$MAX_TRAIN" ]] && MAXT_ARG="--max-train-genomes $MAX_TRAIN"
"$PY" -m bacpredict.engine.gene_lr.build_per_gene_lr_store \
    --split-csv "$SHEET" \
    --drug "$DRUG" \
    --parquet-dir "$PARQUET" \
    --esm-store-dir "$EMB" \
    --embedding-store "$STORE" \
    --out-dir "$OUT" \
    --min-prevalence 0.10 \
    --auroc-filter 0.8 \
    --n-folds 5 \
    --seed 1 \
    $MAXT_ARG \
    --sample-seed 1 \
    --store-dtype "$STORE_DTYPE" \
    $EVAL_ARG \
    $IMPUTE_ARG \
    --n-jobs "${SLURM_CPUS_PER_TASK:-32}"

echo "=== top of the wide ranking table (per_gene_lr_${DRUG}.csv) ==="
head -n 12 "$OUT/per_gene_lr_${DRUG}.csv"
echo "Per-gene LR ranking ($DRUG) finished — table at $OUT/per_gene_lr_${DRUG}.csv"

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
# Usage:  sbatch src/bacpredict/engine/scripts/build_per_gene_lr_ranking.sh                 # ESM (default)
#         EMBEDDING_STORE=baclm sbatch --export=ALL,EMBEDDING_STORE=baclm \
#             src/bacpredict/engine/scripts/build_per_gene_lr_ranking.sh                     # baclm
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
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"

export PYTHONUNBUFFERED=1
# Pin BLAS to 1 thread/process so the joblib per-gene-LR workers don't oversubscribe.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# The 10 TB drugs with a fine-tuned stage_c checkpoint (one drug per array index).
DRUGS=(rifampin isoniazid ethambutol pyrazinamide moxifloxacin levofloxacin streptomycin ethionamide rifabutin kanamycin)
DRUG=${DRUGS[$SLURM_ARRAY_TASK_ID]}
if [[ -z "$DRUG" ]]; then
    echo "ERROR: no drug for array index $SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi

STORE=${EMBEDDING_STORE:-esm}                    # esm (default) | baclm
RDS=$D/processed/train_tb_ast
SHEET=$RDS/binary_ast_with_split.csv            # the FULL cohort split (all 20 drug columns)
PARQUET=$RDS/protein_sequences
EMB=$RDS/$STORE                                  # $RDS/esm or $RDS/baclm — same flat parquet order
# Per-drug + per-store subdir: the module also writes non-drug-specific files (build_summary,
# gene_lr_auroc, gene_prevalence), so concurrent array tasks must not share an out-dir (they'd race
# on those), and esm vs baclm rankings must not overwrite each other.
OUT=$RDS/pangena_predict/per_gene_lr_ranking_$STORE/$DRUG

echo "========================================================================"
echo "Per-gene LR ranking — drug=$DRUG store=$STORE (array task $SLURM_ARRAY_TASK_ID)"
echo "Sheet:   $SHEET"
echo "Emb:     $EMB"
echo "Out dir: $OUT  (subsample 2000 train, min-prevalence 0.10, no panels)"
echo "Job ID:  ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "========================================================================"
mkdir -p "$OUT"

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
    --max-train-genomes 2000 \
    --sample-seed 1 \
    --n-jobs "${SLURM_CPUS_PER_TASK:-32}"

echo "=== top of the wide ranking table (per_gene_lr_${DRUG}.csv) ==="
head -n 12 "$OUT/per_gene_lr_${DRUG}.csv"
echo "Per-gene LR ranking ($DRUG) finished — table at $OUT/per_gene_lr_${DRUG}.csv"

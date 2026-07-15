#!/bin/bash
# Kp per-gene ESM-C LR ranking — ZERO-IMPUTED variant (don't drop absent genomes).
#
# Identical to build_per_gene_lr_ranking.sh but passes --impute-absent-zero: each gene is fit over ALL
# read genomes, with a 0xdim vector for genomes that lack it (instead of dropping them). This lets the LR
# use the presence/absence signal, so ACQUIRED genes (bla*, mph, tet, aac…) are no longer invisible —
# their ESM-LR should now approach the determinant one-hot. ~no change for universal genes (gyrA). Writes
# to per_gene_lr_ranking_imputed_<store>/<drug>/ so it sits beside the drop-absent ranking for comparison.
#
# EMBEDDING_STORE (env, default esm) selects esm (ESM-C) or baclm (baclm coding channel) — the shared
# build_per_gene_lr_store --embedding-store. The baclm ranking is Phase 2's source for the best baclm gene.
#
# Drugs (one per array task) = the chromosomal/intrinsic regime where the per-gene story matters: the
# three where Bacformer beats the Kleborate ceiling (azithromycin, colistin, tetracycline) + ciprofloxacin
# as the chromosomal positive control (expect gyrA/parC to top its ranking — the catalogue already wins
# there, so the ranking should recover the same gene). Writes the wide per_gene_lr_<drug>.csv
# (gene_name, annotation, prevalence, lr_auroc_<drug>, n_train, n_pos, kept_filtered).
#
# Usage:  sbatch src/bacpredict/apps/kleb/scripts/build_per_gene_lr_ranking_imputed.sh
#
#SBATCH --job-name=kleb_per_gene_lr_imp
#SBATCH --output=/scratch/u6fp/dca36.u6fp/logs/%x-%A_%a.out
#SBATCH --error=/scratch/u6fp/dca36.u6fp/logs/%x-%A_%a.out
#SBATCH --array=0-21
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --open-mode=append
# CSD3/UoHPC variant (when it returns): --partition=icelake-himem --account=FLOTO-PROJECT-K-SL2-CPU,
#   logs → a project-tier logs dir (e.g. ~/rds/hpc-work/logs/%x-%A_%a.out).
# CPU-only (sklearn LRs over precomputed ESM-C vectors). At the 2000-genome subsample the in-memory
# footprint is ~20 GB, so one 128 GB task fits all genes — no gene-sharding needed. The wall-time is
# I/O-bound (~20 min of sequential per-genome reads) + a fast parallel fit phase, so ~30 min/task; 16
# cores is ample (the fits are trivial, the reads are sequential). 6 h is a generous ceiling.

set -uo pipefail

# Data root — one env var, cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
: "${BACPREDICT_DATA_ROOT:="$SCRATCHDIR"}"
D="$BACPREDICT_DATA_ROOT"
PY="$SCRATCHDIR/envs/bacpredict-gpu-venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"

export PYTHONUNBUFFERED=1
# Pin BLAS to 1 thread/process so the joblib per-gene-LR workers don't oversubscribe.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Full 22-drug panel (the same set as eval_summary.csv). The per-gene ESM-LR ranking is computed for
# every drug so the causal-gene scorecard (where each one-hot determinant gene lands in the ESM ranking,
# and what it scores) can be drawn across the whole panel — the HGT-vs-chromosomal contrast.
DRUGS=(cefotaxime ertapenem ampicillin-sulbactam ceftriaxone cefuroxime ciprofloxacin ceftazidime \
       gentamicin cefazolin imipenem meropenem trimethoprim-sulfamethoxazole tobramycin amikacin \
       levofloxacin piperacillin-tazobactam cefoxitin tetracycline aztreonam cefepime azithromycin colistin)
DRUG=${DRUGS[$SLURM_ARRAY_TASK_ID]}
if [[ -z "$DRUG" ]]; then
    echo "ERROR: no drug for array index $SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi

STORE=${EMBEDDING_STORE:-esm}                                  # esm (default) | baclm
SHEET=$D/processed/train_kleb_ast/binary_ast_with_split.csv   # the FULL cohort split (all drug columns)
PARQUET=$D/processed/train_kleb_ast/protein_sequences
EMB=$D/processed/train_kleb_ast/$STORE                        # .../esm or .../baclm — same flat parquet order
# Per-drug + per-store subdir: the module also writes non-drug-specific files (build_summary,
# gene_lr_auroc, gene_prevalence), so concurrent array tasks must not share an out-dir (they'd race),
# and esm vs baclm rankings must not overwrite each other.
OUT=$D/processed/train_kleb_ast/pangena_predict/per_gene_lr_ranking_imputed_$STORE/$DRUG

echo "========================================================================"
echo "Kp per-gene LR ranking — drug=$DRUG store=$STORE (array task $SLURM_ARRAY_TASK_ID)"
echo "Sheet:   $SHEET"
echo "Emb:     $EMB"
echo "Out dir: $OUT  (subsample 2000 train, min-prevalence 0.10, no panels)"
echo "Job ID:  ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "========================================================================"
# Idempotent: skip drugs whose ranking already exists (lets the full 0-21 array backfill the panel
# without recomputing the four already done).
if [[ -f "$OUT/per_gene_lr_${DRUG}.csv" ]]; then
    echo "Ranking already exists for $DRUG — skipping."
    exit 0
fi
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
    --n-jobs "${SLURM_CPUS_PER_TASK:-32}" \
    --impute-absent-zero

echo "=== top of the wide ranking table (per_gene_lr_${DRUG}.csv) ==="
head -n 12 "$OUT/per_gene_lr_${DRUG}.csv"
echo "Kp per-gene LR ranking ($DRUG) finished — table at $OUT/per_gene_lr_${DRUG}.csv"

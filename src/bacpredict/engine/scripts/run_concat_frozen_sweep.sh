#!/bin/bash
# Frozen-concat generalisation sweep — does "causal gene ⊕ genome mean → LR" hold across the 10 TB drugs?
#
# The rifampin result (concat 0.975, beats ESM-rpoB 15/15) generalised to all 10 drugs at ZERO GPU cost:
# the FROZEN base Bacformer mean is drug-agnostic (one model for every drug) and already cached
# (frozen_bacformer_vectors_30632514.npz). So per drug we only: (1) auto-pick that drug's top-ranked gene
# from its per-gene LR ranking CSV (--gene-from-ranking), (2) concat its ESM-C vector onto the cached
# frozen mean, (3) honest whole-cohort k=5 × m=3 (frozen mean is label-blind → no leakage). All CPU.
#
# One drug per array task. Reads the ranking written by build_per_gene_lr_ranking.sh (launch that first;
# chain this with --dependency=afterok:<ranking_array_jobid>). The FINE-TUNED-mean refinement is a
# separate per-drug GPU pass (run_concat_ft_sweep.sh) — frozen ≈ FT for rifampin (0.975 vs 0.977), so
# this frozen sweep is the generalisation backbone and the FT pass is the refinement.
#
# Usage:  sbatch --dependency=afterok:<ranking_jobid> src/bacpredict/engine/scripts/run_concat_frozen_sweep.sh
#
#SBATCH --job-name=concat_frozen_sweep
#SBATCH --output=/scratch/u6fp/dca36.u6fp/logs/%x-%A_%a.out
#SBATCH --error=/scratch/u6fp/dca36.u6fp/logs/%x-%A_%a.out
#SBATCH --array=0-9
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --open-mode=append
# CSD3/UoHPC variant (when it returns): --partition=icelake-himem --account=FLOTO-SL2-CPU,
#   logs → concat_frozen_sweep_%A_%a.out/.err (repo-relative).
# CPU-only (cached frozen mean): per drug = genotype the top gene over ~38k parquets + mmap ESM reads +
# 3 frames × (1 single-split + 15 k-fold) LR fits. Observed runtime 30-70 min and I/O-bound (the parquet
# crawl dominates, so CPUs barely help). Modest 8-CPU/3-h request keeps the per-task CPU-minute reservation
# small (~1,440 vs 46,080) so it schedules under the account's AssocGrpCPUMinutes cap.

set -uo pipefail
# Data root + env — cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
: "${BACPREDICT_DATA_ROOT:="$SCRATCHDIR"}"
D="$BACPREDICT_DATA_ROOT"
PY="$SCRATCHDIR/envs/bacpredict-gpu-venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"

export PYTHONUNBUFFERED=1

DRUGS=(rifampin isoniazid ethambutol pyrazinamide moxifloxacin levofloxacin streptomycin ethionamide rifabutin kanamycin)
DRUG=${DRUGS[$SLURM_ARRAY_TASK_ID]}
if [[ -z "$DRUG" ]]; then
    echo "ERROR: no drug for array index $SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi

RDS=$D/processed/train_tb_ast
SHEET=$RDS/binary_ast_with_split.csv
PARQUET_DIR=$RDS/protein_sequences
ESM_STORE_DIR=$RDS/esm
RANKING=$RDS/pangena_predict/per_gene_lr_ranking/$DRUG/per_gene_lr_${DRUG}.csv
BAC_NPZ=$RDS/pangena_predict/concat_rpob_mean/frozen_bacformer_vectors_30632514.npz  # drug-agnostic frozen mean
OUT_DIR=$RDS/pangena_predict/concat_drug_sweep
OUT_JSON=$OUT_DIR/concat_frozen_${DRUG}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json
QC_LOG=$OUT_DIR/gene_presence_qc_${DRUG}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log

if [[ ! -f "$RANKING" ]]; then
    echo "ERROR: ranking CSV not found at $RANKING — run build_per_gene_lr_ranking.sh first." >&2
    exit 1
fi
if [[ ! -f "$BAC_NPZ" ]]; then
    echo "ERROR: cached frozen-mean NPZ not found at $BAC_NPZ" >&2
    exit 1
fi
mkdir -p "$OUT_DIR"

echo "========================================================================"
echo "Frozen-concat sweep — drug=$DRUG (array task $SLURM_ARRAY_TASK_ID)"
echo "Ranking:  $RANKING  (top gene auto-selected)"
echo "Frozen NPZ: $BAC_NPZ"
echo "Out JSON: $OUT_JSON"
echo "========================================================================"

"$PY" -m bacpredict.engine.segment_amr_lr.concat.concatenate_bacformer_genome_esm_protein_emb \
    --ast-sheet-path "$SHEET" \
    --parquet-dir "$PARQUET_DIR" \
    --esm-store-dir "$ESM_STORE_DIR" \
    --output-json "$OUT_JSON" \
    --qc-log "$QC_LOG" \
    --drug "$DRUG" \
    --gene-from-ranking "$RANKING" \
    --device cpu \
    --bacformer-vectors "$BAC_NPZ" \
    --kfold 5 --seeds 1 2 3 \
    --pool-workers "${SLURM_CPUS_PER_TASK:-32}"

echo "Frozen-concat sweep ($DRUG) finished — JSON at $OUT_JSON"

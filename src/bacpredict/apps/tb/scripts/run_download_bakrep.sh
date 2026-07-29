#!/bin/bash
#SBATCH --job-name=download_bakrep
#SBATCH --output=/rds/user/dca36/hpc-work/logs/%x-%j.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/%x-%j.out
#SBATCH --partition=icelake-himem
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --time=06:00:00
#SBATCH --mem=64G
# CPU-only — no --gres. Isambard schedules a GPU-less job on workq normally; --mem must be set
# explicitly (memory defaults are GPU-tied). CSD3/UoHPC variant (when it returns):
#   --partition=icelake-himem --account=FLOTO-SL2-CPU, logs → relative or ~/rds/hpc-work/logs/.

# =============================================================================
# Download BakRep Bakta GFF3 annotations for TB BioSamples
# =============================================================================
#
# Reads BioSample IDs (SAMN... / SAMEA...) from the TB AMR records CSV, batches
# them, and downloads `.bakta.gff3.gz` files from BakRep in parallel via the
# bakrep CLI.
#
# Transient network failures always leave a fraction of samples undownloaded
# on any single pass, so this runs the collect->download cycle in a loop:
# each pass only re-attempts samples still missing on disk (skip-existing),
# and the loop stops when a pass adds zero new files (remaining samples are
# genuinely unavailable in BakRep) or nothing is left to fetch. After the
# loop, a missing-samples sidecar TSV is written.
#
# This script never mutates the input CSV.
#
# Inputs:
#   --metadata path/to/ebi_tb_amr_records.csv (default below)
#
# Outputs (under OUTPUT_DIR = /raw/tb/gff):
#   - <BIOSAMPLE>/<BIOSAMPLE>.bakta.gff3.gz   (per-BioSample subdir, BakRep default)
#   - logs_<timestamp>/pass<NN>/batch_*.log
#   - download_summary_<timestamp>.txt
#   - missing_samples_<timestamp>.tsv
#
# Environment:
#   Reuses the `bakrep_download` micromamba env (pandas + bakrep CLI).
#
# Manual smoke test:
#   bash scripts/run_download_bakrep.sh --n 10 --batch-size 10
#
# Usage:
#   sbatch scripts/run_download_bakrep.sh [OPTIONS]
#   bash   scripts/run_download_bakrep.sh [OPTIONS]
#
# Options:
#   --metadata <path>          TB AMR records CSV (default below)
#   --n <number>               -1 = all (default), >0 = test subset
#   --batch-size <size>        BioSamples per bakrep batch (default: 100)
#   --max-passes <number>      Max collect->download retry passes (default: 10)
#   --overwrite-existing       Re-download even if files exist (default: skip)
# =============================================================================

set -uo pipefail

# Data root — one env var, cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
: "${BACPREDICT_DATA_ROOT:="$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/bac_ast_prediction"}"
D="$BACPREDICT_DATA_ROOT"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"
# NB: collect/download stages run inside the `bakrep_download` micromamba env, not the venv PY.

# Default values
N=-1
METADATA="$D/raw/tb/ebi_tb_amr_records.csv"
OUTPUT_DIR="$D/raw/tb/gff"
FILE_TYPE=gff3
NCORES=76
BATCH_SIZE=100
MAX_PASSES=10
SKIP_EXISTING=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --metadata)
            METADATA="$2"
            shift 2
            ;;
        --n)
            N="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-passes)
            MAX_PASSES="$2"
            shift 2
            ;;
        --overwrite-existing)
            SKIP_EXISTING=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --metadata <path>          : TB AMR records CSV (default: ${METADATA})"
            echo "  --n <number>               : Samples to process (-1=all, default)"
            echo "  --batch-size <size>        : Samples per batch (default: 100)"
            echo "  --max-passes <number>      : Max collect->download retry passes (default: 10)"
            echo "  --overwrite-existing       : Re-download even if files exist"
            exit 1
            ;;
    esac
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === run_download_bakrep.sh START ==="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === run_download_bakrep.sh START ===" >&2

echo "[$(date '+%Y-%m-%d %H:%M:%S')] PWD=$(pwd)" >&2

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Log root (one timestamped tree; one subdir per pass)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_ROOT="${OUTPUT_DIR}/logs_${TIMESTAMP}"
mkdir -p "$LOG_ROOT"

SKIP_ARG=""
[ "$SKIP_EXISTING" = false ] && SKIP_ARG="--no-skip-existing"

# Count .bakta.<filetype>.gz files currently on disk (convergence signal).
count_files() {
    find "$OUTPUT_DIR" -name "*.bakta.${FILE_TYPE}.gz" -type f 2>/dev/null | wc -l
}

# Function to download a batch of BioSamples using the bakrep CLI
download_batch() {
    local BATCH_FILE=$1
    local OUTPUT_DIR=$2
    local LOG_DIR=$3

    local BATCH_NAME
    BATCH_NAME=$(basename "$BATCH_FILE")
    local BATCH_LOG="${LOG_DIR}/${BATCH_NAME}.log"

    local SAMPLE_LIST
    SAMPLE_LIST=$(cat "$BATCH_FILE" | paste -sd,)
    local NUM_SAMPLES
    NUM_SAMPLES=$(cat "$BATCH_FILE" | wc -l)

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting batch: $BATCH_NAME ($NUM_SAMPLES samples)" | tee -a "$BATCH_LOG"
    echo "Samples: $SAMPLE_LIST" >> "$BATCH_LOG"

    if micromamba run -n bakrep_download bakrep download \
        -e "$SAMPLE_LIST" \
        -d "$OUTPUT_DIR" \
        -m "tool:bakta,filetype:${FILE_TYPE}" \
        >> "$BATCH_LOG" 2>&1; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: $BATCH_NAME" | tee -a "$BATCH_LOG"
        return 0
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: $BATCH_NAME" | tee -a "$BATCH_LOG"
        return 1
    fi
}

# Export function and variables for xargs
export -f download_batch
export OUTPUT_DIR
export FILE_TYPE

# -----------------------------------------------------------------------------
# Retry loop: collect (skip-existing) -> parallel download, until a pass adds
# no new files or there is nothing left to download.
# -----------------------------------------------------------------------------
LAST_PASS=0
for ((PASS=1; PASS<=MAX_PASSES; PASS++)); do
    LAST_PASS=$PASS
    echo ""
    echo "============================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] PASS ${PASS}/${MAX_PASSES}"
    echo "============================================"

    BATCH_DIR=$(mktemp -d)
    LOG_DIR=$(printf '%s/pass%02d' "$LOG_ROOT" "$PASS")
    mkdir -p "$LOG_DIR"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running Python collect (micromamba bakrep_download)..." >&2
    micromamba run -n bakrep_download python "$HOME/BacPredict/src/bacpredict/engine/download/scripts/collect_bakrep_samples.py" \
        --metadata "$METADATA" \
        --output-dir "$OUTPUT_DIR" \
        --filetype "$FILE_TYPE" \
        --n "$N" \
        $SKIP_ARG \
        --batch-dir "$BATCH_DIR" \
        --batch-size "$BATCH_SIZE"
    EXIT_COLLECT=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Python collect finished (exit $EXIT_COLLECT)." >&2

    if [[ $EXIT_COLLECT -ne 0 ]]; then
        echo "ERROR: collect_bakrep_samples.py failed (exit $EXIT_COLLECT)."
        rm -rf "$BATCH_DIR"
        exit $EXIT_COLLECT
    fi

    TOTAL=$(find "$BATCH_DIR" -name 'batch_*' -type f -exec cat {} + 2>/dev/null | wc -l)
    NUM_BATCHES=$(find "$BATCH_DIR" -name 'batch_*' -type f 2>/dev/null | wc -l)

    if [[ $TOTAL -eq 0 ]]; then
        echo "Pass ${PASS}: nothing left to download — converged."
        rm -rf "$BATCH_DIR"
        break
    fi

    BEFORE=$(count_files)
    echo "Pass ${PASS}: $TOTAL samples in $NUM_BATCHES batches (files on disk before: $BEFORE)"
    echo "Logs: $LOG_DIR"

    export LOG_DIR
    find "$BATCH_DIR" -name 'batch_*' -type f | sort -V | \
        xargs -I {} -P $NCORES bash -c 'download_batch "$1" "$2" "$3"' _ {} "$OUTPUT_DIR" "$LOG_DIR"
    PASS_EXIT=$?

    AFTER=$(count_files)
    ADDED=$((AFTER - BEFORE))
    rm -rf "$BATCH_DIR"

    echo "Pass ${PASS} complete: added $ADDED files (now $AFTER on disk; xargs exit $PASS_EXIT)"

    if [[ $ADDED -le 0 ]]; then
        echo "Pass ${PASS} added no new files — remaining samples likely unavailable in BakRep. Stopping."
        break
    fi
done

# Summary report
ON_DISK=$(count_files)
SUMMARY_FILE="${OUTPUT_DIR}/download_summary_${TIMESTAMP}.txt"
{
    echo "Download Summary - $(date)"
    echo "==========================================="
    echo ""
    echo "Passes run: $LAST_PASS (max $MAX_PASSES)"
    echo "Batch size: $BATCH_SIZE samples/batch"
    echo "Files on disk: $ON_DISK"
    echo ""
    echo "Per-pass logs: $LOG_ROOT/pass*/"
    echo "Successful batches (all passes): $(grep -rl "SUCCESS:" ${LOG_ROOT}/pass*/batch_*.log 2>/dev/null | wc -l)"
    echo "Failed batches (all passes): $(grep -rl "FAILED:" ${LOG_ROOT}/pass*/batch_*.log 2>/dev/null | wc -l)"
    echo ""
    echo "Log directory: $LOG_ROOT"
} > "$SUMMARY_FILE"

cat "$SUMMARY_FILE"

# Verify downloads and write the missing-samples sidecar TSV (never mutate the input CSV).
echo ""
echo "============================================"
echo "Verifying downloads and writing missing-samples sidecar..."
echo "============================================"

MISSING_OUTPUT="${OUTPUT_DIR}/missing_samples_${TIMESTAMP}.tsv"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running Python verify (micromamba bakrep_download)..." >&2
micromamba run -n bakrep_download python "$HOME/BacPredict/src/bacpredict/engine/download/scripts/collect_bakrep_samples.py" \
    --metadata "$METADATA" \
    --output-dir "$OUTPUT_DIR" \
    --filetype "$FILE_TYPE" \
    --verify \
    --missing-output "$MISSING_OUTPUT"

if [ $? -eq 0 ]; then
    echo "Missing-samples sidecar written to: $MISSING_OUTPUT"
else
    echo "Verification step failed."
fi

# Final summary
echo ""
echo "============================================"
echo "Download job completed at $(date)"
echo "Passes run: $LAST_PASS"
echo "Files on disk: $ON_DISK"
echo ""
echo "Summary report: $SUMMARY_FILE"
echo "Batch logs: $LOG_ROOT/"
echo "============================================"

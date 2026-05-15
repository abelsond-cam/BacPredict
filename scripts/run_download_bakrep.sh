#!/bin/bash
#SBATCH --job-name=download_bakrep
#SBATCH --output=download_bakrep_%j.out
#SBATCH --error=download_bakrep_%j.err
#SBATCH --partition=icelake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --time=02:00:00
#SBATCH --account=FLOTO-SL2-CPU

# =============================================================================
# Download BakRep Bakta GFF3 annotations for TB BioSamples
# =============================================================================
#
# Reads BioSample IDs (SAMN... / SAMEA...) from the TB AMR records CSV, batches
# them, and downloads `.bakta.gff3.gz` files from BakRep in parallel via the
# bakrep CLI. After downloads, emits a missing-samples sidecar TSV listing
# BioSamples for which no GFF3 was retrieved.
#
# This script never mutates the input CSV.
#
# Inputs:
#   --metadata path/to/ebi_tb_amr_records.csv (default below)
#
# Outputs (under OUTPUT_DIR = /raw/tb/gff):
#   - <BIOSAMPLE>/<BIOSAMPLE>.bakta.gff3.gz   (per-BioSample subdir, BakRep default)
#   - logs_<timestamp>/batch_*.log
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
#   --overwrite-existing       Re-download even if files exist (default: skip)
# =============================================================================

# Default values
N=-1
METADATA="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/raw/tb/ebi_tb_amr_records.csv"
OUTPUT_DIR="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/raw/tb/gff"
FILE_TYPE=gff3
NCORES=76
BATCH_SIZE=100
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
            echo "  --overwrite-existing       : Re-download even if files exist"
            exit 1
            ;;
    esac
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === run_download_bakrep.sh START ==="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === run_download_bakrep.sh START ===" >&2

# cd to project root so relative paths resolve consistently
cd /home/dca36/workspace/predict_kleb_by_bacformer
echo "[$(date '+%Y-%m-%d %H:%M:%S')] PWD=$(pwd)" >&2

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Create log and batch directories (same timestamp)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${OUTPUT_DIR}/logs_${TIMESTAMP}"
BATCH_DIR=$(mktemp -d)
mkdir -p "$LOG_DIR"

SKIP_ARG=""
[ "$SKIP_EXISTING" = false ] && SKIP_ARG="--no-skip-existing"

# Collect BioSample IDs and write batch files (uses bakrep_download env's pandas)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running Python collect (micromamba bakrep_download)..." >&2
micromamba run -n bakrep_download python scripts/collect_bakrep_samples.py \
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

# Derive TOTAL and NUM_BATCHES from batch files
TOTAL=$(find "$BATCH_DIR" -name 'batch_*' -type f -exec cat {} + 2>/dev/null | wc -l)
NUM_BATCHES=$(find "$BATCH_DIR" -name 'batch_*' -type f 2>/dev/null | wc -l)
echo "Final sample count for download: $TOTAL"
echo "Created $NUM_BATCHES batches"

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
export LOG_DIR
export FILE_TYPE

# Run batch downloads in parallel
echo ""
echo "Starting parallel batch download with $NCORES cores at $(date)"
echo "Processing $NUM_BATCHES batches ($TOTAL total samples)..."
echo "Logs will be saved to: $LOG_DIR"
echo ""

find "$BATCH_DIR" -name 'batch_*' -type f | sort -V | \
    xargs -I {} -P $NCORES bash -c 'download_batch "$1" "$2" "$3"' _ {} "$OUTPUT_DIR" "$LOG_DIR"

EXIT_CODE=$?

# Create summary report
SUMMARY_FILE="${OUTPUT_DIR}/download_summary_${TIMESTAMP}.txt"
{
    echo "Download Summary - $(date)"
    echo "==========================================="
    echo ""
    echo "Total samples: $TOTAL"
    echo "Total batches: $NUM_BATCHES"
    echo "Batch size: $BATCH_SIZE samples/batch"
    echo ""
    echo "Successful batches: $(grep -l "SUCCESS:" ${LOG_DIR}/batch_*.log 2>/dev/null | wc -l)"
    echo "Failed batches: $(grep -l "FAILED:" ${LOG_DIR}/batch_*.log 2>/dev/null | wc -l)"
    echo ""
    echo "Failed batches:"
    grep -l "FAILED:" ${LOG_DIR}/batch_*.log 2>/dev/null | xargs -I {} basename {} .log
    echo ""
    echo "Log directory: $LOG_DIR"
} > "$SUMMARY_FILE"

cat "$SUMMARY_FILE"

# Verify downloads and write the missing-samples sidecar TSV (never mutate the input CSV).
echo ""
echo "============================================"
echo "Verifying downloads and writing missing-samples sidecar..."
echo "============================================"

MISSING_OUTPUT="${OUTPUT_DIR}/missing_samples_${TIMESTAMP}.tsv"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running Python verify (micromamba bakrep_download)..." >&2
micromamba run -n bakrep_download python scripts/collect_bakrep_samples.py \
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

# Cleanup
rm -rf "$BATCH_DIR"

# Final summary
echo ""
echo "============================================"
echo "Download job completed at $(date)"
echo "Exit code: $EXIT_CODE"
echo ""
echo "Total samples: $TOTAL"
echo "Batches processed: $NUM_BATCHES"
echo ""
echo "Summary report: $SUMMARY_FILE"
echo "Batch logs: $LOG_DIR/"
echo "============================================"

exit $EXIT_CODE

#!/bin/bash
#SBATCH --job-name=ncbi_datasets_download
#SBATCH --output=ncbi_datasets_download_%j.out
#SBATCH --error=ncbi_datasets_download_%j.err
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --time=04:00:00
#SBATCH --account=FLOTO-SL2-CPU

# =============================================================================
# Download NCBI assemblies (FASTA) for TB BioSamples via the `datasets` CLI
# =============================================================================
#
# Reads BioSample IDs (SAMN... / SAMEA...) from the TB AMR records CSV,
# resolves each to a GenBank/RefSeq assembly accession via the NCBI datasets
# CLI, then downloads FASTA assemblies (--include genome) in parallel batches.
# The Bakta GFF3 annotations come from BakRep, not from NCBI - this script
# only fetches assemblies.
#
# Inputs:
#   --metadata path/to/ebi_tb_amr_records.csv (default below)
#
# Outputs (under OUTPUT_DIR = /raw/tb/assemblies):
#   - <ACCESSION>/<ACCESSION>_*_genomic.fna   (per-accession subdir from datasets CLI)
#   - biosample_to_accession_<timestamp>.tsv  (BioSample -> GCF_/GCA_ mapping cache)
#   - missing_samples_<timestamp>.tsv         (BioSamples with no resolvable assembly)
#   - logs_<timestamp>/batch_*.log
#   - ncbi_datasets_summary_<timestamp>.txt
#
# Environment:
#   Reuses the `ncbi-datasets` micromamba env (ncbi-datasets-cli + pandas).
#
# Manual smoke test:
#   bash scripts/download_ncbi_datasets.sh --n 10 --batch-size 10
#
# Usage:
#   sbatch scripts/download_ncbi_datasets.sh [OPTIONS]
#   bash   scripts/download_ncbi_datasets.sh [OPTIONS]
#
# Options:
#   --metadata <path>          TB AMR records CSV (default below)
#   --n <number>               -1 = all (default), >0 = test subset
#   --batch-size <size>        Accessions per batch (default: 100)
# =============================================================================

# Default values
N=-1
METADATA="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/raw/tb/ebi_tb_amr_records.csv"
OUTPUT_DIR="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/raw/tb/assemblies"
NCORES=76
BATCH_SIZE=100

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
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --metadata <path>     : TB AMR records CSV (default: ${METADATA})"
            echo "  --n <number>          : Accessions (10=test, -1=all; default: -1)"
            echo "  --batch-size <size>   : Accessions per batch (default: 100)"
            exit 1
            ;;
    esac
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === download_ncbi_datasets.sh START ==="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === download_ncbi_datasets.sh START ===" >&2

# cd to project root for consistent relative paths
cd /home/dca36/workspace/predict_kleb_by_bacformer
echo "[$(date '+%Y-%m-%d %H:%M:%S')] PWD=$(pwd)" >&2

# Create output directory and log/batch dirs
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${OUTPUT_DIR}/logs_${TIMESTAMP}"
BATCH_DIR=$(mktemp -d)
mkdir -p "$LOG_DIR"

# Sidecar paths (passed into collect script so it can write them)
ACCESSION_MAP="${OUTPUT_DIR}/biosample_to_accession_${TIMESTAMP}.tsv"
MISSING_OUTPUT="${OUTPUT_DIR}/missing_samples_${TIMESTAMP}.tsv"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Using metadata CSV: $METADATA" >&2
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Output directory:   $OUTPUT_DIR" >&2
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Batch directory:    $BATCH_DIR" >&2
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Log directory:      $LOG_DIR" >&2
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Accession map:      $ACCESSION_MAP" >&2
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Missing samples:    $MISSING_OUTPUT" >&2

# Resolve BioSamples -> assembly accessions and write batches
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running Python collect (micromamba ncbi-datasets)..." >&2
micromamba run -n ncbi-datasets python scripts/collect_ncbi_datasets_samples.py \
    --metadata "$METADATA" \
    --output-dir "$OUTPUT_DIR" \
    --accession-map "$ACCESSION_MAP" \
    --missing-output "$MISSING_OUTPUT" \
    --n "$N" \
    --batch-dir "$BATCH_DIR" \
    --batch-size "$BATCH_SIZE"
EXIT_COLLECT=$?
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Python collect finished (exit $EXIT_COLLECT)." >&2

if [[ $EXIT_COLLECT -ne 0 ]]; then
    echo "ERROR: collect_ncbi_datasets_samples.py failed (exit $EXIT_COLLECT)."
    rm -rf "$BATCH_DIR"
    exit $EXIT_COLLECT
fi

# Derive TOTAL and NUM_BATCHES from batch files
TOTAL=$(find "$BATCH_DIR" -name 'batch_*' -type f -exec cat {} + 2>/dev/null | wc -l)
NUM_BATCHES=$(find "$BATCH_DIR" -name 'batch_*' -type f 2>/dev/null | wc -l)
echo "Final accession count for download: $TOTAL"
echo "Created $NUM_BATCHES batches"

# Function to download a batch of accessions using the NCBI datasets CLI.
# Skips the batch entirely if every accession already has a populated dir
# under OUTPUT_DIR (resume behaviour).
download_batch() {
    local BATCH_FILE=$1
    local OUTPUT_DIR=$2
    local LOG_DIR=$3

    local BATCH_NAME
    BATCH_NAME=$(basename "$BATCH_FILE")
    local BATCH_LOG="${LOG_DIR}/${BATCH_NAME}.log"

    local NUM_ACCESSIONS
    NUM_ACCESSIONS=$(grep -c . "$BATCH_FILE" 2>/dev/null || echo 0)

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting batch: $BATCH_NAME ($NUM_ACCESSIONS accessions)" | tee -a "$BATCH_LOG"

    # Resume check: skip if every accession already has a non-empty dir
    local all_present=true
    while IFS= read -r ACC; do
        [[ -z "$ACC" ]] && continue
        if [[ ! -d "${OUTPUT_DIR}/${ACC}" ]] || [[ -z "$(ls -A "${OUTPUT_DIR}/${ACC}" 2>/dev/null)" ]]; then
            all_present=false
            break
        fi
    done < "$BATCH_FILE"
    if $all_present; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP: $BATCH_NAME (all accessions already present)" | tee -a "$BATCH_LOG"
        return 0
    fi

    # Download the whole batch in a single zip
    local TMP_DIR
    TMP_DIR=$(mktemp -d)
    local ZIP_FILE="${TMP_DIR}/${BATCH_NAME}.zip"

    if ! micromamba run -n ncbi-datasets datasets download genome accession \
        --inputfile "$BATCH_FILE" \
        --include genome \
        --filename "$ZIP_FILE" \
        >> "$BATCH_LOG" 2>&1; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED (datasets download): $BATCH_NAME" | tee -a "$BATCH_LOG"
        rm -rf "$TMP_DIR"
        return 1
    fi

    # Unzip and move each accession dir into OUTPUT_DIR
    if ! command -v unzip >/dev/null 2>&1; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: unzip not available" | tee -a "$BATCH_LOG"
        rm -rf "$TMP_DIR"
        return 1
    fi
    if ! unzip -o "$ZIP_FILE" -d "$TMP_DIR" >> "$BATCH_LOG" 2>&1; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED (unzip): $BATCH_NAME" | tee -a "$BATCH_LOG"
        rm -rf "$TMP_DIR"
        return 1
    fi

    local DATA_DIR="${TMP_DIR}/ncbi_dataset/data"
    if [[ ! -d "$DATA_DIR" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED (no ncbi_dataset/data in zip): $BATCH_NAME" | tee -a "$BATCH_LOG"
        rm -rf "$TMP_DIR"
        return 1
    fi

    local moved=0
    for ACC_DIR in "$DATA_DIR"/*/; do
        [[ -d "$ACC_DIR" ]] || continue
        local ACC
        ACC=$(basename "$ACC_DIR")
        mkdir -p "${OUTPUT_DIR}/${ACC}"
        # Move all files (assembly FASTA, etc.) into the target dir.
        mv "${ACC_DIR}"* "${OUTPUT_DIR}/${ACC}/" 2>>"$BATCH_LOG" && moved=$((moved + 1))
    done

    rm -rf "$TMP_DIR"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: $BATCH_NAME ($moved accession dirs moved)" | tee -a "$BATCH_LOG"
    return 0
}

# Export function and variables for xargs
export -f download_batch
export OUTPUT_DIR
export LOG_DIR

echo ""
echo "Starting parallel NCBI datasets download with $NCORES cores at $(date)"
echo "Processing $NUM_BATCHES batches ($TOTAL total accessions)..."
echo "Logs will be saved to: $LOG_DIR"
echo ""

find "$BATCH_DIR" -name 'batch_*' -type f | sort -V | \
    xargs -I {} -P $NCORES bash -c 'download_batch "$1" "$2" "$3"' _ {} "$OUTPUT_DIR" "$LOG_DIR"

EXIT_CODE=$?

# Create summary report
SUMMARY_FILE="${OUTPUT_DIR}/ncbi_datasets_summary_${TIMESTAMP}.txt"
{
    echo "NCBI datasets Download Summary - $(date)"
    echo "==========================================="
    echo ""
    echo "Total accessions: $TOTAL"
    echo "Total batches: $NUM_BATCHES"
    echo "Batch size: $BATCH_SIZE accessions/batch"
    echo ""
    echo "Successful batches: $(grep -l 'SUCCESS:' ${LOG_DIR}/batch_*.log 2>/dev/null | wc -l)"
    echo "Skipped batches:    $(grep -l 'SKIP:'    ${LOG_DIR}/batch_*.log 2>/dev/null | wc -l)"
    echo "Failed batches:     $(grep -l 'FAILED:'  ${LOG_DIR}/batch_*.log 2>/dev/null | wc -l)"
    echo ""
    echo "Accession map: $ACCESSION_MAP"
    echo "Missing samples sidecar: $MISSING_OUTPUT"
    echo "Log directory: $LOG_DIR"
} > "$SUMMARY_FILE"

cat "$SUMMARY_FILE"

# Cleanup
rm -rf "$BATCH_DIR"

echo ""
echo "============================================"
echo "NCBI datasets download job completed at $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================"

exit $EXIT_CODE

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
# Download NCBI genomes (GCF/GCA) using the `datasets` CLI for accessions
# listed in a metadata / missing-samples TSV.
#
# This script is intended as a companion to the BakRep gff3 download pipeline:
# - First run the BakRep pipeline to completion (gff3 mode).
# - Use its enriched missing-samples TSV output (with `Sample` column containing
#   GCF_/GCA_ accessions) as the --metadata input to this script.
# - This script will batch those accessions and download their GFF3 annotations
#   from NCBI using the `datasets` CLI.
#
# Environment: ncbi-datasets (micromamba)
# --------------------------------------
# Create and populate the environment manually with:
#
#   micromamba create -n ncbi-datasets python=3.11 -y
#   micromamba install -n ncbi-datasets -c bioconda -c conda-forge ncbi-datasets-cli pandas -y
#
# This script will then use:
#   micromamba run -n ncbi-datasets python ...
#   micromamba run -n ncbi-datasets datasets ...
#
# Usage:
#   sbatch download_ncbi_datasets.sh --metadata /path/to/missing_samples.tsv [OPTIONS]
#   bash   download_ncbi_datasets.sh --metadata /path/to/missing_samples.tsv [OPTIONS]
#
# Options:
#   --metadata <path>          Required. TSV with a `Sample` column containing
#                              GCF_/GCA_ accessions (e.g. BakRep missing-samples TSV).
#   --n <number>               Number of accessions to process.
#                              10 = test run, -1 = all (default: -1).
#   --batch-size <size>        Accessions per batch (default: 100).
#   --output-base <dir>        Base directory for downloads (default: raw root below).
#   --ncbi-dir-name <name>     Subdirectory under output-base (default: ncbi_gff3).
#
# Output layout:
#   OUTPUT_DIR = ${OUTPUT_BASE}/${ncbi-dir-name}
#   - One directory per accession: ${OUTPUT_DIR}/${ACCESSION}/
#   - Within each directory: contents of the datasets zip (including GFF3).
#   - Logs in: ${OUTPUT_DIR}/logs_<timestamp>/
#   - Summary report: ${OUTPUT_DIR}/ncbi_datasets_summary_<timestamp>.txt
# =============================================================================

# Default values
N=-1  # Process all accessions by default, use --n <number> for test subset
TSV_FILE="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/final/metadata_v2_all_samples_and_columns.tsv"
OUTPUT_BASE="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/raw"
NCBI_DIR_NAME="ncbi_gff3"
OUTPUT_DIR="${OUTPUT_BASE}/${NCBI_DIR_NAME}"
NCORES=76
BATCH_SIZE=100  # Number of accessions per batch download

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --metadata)
            TSV_FILE="$2"
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
        --output-base)
            OUTPUT_BASE="$2"
            OUTPUT_DIR="${OUTPUT_BASE}/${NCBI_DIR_NAME}"
            shift 2
            ;;
        --ncbi-dir-name)
            NCBI_DIR_NAME="$2"
            OUTPUT_DIR="${OUTPUT_BASE}/${NCBI_DIR_NAME}"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--metadata <metadata.tsv>] [OPTIONS]"
            echo "Options:"
            echo "  --metadata <path>          : Metadata TSV with 'Sample' and bakta_gff3_downloaded columns"
            echo "                                (default: ${TSV_FILE})"
            echo "  --n <number>               : Number of accessions (10=test, -1=all; default: -1)"
            echo "  --batch-size <size>        : Accessions per batch (default: 100)"
            echo "  --output-base <dir>        : Base output directory (default: ${OUTPUT_BASE})"
            echo "  --ncbi-dir-name <name>     : Subdirectory under output-base (default: ${NCBI_DIR_NAME})"
            exit 1
            ;;
    esac
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === download_ncbi_datasets.sh START ==="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === download_ncbi_datasets.sh START ===" >&2

# Change to project root for consistent relative paths
echo "[$(date '+%Y-%m-%d %H:%M:%S')] cd to project root..." >&2
cd /home/dca36/workspace/BacPredict
echo "[$(date '+%Y-%m-%d %H:%M:%S')] PWD=$(pwd)" >&2

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Create log and batch directories (same timestamp)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${OUTPUT_DIR}/logs_${TIMESTAMP}"
BATCH_DIR=$(mktemp -d)
mkdir -p "$LOG_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Using metadata TSV: $TSV_FILE" >&2
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Output directory: $OUTPUT_DIR" >&2
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Batch directory:  $BATCH_DIR" >&2
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Log directory:    $LOG_DIR" >&2

# Collect accessions and write batch files
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running Python collect (micromamba ncbi-datasets)..." >&2
micromamba run -n ncbi-datasets python /home/dca36/workspace/BacPredict/src/bacpredict/engine/download/scripts/collect_ncbi_datasets_samples.py \
    --metadata "$TSV_FILE" \
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

# Function to download a batch of accessions using NCBI datasets CLI
download_batch() {
    local BATCH_FILE=$1
    local OUTPUT_DIR=$2
    local LOG_DIR=$3

    local BATCH_NAME
    BATCH_NAME=$(basename "$BATCH_FILE")
    local BATCH_LOG="${LOG_DIR}/${BATCH_NAME}.log"

    local NUM_ACCESSIONS
    NUM_ACCESSIONS=$(cat "$BATCH_FILE" | wc -l)

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting batch: $BATCH_NAME ($NUM_ACCESSIONS accessions)" | tee -a "$BATCH_LOG"

    while IFS= read -r ACC; do
        [[ -z "$ACC" ]] && continue

        echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Downloading accession: $ACC" | tee -a "$BATCH_LOG"

        # Create a per-accession directory
        local ACC_DIR="${OUTPUT_DIR}/${ACC}"
        mkdir -p "$ACC_DIR"

        # Download zip for this accession
        local ZIP_FILE="${ACC_DIR}/${ACC}.zip"

        if micromamba run -n ncbi-datasets datasets download genome accession "$ACC" \
            --include gff3 \
            --filename "$ZIP_FILE" \
            >> "$BATCH_LOG" 2>&1; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')]   SUCCESS: $ACC (downloaded to $ZIP_FILE)" | tee -a "$BATCH_LOG"

            # Optionally unzip into the accession directory
            if command -v unzip >/dev/null 2>&1; then
                unzip -o "$ZIP_FILE" -d "$ACC_DIR" >> "$BATCH_LOG" 2>&1
                echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Unzipped: $ZIP_FILE -> $ACC_DIR" | tee -a "$BATCH_LOG"
            fi
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')]   FAILED: $ACC" | tee -a "$BATCH_LOG"
        fi
    done < "$BATCH_FILE"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished batch: $BATCH_NAME" | tee -a "$BATCH_LOG"
}

# Export function and variables for xargs
export -f download_batch
export OUTPUT_DIR
export LOG_DIR

# Run batch downloads in parallel
echo ""
echo "Starting parallel NCBI datasets download with $NCORES cores at $(date)"
echo "Processing $NUM_BATCHES batches ($TOTAL total accessions)..."
echo "Logs will be saved to: $LOG_DIR"
echo ""

find "$BATCH_DIR" -name 'batch_*' -type f | sort -V | \
    xargs -I {} -P $NCORES bash -c 'download_batch "$1" "$2" "$3"' _ {} "$OUTPUT_DIR" "$LOG_DIR"

EXIT_CODE=$?

# Create summary report
SUMMARY_FILE="${OUTPUT_DIR}/ncbi_datasets_summary_$(date +%Y%m%d_%H%M%S).txt"
echo "NCBI datasets Download Summary - $(date)" > "$SUMMARY_FILE"
echo "===========================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Total accessions: $TOTAL" >> "$SUMMARY_FILE"
echo "Total batches: $NUM_BATCHES" >> "$SUMMARY_FILE"
echo "Batch size: $BATCH_SIZE accessions/batch" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Log directory: $LOG_DIR" >> "$SUMMARY_FILE"

cat "$SUMMARY_FILE"

# Cleanup
rm -rf "$BATCH_DIR"

# Optionally flatten NCBI GFF3 downloads into a top-level directory of .gff files.
DEFAULT_NCBI_GFF_DIR="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/raw/ncbi_gff3"
if [ "$OUTPUT_DIR" = "$DEFAULT_NCBI_GFF_DIR" ]; then
    echo ""
    echo "============================================"
    echo "Flattening ncbi_gff3 directory (moving genomic.gff files to top level)..."
    echo "============================================"
    micromamba run -n ncbi-datasets python /home/dca36/workspace/BacPredict/src/bacpredict/engine/download/scripts/flatten_ncbi_gff3.py
else
    echo ""
    echo "NOTE: Skipping automatic flattening because OUTPUT_DIR ($OUTPUT_DIR)"
    echo "does not match the default ncbi_gff3 directory ($DEFAULT_NCBI_GFF_DIR)."
fi

# Final summary
echo ""
echo "============================================"
echo "NCBI datasets download job completed at $(date)"
echo "Exit code: $EXIT_CODE"
echo ""
echo "Total accessions: $TOTAL"
echo "Batches processed: $NUM_BATCHES"
echo ""
echo "Summary report: $SUMMARY_FILE"
echo "Batch logs: $LOG_DIR/"
echo "============================================"

exit $EXIT_CODE


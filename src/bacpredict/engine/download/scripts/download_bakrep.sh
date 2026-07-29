#!/bin/bash
#SBATCH --job-name=bakrep_download
#SBATCH --output=/rds/user/dca36/hpc-work/logs/%x-%j.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/%x-%j.out
#SBATCH --partition=icelake-himem
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --time=02:00:00
# CSD3/UoHPC variant (when it returns): --partition=icelake --account=FLOTO-SL2-CPU,
#   logs → bakrep_download_%j.out/.err (repo-relative).

# =============================================================================
# Download BakRep bakta-annotated files (GBFF or GFF3) for samples in metadata
# =============================================================================
#
# This script downloads bakta-annotated GBFF or GFF3 files from BakRep for all
# samples in the metadata TSV that have sample_accession starting with "SAM".
# It uses the bakrep CLI in parallel batches, and delegates collection/flag-update
# logic to slurm_scripts/collect_bakrep_samples.py (standalone, pandas only).
#
# File format: Default is filetype:gbff (.bakta.gbff.gz). Use --gff3 for
# filetype:gff3 (.bakta.gff3.gz) per https://github.com/ag-computational-bio/bakrep-cli
#
# Usage:
#   sbatch download_bakrep.sh [OPTIONS]
#   bash download_bakrep.sh [OPTIONS]
#
# Variables / Options:
#   --n <number>              Number of samples to process.
#                             10 = test run (default)
#                             -1 = all filtered samples
#   --batch-size <size>       Samples per bakrep batch (default: 100).
#                             Use 1 to retry failed samples individually.
#   --overwrite-existing      Re-download even if files exist (default: skip).
#   --gff3                    Download GFF3 format instead of GBFF (default: gbff).
#                             Output dir ends in klebsiella_gff3 when set.
#
# Metadata flag update always runs after downloads (no option to disable).
# Outputs:
#   - GBFF or GFF3 files in OUTPUT_DIR (klebsiella_gbff or klebsiella_gff3)
#   - Batch logs in OUTPUT_DIR/logs_<timestamp>/
#   - Summary report in OUTPUT_DIR/download_summary_<timestamp>.txt
#   - Missing sample IDs in OUTPUT_DIR/missing_samples_<timestamp>.txt (if any)
#
# =============================================================================

set -uo pipefail
# Data root — one env var, cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
: "${BACPREDICT_DATA_ROOT:="$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/bac_ast_prediction"}"
D="$BACPREDICT_DATA_ROOT"
PY="$HOME/workspace/BacPredict/.venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"

# Default values
N=-1  # Process all samples by default, use --n <number> to process a test subset
TSV_FILE="$D/final/metadata_v2_all_samples_and_columns.tsv"
OUTPUT_BASE="$D/raw"
FILE_TYPE=gbff  # gbff (default) or gff3
OUTPUT_DIR="${OUTPUT_BASE}/klebsiella_${FILE_TYPE}"
NCORES=76
BATCH_SIZE=100  # Number of samples per batch download
SKIP_EXISTING=true  # Default to skip existing files

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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
        --gff3)
            FILE_TYPE=gff3
            OUTPUT_DIR="${OUTPUT_BASE}/klebsiella_${FILE_TYPE}"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --n <number>              : Number of samples to process"
            echo "                              10 = test run (default)"
            echo "                              -1 = all samples"
            echo "  --batch-size <size>       : Samples per batch (default: 100)"
            echo "                              Use 1 to retry failed samples individually"
            echo "  --overwrite-existing      : Re-download even if files exist (default: skip)"
            echo "  --gff3                    : Download GFF3 instead of GBFF (output dir: klebsiella_gff3)"
            exit 1
            ;;
    esac
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === download_bakrep.sh START ==="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === download_bakrep.sh START ===" >&2

# Change to project root (PYTHONPATH covers imports; the standalone collect scripts run by abs path).
echo "[$(date '+%Y-%m-%d %H:%M:%S')] cd to project root..." >&2
cd "$HOME/BacPredict"
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

# Collect sample IDs and write batch files (standalone script, no uv - uses bakrep_download env)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running Python collect (micromamba bakrep_download)..." >&2
# Execute the python script inside micromamba environment named "bakrep_download", which has pandas installed.
# It is equivilent to micromamba activate bakrep_download and then running the script.
micromamba run -n bakrep_download python "$HOME/BacPredict/src/bacpredict/engine/download/scripts/collect_bakrep_samples.py" \
    --metadata "$TSV_FILE" \
    --n "$N" \
    --filetype "$FILE_TYPE" \
    $SKIP_ARG \
    --batch-dir "$BATCH_DIR" \
    --batch-size "$BATCH_SIZE"
EXIT_COLLECT=$?
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Python collect finished (exit $EXIT_COLLECT)." >&2

# Derive TOTAL and NUM_BATCHES from batch files
TOTAL=$(find "$BATCH_DIR" -name 'batch_*' -type f -exec cat {} + 2>/dev/null | wc -l)
NUM_BATCHES=$(find "$BATCH_DIR" -name 'batch_*' -type f 2>/dev/null | wc -l)
echo "Final sample count for download: $TOTAL"
echo "Created $NUM_BATCHES batches"

# Function to download a batch of samples using bakrep CLI
download_batch() {
    local BATCH_FILE=$1
    local OUTPUT_DIR=$2
    local LOG_DIR=$3
    
    local BATCH_NAME=$(basename "$BATCH_FILE")
    local BATCH_LOG="${LOG_DIR}/${BATCH_NAME}.log"
    
    # Create comma-delimited list from batch file
    local SAMPLE_LIST=$(cat "$BATCH_FILE" | paste -sd,)
    local NUM_SAMPLES=$(cat "$BATCH_FILE" | wc -l)
    
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

# Use xargs for parallel processing (sort for batch order)
find "$BATCH_DIR" -name 'batch_*' -type f | sort -V | \
    xargs -I {} -P $NCORES bash -c 'download_batch "$1" "$2" "$3"' _ {} "$OUTPUT_DIR" "$LOG_DIR"

EXIT_CODE=$?

# Create summary report
SUMMARY_FILE="${OUTPUT_DIR}/download_summary_$(date +%Y%m%d_%H%M%S).txt"
echo "Download Summary - $(date)" > "$SUMMARY_FILE"
echo "===========================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Total samples: $TOTAL" >> "$SUMMARY_FILE"
echo "Total batches: $NUM_BATCHES" >> "$SUMMARY_FILE"
echo "Batch size: $BATCH_SIZE samples/batch" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Successful batches: $(grep -l "SUCCESS:" ${LOG_DIR}/batch_*.log 2>/dev/null | wc -l)" >> "$SUMMARY_FILE"
echo "Failed batches: $(grep -l "FAILED:" ${LOG_DIR}/batch_*.log 2>/dev/null | wc -l)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Failed batches:" >> "$SUMMARY_FILE"
grep -l "FAILED:" ${LOG_DIR}/batch_*.log 2>/dev/null | xargs -I {} basename {} .log >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Log directory: $LOG_DIR" >> "$SUMMARY_FILE"

cat "$SUMMARY_FILE"

# Verify downloaded files and update metadata flags (always run)
echo ""
echo "============================================"
echo "Verifying downloads and updating metadata flags..."
echo "============================================"

MISSING_OUTPUT="${OUTPUT_DIR}/missing_samples_$(date +%Y%m%d_%H%M%S).txt"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running Python update-flags (micromamba bakrep_download)..." >&2
micromamba run -n bakrep_download python "$HOME/BacPredict/src/bacpredict/engine/download/scripts/collect_bakrep_samples.py" \
    --metadata "$TSV_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --filetype "$FILE_TYPE" \
    --missing-output "$MISSING_OUTPUT"

if [ $? -eq 0 ]; then
    echo "✓ Metadata flags updated successfully"
else
    echo "✗ Failed to update metadata flags"
fi

# Cleanup
rm -rf "$BATCH_DIR"

# Optionally flatten GFF3 downloads into top-level directory for easier access.
if [ "$FILE_TYPE" = "gff3" ]; then
    echo ""
    echo "============================================"
    echo "Flattening klebsiella_gff3 directory (moving *.bakta.gff3.gz to top level)..."
    echo "============================================"
    micromamba run -n bakrep_download python "$HOME/BacPredict/src/bacpredict/apps/kleb/scripts/flatten_klebsiella_gff3.py"
fi

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
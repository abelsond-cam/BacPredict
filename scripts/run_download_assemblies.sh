#!/bin/bash
#SBATCH --job-name=download_assemblies
#SBATCH --output=download_assemblies_%j.out
#SBATCH --error=download_assemblies_%j.err
#SBATCH --partition=icelake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --time=06:00:00
#SBATCH --account=FLOTO-SL2-CPU

# =============================================================================
# Download TB genome assemblies, ATB primary + NCBI fallback
# =============================================================================
#
# Stage 1: AllTheBacteria (ATB) v0.2 - https://allthebacteria.org/
#   - Short-read assemblies for ~2.7M public bacterial samples, keyed by
#     INSDC BioSample (SAM*). Direct per-sample HTTPS download from S3.
#
# Stage 2: NCBI (RefSeq + GenBank) - fallback for BioSamples ATB does not hold,
#   resolved via NCBI Entrez and downloaded with the `datasets` CLI. Useful
#   especially for long-read complete genomes that exist in RefSeq but not ATB.
#
# Each downloaded file is normalised to <OUTPUT_DIR>/<BIOSAMPLE>.fa.gz so
# downstream code can locate an assembly by BioSample alone regardless of source.
# A manifest TSV records the source for each BioSample.
#
# Transient network failures always leave a fraction of samples undownloaded
# on any single pass, so the plan->ATB->NCBI cycle runs in a loop: each pass
# re-plans only BioSamples still missing on disk (skip-existing) and the loop
# stops when a pass adds zero new .fa.gz files or nothing is left to fetch.
#
# Inputs:
#   --metadata path/to/ebi_tb_amr_records.csv (default below)
#
# Outputs (under OUTPUT_DIR = /raw/tb/assemblies):
#   - <BIOSAMPLE>.fa.gz                      (per-BioSample FASTA, flat layout)
#   - _atb_file_list.tsv.gz                  (cached ATB index; reused across runs)
#   - manifest_<timestamp>.tsv               BioSample -> source/filename
#   - biosample_to_accession_<timestamp>.tsv NCBI-only mapping
#   - missing_samples_<timestamp>.tsv        in neither ATB nor NCBI
#   - logs_<timestamp>/                      per-batch logs
#   - download_summary_<timestamp>.txt
#
# Environment:
#   Reuses the `ncbi-datasets` micromamba env (ncbi-datasets-cli + pandas).
#
# Usage:
#   sbatch scripts/run_download_assemblies.sh [OPTIONS]
#   bash   scripts/run_download_assemblies.sh [OPTIONS]
#
# Options:
#   --metadata <path>       TB AMR records CSV
#   --n <number>            -1 = all (default); >0 = test subset
#   --batch-size <size>     Items per batch (default: 100)
#   --max-passes <number>   Max plan->download retry passes (default: 10)
#   --skip-atb              Skip the ATB stage
#   --skip-ncbi             Skip the NCBI fallback stage
# =============================================================================

set -u

# Defaults
N=-1
METADATA="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/raw/tb/ebi_tb_amr_records.csv"
OUTPUT_DIR="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/raw/tb/assemblies"
NCORES=76
BATCH_SIZE=100
MAX_PASSES=10
SKIP_ATB=false
SKIP_NCBI=false

ATB_S3_BASE="https://allthebacteria-assemblies.s3.eu-west-2.amazonaws.com"

while [[ $# -gt 0 ]]; do
    case $1 in
        --metadata)   METADATA="$2"; shift 2 ;;
        --n)          N="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --max-passes) MAX_PASSES="$2"; shift 2 ;;
        --skip-atb)   SKIP_ATB=true; shift ;;
        --skip-ncbi)  SKIP_NCBI=true; shift ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [OPTIONS]"
            echo "  --metadata <path>   TB AMR records CSV (default: ${METADATA})"
            echo "  --n <number>        -1=all (default), >0=test subset"
            echo "  --batch-size <size> Items per batch (default: 100)"
            echo "  --max-passes <num>  Max plan->download retry passes (default: 10)"
            echo "  --skip-atb          Skip the ATB stage"
            echo "  --skip-ncbi         Skip the NCBI fallback stage"
            exit 1
            ;;
    esac
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === run_download_assemblies.sh START ===" >&2

cd /home/dca36/workspace/predict_kleb_by_bacformer
echo "[$(date '+%Y-%m-%d %H:%M:%S')] PWD=$(pwd)" >&2

mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_ROOT="${OUTPUT_DIR}/logs_${TIMESTAMP}"
mkdir -p "$LOG_ROOT"

# Canonical (latest) output paths; each pass also writes its own copies under
# LOG_ROOT/passNN.* and the last pass's are copied here at the end.
MANIFEST="${OUTPUT_DIR}/manifest_${TIMESTAMP}.tsv"
ACCESSION_MAP="${OUTPUT_DIR}/biosample_to_accession_${TIMESTAMP}.tsv"
MISSING_OUTPUT="${OUTPUT_DIR}/missing_samples_${TIMESTAMP}.tsv"

EXTRA_FLAGS=""
[ "$SKIP_ATB"  = true ] && EXTRA_FLAGS="$EXTRA_FLAGS --skip-atb"
[ "$SKIP_NCBI" = true ] && EXTRA_FLAGS="$EXTRA_FLAGS --skip-ncbi"

# Count <BIOSAMPLE>.fa.gz files currently on disk (convergence signal).
count_fa_gz() {
    find "$OUTPUT_DIR" -maxdepth 1 -name '*.fa.gz' -type f -size +0 2>/dev/null | wc -l
}

# ── ATB stage helpers ────────────────────────────────────────────────────────
# Each BioSample maps to one HTTPS URL on AWS S3. xargs parallelism over the
# per-BioSample curl calls (one process per sample) inside each batch.

atb_one() {
    local BS=$1
    local OUTPUT_DIR=$2
    local LOG=$3
    local TARGET="${OUTPUT_DIR}/${BS}.fa.gz"
    if [[ -s "$TARGET" ]]; then
        return 0
    fi
    if curl -sfL --max-time 300 \
        "${ATB_S3_BASE}/${BS}.fa.gz" \
        -o "${TARGET}.tmp" 2>>"$LOG"; then
        mv "${TARGET}.tmp" "$TARGET"
        return 0
    else
        rm -f "${TARGET}.tmp"
        echo "ATB miss: $BS" >> "$LOG"
        return 1
    fi
}
export -f atb_one
export ATB_S3_BASE

run_atb_batch() {
    local BATCH_FILE=$1
    local OUTPUT_DIR=$2
    local LOG_DIR=$3

    local BATCH_NAME
    BATCH_NAME=$(basename "$BATCH_FILE")
    local LOG="${LOG_DIR}/atb_${BATCH_NAME}.log"
    local COUNT
    COUNT=$(grep -c . "$BATCH_FILE" 2>/dev/null || echo 0)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ATB batch $BATCH_NAME ($COUNT samples)" | tee -a "$LOG"
    # xargs across samples within the batch (cap concurrency per batch since
    # the outer xargs is already parallelising across batches)
    cat "$BATCH_FILE" | xargs -I {} -P 8 bash -c 'atb_one "$1" "$2" "$3"' _ {} "$OUTPUT_DIR" "$LOG" || true
}
export -f run_atb_batch

# ── NCBI stage helpers ───────────────────────────────────────────────────────
# Per batch: `datasets download genome accession --inputfile X --include genome`
# yields one zip; we unzip, find each <ACC>/*_genomic.fna, look up the matching
# BioSample in the accession_map, gzip and rename to <BIOSAMPLE>.fa.gz.

run_ncbi_batch() {
    local BATCH_FILE=$1
    local OUTPUT_DIR=$2
    local LOG_DIR=$3
    local ACC_MAP=$4

    local BATCH_NAME
    BATCH_NAME=$(basename "$BATCH_FILE")
    local LOG="${LOG_DIR}/ncbi_${BATCH_NAME}.log"
    local COUNT
    COUNT=$(grep -c . "$BATCH_FILE" 2>/dev/null || echo 0)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] NCBI batch $BATCH_NAME ($COUNT accessions)" | tee -a "$LOG"

    # Resume: skip the batch if every accession in it already has its target
    # <BIOSAMPLE>.fa.gz on disk (look up BioSample in ACC_MAP for each accession)
    local all_present=true
    while IFS= read -r ACC; do
        [[ -z "$ACC" ]] && continue
        local BS
        BS=$(awk -F$'\t' -v acc="$ACC" 'NR>1 && $2 == acc {print $1; exit}' "$ACC_MAP")
        if [[ -z "$BS" || ! -s "${OUTPUT_DIR}/${BS}.fa.gz" ]]; then
            all_present=false
            break
        fi
    done < "$BATCH_FILE"
    if $all_present; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP $BATCH_NAME (all targets present)" | tee -a "$LOG"
        return 0
    fi

    local TMP
    TMP=$(mktemp -d)
    local ZIP="$TMP/${BATCH_NAME}.zip"

    if ! micromamba run -n ncbi-datasets datasets download genome accession \
        --inputfile "$BATCH_FILE" \
        --include genome \
        --filename "$ZIP" \
        >> "$LOG" 2>&1; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED (datasets download): $BATCH_NAME" | tee -a "$LOG"
        rm -rf "$TMP"
        return 1
    fi

    if ! command -v unzip >/dev/null 2>&1; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: unzip not available" | tee -a "$LOG"
        rm -rf "$TMP"
        return 1
    fi
    if ! unzip -o "$ZIP" -d "$TMP" >> "$LOG" 2>&1; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED (unzip): $BATCH_NAME" | tee -a "$LOG"
        rm -rf "$TMP"
        return 1
    fi

    local DATA_DIR="${TMP}/ncbi_dataset/data"
    if [[ ! -d "$DATA_DIR" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED (no ncbi_dataset/data in zip): $BATCH_NAME" | tee -a "$LOG"
        rm -rf "$TMP"
        return 1
    fi

    local moved=0
    for ACC_DIR in "$DATA_DIR"/*/; do
        [[ -d "$ACC_DIR" ]] || continue
        local ACC
        ACC=$(basename "$ACC_DIR")
        local BS
        BS=$(awk -F$'\t' -v acc="$ACC" 'NR>1 && $2 == acc {print $1; exit}' "$ACC_MAP")
        if [[ -z "$BS" ]]; then
            echo "WARN: no BioSample mapped to $ACC" >> "$LOG"
            continue
        fi
        local FNA
        FNA=$(find "$ACC_DIR" -name "*_genomic.fna" -type f | head -1)
        if [[ -z "$FNA" ]]; then
            echo "WARN: no *_genomic.fna under $ACC_DIR" >> "$LOG"
            continue
        fi
        gzip -c "$FNA" > "${OUTPUT_DIR}/${BS}.fa.gz" && moved=$((moved + 1))
    done

    rm -rf "$TMP"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: $BATCH_NAME ($moved samples written)" | tee -a "$LOG"
    return 0
}
export -f run_ncbi_batch

# -----------------------------------------------------------------------------
# Retry loop: plan (skip-existing) -> ATB stage -> NCBI stage, until a pass
# adds no new .fa.gz files or there is nothing left to download.
# -----------------------------------------------------------------------------
LAST_PASS=0
for ((PASS=1; PASS<=MAX_PASSES; PASS++)); do
    LAST_PASS=$PASS
    echo ""
    echo "============================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] PASS ${PASS}/${MAX_PASSES}"
    echo "============================================"

    PASS_TAG=$(printf 'pass%02d' "$PASS")
    LOG_DIR="${LOG_ROOT}/${PASS_TAG}"
    mkdir -p "$LOG_DIR"
    ATB_BATCH_DIR=$(mktemp -d)
    NCBI_BATCH_DIR=$(mktemp -d)

    PASS_MANIFEST="${LOG_ROOT}/${PASS_TAG}_manifest.tsv"
    PASS_ACCMAP="${LOG_ROOT}/${PASS_TAG}_biosample_to_accession.tsv"
    PASS_MISSING="${LOG_ROOT}/${PASS_TAG}_missing_samples.tsv"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Planning download (micromamba ncbi-datasets)..." >&2
    micromamba run -n ncbi-datasets python scripts/download_assemblies.py \
        --metadata "$METADATA" \
        --output-dir "$OUTPUT_DIR" \
        --atb-batch-dir "$ATB_BATCH_DIR" \
        --ncbi-batch-dir "$NCBI_BATCH_DIR" \
        --manifest "$PASS_MANIFEST" \
        --accession-map "$PASS_ACCMAP" \
        --missing-output "$PASS_MISSING" \
        --n "$N" \
        --batch-size "$BATCH_SIZE" \
        $EXTRA_FLAGS
    EXIT_PLAN=$?
    if [[ $EXIT_PLAN -ne 0 ]]; then
        echo "ERROR: download_assemblies.py failed (exit $EXIT_PLAN)."
        rm -rf "$ATB_BATCH_DIR" "$NCBI_BATCH_DIR"
        exit $EXIT_PLAN
    fi

    ATB_TOTAL=$(find "$ATB_BATCH_DIR"  -name 'batch_*' -exec cat {} + 2>/dev/null | wc -l)
    NCBI_TOTAL=$(find "$NCBI_BATCH_DIR" -name 'batch_*' -exec cat {} + 2>/dev/null | wc -l)
    ATB_BATCHES=$(find  "$ATB_BATCH_DIR"  -name 'batch_*' -type f 2>/dev/null | wc -l)
    NCBI_BATCHES=$(find "$NCBI_BATCH_DIR" -name 'batch_*' -type f 2>/dev/null | wc -l)
    echo "Pass ${PASS} planned: ATB=${ATB_TOTAL} BioSamples (${ATB_BATCHES} batches); NCBI=${NCBI_TOTAL} accessions (${NCBI_BATCHES} batches)"

    # Copy this pass's planning files to the canonical (latest) names.
    cp -f "$PASS_MANIFEST" "$MANIFEST"     2>/dev/null || true
    cp -f "$PASS_ACCMAP"   "$ACCESSION_MAP" 2>/dev/null || true
    cp -f "$PASS_MISSING"  "$MISSING_OUTPUT" 2>/dev/null || true

    if [[ "$ATB_BATCHES" -eq 0 && "$NCBI_BATCHES" -eq 0 ]]; then
        echo "Pass ${PASS}: nothing left to download — converged."
        rm -rf "$ATB_BATCH_DIR" "$NCBI_BATCH_DIR"
        break
    fi

    BEFORE=$(count_fa_gz)

    if [[ "$SKIP_ATB" != true && "$ATB_BATCHES" -gt 0 ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ATB stage ($ATB_BATCHES batches, $NCORES parallel)"
        find "$ATB_BATCH_DIR" -name 'batch_*' -type f | sort -V | \
            xargs -I {} -P "$NCORES" bash -c 'run_atb_batch "$1" "$2" "$3"' _ {} "$OUTPUT_DIR" "$LOG_DIR"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ATB stage complete"
    else
        echo "Skipping ATB stage (no batches or --skip-atb)"
    fi

    if [[ "$SKIP_NCBI" != true && "$NCBI_BATCHES" -gt 0 ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] NCBI stage ($NCBI_BATCHES batches, $NCORES parallel)"
        find "$NCBI_BATCH_DIR" -name 'batch_*' -type f | sort -V | \
            xargs -I {} -P "$NCORES" bash -c 'run_ncbi_batch "$1" "$2" "$3" "$4"' _ {} "$OUTPUT_DIR" "$LOG_DIR" "$PASS_ACCMAP"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] NCBI stage complete"
    else
        echo "Skipping NCBI stage (no batches or --skip-ncbi)"
    fi

    AFTER=$(count_fa_gz)
    ADDED=$((AFTER - BEFORE))
    rm -rf "$ATB_BATCH_DIR" "$NCBI_BATCH_DIR"

    echo "Pass ${PASS} complete: added $ADDED files (now $AFTER on disk)"
    if [[ $ADDED -le 0 ]]; then
        echo "Pass ${PASS} added no new files — remaining samples likely unavailable. Stopping."
        break
    fi
done

# ── Summary ──────────────────────────────────────────────────────────────────
SUMMARY="${OUTPUT_DIR}/download_summary_${TIMESTAMP}.txt"
FOUND_FA_GZ=$(count_fa_gz)
{
    echo "Download Summary - $(date)"
    echo "==========================================="
    echo ""
    echo "Passes run: $LAST_PASS (max $MAX_PASSES)"
    echo "Total <BIOSAMPLE>.fa.gz files in $OUTPUT_DIR: $FOUND_FA_GZ"
    echo ""
    echo "Manifest:        $MANIFEST"
    echo "Accession map:   $ACCESSION_MAP"
    echo "Missing sidecar: $MISSING_OUTPUT"
    echo "Per-pass logs:   $LOG_ROOT/pass*/"
} > "$SUMMARY"
cat "$SUMMARY"

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === run_download_assemblies.sh DONE ==="

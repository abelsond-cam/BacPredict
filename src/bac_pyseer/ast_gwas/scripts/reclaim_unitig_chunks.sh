#!/usr/bin/env bash
# One-off migration + reclaim for the unitig chunk tree, after the CHUNK_DIR keying fix.
#
# WHY. unitig_lmm_sharded_job.sh used to key CHUNK_DIR on $PAIR. run_drug.sh sets PAIR=$DRUG, so
# every AMR drug re-split the ONE shared unitigs.pyseer.gz into a private ~26.5 GB copy. Twelve
# copies filled /rds/user/dca36 and killed five prep jobs mid-write on 2026-08-21 ("Disk quota
# exceeded"). The job script now derives CHUNK_DIR from the matrix path, so all drugs sharing a
# matrix share one chunk set. This script migrates the surviving chunks to the new matrix-keyed
# names and removes what is now provably redundant.
#
# The duplication is VERIFIED, not assumed: decompressed chunk_07.gz is byte-identical across
# gentamicin / ciprofloxacin / meropenem / trimethoprim-sulfamethoxazole / ceftazidime
# (md5 3cee1b35e1750f5fad14ee0c6b9a4c94 on the first 150 MB), as is chunk_16.gz across
# gentamicin / colistin / ertapenem / ceftazidime. Compressed md5s differ only because gzip stamps
# mtime into its header.
#
# Usage:
#   bash src/bac_pyseer/ast_gwas/scripts/reclaim_unitig_chunks.sh            # dry run (default)
#   DRY_RUN=0 bash src/bac_pyseer/ast_gwas/scripts/reclaim_unitig_chunks.sh  # execute
#   DRY_RUN=0 SCRATCH_LOG_DAYS=14 bash ...                                   # also prune old job logs
set -euo pipefail

U=${U:-/home/dca36/rds/hpc-work/unitig_shards}
DRY_RUN=${DRY_RUN:-1}
SCRATCH_LOGS=${SCRATCH_LOGS:-/home/dca36/rds/hpc-work/pyseer_scratch}
SCRATCH_LOG_DAYS=${SCRATCH_LOG_DAYS:-0}   # 0 = leave job logs alone

# The two matrix-keyed destinations, spelled exactly as the patched job script derives them.
AMR_DST=$U/_matrix_bac_ast_prediction_processed_pyseer_ast_kp_unitigs
ISO_DST=$U/_matrix_processed_pyseer_iso_source_unitigs_blood_faeces

# KEEP one complete set per matrix and promote it; every other copy of the same matrix goes.
AMR_KEEP=gentamicin
AMR_DROP=(ceftazidime ciprofloxacin colistin ertapenem meropenem trimethoprim-sulfamethoxazole)
# Batch-2 casualties: 64 short files each, written until the tier filled. Useless, and the new
# -size +0 count would still treat them as present.
AMR_TRUNC=(amikacin aztreonam cefoxitin ceftriaxone piperacillin-tazobactam)

run () { if [ "$DRY_RUN" = "1" ]; then echo "  DRY  $*"; else echo "  RUN  $*"; eval "$@"; fi; }

gb () { find "$@" -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{printf "%.2f", s/1073741824}'; }

echo "=== unitig chunk reclaim   DRY_RUN=$DRY_RUN   $(date) ==="
if squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -qE '^(uprep|utask|ucomb)_'; then
    echo "REFUSING: a uprep/utask/ucomb job is in the queue — it may be reading these chunks." >&2
    exit 1
fi
echo "no unitig GWAS jobs in the queue"

echo
echo "--- (1) promote one complete set per matrix (mv = rename, instant, same filesystem) ---"
echo "    AMR : $AMR_KEEP  ($(gb "$U/$AMR_KEEP" -maxdepth 1 -name 'chunk_*.gz') GB) -> $(basename "$AMR_DST")"
run "mkdir -p '$AMR_DST'"
run "mv '$U/$AMR_KEEP'/chunk_*.gz '$AMR_DST'/"
echo "    ISO : blood_faeces ($(gb "$U/blood_faeces" -maxdepth 1 -name 'chunk_*.gz') GB) -> $(basename "$ISO_DST")"
echo "          preserved, not deleted: re-splitting this matrix is ~3 h of IO"
run "mkdir -p '$ISO_DST'"
run "mv '$U'/blood_faeces/chunk_*.gz '$ISO_DST'/"

echo
echo "--- (2) DELETE redundant complete AMR chunk sets (content-verified duplicates) ---"
total=0
for d in "${AMR_DROP[@]}"; do
    n=$(find "$U/$d" -maxdepth 1 -name 'chunk_*.gz' | wc -l)
    g=$(gb "$U/$d" -maxdepth 1 -name 'chunk_*.gz')
    total=$(awk -v a="$total" -v b="$g" 'BEGIN{print a+b}')
    echo "    $d  $n files  $g GB"
    run "rm -f '$U/$d'/chunk_*.gz"
done
echo "    subtotal: $total GB"

echo
echo "--- (3) DELETE truncated batch-2 chunk sets (killed mid-write by the full tier) ---"
for d in "${AMR_TRUNC[@]}"; do
    n=$(find "$U/$d" -maxdepth 1 -name 'chunk_*.gz' 2>/dev/null | wc -l)
    g=$(gb "$U/$d" -maxdepth 1 -name 'chunk_*.gz')
    echo "    $d  $n files  $g GB"
    run "rm -f '$U/$d'/chunk_*.gz"
done

echo
echo "--- (4) DELETE stale pre-cohort-keying results at blood_faeces top level ---"
echo "    superseded by blood_faeces/sampled_country_2_1_all_trainval/ which holds all 64"
echo "    $(find "$U/blood_faeces" -maxdepth 1 -name '*.assoc' | wc -l) files  $(gb "$U/blood_faeces" -maxdepth 1 -name '*.assoc') GB"
run "rm -f '$U'/blood_faeces/*.assoc"

echo
echo "--- (5) DELETE leftover pyseer work_* dirs (combine removes these; ceftazidime's combine failed) ---"
w=$(find "$U" -maxdepth 3 -type d -name 'work_*' 2>/dev/null | wc -l)
echo "    $w work_* dirs"
run "find '$U' -maxdepth 3 -type d -name 'work_*' -exec rm -rf {} +"

if [ "$SCRATCH_LOG_DAYS" -gt 0 ]; then
    echo
    # *.out/*.err ONLY. pyseer_scratch is not a pure log dir: a plain -mtime sweep would also have
    # taken three calibration scripts and unitig_subset_stride15.gz (5.12 GB, the strided matrix the
    # lambda-by-allele-frequency work still needs). The dry run is what caught that.
    echo "--- (6) DELETE *.out/*.err older than $SCRATCH_LOG_DAYS days in $SCRATCH_LOGS ---"
    old=$(find "$SCRATCH_LOGS" -maxdepth 1 -type f -mtime "+$SCRATCH_LOG_DAYS" \( -name '*.out' -o -name '*.err' \) 2>/dev/null | wc -l)
    echo "    $old of $(find "$SCRATCH_LOGS" -maxdepth 1 -type f | wc -l) files  $(gb "$SCRATCH_LOGS" -maxdepth 1 -type f -mtime "+$SCRATCH_LOG_DAYS" \( -name '*.out' -o -name '*.err' \)) GB"
    echo "    NOT touched: $(find "$SCRATCH_LOGS" -maxdepth 1 -type f ! -name '*.out' ! -name '*.err' | wc -l) non-log files"
    run "find '$SCRATCH_LOGS' -maxdepth 1 -type f -mtime '+$SCRATCH_LOG_DAYS' \( -name '*.out' -o -name '*.err' \) -delete"
fi

echo
echo "=== done.  df after: ==="
df -h "$U" | tail -1

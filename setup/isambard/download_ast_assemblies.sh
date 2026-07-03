#!/usr/bin/env bash
# Download AllTheBacteria (ATB) assemblies for an AST cohort on Isambard-AI by reusing
# BacPredict's planner (download_assemblies.py: ATB-index intersection keyed on
# phenotype-BioSample_ID) and curling each assembly from S3 in parallel. Idempotent
# (skip-existing), safe to re-run until it converges. NCBI fallback for not-in-ATB
# BioSamples is a separate later pass (--skip-ncbi here).
#
# Runs the planner inside the nuna-tools pixi env (pandas + datasets). Output assemblies
# -> $OUT as <BIOSAMPLE>.fa.gz. Driven by the per-cohort sbatch wrapper via BASE + METADATA.
set -uo pipefail
: "${PROJECTDIR:?PROJECTDIR must be set}"
: "${BASE:?BASE must be set (cohort dir, e.g. \$PROJECTDIR/david/raw/tb)}"
: "${METADATA:?METADATA must be set (per-species EBI records CSV)}"
export PATH="$HOME/.pixi/bin:$PATH"

PIXI="pixi run --manifest-path $HOME/nuna/setup/isambard/tools/pixi.toml"
OUT="$BASE/assemblies"
PLANNER="$HOME/BacPredict/src/tl/genome_download/scripts/download_assemblies.py"
ATB_S3="https://allthebacteria-assemblies.s3.eu-west-2.amazonaws.com"
NCORES="${NCORES:-32}"

mkdir -p "$OUT"
WORK="$(mktemp -d)"

echo "=== plan (ATB index intersection; skip-existing; NCBI deferred) : $METADATA ==="
$PIXI python "$PLANNER" \
  --metadata "$METADATA" --output-dir "$OUT" \
  --atb-batch-dir "$WORK/atb" --ncbi-batch-dir "$WORK/ncbi" \
  --manifest "$OUT/manifest.tsv" --accession-map "$OUT/acc_map.tsv" \
  --missing-output "$OUT/not_in_atb.tsv" \
  --n -1 --batch-size 100 --skip-ncbi 2>&1 | tail -15

atb_one() {
  local BS="$1" T
  T="$OUT/${BS}.fa.gz"
  [[ -s "$T" ]] && return 0
  if curl -sfL --max-time 300 "$ATB_S3/${BS}.fa.gz" -o "${T}.tmp"; then mv "${T}.tmp" "$T"; else rm -f "${T}.tmp"; return 1; fi
}
export -f atb_one; export OUT ATB_S3

echo "=== ATB download ($NCORES parallel) ==="
cat "$WORK"/atb/batch_* 2>/dev/null | xargs -P "$NCORES" -I {} bash -c 'atb_one "$1"' _ {} || true

N_OK=$(find "$OUT" -maxdepth 1 -name '*.fa.gz' -type f -size +0 2>/dev/null | wc -l)
N_MISS=$(( $(wc -l < "$OUT/not_in_atb.tsv" 2>/dev/null || echo 1) - 1 ))
echo "=== assemblies on disk: $N_OK ; not-in-ATB (NCBI fallback pending): ${N_MISS} ==="
rm -rf "$WORK"

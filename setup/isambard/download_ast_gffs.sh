#!/usr/bin/env bash
# Download BakRep Bakta GFF3 annotations for the cohort BioSamples that already have an
# ATB assembly on disk (BakRep annotates ATB assemblies, so seqids match — required by
# extract_proteins_from_gff_fna). Idempotent (skip-existing); safe to re-run.
# Output layout (BakRep default): $GFF/<datasetID>/<BIOSAMPLE>/<BIOSAMPLE>.bakta.gff3.gz
set -uo pipefail
: "${PROJECTDIR:?PROJECTDIR must be set}"
: "${BASE:?BASE must be set (cohort dir)}"
export PATH="$HOME/.pixi/bin:$PATH"

PIXI="pixi run --manifest-path $HOME/nuna/setup/isambard/tools/pixi.toml"
ASM="$BASE/assemblies"
GFF="$BASE/gff"
BATCH="${BATCH:-100}"
mkdir -p "$GFF"

# BioSamples that have an assembly but not yet a GFF anywhere under $GFF.
mapfile -t BS < <(for f in "$ASM"/*.fa.gz; do
  b=$(basename "$f" .fa.gz)
  [[ -n "$(find "$GFF" -name "$b.bakta.gff3.gz" -type f -size +0 -print -quit 2>/dev/null)" ]] || echo "$b"
done)
echo "GFFs to fetch: ${#BS[@]} (batch size $BATCH)"

fetch() {
  local list="$1"
  [[ -z "$list" ]] && return 0
  $PIXI bakrep download -e "$list" -d "$GFF" -m "tool:bakta,filetype:gff3" 2>&1 | tail -3 || true
}

batch=()
for b in "${BS[@]}"; do
  batch+=("$b")
  if [[ ${#batch[@]} -ge $BATCH ]]; then fetch "$(IFS=,; echo "${batch[*]}")"; batch=(); fi
done
[[ ${#batch[@]} -gt 0 ]] && fetch "$(IFS=,; echo "${batch[*]}")"

echo "=== GFFs on disk: $(find "$GFF" -name '*.bakta.gff3.gz' -type f -size +0 2>/dev/null | wc -l) ==="

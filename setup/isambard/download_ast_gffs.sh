#!/usr/bin/env bash
# Download BakRep Bakta GFF3 annotations for the cohort BioSamples that already have an
# ATB assembly on disk (BakRep annotates ATB assemblies, so seqids match — required by
# extract_proteins_from_gff_fna). Output layout (BakRep default):
#   $GFF/<datasetID>/<BIOSAMPLE>/<BIOSAMPLE>.bakta.gff3.gz
#
# Convergence loop: a single bad accession can make `bakrep download -e <batch>` error and
# drop the whole batch, so we don't trust one pass. Each pass recomputes the still-missing
# set (assemblies-on-disk minus GFFs-on-disk) and refetches only those in batches; the loop
# stops when a pass adds zero new GFFs (converged) or MAX_PASSES is hit. Idempotent /
# re-runnable. Driven by the per-cohort sbatch wrapper via BASE.
set -uo pipefail
: "${BASE:?BASE must be set (cohort dir)}"
export PATH="$HOME/.pixi/bin:$PATH"

PIXI="pixi run --manifest-path $HOME/nuna/setup/isambard/tools/pixi.toml"
ASM="$BASE/assemblies"
GFF="$BASE/gff"
BATCH="${BATCH:-100}"
MAX_PASSES="${MAX_PASSES:-10}"
mkdir -p "$GFF"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT

# BioSamples that have an assembly (compute once).
find "$ASM" -maxdepth 1 -name '*.fa.gz' -type f -size +0 -printf '%f\n' | sed 's/\.fa\.gz$//' | sort -u > "$WORK/have_asm"
N_ASM=$(wc -l < "$WORK/have_asm")

count_gff() { find "$GFF" -name '*.bakta.gff3.gz' -type f -size +0 2>/dev/null | wc -l; }
present_bs() {  # BioSamples with a GFF anywhere under $GFF (one whole-tree scan, not per-sample)
  find "$GFF" -name '*.bakta.gff3.gz' -type f -size +0 -printf '%f\n' 2>/dev/null | sed 's/\.bakta\.gff3\.gz$//' | sort -u
}

fetch() {  # one comma-separated batch; || true so a crashed batch doesn't abort the pass (retried next pass)
  local list="$1"; [[ -z "$list" ]] && return 0
  $PIXI bakrep download -e "$list" -d "$GFF" -m "tool:bakta,filetype:gff3" 2>&1 | tail -2 || true
}

for ((PASS=1; PASS<=MAX_PASSES; PASS++)); do
  present_bs > "$WORK/have_gff"
  comm -23 "$WORK/have_asm" "$WORK/have_gff" > "$WORK/missing"
  N_MISS=$(wc -l < "$WORK/missing")
  BEFORE=$(count_gff)
  echo "=== pass $PASS/$MAX_PASSES : $N_ASM assemblies, $BEFORE GFFs on disk, $N_MISS missing ==="
  [[ "$N_MISS" -eq 0 ]] && { echo "converged: nothing missing"; break; }

  batch=()
  while IFS= read -r b; do
    batch+=("$b")
    if [[ ${#batch[@]} -ge $BATCH ]]; then fetch "$(IFS=,; echo "${batch[*]}")"; batch=(); fi
  done < "$WORK/missing"
  [[ ${#batch[@]} -gt 0 ]] && fetch "$(IFS=,; echo "${batch[*]}")"

  AFTER=$(count_gff); ADDED=$((AFTER - BEFORE))
  echo "=== pass $PASS added $ADDED GFFs (now $AFTER on disk) ==="
  [[ "$ADDED" -le 0 ]] && { echo "pass added no new GFFs — remaining likely genuinely unannotated in BakRep; stopping"; break; }
done

echo "=== GFFs on disk: $(count_gff) / $N_ASM assemblies ==="

#!/bin/bash
# Re-run Panaroo for one drug with refinding DISABLED, reusing a prior run's converted GFFs.
#
# Panaroo's gene-refinding step (on by default) threw `KeyError: '0_1_150'` on the imipenem set
# (2280 genomes, high cross-lineage diversity). This re-runs the SAME converted-GFF inputs with
# `--refind-mode off` (keeping --clean-mode strict + --remove-invalid-genes, unchanged) into a fresh
# `<drug>_norefind/` dir — no re-conversion (~1.5h saved), the failed run left intact for reference.
#
# Caveat: a no-refind pangenome differs slightly from the refind-on runs (tetracycline, colistin), so
# cross-drug comparisons involving this drug are apples-to-oranges (accepted for imipenem only).
#
# Usage:  sbatch src/gene_array_lasso/scripts/rerun_panaroo_no_refind.sh <drug>
#   e.g.  sbatch src/gene_array_lasso/scripts/rerun_panaroo_no_refind.sh imipenem
#
#SBATCH --job-name=panaroo_norefind
#SBATCH --partition=icelake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --time=36:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --output=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/gene_array_lasso/logs/panaroo_norefind_%j.out
#SBATCH --error=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/gene_array_lasso/logs/panaroo_norefind_%j.err
set -euo pipefail

DRUG="${1:?usage: sbatch rerun_panaroo_no_refind.sh <drug>}"
GAL=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/gene_array_lasso
SRC="$GAL/panaroo/$DRUG"                 # failed run: converted_gff/, panaroo_input.txt, panaroo_genomes.tsv
DST="$GAL/panaroo/${DRUG}_norefind"      # fresh output
INPUT="$SRC/panaroo_input.txt"

export PYTHONUNBUFFERED=1
echo "=== Panaroo re-run (refind off): $DRUG ==="
echo "  SRC=$SRC  DST=$DST  job=$SLURM_JOB_ID  cpus=$SLURM_CPUS_PER_TASK"
[[ -f "$INPUT" ]] || { echo "ERROR: missing cached input: $INPUT" >&2; exit 1; }
echo "  input genomes: $(wc -l < "$INPUT")"

mkdir -p "$DST"
export TMPDIR="$DST/tmp_${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

if command -v micromamba &>/dev/null; then
  eval "$(micromamba shell hook --shell bash)"
  micromamba activate panaroo
else
  echo "ERROR: micromamba not found; cannot activate panaroo env" >&2; exit 1
fi

echo "panaroo $(panaroo --version 2>&1)"
panaroo \
  -i "$INPUT" \
  -o "$DST" \
  --clean-mode strict \
  --remove-invalid-genes \
  --refind-mode off \
  -t "${SLURM_CPUS_PER_TASK:-36}"

# Carry over the label->Sample map (written by the BacHGT prep step, refind-independent) so Step C
# can map GPA genome columns back to our Sample IDs from the same dir.
cp -n "$SRC/panaroo_genomes.tsv" "$DST/panaroo_genomes.tsv"

echo "=== GPA dims ==="
awk -F',' 'END{print "rows="NR-1", cols="NF}' "$DST/gene_presence_absence.csv"
echo "Done -> $DST"

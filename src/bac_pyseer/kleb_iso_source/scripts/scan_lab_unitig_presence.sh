#!/bin/bash
#SBATCH --job-name=lab_unitig_scan
#SBATCH --output=/rds/user/dca36/hpc-work/logs/lab_unitig_scan_%j.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/lab_unitig_scan_%j.err
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU

# Call hit-unitig presence for the lab collection straight from its assemblies, so the fitted
# invasion unitig model can score genomes the GWAS never saw.
#
# Cost: one Aho-Corasick automaton over 19,622 unitigs (both strands) built once, then a linear scan
# per genome. Measured on a 15-genome validation subset: automaton build dominates the small case;
# ~671 genomes extrapolates to ~2 h. 6 h requested because a wall kill loses the whole scan and buys
# another queue wait. Single-threaded by construction (ahocorasick), so 4 cores is for IO and slack,
# and 32G is generous against a peak dominated by one genome's contigs plus the automaton.
#
# VALIDATED against ground truth before use: 220 of these genomes are also in the GWAS presence
# matrix, and scanning 15 of them reproduced hits_submatrix exactly — 294,330 cells, 100.0000%
# agreement, zero discrepancies either way. Re-run that check if the unitig set ever changes.

set -euo pipefail
export PYTHONUNBUFFERED=1
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/home/dca36/rds/hpc-work/.uv_cache
export TMPDIR=${TMPDIR_OVERRIDE:-/home/dca36/rds/hpc-work/tmp}
mkdir -p "$TMPDIR"
unset PYTHONPATH PYTHONHOME
cd /home/dca36/workspace/BacPredict

DATA=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david
LAB=${LAB_DIR:-$DATA/processed/train_iso_source/lab_collection}
GWAS=${GWAS_DIR:-$DATA/processed/pyseer_iso_source/blood_faeces/sampled_country_2_1_all_trainval/gwas_unitig_lmm}
ASSEMBLIES=${ASSEMBLIES:-$LAB/lab_assemblies.tsv}
UNITIGS=${UNITIGS:-$GWAS/presence_matrix/unitigs.csv}
OUT_DIR=${OUT_DIR:-$LAB/unitig_presence}

for f in "$ASSEMBLIES" "$UNITIGS"; do
  [ -s "$f" ] || { echo "ERROR: missing input $f"; exit 1; }
done
echo "assemblies=$ASSEMBLIES  unitigs=$UNITIGS  out=$OUT_DIR"

uv run python -m bac_pyseer.kleb_iso_source.unitig_presence_from_assemblies \
  --assemblies "$ASSEMBLIES" \
  --unitigs-csv "$UNITIGS" \
  --out-dir "$OUT_DIR"

echo "=== scan done $(date) ==="

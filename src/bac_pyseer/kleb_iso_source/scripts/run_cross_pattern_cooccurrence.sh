#!/bin/bash
#SBATCH --job-name=pyseer_cross_pattern_cooccur
#SBATCH --output=/home/dca36/rds/hpc-work/pyseer_scratch/cross_pattern_cooccur_%j.out
#SBATCH --error=/home/dca36/rds/hpc-work/pyseer_scratch/cross_pattern_cooccur_%j.err
#SBATCH --partition=icelake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00     # dominated by one zcat of the ~77 GB unitig matrix (~40-60 min); padded ~3x
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU

# DESCRIPTIVE carrier co-occurrence between the top unitig pattern groups (NOT a gate on anything).
# Three steps: (1) rank pattern groups + emit representative unitig sequences [cheap python];
#              (2) one zcat|grep of the 77 GB unitig matrix to pull those unitigs' carrier lines [heavy];
#              (3) pairwise between-pattern carrier Jaccard [cheap python].
# Everything runs under PIXI (pandas present; sidesteps the uv.lock breakage, like the unitig permnull).
# Outputs → project_k; SLURM logs → RDS scratch; nothing in /home.
#
# Usage:  TOP=4 sbatch run_cross_pattern_cooccurrence.sh
set -euo pipefail
export PYTHONUNBUFFERED=1
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
unset PYTHONPATH PYTHONHOME

REPO=/home/dca36/workspace/BacPredict
PIXI=$REPO/src/bac_pyseer/pixi.toml
SCRIPT=$REPO/src/bac_pyseer/kleb_iso_source/cross_pattern_cooccurrence.py
DATA=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw
IN=$DATA/david/processed/pyseer_iso_source/blood_faeces/sampled_country_2_1_all
HITS=$IN/gwas_unitig_lmm/blood_vs_faeces_unitig_hits_annotated.tsv
UNITIGS=$DATA/david/processed/pyseer_iso_source/unitigs/blood_faeces/unitigs.pyseer.gz
TOP=${TOP:-4}
OUT=$IN/gwas_unitig_lmm/cross_pattern
mkdir -p "$OUT"
SEQS=$OUT/seqs.txt
META=$OUT/pattern_meta.tsv
LINES=$OUT/carrier_lines.txt
JACC=$OUT/cross_pattern_jaccard.tsv

echo "Job ${SLURM_JOB_ID:-?}  node ${SLURMD_NODENAME:-?}  top=$TOP  $(date)"
[ -s "$HITS" ] || { echo "ERROR: annotated unitig hits $HITS missing/empty"; exit 1; }
[ -s "$UNITIGS" ] || { echo "ERROR: unitig matrix $UNITIGS missing/empty"; exit 1; }

# 1) rank pattern groups + emit representative unitig sequences (one per top group).
pixi run --manifest-path "$PIXI" python "$SCRIPT" select \
    --hits "$HITS" --top "$TOP" --out-seqs "$SEQS" --out-meta "$META"
echo "selected $(wc -l < "$SEQS") representative unitigs  $(date)"

# 2) pull those unitigs' carrier lines from the matrix — ONE pass (grep -F, fixed strings; the zcat is
#    the slow step). grep exits 1 on no match; guard so `set -e` doesn't abort, then require a non-empty file.
zcat "$UNITIGS" | grep -F -f "$SEQS" > "$LINES" || true
[ -s "$LINES" ] || { echo "ERROR: no matrix lines matched the representative unitigs ($LINES empty)"; exit 1; }
echo "pulled $(wc -l < "$LINES") carrier lines  $(date)"

# 3) pairwise between-pattern carrier Jaccard.
pixi run --manifest-path "$PIXI" python "$SCRIPT" jaccard \
    --matrix-lines "$LINES" --meta "$META" --out "$JACC"
echo "CROSS_PATTERN_DONE  $JACC  $(date)"

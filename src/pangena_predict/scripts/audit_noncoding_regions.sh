#!/usr/bin/env bash
# Stage 2 step 4 — genome-wide audit of the baclm non-coding channel (the IGR-vs-RNA architecture
# question). GFF-only (no FASTA load) so it sweeps the whole cohort fast: counts how many non-coding
# runs (and RNA bodies) exceed MAX_LEN (= how many get windowed — the number for the baclm devs), and
# how often a run fuses IGR + RNA into one embedding. CPU-only (NO --gres; --mem required).
#
#   sbatch --export=ALL,TASK=tb   -J audit-nc-tb   src/pangena_predict/scripts/audit_noncoding_regions.sh
#   sbatch --export=ALL,TASK=kleb -J audit-nc-kleb src/pangena_predict/scripts/audit_noncoding_regions.sh
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
#SBATCH --error=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
set -euo pipefail
: "${SCRATCHDIR:?}" "${TASK:=tb}"
S="$SCRATCHDIR"
PY="$S/envs/bacpredict-gpu-venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"
case "$TASK" in
  tb)   DIR=train_tb_ast;   SPECIES=tb ;;
  kleb) DIR=train_kleb_ast; SPECIES=kp ;;
  *) echo "unknown TASK=$TASK (want tb|kleb)"; exit 1 ;;
esac
IN="$S/processed/$DIR/embedding_input.csv"
OUT="$S/processed/$DIR/pangena_predict/audit_noncoding/audit_${SPECIES}_${SLURM_JOB_ID:-local}.json"

echo "=== non-coding audit: species=$SPECIES input=$IN workers=${SLURM_CPUS_PER_TASK:-8} ==="
"$PY" "$HOME/BacPredict/src/pangena_predict/audit_noncoding_regions.py" \
  --input-csv "$IN" --output "$OUT" \
  --workers "${SLURM_CPUS_PER_TASK:-8}"
echo "AUDIT JSON -> $OUT"

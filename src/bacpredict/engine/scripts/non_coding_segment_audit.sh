#!/usr/bin/env bash
# Stage 2 step 4 — genome-wide audit of the baclm non-coding channel (the IGR-vs-RNA architecture
# question). GFF-only (no FASTA load) so it sweeps the whole cohort fast: counts how many non-coding
# runs (and RNA bodies) exceed MAX_LEN (= how many get windowed — the number for the baclm devs), and
# how often a run fuses IGR + RNA into one embedding. CPU-only (NO --gres; --mem required).
#
#   sbatch --export=ALL,TASK=tb   -J audit-nc-tb   src/bacpredict/engine/scripts/non_coding_segment_audit.sh
#   sbatch --export=ALL,TASK=kleb -J audit-nc-kleb src/bacpredict/engine/scripts/non_coding_segment_audit.sh
#SBATCH --partition=icelake-himem
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/rds/user/dca36/hpc-work/logs/%x-%j.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/%x-%j.out
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
"$PY" "$HOME/BacPredict/src/bacpredict/engine/embedding/non_coding_segment_audit.py" \
  --input-csv "$IN" --output "$OUT" \
  --workers "${SLURM_CPUS_PER_TASK:-8}"
echo "AUDIT JSON -> $OUT"

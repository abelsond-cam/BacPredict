#!/usr/bin/env bash
# Full-cohort protein extraction (CPU): build the (Sample, asm, gff) input CSV from the
# downloaded cohort, then extract protein-sequence parquets for every genome with both files.
# Parameterised by TASK; submit with e.g.:
#   sbatch --export=ALL,TASK=tb   -J extract-proteins-tb   setup/isambard/extract_proteins.sbatch
#   sbatch --export=ALL,TASK=kleb -J extract-proteins-kleb setup/isambard/extract_proteins.sbatch
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
#SBATCH --error=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
set -uo pipefail
: "${SCRATCHDIR:?}" "${TASK:?set TASK=tb|kleb}"
S="$SCRATCHDIR"
PY="$S/envs/bacpredict-gpu-venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"
EMB="$HOME/BacPredict/src/tl/embed"

case "$TASK" in
  tb)   RAW="$S/raw/tb";       PROC="$S/processed/train_tb_ast" ;;
  kleb) RAW="$S/raw/kleb_ast"; PROC="$S/processed/train_kleb_ast" ;;
  *) echo "unknown TASK=$TASK (want tb|kleb)"; exit 1 ;;
esac
mkdir -p "$PROC" "$S/tmp"; cd "$S/tmp"

echo "=== [$(date -u +%H:%M:%S)] build input CSV for $TASK ==="
"$PY" "$EMB/build_ast_input_csv.py" --base "$RAW" --out "$PROC/embedding_input.csv"

echo "=== [$(date -u +%H:%M:%S)] extract protein sequences ==="
"$PY" "$EMB/preprocess_assemblies_to_protein_sequences.py" \
  --input-csv "$PROC/embedding_input.csv" --output-dir "$PROC/protein_sequences" \
  --workers 48 --skip-existing

N=$(find "$PROC/protein_sequences" -name '*_protein_sequences.parquet' -type f | wc -l)
echo "=== [$(date -u +%H:%M:%S)] DONE $TASK: $N protein-sequence parquets ==="

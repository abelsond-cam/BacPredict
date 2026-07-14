#!/usr/bin/env bash
# Kp AST labels + 70/10/20 split — the Klebsiella analogue of the TB label prep, so the Kp coding
# ladder / IGR probes have a binary_ast_with_split.csv to read (Stage 2 step 3). Two CPU stages:
#   1. parse the Kp EBI AMR records (133k tests, 10,250 K. pneumoniae BioSamples) -> binary_ast.csv
#      via the canonical organism-agnostic parser (pangena_predict/parse_ebi_ast_to_binary.py).
#   2. add_splits(seed=1) + prune to samples that actually have an ESM embedding (9,724 in esm/)
#      -> binary_ast_with_split.csv, byte-identical recipe to how TB's split was built.
# CPU-only (NO --gres): a no-GPU job schedules normally on workq; --mem is required (mem defaults
# are GPU-tied). Durable sbatch so an SSH drop can't kill it (that lost the TB panel once).
#
#   sbatch -J prepare-kp-ast src/bacpredict/engine/scripts/prepare_kp_ast_labels.sh
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
#SBATCH --error=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
set -euo pipefail
: "${SCRATCHDIR:?}"
S="$SCRATCHDIR"
PY="$S/envs/bacpredict-gpu-venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"

RAW="$S/raw/kleb_ast/ebi_kleb_amr_records.csv"
OUT="$S/processed/train_kleb_ast"
VIZ="$S/processed/train_kleb_ast/label_prep_viz"

echo "=== [1/2] parse EBI AST -> binary_ast.csv ($RAW) ==="
"$PY" "$HOME/BacPredict/src/bacpredict/engine/labels/parse_ebi_ast_to_binary.py" \
  --input "$RAW" --output-dir "$OUT" --viz-dir "$VIZ"

echo "=== [2/2] add 70/10/20 split (seed 1) + prune to embedded samples ==="
"$PY" -m bacpredict.engine.finetune.build_split_csv --task kleb_ast \
  --ast-csv "$OUT/binary_ast.csv" \
  --embeddings-dir "$OUT/esm" \
  --output-base "$OUT" \
  --seed 1

echo "=== done. split CSV -> $OUT/binary_ast_with_split.csv ==="
"$PY" - "$OUT/binary_ast_with_split.csv" <<'PYEOF'
import sys, pandas as pd
df = pd.read_csv(sys.argv[1])
print("rows:", len(df), "unique Sample:", df["Sample"].nunique())
print("split counts:", df.drop_duplicates("Sample")["train_val_eval"].value_counts().to_dict())
drugs = [c for c in df.columns if c not in {"phenotype-BioSample_ID", "Sample", "train_val_eval"}]
print(f"{len(drugs)} drug columns")
top = sorted(((c, int((df[c] == 1).sum()), int((df[c] == 0).sum())) for c in drugs),
             key=lambda t: -(t[1] + t[2]))[:15]
for c, r, s in top:
    print(f"  {c:<20} R={r:<6} S={s:<6} (tested {r + s})")
PYEOF

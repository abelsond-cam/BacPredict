#!/bin/bash
#SBATCH --job-name=build_surprisal_panel_store
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=icelake-himem
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

# Build the per-sample surprisal-panel store from the EXISTING 1000-genome unmasked-surprisal
# scan (raw per-residue dumps already on disk; verified complete: 20/20 shards, 4.07M proteins,
# integrity OK). NO GPU — pure CPU re-key of the dumps into {sample}_panel.npz +
# panel_standardization.json. Also writes a tiny class-balanced smoke AST sheet (manifest
# samples) so the n=10 panel overfit uses store-covered genomes.
#
#   sbatch src/pangena_predict/scripts/build_surprisal_panel_store.sh

cd /home/dca36/workspace/BacPredict
export PYTHONUNBUFFERED=1

RDS=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_tb_ast
SCAN_DIR=$RDS/pangena_predict/unmasked_surprisal_scan
STORE_DIR=$RDS/tb_surprisal_panel
SMOKE_SHEET=$STORE_DIR/tb_rif_smoke_split.csv

mkdir -p "$STORE_DIR"

echo "Building surprisal panel store from $SCAN_DIR/scan_raw_shard*.npz -> $STORE_DIR"
uv run python src/pangena_predict/build_surprisal_store.py \
    --source raw \
    --scan-raw-glob "$SCAN_DIR/scan_raw_shard*.npz" \
    --out-dir "$STORE_DIR"

echo "Writing class-balanced smoke AST sheet (interleaved R/W) -> $SMOKE_SHEET"
uv run python - "$SCAN_DIR/manifest.csv" "$SMOKE_SHEET" <<'PY'
import sys
import pandas as pd

manifest_csv, out_csv = sys.argv[1], sys.argv[2]
m = pd.read_csv(manifest_csv)
r = m[m["role"] == "resistant"]["sample"].astype(str).tolist()
w = m[m["role"] == "wt"]["sample"].astype(str).tolist()
# Interleave R,W,R,W,... so the n=10 dummy-mode head ([:10]) is class-balanced (5R/5W).
interleaved = [s for pair in zip(r, w) for s in pair]
rset = set(r)
rows = [{"Sample": s, "rifampin": 1 if s in rset else 0, "train_val_eval": "train"} for s in interleaved]
out = pd.DataFrame(rows)
out.to_csv(out_csv, index=False)
print(f"Wrote {len(out)} rows -> {out_csv} ({int(out.rifampin.sum())} R / {int((out.rifampin == 0).sum())} S)")
print("First 10 labels (n=10 smoke head):", out["rifampin"].head(10).tolist())
PY

echo "Surprisal panel store build complete. Sample of store dir:"
ls "$STORE_DIR" | head
echo "panel_standardization.json:"
cat "$STORE_DIR/panel_standardization.json"

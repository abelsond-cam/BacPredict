#!/bin/bash
#SBATCH --job-name=tb_eval_panel
#SBATCH --output=/rds/user/dca36/hpc-work/logs/%x-%j.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/%x-%j.out
#SBATCH --time=06:00:00
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --open-mode=append
# CSD3/UoHPC variant (when it returns): --partition=ampere --account=FLOTO-SL2-GPU,
#   logs → relative or ~/rds/hpc-work/logs/, and `module load cuda/12.4 cudnn/8.9_cuda-12.4`.
#
# Evaluate the full TB AST drug panel on its held-out evaluate set and render
# the combined ROC|PR grid + summary CSV. Each drug also gets a Youden-J
# operating threshold (chosen on validation, reported on evaluate).
#
# Single-split mode (no --n-folds): evaluate.py reads the `evaluate` rows from
# binary_ast_with_split.csv. Checkpoint dirs are discovered per-drug by glob
# because the dir names embed the per-drug SLURM jobid.
#
# Submit standalone (once training has finished):
#   sbatch src/bacpredict/apps/tb/scripts/eval_panel_on_slurm.sh
# Or chain after the training jobs:
#   sbatch --dependency=afterany:<jid1>:<jid2>:... src/bacpredict/apps/tb/scripts/eval_panel_on_slurm.sh
# (afterany = run regardless of success; the loop skips drugs with no checkpoint.)

set -uo pipefail

# Data root — one env var, cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
: "${BACPREDICT_DATA_ROOT:="$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/bac_ast_prediction"}"
D="$BACPREDICT_DATA_ROOT"
PY="$HOME/workspace/BacPredict/.venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

BASE=$D/processed/train_tb_ast
SHEET=$BASE/binary_ast_with_split.csv
EMB=$BASE/esm
CKPT=$BASE/checkpoints

# Panel = the 10 Stage C drugs (in resistance-count descending order).
PANEL="rifampin isoniazid ethambutol rifabutin levofloxacin streptomycin moxifloxacin pyrazinamide ethionamide kanamycin"

# 1) Evaluate each drug. Idempotent: drugs with an eval_scores.npz are skipped
# unless FORCE_RECOMPUTE=1.
for d in $PANEL; do
  # Glob since the dir name embeds the per-drug SLURM jobid. Pick the most
  # recent if multiple (sorted lexicographically — jobids are monotonically
  # increasing on CSD3, so tail -n 1 picks the latest).
  CK=$(ls -d "$CKPT"/mycobacterium_tuberculosis_${d}_stage_c_* 2>/dev/null | tail -n 1)
  if [ -z "$CK" ] || [ ! -d "$CK" ]; then
    echo "NO CHECKPOINT (skipping): $d"
    continue
  fi
  if [ -f "$CK/eval_scores.npz" ] && [ "${FORCE_RECOMPUTE:-0}" != "1" ]; then
    echo "ALREADY DONE (skip; set FORCE_RECOMPUTE=1 to re-run): $d  ($CK)"
    continue
  fi
  echo "=== evaluating $d ($CK) ==="
  "$PY" -m bacpredict.engine.finetune.evaluate \
    --checkpoint "$CK" --drug "$d" --task tb_ast \
    --prevalence-label "resistance rate" \
    --ast-sheet-path "$SHEET" --embeddings-dir "$EMB" --num-workers 4 \
    || echo "EVAL FAILED: $d"
done

# 2) Combined ROC|PR grid over every drug that produced scores, in panel order.
ARGS=()
for d in $PANEL; do
  NPZ=$(ls -d "$CKPT"/mycobacterium_tuberculosis_${d}_stage_c_*/eval_scores.npz 2>/dev/null | tail -n 1)
  [ -n "$NPZ" ] && [ -f "$NPZ" ] && ARGS+=("${d}=$NPZ")
done
if [ ${#ARGS[@]} -gt 0 ]; then
  "$PY" -m bacpredict.engine.finetune.evaluate --combine "${ARGS[@]}" \
    --prevalence-label "resistance rate" \
    --combine-out "$BASE/eval_roc_pr_grid_full_panel.png" \
    --bar-out "$BASE/eval_auroc_bar.png" \
    --bar-title "TB AMR panel — held-out AUROC"
else
  echo "WARNING: no eval_scores.npz files found — skipping combine grid."
fi

# 3) Summary CSV (0.5 metrics + Youden operating point), sorted by AUROC desc.
"$PY" - "$CKPT" "$BASE/eval_summary.csv" <<'PY'
import csv, glob, json, os, sys
ckpt_root, out_csv = sys.argv[1], sys.argv[2]
rows = []
for f in glob.glob(os.path.join(ckpt_root, "mycobacterium_tuberculosis_*_stage_c_*", "eval_results.json")):
    d = json.load(open(f))
    m = d["metrics"]
    op = d.get("operating_point") or {}
    rows.append([
        d["drug"], m["n_samples"], round(m["prevalence"], 4),
        round(m["auroc"], 4), round(m["auprc"], 4),
        round(m["sensitivity"], 4), round(m["specificity"], 4), round(m["balanced_accuracy"], 4),
        (round(op["threshold"], 4) if op.get("threshold") is not None else ""),
        (round(op["sensitivity"], 4) if op.get("sensitivity") is not None else ""),
        (round(op["specificity"], 4) if op.get("specificity") is not None else ""),
        (round(op["balanced_accuracy"], 4) if op.get("balanced_accuracy") is not None else ""),
    ])
rows.sort(key=lambda r: (r[3] if isinstance(r[3], float) else -1), reverse=True)
with open(out_csv, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["drug", "n_eval", "resistance_rate", "auroc", "auprc",
                "sens@0.5", "spec@0.5", "balanced_acc@0.5",
                "youden_threshold", "sens@opt", "spec@opt", "balanced_acc@opt"])
    w.writerows(rows)
print(f"wrote {out_csv} with {len(rows)} drugs")
PY

echo "PANEL_EVAL_DONE"

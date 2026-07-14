#!/bin/bash
# Regenerate the TB AMR panel combined ROC|PR grid, AUROC bar chart, and summary
# CSV from existing per-drug eval_scores.npz + eval_results.json files.
#
# No SLURM, no GPU — runs on the login node in seconds. Use this after tweaking
# the figure style (e.g. plot_auroc_bar params) without re-running inference.
# For a full re-eval of one or more drugs, use scripts/eval_panel_on_slurm.sh.
#
# Usage (from anywhere on HPC):
#     bash "$HOME/BacPredict/src/bacpredict/apps/tb/scripts/regen_panel_summary.sh"

set -euo pipefail

# Data root — one env var, cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
: "${BACPREDICT_DATA_ROOT:="$SCRATCHDIR"}"
D="$BACPREDICT_DATA_ROOT"
PY="$SCRATCHDIR/envs/bacpredict-gpu-venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""

BASE=$D/processed/train_tb_ast
CKPT=$BASE/checkpoints

# Panel order: descending R count. Matches eval_panel_on_slurm.sh.
PANEL="rifampin isoniazid ethambutol rifabutin levofloxacin streptomycin moxifloxacin pyrazinamide ethionamide kanamycin"

ARGS=()
for d in $PANEL; do
  # Run dir embeds the SLURM jobid; pick the most recent (lexicographic tail).
  NPZ=$(ls -d "$CKPT"/mycobacterium_tuberculosis_${d}_stage_c_*/eval_scores.npz 2>/dev/null | tail -n 1)
  [ -n "$NPZ" ] && [ -f "$NPZ" ] && ARGS+=("${d}=$NPZ")
done
echo "Combining ${#ARGS[@]} drugs..."

"$PY" -m bacpredict.engine.finetune.evaluate --combine "${ARGS[@]}" \
  --prevalence-label "resistance rate" \
  --combine-out "$BASE/eval_roc_pr_grid_full_panel.png" \
  --bar-out "$BASE/eval_auroc_bar.png" \
  --bar-title "TB AMR panel — held-out AUROC"

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
echo "REGEN_DONE"

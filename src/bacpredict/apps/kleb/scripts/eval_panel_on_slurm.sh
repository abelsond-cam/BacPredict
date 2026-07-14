#!/bin/bash
#SBATCH --job-name=kleb_eval_panel
#SBATCH --output=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
#SBATCH --error=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
#SBATCH --time=02:00:00
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --open-mode=append
# CSD3/UoHPC variant (when it returns): --partition=ampere --account=FLOTO-SL2-GPU,
#   logs → relative or ~/rds/hpc-work/logs/, and `module load cuda/12.4 cudnn/8.9_cuda-12.4`.
#
# Evaluate the full Kp AST drug panel on its held-out evaluate set and render the
# combined ROC|PR grid + summary CSV. Each drug also gets a Youden-J operating
# threshold (chosen on validation, reported on evaluate).
#
# Submit so it fires automatically once the fan-out jobs finish:
#   sbatch --dependency=afterany:<jid1>:<jid2>:... src/bacpredict/apps/kleb/scripts/eval_panel_on_slurm.sh
# (afterany = run regardless of success; the loop skips any drug whose checkpoint is missing.)

set -uo pipefail

# Data root — one env var, cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
: "${BACPREDICT_DATA_ROOT:="$SCRATCHDIR"}"
D="$BACPREDICT_DATA_ROOT"
PY="$SCRATCHDIR/envs/bacpredict-gpu-venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

BASE=$D/processed/train_kleb_ast
SHEET=$BASE/binary_ast_with_split.csv
EMB=$D/processed/train_kleb_ast/esm
FT=$BASE/models/finetune

# Panel order (descending labelled count). ampicillin (intrinsic R) + pentizidone (unverified) excluded; colistin added.
PANEL="gentamicin ceftazidime meropenem ciprofloxacin trimethoprim-sulfamethoxazole amikacin ceftriaxone \
piperacillin-tazobactam cefoxitin aztreonam cefazolin tobramycin cefepime imipenem levofloxacin cefotaxime \
cefuroxime ampicillin-sulbactam ertapenem tetracycline azithromycin colistin"

# 1) Evaluate each drug. Idempotent: drugs that already have an eval_scores.npz
# are skipped unless FORCE_RECOMPUTE=1, so a re-submitted job picks up where a
# timed-out one left off.
for d in $PANEL; do
  CK=$FT/klebsiella_pneumoniae_${d}_lr_0.00015_finetuned_fold00_seed1
  if [ ! -d "$CK" ]; then
    echo "NO CHECKPOINT (skipping): $d"
    continue
  fi
  if [ -f "$CK/eval_scores.npz" ] && [ "${FORCE_RECOMPUTE:-0}" != "1" ]; then
    echo "ALREADY DONE (skip; set FORCE_RECOMPUTE=1 to re-run): $d"
    continue
  fi
  echo "=== evaluating $d ==="
  "$PY" -m bacpredict.engine.finetune.evaluate \
    --checkpoint "$CK" --drug "$d" --task kleb_ast \
    --n-folds 5 --fold 0 --seed 1 --evaluate-seed 1 \
    --prevalence-label "resistance rate" \
    --ast-sheet-path "$SHEET" --embeddings-dir "$EMB" --num-workers 4 \
    || echo "EVAL FAILED: $d"
done

# 2) Combined ROC|PR grid over every drug that produced scores, in panel order.
ARGS=()
for d in $PANEL; do
  NPZ=$FT/klebsiella_pneumoniae_${d}_lr_0.00015_finetuned_fold00_seed1/eval_scores.npz
  [ -f "$NPZ" ] && ARGS+=("${d}=$NPZ")
done
"$PY" -m bacpredict.engine.finetune.evaluate --combine "${ARGS[@]}" \
  --prevalence-label "resistance rate" \
  --combine-out "$BASE/eval_roc_pr_grid_full_panel.png" \
  --bar-out "$BASE/eval_auroc_bar.png" \
  --bar-title "Kp AMR panel — held-out AUROC"

# 3) Summary CSV (0.5 metrics + Youden operating point), sorted by AUROC desc.
"$PY" - "$FT" "$BASE/eval_summary.csv" <<'PY'
import csv, glob, json, os, sys
ft, out_csv = sys.argv[1], sys.argv[2]
rows = []
for f in glob.glob(os.path.join(ft, "klebsiella_pneumoniae_*_fold00_seed1", "eval_results.json")):
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

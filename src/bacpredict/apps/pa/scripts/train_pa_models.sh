#!/usr/bin/env bash
# The PsA-only AMR models (the baseline arm: one model per drug), one stage at a time, from a CSD3 login node:
#
#   bash src/bacpredict/apps/pa/scripts/train_pa_models.sh splits        # split tables → processed/paeruginosa/splits/
#   bash src/bacpredict/apps/pa/scripts/train_pa_models.sh check-splits  # what the trainer will read, per drug
#   ARRAY_REPO=<worktree> bash …/train_pa_models.sh smoke                # Stage A: one drug, 10 genomes, one GPU
#   bash src/bacpredict/apps/pa/scripts/train_pa_models.sh check-smoke   # Stage A: every output written
#   bash src/bacpredict/apps/pa/scripts/train_pa_models.sh overfit       # Stage B: the smoke's loss → ~0, AUROC 1
#   ARRAY_REPO=<worktree> bash …/train_pa_models.sh submit               # Stage C: GPU array, one task per drug
#   bash src/bacpredict/apps/pa/scripts/train_pa_models.sh collect       # holdout + holdout_st metrics, every drug
#
# Splits are engine/splits/build_amr_holdout_splits (PROJECT_STATE §6, 2026-11-20): ST235 held out whole as
# holdout_st (exact ST), a random 20 % holdout, 5 folds over the rest (fold 0 validates), the 15 Ecuador long-read
# assemblies excluded (apps/pa/excluded_genomes.tsv), the 3 two-BioSample CDC strains grouped
# (apps/pa/duplicate_isolate_groups.csv). Eligible = every genome of genomes/bakta_annotation.csv, and `splits`
# refuses unless each has a .pt: the builder shuffles the eligible set as a whole, so a genome added later would move
# every other genome's split.
#
# Training is engine/finetune/scripts/finetune_amr_array.sh: finetune_amr --split-table, bf16, early stopping, the
# deployed TB/Kp hyperparameters. Stage A and B are one run: the trainer's 10-genome path trains, validates and
# scores on the same balanced 10 for 100 epochs, so `smoke` submits it, `check-smoke` checks its outputs (Stage A)
# and `overfit` reads its loss and AUROC (Stage B). smoke and submit need ARRAY_REPO, a detached worktree at the SHA
# to run (.venv symlinked to the shared env), whose trainer code must be byte-identical to this checkout's; the
# array is pinned to its SHA (EXPECT_SHA) and to the split tables (split_manifest.json md5).
#
# Settings: SMOKE_DRUG (default ciprofloxacin), ONLY (comma-separated subset of drugs for submit/collect),
# ARRAY_CONCURRENT (a %N throttle on the array), SMOKE_TIME / TRAIN_TIME (walls), OVERWRITE=1 (let the builder replace
# tables whose content differs: every model trained on the old tables is then invalid).
set -euo pipefail

# What the array reads from the environment (sbatch --export=ALL) must come from this driver, not a stale login shell
unset N_SAMPLES MODEL LR EVAL_STEPS MAX_EPOCHS PATIENCE THRESHOLD SEED NUM_WORKERS EXTRA_HOLDOUTS \
      TASK DRUGS SPLIT_DIR SPLIT_MANIFEST_MD5 EMBEDDINGS_DIR OUT_ROOT REPO EXPECT_SHA

ROOT=${BACPREDICT_AMR_ROOT:-$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/amr_prediction}
[[ -d $ROOT ]] || { echo "no AMR root at $ROOT (set BACPREDICT_AMR_ROOT)"; exit 1; }
HERE=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
# this checkout's code on the login node too, even where .venv's editable install points at another checkout
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
T=$ROOT/processed/paeruginosa
GEN=$T/genomes
COHORT=$GEN/bakta_annotation.csv
LABELS=$T/binary_ast.csv
QC=$GEN/genome_qc.csv
ESM=$ROOT/processed/esm
SPLITS=$T/splits
ELIGIBLE=$SPLITS/eligible_genomes.csv
MODELS=$T/models
EXCLUDE_FILE=$HERE/src/bacpredict/apps/pa/excluded_genomes.tsv
GROUP_FILE=$HERE/src/bacpredict/apps/pa/duplicate_isolate_groups.csv
HOLDOUT_ST=paeruginosa:235
PA_DRUGS=(ciprofloxacin ceftazidime meropenem tobramycin amikacin imipenem piperacillin-tazobactam aztreonam
          levofloxacin cefepime)
if [[ -n ${ONLY:-} ]]; then IFS=, read -r -a PA_DRUGS <<<"$ONLY"; fi
SMOKE_DRUG=${SMOKE_DRUG:-ciprofloxacin}
ARRAY=src/bacpredict/engine/finetune/scripts/finetune_amr_array.sh
# what the array runs in ARRAY_REPO: it must be this driver's own copy of each (same_array_code)
ARRAY_CODE=("$ARRAY" src/bacpredict/engine/finetune/finetune_amr.py src/bacpredict/engine/finetune/datasets.py
            src/bacpredict/engine/finetune/metrics.py src/bacpredict/engine/finetune/attention_pool.py
            src/bacpredict/engine/splits/load_splits.py src/bacpredict/engine/splits/load_split_species.py
            src/bacpredict/engine/splits/subsample.py
            src/bacpredict/engine/config.py)
cd "$HERE"

submit() {  # sbatch, then print the resource line the global rules require (read back from SLURM)
    local id job kind; id=$(sbatch --parsable "$@"); id=${id%%;*}
    job=$(scontrol show job "$id")
    kind=CPU; grep -q 'gres/gpu' <<<"$job" && kind="GPU x$(grep -m1 -oE 'gres/gpu=[0-9]+' <<<"$job" | cut -d= -f2)"
    echo "JOB $id $(grep -m1 -oE 'JobName=[^ ]+' <<<"$job") | $kind | $(grep -m1 -oE 'ReqTRES=[^ ]+' <<<"$job")" \
         "| $(grep -m1 -oE 'TimeLimit=[^ ]+' <<<"$job") | $(grep -m1 -oE 'Partition=[^ ]+' <<<"$job")" \
         "| $(grep -m1 -oE 'Account=[^ ]+' <<<"$job")"
}

# same_array_code <array repo>: refuse unless the array script and the trainer modules it runs are this checkout's
same_array_code() {
    local f
    for f in "${ARRAY_CODE[@]}"; do
        cmp -s "$HERE/$f" "$1/$f" || {
            echo "ARRAY_REPO's $1/$f is not this driver's $HERE/$f (differs or missing):"
            echo "check out this driver's SHA in ARRAY_REPO, or run the driver from ARRAY_REPO"; exit 1; }
    done
}

# no_live <job name>: a failed squeue is not "none live"
no_live() {
    local jobs live
    jobs=$(squeue -u "$USER" -h -o '%j') || { echo "squeue failed: cannot tell whether a $1 job is live"; exit 1; }
    live=$(grep -cx -- "$1" <<<"$jobs" || true)
    (( live == 0 )) || { echo "a $1 job is queued or running: wait for it (or scancel it) first"; exit 1; }
}

need_tables() {  # every drug's table and the manifest, before anything is submitted
    local d
    [[ -s $SPLITS/split_manifest.json ]] || { echo "no $SPLITS/split_manifest.json: run splits first"; exit 1; }
    for d in "$@"; do [[ -s $SPLITS/${d}_split.csv ]] || { echo "no $SPLITS/${d}_split.csv: run splits"; exit 1; }; done
}

# the smoke run the Stage A/B checks read: the newest models/smoke/<stamp>/<drug>
latest_smoke() {
    [[ -s $MODELS/smoke/LATEST ]] || { echo "no $MODELS/smoke/LATEST: run smoke first" >&2; exit 1; }
    cat "$MODELS/smoke/LATEST"
}

case "${1:-}" in
splits)
    for f in "$COHORT" "$LABELS" "$QC" "$EXCLUDE_FILE" "$GROUP_FILE"; do [[ -s $f ]] || { echo "no $f"; exit 1; }; done
    mkdir -p "$SPLITS"
    # eligible = the cohort genomes with a .pt, written only when it changes (its md5 goes into the manifest). Every
    # cohort genome must have one: run build_pa_embeddings.sh check first if this refuses
    uv run --no-sync python -I - "$COHORT" "$ESM" "$ELIGIBLE" <<'EOF'
import os, sys
from pathlib import Path
import pandas as pd
cohort, esm, out = sys.argv[1:4]
samples = sorted(pd.read_csv(cohort, dtype={"Sample": str})["Sample"])
missing = [s for s in samples if not (Path(esm) / f"{s}_esm_embeddings.pt").exists()]
if missing:
    sys.exit(f"{len(missing)} of {len(samples)} cohort genomes have no .pt in {esm}: {missing[:10]}")
text = "Sample\n" + "".join(f"{s}\n" for s in samples)
if not os.path.exists(out) or open(out).read() != text:
    tmp = f"{out}.{os.getpid()}.part"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, out)
print(f"eligible: {len(samples)} genomes, each with a .pt -> {out}")
EOF
    uv run --no-sync python -m bacpredict.engine.splits.build_amr_holdout_splits \
        --species "paeruginosa=$LABELS,$ELIGIBLE,$QC" --holdout-st "paeruginosa=$HOLDOUT_ST" \
        --exclude "$EXCLUDE_FILE" --groups "$GROUP_FILE" --drugs "${PA_DRUGS[@]}" --out-dir "$SPLITS" \
        ${OVERWRITE:+--overwrite}
    ;;
check-splits)
    need_tables "${PA_DRUGS[@]}"
    # exactly the trainer's own read (resolve_split_table, missing_embeddings): what each array task will see
    uv run --no-sync python - "$SPLITS" "$ESM" "${PA_DRUGS[@]}" <<'EOF'
import sys
from bacpredict.engine.finetune.finetune_amr import missing_embeddings, resolve_split_table
splits, esm, *drugs = sys.argv[1:]
bad = 0
print(f"{'drug':<24}{'train':>7}{'valid':>7}{'hold':>7}{'%R':>6}{'ST':>6}{'%R':>6}  missing .pt")
for d in drugs:
    s = resolve_split_table(f"{splits}/{d}_split.csv")
    pct = lambda ids: 100 * sum(s.label_map[i] for i in ids) / len(ids) if ids else float("nan")
    st = s.extra["holdout_st"]
    miss = missing_embeddings(s.train + s.validate + s.holdout + st, esm)
    bad += bool(miss)
    print(f"{d:<24}{len(s.train):>7}{len(s.validate):>7}{len(s.holdout):>7}{pct(s.holdout):>6.1f}{len(st):>6}"
          f"{pct(st):>6.1f}  {len(miss)} {miss[:3]}")
sys.exit(1 if bad else 0)
EOF
    ;;
smoke)
    array_repo=${ARRAY_REPO:?set ARRAY_REPO to a detached worktree at the SHA to run (see the header)}
    same_array_code "$array_repo"
    need_tables "$SMOKE_DRUG"
    out_root=$MODELS/smoke/$(date +%Y%m%d_%H%M%S)
    md5=$(md5sum < "$SPLITS/split_manifest.json" | cut -d' ' -f1)
    sha=$(git -C "$array_repo" rev-parse HEAD)
    # ~100 epochs of 10 genomes plus a checkpoint each epoch: well under an hour (short job: a tight wall)
    submit --array=0 --time="${SMOKE_TIME:-01:00:00}" --job-name=pa-finetune-smoke \
        --export="ALL,REPO=$array_repo,EXPECT_SHA=$sha,TASK=pa_ast,DRUGS=$SMOKE_DRUG,SPLIT_DIR=$SPLITS,SPLIT_MANIFEST_MD5=$md5,EMBEDDINGS_DIR=$ESM,OUT_ROOT=$out_root,N_SAMPLES=10" \
        "$array_repo/$ARRAY"
    mkdir -p "$MODELS/smoke"
    echo "$out_root/$SMOKE_DRUG" > "$MODELS/smoke/LATEST"
    echo "smoke run: $out_root/$SMOKE_DRUG"
    ;;
check-smoke)
    run=$(latest_smoke)
    uv run --no-sync python -I - "$run" <<'EOF'
import json, sys
from pathlib import Path
import numpy as np
run = Path(sys.argv[1])
res = run / "results.json"
if not res.exists():
    sys.exit(f"no {res}: the smoke has not finished (or failed; read its log)")
r = json.loads(res.read_text())
problems = []
if r["split"]["source"] != "smoke":
    problems.append(f"split.source is {r['split']['source']!r}, not 'smoke'")
if list(r.get("holdouts", {})) != ["holdout", "holdout_st"]:
    problems.append(f"holdouts are {list(r.get('holdouts', {}))}, not ['holdout', 'holdout_st']")
for name, block in r.get("holdouts", {}).items():
    npz = run / block["scores"]
    if not npz.exists():
        problems.append(f"no {npz}")
        continue
    z = np.load(npz)
    if len(z["sample_ids"]) != block["n_samples"] or len(z["y_prob"]) != block["n_samples"]:
        problems.append(f"{npz.name}: {len(z['sample_ids'])} rows for n_samples {block['n_samples']}")
    m = block["metrics"]
    print(f"{name}: n={block['n_samples']} R={m['per_class']['1']['n']} AUROC={m['auroc']:.3f} "
          f"bal_acc={m['balanced_accuracy']:.3f} scores={npz.name}")
if not list(run.glob("checkpoint-*/config.json")):
    problems.append("no checkpoint-*/config.json")
for p in problems:
    print(f"FAIL {p}")
print("Stage A: PASS" if not problems else "Stage A: FAIL")
sys.exit(1 if problems else 0)
EOF
    ;;
overfit)
    run=$(latest_smoke)
    uv run --no-sync python -I - "$run" "${OVERFIT_MAX_LOSS:-0.1}" <<'EOF'
import json, sys
from pathlib import Path
run, max_loss = Path(sys.argv[1]), float(sys.argv[2])
states = sorted(run.glob("checkpoint-*/trainer_state.json"), key=lambda p: json.loads(p.read_text())["global_step"])
if not states:
    sys.exit(f"no checkpoint-*/trainer_state.json under {run}: has the smoke finished?")
history = json.loads(states[-1].read_text())["log_history"]
loss = [h["loss"] for h in history if "loss" in h]
auroc = [h["eval_auroc"] for h in history if "eval_auroc" in h]
holdout = json.loads((run / "results.json").read_text())["holdouts"]["holdout"]["metrics"]["auroc"]
print(f"train loss: first {loss[0]:.4f}, min {min(loss):.4f}, last {loss[-1]:.4f} ({len(loss)} logged)")
print(f"eval AUROC (validate = train): last {auroc[-1]:.3f}, max {max(auroc):.3f}; holdout (= train) AUROC {holdout:.3f}")
ok = loss[-1] <= max_loss and holdout >= 0.99
print(f"Stage B ({'PASS' if ok else 'FAIL'}): last train loss <= {max_loss} and train = holdout AUROC >= 0.99")
sys.exit(0 if ok else 1)
EOF
    ;;
submit)
    array_repo=${ARRAY_REPO:?set ARRAY_REPO to a detached worktree at the SHA to run (see the header)}
    same_array_code "$array_repo"
    need_tables "${PA_DRUGS[@]}"
    no_live pa-finetune
    for d in "${PA_DRUGS[@]}"; do
        [[ ! -e $MODELS/$d ]] || { echo "$MODELS/$d exists: move it aside (or set ONLY to the drugs still to run)"; exit 1; }
    done
    md5=$(md5sum < "$SPLITS/split_manifest.json" | cut -d' ' -f1)
    sha=$(git -C "$array_repo" rev-parse HEAD)
    drugs=$(IFS=:; echo "${PA_DRUGS[*]}")
    # ~16 min per epoch at ~2,500 training genomes, 40-epoch cap, plus validation passes: ≤ ~14 h; 35 h stays under
    # FLOTO-SL2-GPU's 36 h QOS cap. mem 250G = 32 cores × 8,000 MB, ampere's one-GPU ceiling
    submit --array=0-$((${#PA_DRUGS[@]} - 1))${ARRAY_CONCURRENT:+%$ARRAY_CONCURRENT} \
        --time="${TRAIN_TIME:-35:00:00}" --job-name=pa-finetune \
        --export="ALL,REPO=$array_repo,EXPECT_SHA=$sha,TASK=pa_ast,DRUGS=$drugs,SPLIT_DIR=$SPLITS,SPLIT_MANIFEST_MD5=$md5,EMBEDDINGS_DIR=$ESM,OUT_ROOT=$MODELS" \
        "$array_repo/$ARRAY"
    echo "drugs (array index order): ${PA_DRUGS[*]}"
    echo "models: $MODELS/<drug>/ (split tables md5-pinned: split_manifest.json $md5)"
    ;;
collect)
    uv run --no-sync python -m bacpredict.engine.finetune.collect_holdout_metrics --models-dir "$MODELS" \
        --drugs "${PA_DRUGS[@]}" --out "$MODELS/holdout_metrics.csv"
    ;;
*)
    sed -n '2,10p' "$0"; exit 2 ;;
esac

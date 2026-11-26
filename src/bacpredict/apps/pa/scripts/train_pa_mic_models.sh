#!/usr/bin/env bash
# Does training on the MIC beat training on the R/S call? The PsA-only MIC arms, one stage at a time, from a
# CSD3 login node:
#
#   bash src/bacpredict/apps/pa/scripts/train_pa_mic_models.sh intervals       # censored-MIC tables -> psaonly_mic/intervals
#   bash src/bacpredict/apps/pa/scripts/train_pa_mic_models.sh check-intervals # what each arm will read, per drug
#   ARRAY_REPO=<worktree> bash …/train_pa_mic_models.sh smoke                  # Stage A: one drug, 10 genomes, one GPU
#   bash src/bacpredict/apps/pa/scripts/train_pa_mic_models.sh check-smoke     # Stage A: every output written
#   bash src/bacpredict/apps/pa/scripts/train_pa_mic_models.sh overfit         # Stage B: the smoke's loss -> ~0
#   ARRAY_REPO=<worktree> bash …/train_pa_mic_models.sh submit                 # Stage C: one GPU array per arm
#   bash src/bacpredict/apps/pa/scripts/train_pa_mic_models.sh collect         # every arm x drug, side by side
#
# ---------------------------------------------------------------------------------------------------------
# THE QUESTION. A binary R/S label is a continuous measurement cut at one threshold. Cox's 1957 grouping
# result says dichotomising a Gaussian keeps 2/pi ~ 64 % of its information and ~10 ordered bins keep 97-98 %,
# so the MIC carries more about each genome than the call derived from it. Whether a model trained on it is
# *better at the call* is a different question, and an empirical one — this is the experiment that asks it.
#
# THE ARMS. Four, over five drugs, one seed (David, 2026-11-26: seed-to-seed differences are small next to
# the effect being looked for; five drugs on one seed shows whether it is there). Every arm reads the SAME
# split tables, so no genome changes split and the holdouts are nested.
#
#   arm 0   BCE, every cell                 THE INCUMBENT — already trained, at cross_species_psaonly_v2/models
#   arm 0'  BCE, MIC-bearing cells only     the control: isolates the cohort change from the loss change
#   arm A   least squares on a point MIC    David's hand rule (">= X" -> X, "> X" -> one dilution up), the
#                                           comparator this whole design exists to test censoring against
#   arm B   interval-censored Gaussian      the proposed estimator: every cell, each as the interval the
#                                           assay actually reports, bare R/S calls included as open intervals
#
# Arm 0 is NOT re-run. It exists, it was trained by this same trainer from these same tables at this same
# seed, and re-running it would only add noise to the comparison.
#
# WHY arm 0' EXISTS. Dropping the cells with no measured MIC is not a random thinning: 0.6-2.3 points of
# resistant share move with it. Without 0', a difference between arm 0 and arm A/B could be the loss or could
# be the cohort, and nothing in the result would say which.
#
# THE FIVE DRUGS were chosen before any of this was built, on the v1 -> v2 PsA-only AUROCs: moderate-or-poor
# in v1 and moderate-or-good in v2, where there is room to move. Excluded: tobramycin (0.933 -> 0.941, David's
# instruction — already excellent) and imipenem (0.884 -> 0.915, already strong).
#
# SIGMA. The censored model needs a residual spread for log2 MIC, in doubling dilutions. It is learned during
# training and then **re-fitted once, on validate, at the selected checkpoint** (David, 2026-11-26). Fitting it
# on train reads the scale of residuals the weights are already fitted to; fitting it on holdout is leakage.
# Re-fitting one scale parameter on held-out data is temperature scaling, and it provably cannot move AUROC,
# AUPRC or the R/S call — only the calibration. engine/finetune/mic_trainer says why in full.
#
# WHAT IS REPORTED, and why it is not AUROC alone. On known-truth simulation (tests/engine/finetune/
# test_mic_estimator_recovery.py) the point-imputed arm recovers barely half the latent MIC scale and still
# scores an AUROC within 0.03 of the censored arm, while its balanced accuracy falls ~0.1 and its calibration
# error triples. AUROC is rank-only and survives the distortion. So `collect` reports, per arm: AUROC and
# AUPRC; the R/S call (balanced accuracy, sensitivity, specificity); calibration (Brier, ECE) both at the
# validate sigma and at the training one, so the size of the temperature correction is visible; and for the
# MIC arms the censored NLL, interval coverage, and absolute and essential (+-1 dilution) agreement. Essential
# agreement is reported because it is the field's convention and the assay's own reproducibility ceiling
# (ISO 20776-2) — never as the headline.
#
# NOT IN THIS EXPERIMENT, and worth knowing. A censored MIC that straddles the breakpoint is `mic_undecided`:
# it has no binary label, so it is not in the split tables, so it is not here. The censored likelihood could
# use those cells — their interval is informative even when their call is not — but adding them would change
# the cohort and break the pairing with arm 0. `intervals` counts them so the size of that future gain is on
# the record.
#
# Settings: SMOKE_DRUG / SMOKE_ARM, ONLY (comma-separated drugs), ARMS (comma-separated arm names),
# ARRAY_CONCURRENT (a %N throttle), SMOKE_TIME / TRAIN_TIME (walls), OVERWRITE=1 (rebuild interval tables).
set -euo pipefail

# What the array reads from the environment (sbatch --export=ALL) must come from this driver, not a stale shell
unset N_SAMPLES MODEL LR EVAL_STEPS MAX_STEPS MAX_EPOCHS PATIENCE THRESHOLD SEED NUM_WORKERS EXTRA_HOLDOUTS \
      TASK DRUGS SPLIT_DIR SPLIT_MANIFEST_MD5 EMBEDDINGS_DIR OUT_ROOT REPO EXPECT_SHA \
      MIC_OBJECTIVE MIC_DIR MIC_CELLS MIC_SIGMA_LR

ROOT=${BACPREDICT_AMR_ROOT:-$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/amr_prediction}
[[ -d $ROOT ]] || { echo "no AMR root at $ROOT (set BACPREDICT_AMR_ROOT)"; exit 1; }
HERE=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"

# The split tables and the incumbent arm 0 are the v2 PsA-only run's; nothing here rebuilds either.
BASE=$ROOT/processed/cross_species_psaonly_v2
SPLITS=$BASE/splits
ARM0=$BASE/models
# The MIC intervals come from the CABBAGE v2 test table, which covers 100 % of those split tables (the older
# processed/paeruginosa one covers 35 %, so it is the wrong cohort and would silently drop two thirds of them).
TESTS=$ROOT/processed/paeruginosa_cabbage_v2/ast_tests_called.parquet
ESM=$ROOT/processed/esm
T=$ROOT/processed/psaonly_mic
INTERVALS=$T/intervals
MODELS=$T/models

PA_DRUGS=(amikacin cefepime ceftazidime ciprofloxacin meropenem)
if [[ -n ${ONLY:-} ]]; then IFS=, read -r -a PA_DRUGS <<<"$ONLY"; fi
# arm name -> "MIC_OBJECTIVE MIC_CELLS": the three arms this driver trains (arm 0 already exists)
declare -A ARM_SPEC=(
    [bce_mic_cells]="bce mic_only"          # arm 0': the incumbent loss, the MIC arms' cohort
    [mse_imputed_point]="mse mic_only"      # arm A : least squares on David's hand-imputed point
    [censored_interval]="censored all"      # arm B : the proposed estimator, every cell
)
ARM_ORDER=(bce_mic_cells mse_imputed_point censored_interval)
if [[ -n ${ARMS:-} ]]; then IFS=, read -r -a ARM_ORDER <<<"$ARMS"; fi
SMOKE_DRUG=${SMOKE_DRUG:-ciprofloxacin}
SMOKE_ARM=${SMOKE_ARM:-censored_interval}
TASK_LABEL=pa_mic_ast

ARRAY=src/bacpredict/engine/finetune/scripts/finetune_amr_array.sh
# what the array runs in ARRAY_REPO: it must be this driver's own copy of each (same_array_code)
ARRAY_CODE=("$ARRAY" src/bacpredict/engine/finetune/finetune_amr.py src/bacpredict/engine/finetune/datasets.py
            src/bacpredict/engine/finetune/metrics.py src/bacpredict/engine/finetune/attention_pool.py
            src/bacpredict/engine/finetune/mic_trainer.py
            src/bacpredict/engine/finetune/censored_mic_loss.py
            src/bacpredict/engine/ast_labels/mic_intervals.py
            src/bacpredict/engine/splits/load_splits.py
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

same_array_code() {  # refuse unless the array script and the trainer modules it runs are this checkout's
    local f
    for f in "${ARRAY_CODE[@]}"; do
        cmp -s "$HERE/$f" "$1/$f" || {
            echo "ARRAY_REPO's $1/$f is not this driver's $HERE/$f (differs or missing):"
            echo "check out this driver's SHA in ARRAY_REPO, or run the driver from ARRAY_REPO"; exit 1; }
    done
}

no_live() {  # a failed squeue is not "none live"
    local jobs live
    jobs=$(squeue -u "$USER" -h -o '%j') || { echo "squeue failed: cannot tell whether a $1 job is live"; exit 1; }
    live=$(grep -cx -- "$1" <<<"$jobs" || true)
    (( live == 0 )) || { echo "a $1 job is queued or running: wait for it (or scancel it) first"; exit 1; }
}

need_tables() {  # the split tables AND the interval tables, before anything is submitted
    local d
    [[ -s $SPLITS/split_manifest.json ]] || { echo "no $SPLITS/split_manifest.json"; exit 1; }
    for d in "$@"; do
        [[ -s $SPLITS/${d}_split.csv ]] || { echo "no $SPLITS/${d}_split.csv"; exit 1; }
        [[ -s $INTERVALS/${d}_mic_intervals.csv ]] || { echo "no $INTERVALS/${d}_mic_intervals.csv: run intervals"; exit 1; }
    done
}

arm_spec() {  # arm name -> "objective cells"; an unknown name is a typo, not an empty arm
    [[ -n ${ARM_SPEC[$1]:-} ]] || { echo "unknown arm '$1'; known: ${!ARM_SPEC[*]}" >&2; exit 1; }
    echo "${ARM_SPEC[$1]}"
}

latest_smoke() {
    [[ -s $MODELS/smoke/LATEST ]] || { echo "no $MODELS/smoke/LATEST: run smoke first" >&2; exit 1; }
    cat "$MODELS/smoke/LATEST"
}

case "${1:-}" in
intervals)
    [[ -s $TESTS ]] || { echo "no $TESTS"; exit 1; }
    mkdir -p "$INTERVALS"
    if [[ -z ${OVERWRITE:-} ]]; then
        for d in "${PA_DRUGS[@]}"; do
            [[ ! -e $INTERVALS/${d}_mic_intervals.csv ]] || {
                echo "$INTERVALS/${d}_mic_intervals.csv exists: set OVERWRITE=1 to rebuild (every model trained"
                echo "on the old table is then invalid)"; exit 1; }
        done
    fi
    # light: one parquet filter plus pandas over ~73k test rows. NOT `python -I`, which ignores PYTHONPATH
    uv run --no-sync python -m bacpredict.engine.ast_labels.mic_intervals \
        --tests "$TESTS" --split-dir "$SPLITS" --species paeruginosa \
        --drugs "${PA_DRUGS[@]}" --out-dir "$INTERVALS"
    # how many cells the breakpoint-straddling censored MICs would add — not used here (see the header), counted
    uv run --no-sync python - "$TESTS" "${PA_DRUGS[@]}" <<'EOF'
import sys
import pandas as pd
t = pd.read_parquet(sys.argv[1])
t = t[t.drug.isin(sys.argv[2:]) & (t.species.astype(str) == "paeruginosa")]
u = t[t.basis.astype(str) == "mic_undecided"]
print(f"\nnot in this experiment: {len(u)} mic_undecided tests over "
      f"{u.groupby(['Sample', 'drug']).ngroups} (genome, drug) cells — censored MICs that straddle the "
      f"breakpoint, so they have no binary label and are in no split table. The censored likelihood could "
      f"use them; adding them would change the cohort.")
EOF
    ;;
check-intervals)
    need_tables "${PA_DRUGS[@]}"
    # exactly what each array task will read: resolve_split_table -> resolve_mic_cells -> missing_embeddings
    uv run --no-sync python - "$SPLITS" "$INTERVALS" "$ESM" "$(IFS=,; echo "${ARM_ORDER[*]}")" \
        "$(for a in "${ARM_ORDER[@]}"; do arm_spec "$a" | tr ' ' ':'; echo -n ","; done)" "${PA_DRUGS[@]}" <<'EOF'
import sys
from bacpredict.engine.finetune.finetune_amr import missing_embeddings, resolve_mic_cells, resolve_split_table
splits, intervals, esm, arm_names, arm_specs, *drugs = sys.argv[1:]
arms = list(zip(arm_names.split(","), [s for s in arm_specs.split(",") if s], strict=True))
bad = 0
print(f"{'drug':<16}{'arm':<20}{'train':>7}{'valid':>7}{'hold':>7}{'%R':>7}{'ST':>5}  missing .pt")
for d in drugs:
    s = resolve_split_table(f"{splits}/{d}_split.csv")
    pct = lambda ids: 100 * sum(s.label_map[i] for i in ids) / len(ids) if ids else float("nan")
    print(f"{d:<16}{'0  (incumbent)':<20}{len(s.train):>7}{len(s.validate):>7}{len(s.holdout):>7}"
          f"{pct(s.holdout):>7.1f}{len(s.extra['holdout_st']):>5}")
    for name, spec in arms:
        objective, cells = spec.split(":")
        f, intervals_map, _, _ = resolve_mic_cells(s, f"{intervals}/{d}_mic_intervals.csv", objective, cells)
        miss = missing_embeddings(f.train + f.validate + f.holdout + f.extra["holdout_st"], esm)
        bad += bool(miss)
        print(f"{'':<16}{name:<20}{len(f.train):>7}{len(f.validate):>7}{len(f.holdout):>7}"
              f"{pct(f.holdout):>7.1f}{len(f.extra['holdout_st']):>5}  {len(miss)} {miss[:3]}")
sys.exit(1 if bad else 0)
EOF
    ;;
smoke)
    array_repo=${ARRAY_REPO:?set ARRAY_REPO to a detached worktree at the SHA to run (see the header)}
    same_array_code "$array_repo"
    need_tables "$SMOKE_DRUG"
    read -r objective cells <<<"$(arm_spec "$SMOKE_ARM")"
    out_root=$MODELS/smoke/$(date +%Y%m%d_%H%M%S)/$SMOKE_ARM
    md5=$(md5sum < "$SPLITS/split_manifest.json" | cut -d' ' -f1)
    sha=$(git -C "$array_repo" rev-parse HEAD)
    # 200 optimizer steps at a constant LR on a balanced 10 genomes (train = validate = holdout), an evaluation
    # and a checkpoint each epoch: well under an hour, so a tight wall. PATIENCE=1000 turns early stopping off —
    # on train = validate the AUROC saturates within a few epochs and patience 15 would end the run before the
    # loss had fallen, which is the thing Stage B is there to see.
    submit --array=0 --time="${SMOKE_TIME:-01:00:00}" --job-name=pa-mic-finetune-smoke \
        --export="ALL,REPO=$array_repo,EXPECT_SHA=$sha,TASK=$TASK_LABEL,DRUGS=$SMOKE_DRUG,SPLIT_DIR=$SPLITS,SPLIT_MANIFEST_MD5=$md5,EMBEDDINGS_DIR=$ESM,OUT_ROOT=$out_root,N_SAMPLES=10,PATIENCE=1000,MIC_DIR=$INTERVALS,MIC_OBJECTIVE=$objective,MIC_CELLS=$cells" \
        "$array_repo/$ARRAY"
    mkdir -p "$MODELS/smoke"
    echo "$out_root/$SMOKE_DRUG" > "$MODELS/smoke/LATEST"
    echo "smoke run: $out_root/$SMOKE_DRUG (arm $SMOKE_ARM = $objective on $cells cells)"
    ;;
check-smoke)
    run=$(latest_smoke)
    uv run --no-sync python -I - "$run" <<'EOF'
import json, math, sys
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
mic = r.get("mic")
if not mic:
    problems.append("no top-level 'mic' block: this run did not go through the MIC path at all")
elif not math.isfinite(mic.get("sigma", float("nan"))):
    problems.append(f"sigma is {mic.get('sigma')}: the validate re-fit did not produce a number")
else:
    print(f"sigma: train {mic['sigma_train']:.4f} -> validate re-fit {mic['sigma']:.4f} "
          f"(at_bound={mic['sigma_validate_fit']['at_bound']}, n={mic['sigma_validate_fit']['n']})")
    print(f"cells: {mic['objective']} on {mic['cells']}; " + ", ".join(
        f"{k} {v['n_before']}->{v['n_after']}" for k, v in mic["by_split"].items()))
for name, block in r.get("holdouts", {}).items():
    npz = run / block["scores"]
    if not npz.exists():
        problems.append(f"no {npz}"); continue
    z = np.load(npz)
    for key in ("mu", "lo", "hi", "sigma"):
        if key not in z:
            problems.append(f"{npz.name}: no '{key}' array (a MIC arm's scores must carry the MIC scale)")
    if len(z["sample_ids"]) != block["n_samples"]:
        problems.append(f"{npz.name}: {len(z['sample_ids'])} rows for n_samples {block['n_samples']}")
    m, mb = block["metrics"], block.get("mic", {})
    if not mb:
        problems.append(f"{name}: no per-holdout 'mic' block")
    else:
        print(f"{name}: n={block['n_samples']} AUROC={m['auroc']:.3f} bal_acc={m['balanced_accuracy']:.3f} "
              f"NLL={mb['censored_nll']:.3f} ECE={mb['calibration']['ece']:.3f} "
              f"coverage={mb['interval_coverage']:.3f} EA={mb['essential_agreement']} scores={npz.name}")
if not list(run.glob("checkpoint-*/config.json")):
    problems.append("no checkpoint-*/config.json")
if not list(run.glob("checkpoint-*/mic_sigma.json")):
    problems.append("no checkpoint-*/mic_sigma.json: sigma would not survive a resume")
if not (run / "trainer_state.json").exists():
    problems.append("no trainer_state.json (the whole run's history, which overfit reads)")
for p in problems:
    print(f"FAIL {p}")
print("Stage A: PASS" if not problems else "Stage A: FAIL")
sys.exit(1 if problems else 0)
EOF
    ;;
overfit)
    run=$(latest_smoke)
    # engine/finetune/check_overfit: the whole run's history (<run>/trainer_state.json), the full schedule run,
    # last train loss <= OVERFIT_MAX_LOSS, train = holdout AUROC >= 0.99. The censored NLL reaches ~0 on an
    # overfit for the same reason BCE does — the interval probability goes to 1 as sigma shrinks onto mu.
    uv run --no-sync python -m bacpredict.engine.finetune.check_overfit "$run" --max-loss "${OVERFIT_MAX_LOSS:-0.1}"
    ;;
submit)
    array_repo=${ARRAY_REPO:?set ARRAY_REPO to a detached worktree at the SHA to run (see the header)}
    same_array_code "$array_repo"
    need_tables "${PA_DRUGS[@]}"
    no_live pa-mic-finetune
    for a in "${ARM_ORDER[@]}"; do
        arm_spec "$a" >/dev/null
        for d in "${PA_DRUGS[@]}"; do
            [[ ! -e $MODELS/$a/$d ]] || { echo "$MODELS/$a/$d exists: move it aside (or set ONLY/ARMS)"; exit 1; }
        done
    done
    md5=$(md5sum < "$SPLITS/split_manifest.json" | cut -d' ' -f1)
    sha=$(git -C "$array_repo" rev-parse HEAD)
    drugs=$(IFS=:; echo "${PA_DRUGS[*]}")
    # One array per arm, not one array over arm x drug: each arm needs its own MIC_OBJECTIVE/MIC_CELLS/OUT_ROOT,
    # and a separate job name per arm is what makes the queue readable.
    #
    # The wall is 12 h. The measured PsA-only v2 fine-tunes took 4.0-7.2 h per drug on these cohorts, and arms
    # 0'/A train on FEWER cells than that, so 12 h is ~1.7x the slowest measured run. It is deliberately not
    # 35 h: on 2026-11-24 four jobs asking 35 h on ampere were given start estimates a day out and the same
    # jobs at 12 h started within five minutes. A run that does hit the wall resumes from its checkpoint
    # (finetune_amr --resume-from-checkpoint), it does not restart.
    for a in "${ARM_ORDER[@]}"; do
        read -r objective cells <<<"$(arm_spec "$a")"
        echo "arm $a: --mic-objective $objective --mic-cells $cells -> $MODELS/$a/<drug>"
        submit --array=0-$((${#PA_DRUGS[@]} - 1))${ARRAY_CONCURRENT:+%$ARRAY_CONCURRENT} \
            --time="${TRAIN_TIME:-12:00:00}" --job-name=pa-mic-finetune \
            --export="ALL,REPO=$array_repo,EXPECT_SHA=$sha,TASK=$TASK_LABEL,DRUGS=$drugs,SPLIT_DIR=$SPLITS,SPLIT_MANIFEST_MD5=$md5,EMBEDDINGS_DIR=$ESM,OUT_ROOT=$MODELS/$a,MIC_DIR=$INTERVALS,MIC_OBJECTIVE=$objective,MIC_CELLS=$cells" \
            "$array_repo/$ARRAY"
    done
    echo "drugs (array index order): ${PA_DRUGS[*]}"
    echo "arm 0 (the incumbent) is NOT re-run: it is at $ARM0/<drug>/results.json"
    ;;
collect)
    uv run --no-sync python -m bacpredict.apps.pa.collect_mic_arms \
        --arm "0_bce_all_cells=$ARM0" \
        $(for a in "${ARM_ORDER[@]}"; do echo --arm "$a=$MODELS/$a"; done) \
        --drugs "${PA_DRUGS[@]}" --intervals-dir "$INTERVALS" --out "$MODELS/mic_arm_metrics.csv"
    ;;
*)
    sed -n '2,8p' "$0"; exit 2 ;;
esac

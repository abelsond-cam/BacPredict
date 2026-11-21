#!/usr/bin/env bash
# The Klebsiella-only AMR models (a single-species baseline arm: one model per drug), one stage at a time, from a
# CSD3 login node:
#
#   bash src/bacpredict/apps/entero/scripts/train_kpsc_models.sh splits        # split tables → processed/kpsc/splits/
#   bash src/bacpredict/apps/entero/scripts/train_kpsc_models.sh check-splits  # what the trainer will read, per drug
#   ARRAY_REPO=<worktree> bash …/train_kpsc_models.sh smoke                    # Stage A: one drug, 10 genomes, one GPU
#   bash src/bacpredict/apps/entero/scripts/train_kpsc_models.sh check-smoke   # Stage A: every output written
#   bash src/bacpredict/apps/entero/scripts/train_kpsc_models.sh overfit       # Stage B: the smoke's loss -> ~0, AUROC 1
#   ARRAY_REPO=<worktree> bash …/train_kpsc_models.sh submit                   # Stage C: GPU array, one task per drug
#   bash src/bacpredict/apps/entero/scripts/train_kpsc_models.sh collect       # holdout + holdout_st metrics, every drug
#
# The same arm as apps/pa/scripts/train_pa_models.sh, for the KpSC genomes of the Enterobacterales cohort, so that
# "does Klebsiella improve under cross-species training?" is answered the way it is for PsA (David, 2026-11-21).
# Every stage below is that driver's, unchanged but for what the organism decides.
#
# Splits are engine/splits/build_amr_holdout_splits (PROJECT_STATE §6, 2026-11-21): ST37 held out whole as
# holdout_st (exact ST, 1.8-2.5 % of each drug's labelled set, inside the <= 5 % cap on a lineage holdout), a random
# 20 % holdout of what is left, 5 folds over the rest (fold 0 validates). There is NO exclusion list and NO
# duplicate-isolate group file for this cohort - PsA has both and Klebsiella has neither, which is a real difference
# between the two arms, not an oversight (PROJECT_STATE §3.6 Caveats).
#
# Eligible = the `kpsc` unit of apps/entero/species_units.py, which NAMES the five KpSC species rather than matching
# the genus: K. aerogenes is not in the complex, and the 2026-07 and CABBAGE v2 label sheets spell it differently.
# `splits` refuses unless each eligible genome has a .pt, because the builder shuffles the eligible set as a whole:
# a genome added later would move every other genome's split, and the Klebsiella rows must stay identical to the
# cross-species arm's (--assert-same-as).
#
# Training is engine/finetune/scripts/finetune_amr_array.sh: finetune_amr --split-table, bf16, early stopping, the
# deployed TB/Kp hyperparameters - the same settings as the PsA arm, so the two baselines differ only in organism.
# Stage A and B are one run: the trainer's 10-genome path trains, validates and scores on the same balanced 10 for
# 200 optimizer steps at a constant LR (the smoke sets PATIENCE=1000: early stopping off). `smoke` submits it,
# `check-smoke` checks its outputs (Stage A) and `overfit` reads its whole loss history from
# <run>/trainer_state.json (Stage B). smoke and submit need ARRAY_REPO, a detached worktree at the SHA to run
# (.venv symlinked to the shared env), whose trainer code must be byte-identical to this checkout's; the array is
# pinned to its SHA (EXPECT_SHA) and to the split tables (split_manifest.json md5).
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
# inputs are the Enterobacterales cohort's; outputs are this arm's own folder
ENTERO=$ROOT/processed/enterobacterales
LABELS=$ENTERO/binary_ast.csv
QC=$ENTERO/genomes/genome_qc.csv
T=$ROOT/processed/kpsc
ESM=$ROOT/processed/esm
SPLITS=$T/splits
ELIGIBLE=$SPLITS/eligible_genomes.csv
MODELS=$T/models
# the split-builder unit name. It is the split tables' species column AND seeds the unit's shuffle, so it must be
# spelled identically here, in the cross-species arm and in CABBAGE v2 (§3.7) or --assert-same-as fails
UNIT=kpsc
HOLDOUT_ST=klebsiella:37
KP_DRUGS=(ciprofloxacin ceftazidime meropenem tobramycin amikacin imipenem piperacillin-tazobactam aztreonam
          levofloxacin cefepime)
if [[ -n ${ONLY:-} ]]; then IFS=, read -r -a KP_DRUGS <<<"$ONLY"; fi
SMOKE_DRUG=${SMOKE_DRUG:-ciprofloxacin}
ARRAY=src/bacpredict/engine/finetune/scripts/finetune_amr_array.sh
# what the array runs in ARRAY_REPO: it must be this driver's own copy of each (same_array_code)
ARRAY_CODE=("$ARRAY" src/bacpredict/engine/finetune/finetune_amr.py src/bacpredict/engine/finetune/datasets.py
            src/bacpredict/engine/finetune/metrics.py src/bacpredict/engine/finetune/attention_pool.py
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
    for f in "$LABELS" "$QC"; do [[ -s $f ]] || { echo "no $f"; exit 1; }; done
    mkdir -p "$SPLITS"
    # eligible = the unit's QC-passing genomes (apps/entero/species_units), each with a .pt, written only when it
    # changes (its md5 goes into the manifest). NOT `python -I`: that ignores PYTHONPATH, so the import below would
    # silently come from whatever checkout the shared .venv's editable install points at
    uv run --no-sync python - "$QC" "$UNIT" "$ESM" "$ELIGIBLE" <<'EOF'
import os, sys
from pathlib import Path
import pandas as pd
from bacpredict.apps.entero.species_units import unit_samples
qc_path, unit, esm, out = sys.argv[1:5]
samples = unit_samples(pd.read_csv(qc_path, dtype=str, low_memory=False), unit)
if not samples:
    sys.exit(f"no QC-passing genomes in unit {unit!r}")
missing = [s for s in samples if not (Path(esm) / f"{s}_esm_embeddings.pt").exists()]
if missing:
    sys.exit(f"{len(missing)} of {len(samples)} {unit} genomes have no .pt in {esm}: {missing[:10]}")
text = "Sample\n" + "".join(f"{s}\n" for s in samples)
if not os.path.exists(out) or open(out).read() != text:
    tmp = f"{out}.{os.getpid()}.part"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, out)
print(f"eligible: {len(samples)} {unit} genomes, each with a .pt -> {out}")
EOF
    # no --exclude and no --groups: neither file exists for this cohort (see the header)
    uv run --no-sync python -m bacpredict.engine.splits.build_amr_holdout_splits \
        --species "$UNIT=$LABELS,$ELIGIBLE,$QC" --holdout-st "$UNIT=$HOLDOUT_ST" \
        --drugs "${KP_DRUGS[@]}" --out-dir "$SPLITS" ${OVERWRITE:+--overwrite}
    ;;
check-splits)
    need_tables "${KP_DRUGS[@]}"
    # exactly the trainer's own read (resolve_split_table, missing_embeddings): what each array task will see
    uv run --no-sync python - "$SPLITS" "$ESM" "${KP_DRUGS[@]}" <<'EOF'
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
    # 200 optimizer steps at a constant LR (10 genomes at grad-accum 8 = 1 per epoch) plus an evaluation and a checkpoint
    # each epoch: well under an hour (short job: a tight wall). PATIENCE=1000 turns early stopping off: on train =
    # validate the AUROC saturates within a few epochs, and patience 15 would end the run long before the loss falls
    submit --array=0 --time="${SMOKE_TIME:-01:00:00}" --job-name=kpsc-finetune-smoke \
        --export="ALL,REPO=$array_repo,EXPECT_SHA=$sha,TASK=kpsc_ast,DRUGS=$SMOKE_DRUG,SPLIT_DIR=$SPLITS,SPLIT_MANIFEST_MD5=$md5,EMBEDDINGS_DIR=$ESM,OUT_ROOT=$out_root,N_SAMPLES=10,PATIENCE=1000" \
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
    # engine/finetune/check_overfit: the whole run's history (<run>/trainer_state.json), the full schedule run, last
    # train loss <= OVERFIT_MAX_LOSS, train = holdout AUROC >= 0.99
    uv run --no-sync python -m bacpredict.engine.finetune.check_overfit "$run" --max-loss "${OVERFIT_MAX_LOSS:-0.1}"
    ;;
submit)
    array_repo=${ARRAY_REPO:?set ARRAY_REPO to a detached worktree at the SHA to run (see the header)}
    same_array_code "$array_repo"
    need_tables "${KP_DRUGS[@]}"
    no_live kpsc-finetune
    for d in "${KP_DRUGS[@]}"; do
        [[ ! -e $MODELS/$d ]] || { echo "$MODELS/$d exists: move it aside (or set ONLY to the drugs still to run)"; exit 1; }
    done
    md5=$(md5sum < "$SPLITS/split_manifest.json" | cut -d' ' -f1)
    sha=$(git -C "$array_repo" rev-parse HEAD)
    drugs=$(IFS=:; echo "${KP_DRUGS[*]}")
    # KpSC trains on 2,286-4,885 genomes per drug, about twice PsA's, so size TRAIN_TIME from PsA's measured
    # 4.0-7.2 h per drug rather than from this default. 35 h stays under FLOTO-SL2-GPU's 36 h QOS cap, but a wall
    # that long barely backfills on a busy ampere: pass TRAIN_TIME. mem 250G = 32 cores x 8,000 MB, its one-GPU ceiling
    submit --array=0-$((${#KP_DRUGS[@]} - 1))${ARRAY_CONCURRENT:+%$ARRAY_CONCURRENT} \
        --time="${TRAIN_TIME:-35:00:00}" --job-name=kpsc-finetune \
        --export="ALL,REPO=$array_repo,EXPECT_SHA=$sha,TASK=kpsc_ast,DRUGS=$drugs,SPLIT_DIR=$SPLITS,SPLIT_MANIFEST_MD5=$md5,EMBEDDINGS_DIR=$ESM,OUT_ROOT=$MODELS" \
        "$array_repo/$ARRAY"
    echo "drugs (array index order): ${KP_DRUGS[*]}"
    echo "models: $MODELS/<drug>/ (split tables md5-pinned: split_manifest.json $md5)"
    ;;
collect)
    uv run --no-sync python -m bacpredict.engine.finetune.collect_holdout_metrics --models-dir "$MODELS" \
        --drugs "${KP_DRUGS[@]}" --out "$MODELS/holdout_metrics.csv"
    ;;
*)
    sed -n '2,11p' "$0"; exit 2 ;;
esac

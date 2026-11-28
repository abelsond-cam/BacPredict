#!/usr/bin/env bash
# TB rifampicin, three fine-tunes that differ ONLY in how the master weights are held, from a CSD3 login node:
#
#   bash src/bacpredict/apps/tb/scripts/train_rifampicin_precision_arms.sh check          # store, table, worktree
#   ARRAY_REPO=<worktree> bash …/train_rifampicin_precision_arms.sh smoke                 # Stage A/B, 10 genomes
#   ARRAY_REPO=<worktree> BEGIN=now+3hours bash …/train_rifampicin_precision_arms.sh submit
#   bash src/bacpredict/apps/tb/scripts/train_rifampicin_precision_arms.sh collect        # the three AUROCs
#
# WHY. The deployed TB rifampicin fine-tune reads 0.9642 (July, bf16, Isambard); its May predecessor read 0.9046
# (fp32, CSD3). The base weights were identical — all three cached revisions of the complete-genomes repo symlink
# one safetensors blob, and `bacformer.embeddings.mask_embed` is byte-identical to that blob in BOTH checkpoints —
# so the gap is not a model refresh. What differs is the master-weight dtype, and comparing each checkpoint against
# the base shows why that is not merely "less precise": under bf16, 90 of the 92 backbone LayerNorm gains never
# moved at all (each step is ~1e-4 against a bf16 half-ULP of ~2e-3 near 1.0, so every update rounds back), and a
# median 62 % of individual elements in the tensors that did move never changed. Under fp32, none of the 92 stood
# still. bf16 was training far fewer effective degrees of freedom — an implicit regulariser.
#
# THE ARMS. One drug, one split table, one schedule; the only differences are named here.
#   bf16            master weights bf16          — what today's deployed setting gives, the control
#   fp32            master weights fp32 + bf16 AMP — the May condition, which has NEVER been scored on a holdout
#   fp32_lnfrozen   fp32, backbone LayerNorm gains frozen — the regularisation without the precision
# If fp32_lnfrozen lands near bf16, the freezing is the mechanism; if it lands near fp32, precision does something
# else and the hypothesis is wrong. Either answer is worth having, and it generalises: every AMR fine-tune in this
# repo, and the invasion arm, trains in bf16.
#
# COMPARABILITY. All three pin the base to one hub revision (BASE_REVISION) — nothing pinned it before, and the
# complete-genomes modeling_bacformer.py differs across revisions (the 28 Apr → 25 May one moved norm1 from after
# the projection to before it). All three read rifampin_split.csv, whose md5 is checked here and recorded by the
# trainer, and whose holdout (7,127) and validate (5,702) match the deployed run's genome-for-genome counts.
# All three run the same MAX_STEPS, so all three get the same linear schedule and anneal fully — which the July run
# did not, having been cut off at step 31,000 of a 100,000-step schedule. So these are comparable to EACH OTHER,
# and are not a literal reproduction of 0.9642.
set -euo pipefail

ROOT=${BACPREDICT_DATA_ROOT:-$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/bac_ast_prediction}
T=$ROOT/processed/train_tb_ast
SPLIT_DIR=$T/splits
TABLE=$SPLIT_DIR/rifampin_split.csv
TABLE_MD5=${TABLE_MD5:-da18a9e73c19bf225d92a9828a5a2f31}
ESM=$T/esm
OUT=$T/models/precision_arms
N_STORE=${N_STORE:-38317}          # esm/tb after the 2026-11-28 top-up: the 38,248 the RDV note
                                   # recorded, plus the 69 parquets that had never been embedded
HERE=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
ARRAY=src/bacpredict/engine/finetune/scripts/finetune_amr_array.sh
# 35,000 steps at ~2,850 steps/epoch is ~12.3 epochs, past the 10.9 where the July bf16 run was still climbing,
# and ~30 h at ~3 s/step plus ~2.4 h of validation — inside the 35 h wall, so each arm is ONE job and no chain.
MAX_STEPS=${MAX_STEPS:-35000}
EVAL_STEPS=${EVAL_STEPS:-1000}     # the deployed TB setting: ~2.8 evals/epoch on this cohort
PATIENCE=${PATIENCE:-45}           # ~15 epochs of no improvement, per root CLAUDE.md §0.2
BASE_REVISION=${BASE_REVISION:-84f85c37a05e55559142b842c1569e265183b554}
ARMS=(bf16 fp32 fp32_lnfrozen)
cd "$HERE"

submit() {  # sbatch, then print the resource line the global rules require, read back from SLURM not from intent
    local id job kind; id=$(sbatch --parsable "$@"); id=${id%%;*}
    job=$(scontrol show job "$id")
    kind=CPU; grep -q 'gres/gpu' <<<"$job" && kind="GPU x$(grep -m1 -oE 'gres/gpu=[0-9]+' <<<"$job" | cut -d= -f2)"
    echo "JOB $id $(grep -m1 -oE 'JobName=[^ ]+' <<<"$job") | $kind | $(grep -m1 -oE 'ReqTRES=[^ ]+' <<<"$job")" \
         "| $(grep -m1 -oE 'TimeLimit=[^ ]+' <<<"$job") | $(grep -m1 -oE 'Partition=[^ ]+' <<<"$job")" \
         "| $(grep -m1 -oE 'StartTime=[^ ]+' <<<"$job")"
}

arm_env() {  # the ONE place an arm differs from its siblings
    case $1 in
        bf16)          echo "PRECISION=bf16" ;;
        fp32)          echo "PRECISION=fp32" ;;
        fp32_lnfrozen) echo "PRECISION=fp32,FREEZE_LAYERNORM_GAINS=1" ;;
        *) echo "unknown arm $1" >&2; exit 1 ;;
    esac
}

check_store() {
    local n; n=$(ls "$ESM" 2>/dev/null | wc -l)
    echo "embedding store $ESM: $n files (expect $N_STORE)"
    (( n == N_STORE )) || { echo "store incomplete — the RDV restore has not finished or did not verify"; return 1; }
}

case ${1:-check} in
check)
    echo "split table : $TABLE"
    [[ -s $TABLE ]] || { echo "no split table"; exit 1; }
    now=$(md5sum < "$TABLE" | cut -d' ' -f1)
    echo "table md5   : $now (expect $TABLE_MD5)"
    [[ $now == "$TABLE_MD5" ]] || { echo "the split table changed: every arm would train a different holdout"; exit 1; }
    awk -F, 'NR>1{c[$NF]++} END{for (k in c) printf "  %-9s %d\n", k, c[k]}' "$TABLE"
    check_store || true
    echo "base pinned : $BASE_REVISION"
    echo "schedule    : max_steps=$MAX_STEPS eval_steps=$EVAL_STEPS patience=$PATIENCE"
    for a in "${ARMS[@]}"; do printf "  %-14s %s  -> %s\n" "$a" "$(arm_env "$a")" "$OUT/$a/rifampin"; done
    ;;
smoke|submit)
    : "${ARRAY_REPO:?set ARRAY_REPO to a detached worktree at this SHA (.venv symlinked to the shared env)}"
    sha=$(git -C "$ARRAY_REPO" rev-parse HEAD)
    [[ $sha == "$(git rev-parse HEAD)" ]] || { echo "ARRAY_REPO is at ${sha:0:7}, this driver at $(git rev-parse --short HEAD)"; exit 1; }
    common="TASK=tb_ast,DRUGS=rifampin,SPLIT_DIR=$SPLIT_DIR,EMBEDDINGS_DIR=$ESM,EXTRA_HOLDOUTS=,REPO=$ARRAY_REPO"
    common="$common,EXPECT_SHA=$sha,MODEL_REVISION=$BASE_REVISION,REQUIRE_ALL_EMBEDDINGS=1"
    if [[ $1 == smoke ]]; then
        # Stage A and B in one run (root §0.2): 10 balanced genomes, train = validate = holdout, early stopping off.
        # A short wall backfills sooner, which matters on a queue 600 deep.
        submit --array=0 --job-name=tb-rif-smoke --time=00:30:00 ${BEGIN:+--begin="$BEGIN"} \
            ${DEPEND:+--dependency="$DEPEND"} \
            --export="ALL,$common,OUT_ROOT=$OUT/_smoke,N_SAMPLES=10,PATIENCE=1000,PRECISION=bf16" "$ARRAY"
    else
        # The store may still be filling at submit time (the RDV restore). With BEGIN set, that is fine and
        # intended: the job itself re-checks every genome (REQUIRE_ALL_EMBEDDINGS) and exits 6 rather than
        # training a silently smaller cohort. Without BEGIN, a job could start immediately, so refuse.
        check_store || {
            [[ -n ${BEGIN:-}${DEPEND:-} ]] || {
                echo "refusing to submit against an incomplete store with no --begin and no --dependency"; exit 1; }
            echo "store incomplete now; queued behind ${DEPEND:-$BEGIN} and the job re-checks before it trains"; }
        for a in "${ARMS[@]}"; do
            submit --array=0 --job-name="tb-rif-$a" ${BEGIN:+--begin="$BEGIN"} ${DEPEND:+--dependency="$DEPEND"} \
                --export="ALL,$common,OUT_ROOT=$OUT/$a,MAX_STEPS=$MAX_STEPS,EVAL_STEPS=$EVAL_STEPS,PATIENCE=$PATIENCE,$(arm_env "$a")" \
                "$ARRAY"
        done
    fi
    ;;
collect)
    for a in "${ARMS[@]}"; do
        f=$OUT/$a/rifampin/results.json
        if [[ -s $f ]]; then
            python3 -c "
import json,sys
d=json.load(open(sys.argv[1])); m=d['metrics']; r=d.get('run',{})
print(f\"{sys.argv[2]:<14} AUROC {m['auroc']:.4f}  AUPRC {m['auprc']:.4f}  bal.acc {m['balanced_accuracy']:.4f}\"
      f\"  n={m['n_samples']}  precision={r.get('precision')}  frozen_gains={r.get('n_frozen_layernorm_gains')}\"
      f\"  rev={d['model'].get('revision','?')[:8]}\")" "$f" "$a"
        else
            echo "$a: no results.json yet"
        fi
    done
    ;;
*) echo "usage: $0 {check|smoke|submit|collect}"; exit 1 ;;
esac

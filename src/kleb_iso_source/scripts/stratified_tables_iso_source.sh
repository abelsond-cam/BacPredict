#!/bin/bash
#SBATCH --job-name=strat_tables_iso
#SBATCH --output=/rds/user/dca36/hpc-work/logs/strat_tables_iso_%j.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/strat_tables_iso_%j.err
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#
# Per-stratum AUROC tables + forest plots for the blood-vs-faeces model, at three scopes.
#
# SUBMIT WITH sbatch, do not run inline. The arithmetic is trivial (a 2,000-draw bootstrap over at
# most 14k rows), but the script makes twelve separate `uv run` calls and on this cluster each pays
# ~10 min of interpreter+import startup against a cold NFS uv cache — measured, the first inline run
# took ~12 min per table. That is well past the login-node budget, and a login-node process can be
# killed out from under a half-written table. mem=16G: peak is the 14k-row score frame, a few hundred
# MB, so this is ~30x headroom on a short job; 4 cores for the sklearn/BLAS calls.
#
# Two group columns x three split scopes = six tables:
#   Sublineage    x {evaluate, heldout, all}
#   Clonal group  x {evaluate, heldout, all}    <- note the SPACE in the column name
#
# WHY THREE SCOPES. The holdout alone leaves only 5 sublineages / 4 clonal groups at n>=100, too few
# to say the signal holds *within* a clone; the whole cohort lifts that to 20 / 15. But this model
# memorises hard — cohort scoring gives train 0.959 vs evaluate 0.786 — and train is 70% of the
# cohort, so an `all` AUROC is mostly recall of fitted rows and cannot show that a clone generalises.
# Hence `heldout` = validate + evaluate: neither was fitted on, n nearly doubles (1,412 + 2,822), and
# more groups clear n>=100 while the number stays honest. Quote `evaluate`; read `heldout` for the
# within-clone claim; treat `all` as the pattern check only. stratified_metrics stamps `split_scope`
# into every row and warns loudly when train rows are included.
#
# Needs cohort_scores.npz from score_cohort.py (it carries the per-genome `split` array);
# eval_scores.npz alone can only serve --restrict-split all over the holdout.
#
# Re-running is safe and idempotent: every table is rewritten from the npz, nothing accumulates.
#
# Usage:  sbatch src/kleb_iso_source/scripts/stratified_tables_iso_source.sh
#         SCORES_NPZ=<npz> sbatch ...     # e.g. the bf16 cohort scores when they land

set -euo pipefail
export MPLBACKEND=Agg
cd /home/dca36/workspace/BacPredict

DATA=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_iso_source
COHORT=${COHORT:-sampled_country_2_1_all}
CO=$DATA/blood_faeces/$COHORT/kpsc_human
MODELS=${MODELS:-$CO/models}
SCORES=${SCORES_NPZ:-${1:-$MODELS/cohort_scores.npz}}
META=${META:-$CO/binary_blood_vs_faeces_with_split.csv}

[ -s "$SCORES" ] || { echo "ERROR: missing scores npz $SCORES (run score_cohort_iso_source.sh first)"; exit 1; }
[ -s "$META" ]   || { echo "ERROR: missing metadata/split csv $META"; exit 1; }
echo "scores=$SCORES"
echo "meta=$META"

run_one () {
    local col="$1" slug="$2" scope="$3"
    local stem="$MODELS/per_${slug}_metrics_${scope}"
    echo "=== $col / split_scope=$scope ==="
    uv run python -m bacpredict.engine.finetune.stratified_metrics \
        --eval-scores "$SCORES" --metadata "$META" \
        --group-column "$col" --restrict-split "$scope" \
        --out "${stem}.csv"
    uv run python -m bacpredict.engine.plots.plot_stratified_auroc \
        --csv "${stem}.csv" --out "${stem}.png" --group-label "$col" \
        --title "blood vs faeces — AUROC by $col ($scope scope)"
}

for scope in evaluate heldout all; do
    run_one "Sublineage"   sublineage   "$scope"
    run_one "Clonal group" clonal_group "$scope"
done

echo "=== done — six tables + six plots in $MODELS ==="
ls -1 "$MODELS"/per_*_metrics_*.csv "$MODELS"/per_*_metrics_*.png

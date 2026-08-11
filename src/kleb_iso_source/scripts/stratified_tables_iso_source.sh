#!/bin/bash
# Per-stratum AUROC tables + forest plots for the blood-vs-faeces model, at two scopes.
#
# LOGIN-NODE SAFE: pure pandas + a bootstrap over <=14k rows, a couple of minutes single-process.
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
# Usage:  bash src/kleb_iso_source/scripts/stratified_tables_iso_source.sh [SCORES_NPZ]

set -euo pipefail
export MPLBACKEND=Agg
cd /home/dca36/workspace/BacPredict

DATA=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_iso_source
COHORT=${COHORT:-sampled_country_2_1_all}
CO=$DATA/blood_faeces/$COHORT/kpsc_human
MODELS=${MODELS:-$CO/models}
SCORES=${1:-$MODELS/cohort_scores.npz}
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

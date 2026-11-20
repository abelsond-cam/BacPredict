#!/usr/bin/env bash
# Bakta 1.9.4 annotation of the P. aeruginosa cohort's QC-passing genomes, one stage at a time, from a CSD3 login node:
#
#   bash src/bacpredict/apps/pa/scripts/annotate_pa_genomes.sh plan     # which genomes BakRep covers, which we run
#   bash src/bacpredict/apps/pa/scripts/annotate_pa_genomes.sh ingest   # CPU job: BakRep's GFF3s, checked on our contigs
#   bash src/bacpredict/apps/pa/scripts/annotate_pa_genomes.sh submit   # CPU array: our Bakta on the rest, and on
#                                                                        #  any genome ingest refused
#   bash src/bacpredict/apps/pa/scripts/annotate_pa_genomes.sh collate  # genomes/bakta_annotation.csv (exit 1 if any
#                                                                        #  genome lacks one current annotation)
#
# Bakta is the annotation of record (PROJECT_STATE §6). BakRep's GFF3 is used where BakRep holds the genome and our
# contigs are ATB's (its coordinates sit on ATB's contigs; ingest re-checks every file base for base); every other
# genome is annotated by us with the same release, flags and database. That our run reproduces BakRep's was gated on
# 18 genomes (engine/genomes/scripts/validate_bakta_against_bakrep.sh), so a genome whose BakRep file ingest refuses
# is simply annotated by us. The refusals seen: AllTheBacteria contig numbering with gaps (contig00130, 00132, …),
# which our store renumbers consecutively, so BakRep's seqids name different contigs of ours (same sequences). The cohort is genomes/embedding_input.csv
# (build_pa_genomes.sh collate); bakta_annotation.csv is what build_pa_embeddings.sh then reads.
set -euo pipefail

ROOT=${BACPREDICT_DATA_ROOT:-$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/bac_ast_prediction}
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
export REPO
T=$ROOT/processed/train_pa_ast
GEN=$T/genomes
ANN=$ROOT/raw/ast_cohort_genomes/bakta
PLAN=$GEN/bakta
DB=$ROOT/raw/bakta_db/v5.1_full
IN_BAKREP=$ROOT/processed/bakrep_coverage_20261119/in_bakrep_ids.txt
LOGS=/rds/user/dca36/hpc-work/logs
cd "$REPO"

submit() {  # sbatch, then print the resource line the global rules require (read back from SLURM)
    local id job; id=$(sbatch --parsable "$@"); id=${id%%;*}
    job=$(scontrol show job "$id")
    echo "JOB $id $(grep -m1 -oE 'JobName=[^ ]+' <<<"$job") | CPU | $(grep -m1 -oE 'ReqTRES=[^ ]+' <<<"$job")" \
         "| $(grep -m1 -oE 'TimeLimit=[^ ]+' <<<"$job") | $(grep -m1 -oE 'Partition=[^ ]+' <<<"$job")" \
         "| $(grep -m1 -oE 'Account=[^ ]+' <<<"$job")"
}
tools_bin() { echo "$(cd "$REPO/src/bacpredict/engine/genomes" && pixi run -e bakta -- printenv CONDA_PREFIX)/bin"; }

case "${1:-}" in
plan)
    mkdir -p "$PLAN"
    uv run --no-sync python -I - "$GEN" "$IN_BAKREP" "$PLAN" <<'EOF'
import sys
from pathlib import Path
import pandas as pd
gen, in_bakrep, plan = Path(sys.argv[1]), set(Path(sys.argv[2]).read_text().split()), Path(sys.argv[3])
c = pd.read_csv(gen / "embedding_input.csv", dtype={"Sample": str})
q = pd.read_csv(gen / "genome_qc.csv", dtype={"Sample": str}, usecols=["Sample", "source"]).set_index("Sample")
c["source"] = c["Sample"].map(q["source"])
assert c["source"].notna().all() and c["Sample"].is_unique
bakrep = c["source"].eq("atb") & c["Sample"].isin(in_bakrep)
for name, rows in (("bakrep_roster.tsv", c[bakrep]), ("run_roster.tsv", c[~bakrep])):
    rows.sort_values("Sample")[["Sample", "sr_assembly_file"]].to_csv(plan / name, sep="\t", header=False, index=False)
print(f"cohort {len(c)}: BakRep {int(bakrep.sum())}, our Bakta {int((~bakrep).sum())} "
      f"{c.loc[~bakrep, 'source'].value_counts().to_dict()}")
EOF
    wc -l "$PLAN"/*.tsv
    ;;
ingest)
    [[ -s "$PLAN/bakrep_roster.tsv" ]] || { echo "run plan first"; exit 1; }
    # network-bound: 8 BakRep streams (~0.6 MB/s each; a 76-way pull was throttled). ~2 MB per GFF3 → ~5 s per
    # genome per stream, so ~20 min for ~1,800; resumable (a current ingest record is skipped)
    run_dir=$LOGS/pa_bakta_ingest_$(date +%Y%m%d_%H%M%S); mkdir -p "$run_dir"
    submit --account=FLOTO-PROJECT-K-SL2-CPU --partition=icelake-himem --cpus-per-task=8 --mem=16G \
        --time=02:00:00 --job-name=pa-bakta-ingest --chdir="$run_dir" \
        --output="$run_dir/%x_%j.out" --error="$run_dir/%x_%j.out" \
        --wrap="echo repo=$REPO@$(git rev-parse --short HEAD); PYTHONUNBUFFERED=1 uv run --project $REPO --no-sync \
python -m bacpredict.engine.genomes.bakta ingest --roster $PLAN/bakrep_roster.tsv --out-dir $ANN --workers 8"
    echo "logs: $run_dir"
    ;;
submit)
    [[ -s "$PLAN/run_roster.tsv" && -s "$DB/version.json" ]] || { echo "run plan first; DB at $DB"; exit 1; }
    # the run roster = the planned one + every BakRep-roster genome without a current ingest record
    uv run --no-sync python -I - "$PLAN" "$ANN" <<'EOF'
import sys
from pathlib import Path
import pandas as pd
from bacpredict.engine.genomes.bakta import _record_if_current, ingest_record
from bacpredict.engine.genomes.build_genomes import _md5
plan, ann = Path(sys.argv[1]), Path(sys.argv[2])
cols = ["Sample", "contigs"]
planned = pd.read_csv(plan / "run_roster.tsv", sep="\t", header=None, names=cols, dtype=str)
bakrep = pd.read_csv(plan / "bakrep_roster.tsv", sep="\t", header=None, names=cols, dtype=str)
refused = bakrep[[_record_if_current(ingest_record(ann, s), _md5(Path(c))) is None
                  for s, c in zip(bakrep["Sample"], bakrep["contigs"], strict=True)]]
roster = pd.concat([planned, refused]).sort_values("Sample")
assert roster["Sample"].is_unique
roster.to_csv(plan / "run_roster_with_refused.tsv", sep="\t", header=False, index=False)
print(f"run roster: {len(planned)} planned + {len(refused)} BakRep genomes without a current ingest = {len(roster)}")
EOF
    roster=$PLAN/run_roster_with_refused.tsv
    chunk=4
    n=$(( ($(wc -l < "$roster") + chunk - 1) / chunk ))
    # 8 cores / 52 GB per task: sacct MaxRSS was 38-42.5 GB per genome in the gate (52G is under 8 x 6,760 MB).
    # 7.5-9.5 min per genome at 10 threads in the gate, 16-18 min at 1,677-3,010 contigs: 4 genomes ≈ 40-90 min → 3 h.
    # ARRAY_REPO: the checkout the tasks run (default this one) — a detached worktree keeps hours of array immune to
    # other agents' pulls of the shared checkout (see run_bakta_array.sh)
    array_repo=${ARRAY_REPO:-$REPO}
    submit --array=0-$((n - 1))%100 --cpus-per-task=8 --mem=52G --time=03:00:00 --job-name=pa-bakta \
        --export=ALL,REPO=$array_repo,ROSTER=$roster,OUT_DIR=$ANN,TOOLS_BIN=$(tools_bin),DB=$DB,CHUNK=$chunk,KEEP=gff3+faa+txt,EXPECT_SHA=$(git -C "$array_repo" rev-parse HEAD) \
        src/bacpredict/engine/genomes/scripts/run_bakta_array.sh
    ;;
collate)
    uv run --no-sync python -m bacpredict.engine.genomes.bakta collate --cohort "$GEN/embedding_input.csv" \
        --annotation-dir "$ANN" --tools-bin "$(tools_bin)" --db "$DB" --out "$GEN/bakta_annotation.csv"
    ;;
*)
    sed -n '2,8p' "$0"; exit 2 ;;
esac

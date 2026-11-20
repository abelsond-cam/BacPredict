#!/usr/bin/env bash
# Bakta 1.9.4 annotation of the Enterobacterales genomes, one stage at a time, from a CSD3 login node:
#
#   bash src/bacpredict/apps/entero/scripts/annotate_entero_genomes.sh plan     # BakRep roster from the build shards
#   bash src/bacpredict/apps/entero/scripts/annotate_entero_genomes.sh ingest   # CPU job: BakRep's GFF3s, every genome
#                                                                                #  classified (ingest_status.tsv)
#   bash src/bacpredict/apps/entero/scripts/annotate_entero_genomes.sh submit [all|planned|refused]
#                                                                                # CPU array: our Bakta on the rest
#   bash src/bacpredict/apps/entero/scripts/annotate_entero_genomes.sh collate  # CPU job: genomes/bakta_annotation.csv
#                                                                                #  (_<cohort>.csv for another COHORT)
#   bash src/bacpredict/apps/entero/scripts/annotate_entero_genomes.sh status   # ingest outcomes, run records, rosters
#
# The PsA annotation (apps/pa/scripts/annotate_pa_genomes.sh; mechanics in apps/pa/CLAUDE.md *Annotation*) at
# Enterobacterales scale. BakRep's GFF3 where BakRep holds the genome on our ATB contigs; our run of the same release,
# flags, tool env and database otherwise (submit refuses unless the env's fingerprint is PsA's). What differs from PsA:
# - plan reads the build shards, not the QC'd cohort, so ingest can start as soon as ATB contigs exist (pre-QC);
# - ingest classifies every genome (engine/genomes/bakta.py INGEST_STATUSES) and the run roster comes from that file,
#   with no md5 pass over the cohort on the login node;
# - each submit writes a new, dated, read-only run roster; `submit planned` runs the genomes BakRep cannot cover
#   while ingest is still going;
# - COHORT (default genomes/embedding_input.csv, Sample + sr_assembly_file) says which genomes submit and collate
#   cover. Before species QC exists, COHORT=$GEN/bakta/built_genomes.csv (every built genome, written by plan). Only
#   the default cohort's collate writes genomes/bakta_annotation.csv (the protein stage's input); any other COHORT's
#   writes genomes/bakta_annotation_<COHORT file stem>.csv.
# - submit refuses a run roster of more than MAX_RUN genomes (default 6000; raise it to mean it), and refuses while any
#   roster genome's ingest ended error:* unless ALLOW_ERRORS=1, which runs ours on them (the same annotation of record).
# ingest and the run array execute ARRAY_REPO, a detached worktree at the SHA to run with .venv symlinked to the shared
# env (apps/pa/CLAUDE.md *Annotation*): the shared checkout moves under other agents.
set -euo pipefail

# the multi-species AMR root on project_k (PROJECT_STATE §3.6 *Data layout*): per-sample stores flat, keyed by
# BioSample; per-species tables under processed/<species>/
ROOT=${BACPREDICT_AMR_ROOT:-$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/amr_prediction}
[[ -d $ROOT ]] || { echo "no AMR root at $ROOT (set BACPREDICT_AMR_ROOT)"; exit 1; }
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
export REPO
T=$ROOT/processed/enterobacterales
GEN=$T/genomes
PLAN=$GEN/bakta
STATUS=$PLAN/ingest_status.tsv
# resolved once, to an absolute path, before the cd below: submit and collate refuse if it does not exist (the default
# exists only once species QC has run)
COHORT_ARG=${COHORT:-$GEN/embedding_input.csv}
COHORT=$(realpath -e -- "$COHORT_ARG" 2>/dev/null) || COHORT=
need_cohort() {
    [[ -n $COHORT && -f $COHORT && -s $COHORT ]] || {
        echo "no cohort at $COHORT_ARG (set COHORT; before QC, COHORT=$GEN/bakta/built_genomes.csv)"; exit 1; }
}
ANN=$ROOT/raw/bakta  # shared with PsA, keyed by BioSample
DB=$ROOT/raw/bakta_db/v5.1_full
IN_BAKREP=$ROOT/processed/bakrep_coverage_20261119/in_bakrep_ids.txt
# whose run records fix the tool env and database every Entero run must match
PSA_RUN_ROSTER=$ROOT/processed/paeruginosa/genomes/bakta/run_roster_with_refused.tsv
LOGS=/rds/user/dca36/hpc-work/logs
ACCOUNT=FLOTO-PROJECT-K-SL2-CPU
cd "$REPO"

submit() {  # sbatch, then print the resource line the global rules require (read back from SLURM)
    local id job; id=$(sbatch --parsable "$@"); id=${id%%;*}
    job=$(scontrol show job "$id")
    echo "JOB $id $(grep -m1 -oE 'JobName=[^ ]+' <<<"$job") | CPU | $(grep -m1 -oE 'ReqTRES=[^ ]+' <<<"$job")" \
         "| $(grep -m1 -oE 'TimeLimit=[^ ]+' <<<"$job") | $(grep -m1 -oE 'Partition=[^ ]+' <<<"$job")" \
         "| $(grep -m1 -oE 'Account=[^ ]+' <<<"$job")"
}
tools_bin() { echo "$(cd "$REPO/src/bacpredict/engine/genomes" && pixi run -e bakta -- printenv CONDA_PREFIX)/bin"; }
pinned() {  # the checkout a job runs: ARRAY_REPO, which must exist with the shared env linked in
    local ar=${ARRAY_REPO:?set ARRAY_REPO to a detached worktree at the SHA to run (see the header)}
    [[ -d $ar/src/bacpredict && -e $ar/.venv ]] || { echo "ARRAY_REPO=$ar: no src/bacpredict or no .venv" >&2; exit 1; }
    echo "$ar"
}
# The start of a --wrap job run from checkout $1: refuse unless it is still at the SHA submitted, and put its code
# first on PYTHONPATH (a worktree's .venv is the shared env, whose editable install points at the shared checkout).
pin_guard() {
    local sha; sha=$(git -C "$1" rev-parse HEAD)
    echo "EXPECT_SHA=$sha; h=\$(git -C $1 rev-parse HEAD); [ \"\$h\" = \"\$EXPECT_SHA\" ] ||" \
         "{ echo \"$1 moved: \$h, submitted at \$EXPECT_SHA\"; exit 3; }; echo repo=$1@\$h; export PYTHONPATH=$1/src;"
}
refuse_if_live() {  # refuse_if_live <job names, comma-separated> <why>: stop if any of mine is queued or running
    local ids
    ids=$(squeue -u "$USER" -h -n "$1" -o '%i') || { echo "squeue failed: cannot tell whether $1 is live"; exit 1; }
    [[ -z $ids ]] || { echo "queued or running ($1): $(echo $ids) — $2"; exit 1; }
}

case "${1:-}" in
plan)
    # pre-QC: every genome the build array has written so far (re-run plan, then ingest, as more shards land)
    mkdir -p "$PLAN"
    uv run --no-sync python -I - "$GEN/shards" "$GEN/genome_manifest.csv" "$IN_BAKREP" "$PLAN" <<'EOF'
import os
import sys
from pathlib import Path

import pandas as pd

shard_dir, manifest, in_bakrep, plan = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
files = sorted(shard_dir.glob("*.tsv"))
if not files:
    sys.exit(f"no shards in {shard_dir}: build_entero_genomes.sh submit writes them")
cols = ["Sample", "source", "status", "contigs_path"]
s = pd.concat([pd.read_csv(f, sep="\t", dtype=str, usecols=cols) for f in files], ignore_index=True)
# a shard left from an earlier plan may name another source: the manifest's is the one the contigs were built from
planned = pd.read_csv(manifest, dtype=str, usecols=["Sample", "source"]).set_index("Sample")["source"]
ok = s[s["status"].eq("ok") & s["source"].eq(s["Sample"].map(planned))]
ok = ok.drop_duplicates(["Sample", "source", "contigs_path"])
dup = ok["Sample"].duplicated()
if dup.any():
    sys.exit(f"genomes ok in two shards under different contigs paths, e.g. {ok.loc[dup, 'Sample'].head(3).tolist()}")
built = (ok.rename(columns={"contigs_path": "sr_assembly_file"})[["Sample", "sr_assembly_file", "source"]]
         .sort_values("Sample"))
bakrep = built[built["source"].eq("atb") & built["Sample"].isin(set(in_bakrep.read_text().split()))]


def write(df: pd.DataFrame, out: Path, **kw) -> None:  # atomically: a running ingest may be reading the old one
    tmp = out.with_name(f".{out.name}.part")
    df.to_csv(tmp, index=False, **kw)
    os.replace(tmp, out)


write(built, plan / "built_genomes.csv")
write(bakrep[["Sample", "sr_assembly_file"]], plan / "bakrep_roster.tsv", sep="\t", header=False)
n_to_build = int(planned.ne("drop").sum())
print(f"{len(files)} shards: {len(built)} of {n_to_build} genomes to build are built "
      f"{built['source'].value_counts().to_dict()}")
print(f"BakRep roster: {len(bakrep)} ATB genomes in BakRep → {plan / 'bakrep_roster.tsv'}")
print(f"built genomes (a pre-QC COHORT) → {plan / 'built_genomes.csv'}")
EOF
    ;;
ingest)
    [[ -s $PLAN/bakrep_roster.tsv ]] || { echo "run plan first"; exit 1; }
    ar=$(pinned)
    refuse_if_live entero-bakta-ingest,entero-bakta-all,entero-bakta-refused \
        "an ingest racing our runs on the same genomes could leave two annotations"
    # network-bound: 8 BakRep streams (a 76-way pull was throttled). PsA took 4 min for 1,792 genomes, so ~2.5 h for
    # ~64k at that rate; 24 h covers BakRep slowing down. Resumable: a genome with a current ingest record is kept,
    # so a re-run (after a wall, or to clear error:*) re-fetches only what is not ingested.
    run_dir=$LOGS/entero_bakta_ingest_$(date +%Y%m%d_%H%M%S); mkdir -p "$run_dir"
    submit --account=$ACCOUNT --partition=icelake --cpus-per-task=8 --mem=16G --time=24:00:00 \
        --job-name=entero-bakta-ingest --chdir="$run_dir" \
        --output="$run_dir/%x_%j.out" --error="$run_dir/%x_%j.out" \
        --wrap="$(pin_guard "$ar") PYTHONUNBUFFERED=1 uv run --project $ar --no-sync \
python -m bacpredict.engine.genomes.bakta ingest --roster $PLAN/bakrep_roster.tsv --out-dir $ANN --workers 8 \
--status-out $STATUS"
    echo "logs: $run_dir"
    ;;
submit)  # submit [all|planned|refused]: all = planned + refused (the default)
    mode=${2:-all}
    case $mode in
        all) clash=entero-bakta-all,entero-bakta-planned,entero-bakta-refused ;;
        planned|refused) clash=entero-bakta-all,entero-bakta-$mode ;;
        *) echo "submit [all|planned|refused]"; exit 2 ;;
    esac
    ar=$(pinned)
    need_cohort
    for f in "$PLAN/built_genomes.csv" "$DB/version.json" "$PSA_RUN_ROSTER"; do
        [[ -s $f ]] || { echo "missing $f (built_genomes.csv: run plan; else check the DB)"; exit 1; }
    done
    # a guard on cost: a roster this large is more likely a wrong COHORT or a failed ingest than the plan
    max_run=${MAX_RUN:-6000}
    [[ $max_run =~ ^[0-9]+$ ]] || { echo "MAX_RUN=$max_run: not a whole number"; exit 2; }
    allow_errors=${ALLOW_ERRORS:-0}
    [[ $allow_errors == [01] ]] || { echo "ALLOW_ERRORS=$allow_errors: 0 or 1"; exit 2; }
    # planned: cohort genomes BakRep cannot cover (not ATB, or ATB but not in BakRep); needs no ingest status, so it
    #   can run while ingest does. refused: BakRep-roster genomes ingest did not take (absent, refused:*, skipped:run;
    #   error:* too under ALLOW_ERRORS=1).
    refuse_if_live "$clash" "an overlapping run array would run the same genomes twice"
    if [[ $mode != planned ]]; then
        [[ -s $STATUS ]] || {
            echo "no $STATUS: run ingest first, or re-run it (a killed ingest leaves only $STATUS.part)," \
                 "or: submit planned"
            exit 1; }
        refuse_if_live entero-bakta-ingest "its status file is not final yet"
    fi
    # one annotation of record across species: our runs must use the tool env and database PsA's runs used
    tb=$(tools_bin)
    read -r ref _ < "$PSA_RUN_ROSTER"
    fp=$(PYTHONPATH=$ar/src uv run --project "$ar" --no-sync python -m bacpredict.engine.genomes.bakta fingerprint \
        --tools-bin "$tb" --db "$DB")
    psa_fp=$(uv run --no-sync python -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["fingerprint"])' \
        "$ANN/$ref.bakta.run.json")
    [[ $fp == "$psa_fp" ]] || {
        echo "fingerprint of $tb + $DB is $fp, PsA's runs ($ANN/$ref.bakta.run.json) used $psa_fp: not submitting"
        exit 1; }
    roster=$PLAN/run_roster_$(date +%Y%m%d_%H%M%S).tsv
    uv run --no-sync python -I - "$COHORT" "$PLAN/built_genomes.csv" "$IN_BAKREP" "$STATUS" "$ANN" "$fp" "$mode" \
        "$roster" "$max_run" "$allow_errors" <<'EOF'
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

cohort_csv, built_csv, in_bakrep, status_tsv, ann, fp, mode, out, max_run, allow_errors = sys.argv[1:11]
ann = Path(ann)
c = pd.read_csv(cohort_csv, dtype={"Sample": str}, usecols=["Sample", "sr_assembly_file"])
if not c["Sample"].is_unique or c["sr_assembly_file"].isna().any():
    sys.exit(f"{cohort_csv}: a Sample twice, or a genome without contigs")
source = c["Sample"].map(pd.read_csv(built_csv, dtype=str).set_index("Sample")["source"])
if source.isna().any():
    sys.exit(f"{int(source.isna().sum())} cohort genomes are not in {built_csv} "
             f"(e.g. {c.loc[source.isna(), 'Sample'].head(3).tolist()}): re-run plan")
bakrep = source.eq("atb") & c["Sample"].isin(set(Path(in_bakrep).read_text().split()))
if mode == "planned":
    status = pd.Series(np.nan, index=c.index, dtype=object)
    take = ~bakrep
else:
    status = c["Sample"].map(pd.read_csv(status_tsv, sep="\t", dtype=str).set_index("Sample")["status"])
    none = bakrep & status.isna()
    if none.any():
        sys.exit(f"{int(none.sum())} BakRep-roster cohort genomes have no ingest status "
                 f"(e.g. {c.loc[none, 'Sample'].head(3).tolist()}): re-run plan, then ingest")
    err = bakrep & status.str.startswith("error:", na=False)
    if err.any() and allow_errors != "1":
        sys.exit(f"ingest ended {status[err].value_counts().to_dict()}: re-run ingest (it resumes) until none do, or "
                 f"ALLOW_ERRORS=1 to run ours on them (the same annotation of record); details in {status_tsv}")
    rest = bakrep & status.ne("ingested")  # absent, refused:*, skipped:run; error:* only under ALLOW_ERRORS=1
    take = rest if mode == "refused" else rest | ~bakrep
why = pd.Series(np.where(bakrep, "BakRep " + status.fillna(""), "not in BakRep: " + source), index=c.index)
want = c[take]
names = set(os.listdir(ann)) if ann.is_dir() else set()  # one directory read, not a stat per genome


def done(sample: str, contigs: str) -> bool:
    """Our run of this env and db, written after these contigs. Not an md5 check: collate makes that one."""
    rec = ann / f"{sample}.bakta.run.json"
    if rec.name not in names or f"{sample}.bakta.gff3.gz" not in names:
        return False
    try:
        return json.loads(rec.read_text()).get("fingerprint") == fp and rec.stat().st_mtime > os.stat(contigs).st_mtime
    except (OSError, ValueError):
        return False


have = pd.Series([done(s, p) for s, p in zip(want["Sample"], want["sr_assembly_file"], strict=True)],
                 index=want.index, dtype=bool)
todo = want[~have].sort_values("Sample")
ingested = "" if mode == "planned" else f"ingested from BakRep {int((bakrep & status.eq('ingested')).sum())}; "
print(f"cohort {len(c)} ({cohort_csv}): {ingested}{mode}: {len(want)}, of which {int(have.sum())} already have "
      f"our run → {len(todo)} to run")
if len(todo):
    print(why[todo.index].value_counts().to_string())
    if len(todo) > int(max_run):
        sys.exit(f"{len(todo)} genomes to run is more than MAX_RUN={max_run}: no roster written. If the counts above "
                 f"are meant, re-run with MAX_RUN={len(todo)}")
    with open(out, "x") as fh:  # dated, never overwritten
        todo[["Sample", "sr_assembly_file"]].to_csv(fh, sep="\t", header=False, index=False)
EOF
    [[ -s $roster ]] || { echo "nothing to run"; exit 0; }
    chmod a-w "$roster"
    chunk=4
    n=$(( ($(wc -l < "$roster") + chunk - 1) / chunk ))
    # as PsA (sacct, PROJECT_STATE §3.6): 33 min mean / 61 min max per 4-genome task, MaxRSS ≤ 42.6 GB per genome, so
    # 8 cores / 52 GB (under 8 × 6,760 MB) on icelake-himem and 3 h
    export_vars="ALL,REPO=$ar,ROSTER=$roster,OUT_DIR=$ANN,TOOLS_BIN=$tb,DB=$DB,CHUNK=$chunk,KEEP=gff3+faa+txt"
    export_vars+=",EXPECT_SHA=$(git -C "$ar" rev-parse HEAD)"
    submit --array=0-$((n - 1))%100 --account=$ACCOUNT --partition=icelake-himem --cpus-per-task=8 --mem=52G \
        --time=03:00:00 --job-name=entero-bakta-$mode --export="$export_vars" \
        "$ar/src/bacpredict/engine/genomes/scripts/run_bakta_array.sh"
    echo "roster: $roster"
    ;;
collate)
    # exit 1, and no table, unless every COHORT genome has exactly one current annotation (md5 of its contigs now).
    # Only the canonical cohort may write bakta_annotation.csv, the protein stage's input: a pre-QC collate must
    # neither replace it nor (failing) delete it
    need_cohort
    canon=$(realpath -e -- "$GEN/embedding_input.csv" 2>/dev/null) || canon=
    if [[ $COHORT == "$canon" ]]; then
        out=$GEN/bakta_annotation.csv
    else
        stem=$(basename -- "$COHORT"); out=$GEN/bakta_annotation_${stem%.*}.csv
    fi
    cr=${ARRAY_REPO:-$REPO}
    run_dir=$LOGS/entero_bakta_collate_$(date +%Y%m%d_%H%M%S); mkdir -p "$run_dir"
    submit --account=$ACCOUNT --partition=icelake --cpus-per-task=16 --mem=32G --time=02:00:00 \
        --job-name=entero-bakta-collate --chdir="$run_dir" \
        --output="$run_dir/%x_%j.out" --error="$run_dir/%x_%j.out" \
        --wrap="$(pin_guard "$cr") PYTHONUNBUFFERED=1 uv run --project $cr --no-sync \
python -m bacpredict.engine.genomes.bakta collate --cohort $COHORT --annotation-dir $ANN --tools-bin $(tools_bin) \
--db $DB --out $out"
    echo "cohort: $COHORT → $out"
    echo "logs: $run_dir"
    ;;
status)
    uv run --no-sync python -I - "$PLAN" "$ANN" "${COHORT:-$COHORT_ARG}" <<'EOF'
import os
import sys
from pathlib import Path

import pandas as pd

plan, ann, cohort_csv = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
roster_tsv, status_tsv = plan / "bakrep_roster.tsv", plan / "ingest_status.tsv"
part_tsv = status_tsv.with_name(status_tsv.name + ".part")  # appended as each genome completes; gone once final
roster = (pd.read_csv(roster_tsv, sep="\t", header=None, names=["Sample", "contigs"], dtype=str)
          if roster_tsv.exists() else None)
src = status_tsv if status_tsv.exists() else part_tsv if part_tsv.exists() else None
st = pd.read_csv(src, sep="\t", dtype=str) if src is not None else None
if st is None:
    print("no ingest status yet")
else:
    what = "final" if src == status_tsv else "NOT final: ingest is running, or was killed (re-run it)"
    print(f"ingest status ({src}, {what}): {len(st)} genomes")
    print(st["status"].value_counts().to_string())
if roster is not None:
    n = len(roster) if st is None else int((~roster["Sample"].isin(st["Sample"])).sum())
    print(f"BakRep roster: {len(roster)} genomes, {n} without an ingest status")
names = set(os.listdir(ann)) if ann.is_dir() else set()
if cohort_csv.exists():
    c = pd.read_csv(cohort_csv, dtype={"Sample": str}, usecols=["Sample"])["Sample"]
    ing = c.isin(st.loc[st["status"].eq("ingested"), "Sample"]) if st is not None else pd.Series(False, index=c.index)
    run = (c + ".bakta.run.json").isin(names)
    print(f"cohort {len(c)} ({cohort_csv}): ingested {int(ing.sum())}, our run record {int(run.sum())}, "
          f"both {int((ing & run).sum())}, neither {int((~ing & ~run).sum())}")
else:
    print(f"no cohort at {cohort_csv} (set COHORT)")
for r in sorted(plan.glob("run_roster_*.tsv")):
    with open(r) as fh:
        print(f"  {r.name}: {sum(1 for _ in fh)} genomes")
EOF
    q=$(squeue -u "$USER" -h -o '%.12i %.24j %.9T %.11M') \
        || { echo "squeue failed: cannot tell which jobs are live"; exit 1; }
    grep -E 'entero-bakta' <<<"$q" || echo "no entero-bakta job queued or running"
    ;;
*)
    sed -n '2,12p' "$0"; exit 2 ;;
esac

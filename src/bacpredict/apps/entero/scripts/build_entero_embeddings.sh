#!/usr/bin/env bash
# ESM-C embeddings for the Enterobacterales cohort, one stage at a time, from a CSD3 login node:
#
#   bash src/bacpredict/apps/entero/scripts/build_entero_embeddings.sh proteins        # CPU job: genomes → parquets
#   bash src/bacpredict/apps/entero/scripts/build_entero_embeddings.sh check-proteins  # CPU job: every parquet checked
#   bash src/bacpredict/apps/entero/scripts/build_entero_embeddings.sh smoke           # GPU job: 10 genomes, every genus
#   bash src/bacpredict/apps/entero/scripts/build_entero_embeddings.sh check-smoke     # login node, those 10
#   ARRAY_REPO=<worktree> bash …/build_entero_embeddings.sh submit                    # GPU array: every genome → esm/
#   bash src/bacpredict/apps/entero/scripts/build_entero_embeddings.sh check           # CPU job: end to end
#
# The PsA stage (apps/pa/scripts/build_pa_embeddings.sh; mechanics in apps/pa/CLAUDE.md) at ~68k genomes. The cohort
# is genomes/bakta_annotation.csv (Bakta collate: the QC-passing genomes, each with its one Bakta 1.9.4 GFF3), and
# every check counts against it, never against the files that exist. The checks are
# engine/embedding/check_cohort_store.py run as 32-core jobs (contigs md5, parquet protein counts and .pt reads over
# ~68k genomes are too much for a login node); each prints PASS or FAIL last and writes every problem to a CSV.
#
# Parquets and embeddings go to the flat per-BioSample stores shared with PsA ($ROOT/processed/{protein_sequences,
# esm}); the ESM-C array is therefore given a sample list, genomes/esm_samples.txt (the sorted Samples of
# bakta_annotation.csv), pinned by its md5, and never the store's listing. The GPU array runs from a pinned worktree
# (ARRAY_REPO, a detached worktree at the SHA to run with .venv symlinked to the shared env), because array tasks
# otherwise run whatever the shared checkout's HEAD is when they start. Settings as for PsA: --max-proteins 9000
# and bf16 storage (PROJECT_STATE.md §6).
set -euo pipefail

# the multi-species AMR root on project_k (PROJECT_STATE §3.6 *Data layout*): per-sample stores flat, keyed by
# BioSample; per-species tables under processed/<species>/
ROOT=${BACPREDICT_AMR_ROOT:-$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/amr_prediction}
[[ -d $ROOT ]] || { echo "no AMR root at $ROOT (set BACPREDICT_AMR_ROOT)"; exit 1; }
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
export REPO
T=$ROOT/processed/enterobacterales
GEN=$T/genomes
COHORT=$GEN/bakta_annotation.csv
QC_COHORT=$GEN/embedding_input.csv
SAMPLES=$GEN/esm_samples.txt
SMOKE_SAMPLES=$GEN/esm_samples_smoke.txt
PROT=$ROOT/processed/protein_sequences
ESM=$ROOT/processed/esm
ESM_SMOKE=$ROOT/processed/esm_smoke
MAX_PROTEINS=9000
SAVE_DTYPE=bfloat16
# genomes per 4 h GPU task: ~13 s each (2.8 ms per protein measured on PsA, ~4,700 proteins per genome) ≈ 65 min,
# so the wall is ~3.7× the estimate; NTASKS = ceil(cohort / PER_TASK) (~225 tasks, ~250 A100-h)
PER_TASK=${ESM_PER_TASK:-300}
LOGS=/rds/user/dca36/hpc-work/logs
ARRAY=src/bacpredict/engine/embedding/scripts/esm_embeddings_array.sh
cd "$REPO"

# The start of a --wrap job: the code it runs is the checkout's HEAD when it STARTS, so refuse to run if that is
# no longer the SHA submitted (the shared checkout moves under other agents), and log the SHA it did run.
sha_guard() {
    local sha; sha=$(git -C "$REPO" rev-parse HEAD)
    echo "h=\$(git -C $REPO rev-parse HEAD); [ \"\$h\" = $sha ] || { echo \"checkout moved: \$h, submitted at $sha\"; exit 3; }; echo repo=$REPO@\$h;"
}

submit() {  # sbatch, then print the resource line the global rules require (read back from SLURM)
    local id job kind; id=$(sbatch --parsable "$@"); id=${id%%;*}
    job=$(scontrol show job "$id")
    kind=CPU; grep -q 'gres/gpu' <<<"$job" && kind="GPU x$(grep -m1 -oE 'gres/gpu=[0-9]+' <<<"$job" | cut -d= -f2)"
    echo "JOB $id $(grep -m1 -oE 'JobName=[^ ]+' <<<"$job") | $kind | $(grep -m1 -oE 'ReqTRES=[^ ]+' <<<"$job")" \
         "| $(grep -m1 -oE 'TimeLimit=[^ ]+' <<<"$job") | $(grep -m1 -oE 'Partition=[^ ]+' <<<"$job")" \
         "| $(grep -m1 -oE 'Account=[^ ]+' <<<"$job")"
}

live() { squeue -u "$USER" -h -o '%j' | grep -cE "$1" || true; }  # queued or running jobs whose name matches

# write_list <out> <python printing the sorted, unique samples from the cohort table (argv 1) and cohort.csv (argv 2)>:
# (re)write <out>, atomically and only if it changed; print its md5 (what the array's SAMPLES_MD5 pins). Called as
# md5=$(write_list …), where bash drops set -e: hence the explicit return, or a failed write would hand on the md5
# of a stale list. generate_embeddings re-checks the order and uniqueness before loading the model.
write_list() {
    local out=$1 tmp; tmp=$1.$$.part
    uv run --no-sync python -I -c "$2" "$COHORT" "$T/cohort.csv" > "$tmp" || { rm -f "$tmp"; return 1; }
    if cmp -s "$tmp" "$out"; then rm -f "$tmp"; else mv "$tmp" "$out"; fi
    md5sum < "$out" | cut -d' ' -f1
}
ALL_SAMPLES='
import sys
import pandas as pd
s = pd.read_csv(sys.argv[1], dtype={"Sample": str})["Sample"]
if s.isna().any() or not s.is_unique or s.str.strip().ne(s).any() or s.eq("").any():
    sys.exit(f"{sys.argv[1]}: Sample must be unique, non-empty and unpadded")
print("\n".join(sorted(s)))'
# one genome of every genus (the first by Sample), plus the genome with the most proteins (the 9,000 cap and the
# largest GPU batch), then genera in turn up to 10
SMOKE_SAMPLES_PY='
import sys
import pandas as pd
c = pd.read_csv(sys.argv[1], dtype={"Sample": str}).sort_values("Sample")
genus = pd.read_csv(sys.argv[2], dtype={"Sample": str}).set_index("Sample")["species"].str.split().str[0]
c["genus"] = c["Sample"].map(genus).fillna("unknown")
c["turn"] = c.groupby("genus").cumcount()
pick = list(c.loc[c["turn"].eq(0), "Sample"])
pick.append(c.loc[(c["n_cds"] - c["n_pseudo"]).idxmax(), "Sample"])
rest = c[~c["Sample"].isin(pick)].sort_values(["turn", "Sample"])
pick += list(rest["Sample"].head(max(0, 10 - len(set(pick)))))
pick = sorted(set(pick))
print(c.set_index("Sample").loc[pick].assign(proteins=lambda d: d.n_cds - d.n_pseudo)[["genus", "proteins"]]
      .to_string(), file=sys.stderr)
print("\n".join(pick))'

# check_job <stage> <args…>: check_cohort_store as a 32-core icelake job (CHECK_* override the shape)
check_job() {
    local stage=$1 run_dir; shift
    run_dir=$LOGS/entero_esm_${stage}_$(date +%Y%m%d_%H%M%S); mkdir -p "$run_dir"
    # I/O-bound: per genome a contigs md5 (~1.5 MB), a parquet column (~1.5 MB) and a .pt zip check. ~190 GB over
    # ~68k genomes is minutes at Lustre rates and under 1 h at 100 MB/s; ~20 GB for 32 workers, so 48G
    submit --account=FLOTO-PROJECT-K-SL2-CPU --partition=${CHECK_PARTITION:-icelake} --cpus-per-task=32 \
        --mem=${CHECK_MEM:-48G} --time=${CHECK_TIME:-01:00:00} --job-name=entero-esm-$stage --chdir="$run_dir" \
        --output="$run_dir/%x_%j.out" --error="$run_dir/%x_%j.out" \
        --wrap="$(sha_guard) PYTHONUNBUFFERED=1 uv run --project $REPO --no-sync \
python -m bacpredict.engine.embedding.check_cohort_store --cohort $COHORT --qc-cohort $QC_COHORT --protein-dir $PROT \
--max-proteins $MAX_PROTEINS --save-dtype $SAVE_DTYPE --workers 32 --report $run_dir/problems.csv $*"
    echo "logs: $run_dir (the .out ends PASS or FAIL; every problem is in problems.csv)"
}

case "${1:-}" in
proteins)
    [[ -s "$COHORT" ]] || { echo "no $COHORT — annotate the cohort first (Bakta collate)"; exit 1; }
    # A PsA ESM-C array submitted before the sample list (esm_embeddings_array.sh without SAMPLES) slices the store's
    # whole listing: parquets landing now would shift its pending tasks' slices and leave PsA genomes unembedded
    n=$(live '^(pa-esm-embeddings|pa-esm-smoke|esm-embeddings)$')
    if (( n > 0 )) && [[ ${IGNORE_LIVE_PSA_ESM:-0} != 1 ]]; then
        echo "$n PsA ESM-C job(s) queued or running. If every one was submitted with SAMPLES (its .out header shows"
        echo "samples=…), rerun with IGNORE_LIVE_PSA_ESM=1; otherwise wait for them to finish."; exit 1
    fi
    run_dir=$LOGS/entero_protein_sequences_$(date +%Y%m%d_%H%M%S); mkdir -p "$run_dir"
    # one job: 32 cores (one genome per worker) on icelake, which has idle cores; 96G is under icelake's 3,380 MB per
    # core × 32. ~2 h for ~68k genomes; resumable (--skip-existing, atomic parquets), so a wall kill only means
    # resubmitting. PROT_PARTITION/PROT_CPUS/PROT_MEM/PROT_TIME override the shape.
    cpus=${PROT_CPUS:-32}
    submit --account=FLOTO-PROJECT-K-SL2-CPU --partition=${PROT_PARTITION:-icelake} --cpus-per-task=$cpus \
        --mem=${PROT_MEM:-96G} --time=${PROT_TIME:-02:00:00} --job-name=entero-protein-sequences --chdir="$run_dir" \
        --output="$run_dir/%x_%j.out" --error="$run_dir/%x_%j.err" \
        --wrap="$(sha_guard) PYTHONUNBUFFERED=1 uv run --project $REPO --no-sync \
python -m bacpredict.engine.embedding.preprocess_assemblies_to_protein_sequences --input-csv $COHORT \
--output-dir $PROT --workers $cpus --skip-existing"
    echo "cohort: $(($(wc -l < "$COHORT") - 1)) genomes; logs: $run_dir"
    ;;
check-proteins)
    check_job check-proteins
    ;;
smoke)
    [[ -s "$COHORT" ]] || { echo "no $COHORT"; exit 1; }
    md5=$(write_list "$SMOKE_SAMPLES" "$SMOKE_SAMPLES_PY")
    # those 10 parquets must exist and hold the GFF3's protein count (the full check is check-proteins)
    uv run --no-sync python -m bacpredict.engine.embedding.check_cohort_store --cohort "$COHORT" --protein-dir "$PROT" \
        --samples-file "$SMOKE_SAMPLES" --workers 4 \
        || { echo "smoke genomes lack proteins: run proteins and check-proteins"; exit 1; }
    sha=$(git rev-parse HEAD)
    submit --array=0 --time=00:20:00 --job-name=entero-esm-smoke \
        --export="ALL,REPO=$REPO,EXPECT_SHA=$sha,INPUT_DIR=$PROT,ESM_DIR=$ESM_SMOKE,SAMPLES=$SMOKE_SAMPLES,SAMPLES_MD5=$md5,MAX_PROTEINS=$MAX_PROTEINS,SAVE_DTYPE=$SAVE_DTYPE,NTASKS=1" \
        "$ARRAY"
    echo "samples: $SMOKE_SAMPLES (md5 $md5)"
    ;;
check-smoke)
    [[ -s $SMOKE_SAMPLES ]] || { echo "no $SMOKE_SAMPLES: run smoke first"; exit 1; }
    uv run --no-sync python -m bacpredict.engine.embedding.check_cohort_store --cohort "$COHORT" --protein-dir "$PROT" \
        --esm-dir "$ESM_SMOKE" --samples-file "$SMOKE_SAMPLES" --max-proteins $MAX_PROTEINS --save-dtype $SAVE_DTYPE \
        --workers 4
    ;;
submit)
    array_repo=${ARRAY_REPO:?set ARRAY_REPO to a detached worktree at the SHA to run (see the header)}
    # a second array into the same store would share each sample's temporary .pt name with the first
    (( $(live '^entero-esm-embeddings$') == 0 )) || { echo "an entero-esm-embeddings array is queued or running"; exit 1; }
    md5=$(write_list "$SAMPLES" "$ALL_SAMPLES")
    # every listed genome has a parquet (one directory read; the counts are check-proteins' job, run it first)
    uv run --no-sync python -m bacpredict.engine.embedding.check_cohort_store --cohort "$COHORT" --protein-dir "$PROT" \
        --samples-file "$SAMPLES" --presence-only \
        || { echo "proteins incomplete: run proteins, then check-proteins"; exit 1; }
    total=$(awk '{ gsub(/[[:space:]]/, "") } length($0) { n++ } END { print n + 0 }' "$SAMPLES")
    ntasks=$(( (total + PER_TASK - 1) / PER_TASK ))
    # 8 cores / 64 GB / 1 A100 per task (the array's defaults); ESM_CONCURRENT caps how many run at once.
    # Re-running submit is safe: each task skips every genome whose .pt exists, and a task with none left exits
    # before loading the model.
    sha=$(git -C "$array_repo" rev-parse HEAD)
    submit --array="0-$((ntasks - 1))%${ESM_CONCURRENT:-64}" --time=04:00:00 --job-name=entero-esm-embeddings \
        --export="ALL,REPO=$array_repo,EXPECT_SHA=$sha,INPUT_DIR=$PROT,ESM_DIR=$ESM,SAMPLES=$SAMPLES,SAMPLES_MD5=$md5,MAX_PROTEINS=$MAX_PROTEINS,SAVE_DTYPE=$SAVE_DTYPE,NTASKS=$ntasks" \
        "$array_repo/$ARRAY"
    echo "samples: $SAMPLES ($total genomes, md5 $md5); $ntasks tasks of ≤$(( (total + ntasks - 1) / ntasks ))"
    ;;
check)
    check_job check --esm-dir "$ESM"
    ;;
*)
    sed -n '2,9p' "$0"; exit 2 ;;
esac

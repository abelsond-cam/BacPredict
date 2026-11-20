#!/usr/bin/env bash
# ESM-C embeddings for the P. aeruginosa cohort's QC-passing genomes, one stage at a time, from a CSD3 login node:
#
#   bash src/bacpredict/apps/pa/scripts/build_pa_embeddings.sh proteins     # CPU job: genomes → protein parquets
#   bash src/bacpredict/apps/pa/scripts/build_pa_embeddings.sh check-proteins  # every cohort genome has a parquet
#   bash src/bacpredict/apps/pa/scripts/build_pa_embeddings.sh smoke        # GPU job: first 10 genomes → esm_smoke/
#   bash src/bacpredict/apps/pa/scripts/build_pa_embeddings.sh check-smoke
#   bash src/bacpredict/apps/pa/scripts/build_pa_embeddings.sh submit       # GPU array: every genome → esm/
#   bash src/bacpredict/apps/pa/scripts/build_pa_embeddings.sh check        # every cohort genome, end to end
#
# The cohort is genomes/bakta_annotation.csv from annotate_pa_genomes.sh collate (the QC-passing genomes, each with
# its one Bakta 1.9.4 GFF3), and every check counts against it (never against whatever files happen to exist): the
# protein extractor logs a failed genome and exits 0, so a parquet-based count would shrink with the cohort and
# still read complete. check-proteins also requires each parquet to hold exactly the GFF3's non-pseudogene CDS: a
# contig-name mismatch between GFF3 and contigs would otherwise drop proteins silently. smoke and submit refuse to
# start until every cohort genome has a parquet, which also keeps the array's slices tiling a fixed list.
#
# Two settings are PsA-specific and deliberate: --max-proteins 9000 (a PsA genome has ~6,400 CDS;
# generate_embeddings' default of 6,000 would cut the smallest contigs — plasmids — first) and bf16 storage
# (decision of record, PROJECT_STATE.md §6).
set -euo pipefail

# the multi-species AMR root on project_k (PROJECT_STATE §3.6 *Data layout*): per-sample stores flat, keyed by
# BioSample; per-species tables under processed/<species>/
ROOT=${BACPREDICT_AMR_ROOT:-$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/amr_prediction}
[[ -d $ROOT ]] || { echo "no AMR root at $ROOT (set BACPREDICT_AMR_ROOT)"; exit 1; }
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
export REPO
T=$ROOT/processed/paeruginosa
GEN=$T/genomes
COHORT=$GEN/bakta_annotation.csv
# the per-sample stores are flat and shared across species. ⚠ esm_embeddings_array.sh tiles INPUT_DIR's whole
# listing (and the smoke its first N files), so it needs a sample list before a second species writes here
PROT=$ROOT/processed/protein_sequences
ESM=$ROOT/processed/esm
ESM_SMOKE=$ROOT/processed/esm_smoke
MAX_PROTEINS=9000
SAVE_DTYPE=bfloat16
NTASKS=16
LOGS=/rds/user/dca36/hpc-work/logs
ARRAY=$REPO/src/bacpredict/engine/embedding/scripts/esm_embeddings_array.sh
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

# check_store <esm dir> <all|N> [proteins-only]
#   all: the cohort is every Sample of bakta_annotation.csv; N: the first N sorted parquets (what the smoke embeds).
#   Fails on: an empty cohort; a genome with no parquet, a parquet older than its GFF (built from a genome since
#   rebuilt), or, at scope all, a parquet whose protein count is not the GFF3's CDS minus pseudogenes; with an
#   embedding stage, a missing, unreadable or stale .pt, or a sampled file of the wrong shape, width or dtype, or
#   holding NaN.
check_store() {
    uv run --no-sync python -I - "$COHORT" "$PROT" "$1" "$2" "${3:-}" "$MAX_PROTEINS" "$SAVE_DTYPE" <<'EOF'
import random, sys, zipfile
from pathlib import Path
import pandas as pd, torch
cohort_csv, prot, esm, scope, proteins_only, cap, dtype = sys.argv[1:8]
prot, esm, cap, dtype = Path(prot), Path(esm), int(cap), getattr(torch, dtype)
c = pd.read_csv(cohort_csv, dtype={"Sample": str}).set_index("Sample")
if scope == "all":
    samples = sorted(c.index)
else:
    samples = sorted(f.name.removesuffix("_protein_sequences.parquet") for f in prot.glob("*_protein_sequences.parquet"))
    samples = samples[: int(scope)]
pq = {s: prot / f"{s}_protein_sequences.parquet" for s in samples}
pt = {s: esm / f"{s}_esm_embeddings.pt" for s in samples}
problems = {}
problems["no parquet"] = [s for s in samples if not pq[s].exists()]
if scope == "all":  # the ESM array slices the store's whole listing, so the store must hold this cohort only
    listed = {f.name.removesuffix("_protein_sequences.parquet") for f in prot.glob("*_protein_sequences.parquet")}
    problems["parquets outside this cohort (the ESM array would embed them too)"] = sorted(listed - set(samples))
have_pq = [s for s in samples if pq[s].exists()]
problems["parquet older than its GFF"] = [
    s for s in have_pq if s in c.index and pq[s].stat().st_mtime < Path(c.at[s, "sr_gff_file"]).stat().st_mtime]
print(f"cohort: {len(samples)} genomes ({Path(cohort_csv).name if scope == 'all' else f'first {scope} parquets'}); "
      f"with a protein parquet: {len(have_pq)}")
if scope == "all" and {"n_cds", "n_pseudo", "contigs_md5"} <= set(c.columns):
    import hashlib
    from concurrent.futures import ThreadPoolExecutor
    import pyarrow.parquet as papq
    # the annotation table must still describe the QC cohort and the contigs it was made on
    qc = pd.read_csv(Path(cohort_csv).with_name("embedding_input.csv"), dtype={"Sample": str})
    problems["cohort differs from embedding_input.csv"] = sorted(set(qc["Sample"]) ^ set(c.index))
    def md5(p):
        h = hashlib.md5()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        return h.hexdigest()
    with ThreadPoolExecutor(16) as ex:
        now = dict(zip(c.index, ex.map(md5, c["sr_assembly_file"])))
    problems["contigs changed since annotation"] = [s for s in c.index if now[s] != c.at[s, "contigs_md5"]]
    expect = (c["n_cds"] - c["n_pseudo"]).astype(int)
    got, unreadable = {}, []
    for s in have_pq:
        try:
            got[s] = sum(len(x) for x in papq.read_table(pq[s], columns=["protein_sequence"]).column(0).to_pylist()[0])
        except Exception:
            unreadable.append(s)
    problems["unreadable parquet (delete it and rerun proteins)"] = unreadable
    problems["protein count != CDS - pseudogenes"] = [s for s in got if got[s] != expect[s]]
    print(f"protein count = CDS - pseudogenes: {len(got) - len(problems['protein count != CDS - pseudogenes'])}"
          f"/{len(have_pq)}; contigs unchanged since annotation: "
          f"{len(c) - len(problems['contigs changed since annotation'])}/{len(c)}")
if not proteins_only:
    problems["no embedding"] = [s for s in have_pq if not pt[s].exists()]
    have_pt = [s for s in have_pq if pt[s].exists()]
    problems["unreadable embedding"] = [s for s in have_pt if not zipfile.is_zipfile(pt[s])]
    problems["embedding older than its parquet"] = [s for s in have_pt if pt[s].stat().st_mtime < pq[s].stat().st_mtime]
    print(f"with an embedding in {esm}: {len(have_pt)}")
    good = [s for s in have_pt if s not in problems["unreadable embedding"]]
    random.seed(0)
    bad = []
    for s in random.sample(good, min(10, len(good))):
        d = torch.load(pt[s], weights_only=False)
        pe = d["protein_embeddings"]
        n_prot = sum(len(x) for x in pd.read_parquet(pq[s])["protein_sequence"].iloc[0])
        ok = (pe.shape[1] == min(n_prot, cap) and pe.shape[2] == 960 and pe.dtype == dtype
              and not torch.isnan(pe.float()).any())
        print(f"  {s}: {tuple(pe.shape)} {pe.dtype} proteins={n_prot} contigs={int(d['contig_ids'].max()) + 1} "
              f"{'ok' if ok else 'MISMATCH'}")
        if not ok:
            bad.append(s)
    problems["sampled embedding mismatch"] = bad
for k, v in problems.items():
    if v:
        print(f"FAIL {k}: {len(v)} {v[:10]}")
sys.exit(1 if not samples or any(problems.values()) else 0)
EOF
}

case "${1:-}" in
proteins)
    [[ -s "$COHORT" ]] || { echo "no $COHORT — run build_pa_genomes.sh collate"; exit 1; }
    # runs $REPO's code (not whatever ~/workspace holds) and keeps the extractor's own logs out of the repo
    run_dir=$LOGS/pa_protein_sequences_$(date +%Y%m%d_%H%M%S)
    mkdir -p "$run_dir"
    # PROT_PARTITION/PROT_CPUS/PROT_MEM/PROT_TIME: a smaller shape backfills sooner on a busy queue
    cpus=${PROT_CPUS:-32}
    submit --account=FLOTO-PROJECT-K-SL2-CPU --partition=${PROT_PARTITION:-icelake-himem} --cpus-per-task=$cpus \
        --mem=${PROT_MEM:-128G} --time=${PROT_TIME:-00:45:00} --job-name=pa-protein-sequences --chdir="$run_dir" \
        --output="$run_dir/%x_%j.out" --error="$run_dir/%x_%j.err" \
        --wrap="$(sha_guard) PYTHONUNBUFFERED=1 uv run --project $REPO --no-sync \
python -m bacpredict.engine.embedding.preprocess_assemblies_to_protein_sequences --input-csv $COHORT \
--output-dir $PROT --workers $cpus --skip-existing"
    echo "logs: $run_dir"
    ;;
check-proteins)
    check_store "$ESM" all proteins-only
    ;;
smoke)
    check_store "$ESM" all proteins-only >/dev/null || { echo "proteins incomplete: run check-proteins"; exit 1; }
    submit --array=0 --time=00:20:00 --job-name=pa-esm-smoke \
        --export=ALL,INPUT_DIR=$PROT,ESM_DIR=$ESM_SMOKE,MAX_PROTEINS=$MAX_PROTEINS,SAVE_DTYPE=$SAVE_DTYPE,NTASKS=1,N=10 \
        "$ARRAY"
    ;;
check-smoke)
    check_store "$ESM_SMOKE" 10
    ;;
submit)
    check_store "$ESM" all proteins-only >/dev/null || { echo "proteins incomplete: run check-proteins"; exit 1; }
    submit --array=0-$((NTASKS - 1)) --time=04:00:00 --job-name=pa-esm-embeddings \
        --export=ALL,INPUT_DIR=$PROT,ESM_DIR=$ESM,MAX_PROTEINS=$MAX_PROTEINS,SAVE_DTYPE=$SAVE_DTYPE,NTASKS=$NTASKS \
        "$ARRAY"
    ;;
check)
    check_store "$ESM" all
    ;;
*)
    sed -n '2,9p' "$0"; exit 2 ;;
esac

#!/usr/bin/env bash
# The gate before we annotate any genome BakRep lacks: does our Bakta 1.9.4 reproduce BakRep's, and does the
# protein extractor reproduce Bakta's own proteins? One stage at a time, from a CSD3 login node:
#
#   bash src/bacpredict/engine/genomes/scripts/validate_bakta_against_bakrep.sh roster   # pick the genomes
#   bash src/bacpredict/engine/genomes/scripts/validate_bakta_against_bakrep.sh fetch    # contigs + BakRep files
#   bash src/bacpredict/engine/genomes/scripts/validate_bakta_against_bakrep.sh submit   # our Bakta, one task each
#   bash src/bacpredict/engine/genomes/scripts/validate_bakta_against_bakrep.sh compare  # the gate (exit 1 = fail)
#
# Genomes (seeded, so a re-run picks the same): 4 QC-passing P. aeruginosa that BakRep holds — fewest contigs,
# median, most, one random — on our contig store; 11 Enterobacterales BakRep holds (E. coli, S. enterica and
# K. pneumoniae 3 each, one meropenem-resistant where there is one; Enterobacter and Shigella 1 each); and 3 of
# the 283 whose AllTheBacteria Bakta run failed (2 S. enterica, 1 Escherichia), to see whether ours finishes.
# Enterobacterales contigs are fetched from ATB and renamed exactly as engine/genomes/build_genomes names them.
# What is compared: engine/genomes/compare_bakta_to_bakrep.py.
set -euo pipefail

# the multi-species AMR root on project_k (PROJECT_STATE §3.6 *Data layout*): per-sample stores flat, keyed by
# BioSample; per-species tables under processed/<species>/
ROOT=${BACPREDICT_AMR_ROOT:-$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/amr_prediction}
[[ -d $ROOT ]] || { echo "no AMR root at $ROOT (set BACPREDICT_AMR_ROOT)"; exit 1; }
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
export REPO
GATE=$ROOT/processed/bakta_vs_bakrep_validation
DB=$ROOT/raw/bakta_db/v5.1_full
COVERAGE=$ROOT/processed/bakrep_coverage_20261119
PA=$ROOT/processed/paeruginosa
EN=$ROOT/processed/enterobacterales
ATB_INDEX=$ROOT/raw/allthebacteria/assemblies_list_20261115
cd "$REPO"

submit() {  # sbatch, then print the resource line the global rules require (read back from SLURM)
    local id job; id=$(sbatch --parsable "$@"); id=${id%%;*}
    job=$(scontrol show job "$id")
    echo "JOB $id $(grep -m1 -oE 'JobName=[^ ]+' <<<"$job") | CPU | $(grep -m1 -oE 'ReqTRES=[^ ]+' <<<"$job")" \
         "| $(grep -m1 -oE 'TimeLimit=[^ ]+' <<<"$job") | $(grep -m1 -oE 'Partition=[^ ]+' <<<"$job")" \
         "| $(grep -m1 -oE 'Account=[^ ]+' <<<"$job")"
}

case "${1:-}" in
roster)
    mkdir -p "$GATE"
    uv run --no-sync python -I - "$GATE" "$COVERAGE" "$PA" "$EN" "$ATB_INDEX" <<'EOF'
import sys
from pathlib import Path
import pandas as pd
from bacpredict.engine.genomes.plan_genome_sources import read_atb_index
gate, cov, pa, en, atb = map(Path, sys.argv[1:6])
in_bakrep = set((cov / "in_bakrep_ids.txt").read_text().split())
atb_fail = set((cov / "entero_early_miss_ids.txt").read_text().split())
rows = []
q = pd.read_csv(pa / "genomes" / "genome_qc.csv", dtype={"Sample": str})
q = q[q["qc_pass"].astype(str).eq("True") & q["source"].eq("atb") & q["Sample"].isin(in_bakrep)]
q = q.sort_values(["n_contigs", "Sample"]).reset_index(drop=True)
pick = [0, len(q) // 2, len(q) - 1]
pick.append(q.drop(index=pick).sample(1, random_state=0).index[0])
for i in pick:
    rows.append({"Sample": q.at[i, "Sample"], "group": "psa", "species": "Pseudomonas aeruginosa",
                 "contigs_source": "store", "contigs_path": q.at[i, "contigs_path"]})
c = pd.read_csv(en / "cohort.csv", dtype={"Sample": str})
b = pd.read_csv(en / "binary_ast.csv", dtype={"phenotype-BioSample_ID": str}).set_index("phenotype-BioSample_ID")
c["mem_r"] = c["Sample"].map(b["meropenem"]).eq(1)
held = c[c["Sample"].isin(in_bakrep)].sort_values("Sample")
for label, mask, n in [("Escherichia coli", held["species"].eq("Escherichia coli"), 3),
                       ("Salmonella enterica", held["species"].eq("Salmonella enterica"), 3),
                       ("Klebsiella pneumoniae", held["species"].eq("Klebsiella pneumoniae"), 3),
                       ("Enterobacter", held["species"].str.startswith("Enterobacter"), 1),
                       ("Shigella", held["species"].str.startswith("Shigella"), 1)]:
    g = held[mask]
    first = g[g["mem_r"]].sample(1, random_state=0) if g["mem_r"].any() and n > 1 else g.iloc[:0]
    rest = g.drop(index=first.index).sample(n - len(first), random_state=0)
    for _, r in pd.concat([first, rest]).iterrows():
        rows.append({"Sample": r["Sample"], "group": "entero", "species": r["species"], "contigs_source": "atb"})
fail = c[c["Sample"].isin(atb_fail)].sort_values("Sample")
for genus, n in [("Salmonella", 2), ("Escherichia", 1)]:
    for _, r in fail[fail["species"].str.startswith(genus)].sample(n, random_state=0).iterrows():
        rows.append({"Sample": r["Sample"], "group": "atb_bakta_fail", "species": r["species"],
                     "contigs_source": "atb"})
roster = pd.DataFrame(rows)
idx = read_atb_index(sorted(atb.glob("*.csv.gz")))
is_atb = roster["contigs_source"].eq("atb")
missing = roster.loc[is_atb & ~roster["Sample"].isin(idx.index), "Sample"].tolist()
assert not missing, f"not in the ATB index: {missing}"
roster.loc[is_atb, "contigs_path"] = [str(gate / "contigs" / f"{s}.fna.gz") for s in roster.loc[is_atb, "Sample"]]
roster["atb_md5"] = roster["Sample"].map(idx["atb_md5"]).where(is_atb)
roster["atb_bytes"] = roster["Sample"].map(idx["atb_bytes"]).where(is_atb)
assert roster["Sample"].is_unique and len(roster) == 18, len(roster)
roster.to_csv(gate / "roster.csv", index=False)
roster[["Sample", "contigs_path"]].to_csv(gate / "bakta_roster.tsv", sep="\t", header=False, index=False)
print(roster[["Sample", "group", "species", "contigs_source"]].to_string(index=False))
EOF
    ;;
fetch)
    [[ -s "$GATE/roster.csv" ]] || { echo "run roster first"; exit 1; }
    mkdir -p "$GATE/contigs" "$GATE/bakrep" "$GATE/atb"
    uv run --no-sync python -I - "$GATE" <<'EOF'
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
from bacpredict.engine.genomes.bakta import fetch_bakrep
from bacpredict.engine.genomes.build_genomes import download, read_fasta, write_renamed_contigs
from bacpredict.engine.genomes.plan_genome_sources import ATB_URL
gate = Path(sys.argv[1])
roster = pd.read_csv(gate / "roster.csv", dtype=str)

def one(r):
    s, msgs = r["Sample"], []
    if r["contigs_source"] == "atb" and not Path(r["contigs_path"]).exists():
        raw = gate / "atb" / f"{s}.fa.gz"
        download(ATB_URL.format(sample=s), raw, md5=r["atb_md5"] if isinstance(r["atb_md5"], str) else None,
                 nbytes=r["atb_bytes"])
        write_renamed_contigs(read_fasta(raw), s, Path(r["contigs_path"]), "atb")
        raw.unlink()
        msgs.append("contigs")
    if r["group"] != "atb_bakta_fail" and not (gate / "bakrep" / f"{s}.bakta.json.gz").exists():
        fetch_bakrep(s, gate / "bakrep", ("gff3", "faa", "json"))
        msgs.append("bakrep")
    return f"{s}: {', '.join(msgs) or 'already here'}"

with ThreadPoolExecutor(4) as ex:
    for m in ex.map(one, [r for _, r in roster.iterrows()]):
        print(m, flush=True)
EOF
    ;;
submit)
    [[ -s "$GATE/bakta_roster.tsv" && -s "$DB/version.json" ]] || { echo "run roster and fetch; DB at $DB"; exit 1; }
    n=$(wc -l < "$GATE/bakta_roster.tsv")
    tools_bin="$(cd "$REPO/src/bacpredict/engine/genomes" && pixi run -e bakta -- printenv CONDA_PREFIX)/bin"
    # 10 cores / 64 GB: the partition's memory per core; 1 h 30 min for a genome expected to take 10-30 min
    submit --array=0-$((n - 1)) --time=01:30:00 --job-name=bakta-vs-bakrep \
        --export=ALL,ROSTER=$GATE/bakta_roster.tsv,OUT_DIR=$GATE/ours,TOOLS_BIN=$tools_bin,DB=$DB,CHUNK=1,EXPECT_SHA=$(git rev-parse HEAD) \
        src/bacpredict/engine/genomes/scripts/run_bakta_array.sh
    ;;
compare)
    uv run --no-sync python -m bacpredict.engine.genomes.compare_bakta_to_bakrep --gate-dir "$GATE"
    ;;
*)
    sed -n '2,9p' "$0"; exit 2 ;;
esac

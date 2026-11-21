#!/usr/bin/env bash
# CABBAGE v2 (PROJECT_STATE §3.7): the Enterobacterales cohort's genomes, built as build_entero_genomes.sh built §3.6's —
# ATB → portal GFF → shovill (--prefer-reads, 10×, §3.6's genome size per genus), Prodigal + mlst, CheckM2 + skani
# against the same panel, collate in report mode — for the genomes no §3.6 (or v2 P. aeruginosa) manifest names; every
# other genome is §3.6's own, reused with its QC. Stages and mechanics: engine/genomes/scripts/build_genomes_reusing_built.sh.
#
# What differs from §3.6's first build, and why:
# - A genome with no ATB assembly, no portal assembly and no usable reads takes its submitter's GenBank assembly from
#   NCBI (ncbi_gca; David, 2026-11-21: "use NCBI when no reads exist") — ~15,500 v2 genomes have no reads in ENA or SRA.
# - The MLST scheme is pinned at submission (one array per scheme), the fix §3.6 then applied to its own genomes
#   (PROJECT_STATE §6, 2026-11-21): per genus, the scheme each genus ended on in §3.6's re-typed genome_qc.csv, plus
#   v2 genera §3.6 did not have (Citrobacter, Yersinia, Cronobacter); a genus with no scheme (Morganella, Kluyvera,
#   Hafnia, …) is built without one, as in §3.6. Per species, over the genus: aerogenes (§3.6 labels it Enterobacter,
#   v2 Klebsiella) on kaerogenes, as §3.6's are; the K. oxytoca complex, which §3.6 did not have, on koxytoca rather
#   than the KpSC scheme its alleles are not in.
# - Aerogenes is sized as §3.6 sized it (Enterobacter, 4.9 Mbp) under its v2 name too, so its read depth is computed
#   the same way in both experiments.
#
#   ARRAY_REPO=<detached worktree> bash src/bacpredict/apps/entero/scripts/build_entero_genomes_cabbage_v2.sh <stage> …
set -euo pipefail

ROOT=${BACPREDICT_AMR_ROOT:-$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/amr_prediction}
[[ -d $ROOT ]] || { echo "no AMR root at $ROOT (set BACPREDICT_AMR_ROOT)"; exit 1; }
P=$ROOT/processed
E=$P/enterobacterales/genomes
REPO=$(cd "$(dirname "$0")" && git rev-parse --show-toplevel)
export ROOT T=$P/enterobacterales_cabbage_v2 JOB=v2-entero
export OWNER_MANIFESTS="$P/paeruginosa/genomes/genome_manifest.csv $E/genome_manifest.csv $P/paeruginosa_cabbage_v2/genomes/genome_manifest.csv"
export REUSE_SHARD_DIRS=$E/shards  # holds the MLST re-type's shards too: read in place, never through a copy (mtimes)
# §3.6's CheckM2 reports, every one its frozen lists call for (2,000 genomes per task): a missing one stops here
reports=()
for pair in checkm2_list.tsv:checkm2 checkm2_list_topup.tsv:topup/checkm2; do
    list=$E/qc/${pair%%:*}; dir=$E/qc/${pair#*:}
    if [[ ! -s $list ]]; then
        [[ $pair == checkm2_list.tsv:* ]] && { echo "no §3.6 CheckM2 list at $list"; exit 1; }
        continue
    fi
    for ((k = 0; k < ($(wc -l < "$list") + 1999) / 2000; k++)); do
        f=$(printf '%s/checkm2_quality_report_%05d.tsv' "$dir" "$k")
        [[ -s $f ]] || { echo "§3.6 CheckM2 report missing: $f"; exit 1; }
        reports+=("$f")
    done
done
export REUSE_CHECKM2="${reports[*]}"
export REUSE_SKANI=$E/qc/skani_dist.tsv
export REUSE_GENOME_QC=$E/genome_qc.csv
export SCHEME_BY_SPECIES="Klebsiella aerogenes=kaerogenes,Enterobacter aerogenes=kaerogenes,Klebsiella oxytoca=koxytoca,Klebsiella michiganensis=koxytoca,Klebsiella grimontii=koxytoca,Klebsiella pasteurii=koxytoca"
export SCHEME_BY_GENUS=Escherichia=ecoli_achtman_4,Shigella=ecoli_achtman_4,Salmonella=salmonella,Klebsiella=klebsiella,Enterobacter=ecloacae,Serratia=serratia,Proteus=proteus,Providencia=providencia,Citrobacter=cfreundii,Yersinia=yersinia,Cronobacter=cronobacter
# §3.6's: typical complete-genome lengths per genus; any other genus 5.0 Mbp
export GENOME_SIZE=5.0e6
export GENOME_SIZES=Escherichia=5.1e6,Shigella=4.6e6,Salmonella=4.9e6,Klebsiella=5.6e6,Enterobacter=4.9e6,Serratia=5.2e6,Proteus=4.1e6,Morganella=3.9e6,Providencia=4.3e6
export GENOME_SIZES_BY_SPECIES="Klebsiella aerogenes=4.9e6"
# the reference list §3.6's own skani job published beside its result: v2 is called against exactly that panel
[[ -s $E/qc/skani_dist.refs.tsv ]] || { echo "no §3.6 skani reference list at $E/qc/skani_dist.refs.tsv"; exit 1; }
REFS=$(cat "$E/qc/skani_dist.refs.tsv")
export REFS
export SPECIES_MODE=report
export ATB_INDEX=$ROOT/raw/allthebacteria/assemblies_list_20261115
exec bash "$REPO/src/bacpredict/engine/genomes/scripts/build_genomes_reusing_built.sh" "$@"

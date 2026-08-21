#!/usr/bin/env bash
# Per-drug progress across the AST GWAS fan-out, plus the two ways a chain stalls silently.
#
# The chain is prep -> array(64) -> combine, with combine on `afterok`. Two failure modes leave a
# drug looking "in progress" forever:
#
#   1. A shard hits a transient error (on this cluster, "failed to map segment from shared object"
#      when many jobs mmap the same conda .so at once). SLURM requeues it HELD, so it never runs
#      and neither does the combine. `scontrol release` restarts it.
#   2. A shard genuinely FAILED once. Even if a later run wrote every chunk_i.assoc, `afterok` has
#      already been dissatisfied, so the combine never fires and the drug sits at 64/64 with no
#      .assoc. That one needs the combine resubmitting by hand.
#
# Both are invisible from squeue alone, which is why this reports shards-on-disk next to job state.
#
# Usage:
#   BACPREDICT_DATA_ROOT=<root> bash src/bac_pyseer/ast_gwas/scripts/fanout_status.sh [drug...]
set -euo pipefail

ORGANISM=${ORGANISM:-kp}
DATA_ROOT=${BACPREDICT_DATA_ROOT:?set BACPREDICT_DATA_ROOT}
NSHARDS=${NSHARDS:-64}
COHORT=${COHORT:-sampled_country_2_1_all}   # the shard job's default; shard results are keyed on it
SHARD_ROOT=${SHARD_ROOT:-/home/dca36/rds/hpc-work/unitig_shards}

OUT_DIR=$DATA_ROOT/processed/pyseer_ast/$ORGANISM
case "$ORGANISM" in
    kp) TASK=train_kleb_ast ;;
    tb) TASK=train_tb_ast ;;
    *)  echo "ORGANISM must be kp or tb" >&2; exit 1 ;;
esac

if [ "$#" -gt 0 ]; then
    drugs=("$@")
else
    mapfile -t drugs < <(ls "$DATA_ROOT/processed/$TASK/splits"/*_split.csv 2>/dev/null \
        | xargs -n1 basename | sed 's/_split\.csv$//' | sort)
fi

printf "%-32s %-9s %-8s %-8s %-7s %-6s %s\n" DRUG PHENO SHARDS ASSOC HITS LR NOTE
for drug in "${drugs[@]}"; do
    d=$OUT_DIR/$drug
    # Count NON-EMPTY shard results only. A shard that dies mid-pyseer leaves a 0-byte
    # chunk_NN.assoc behind, so counting files reports 64/64 while the combine still refuses with
    # "missing shard assoc". Counting files is exactly how I misread ceftazidime as complete.
    # Both the `find` and the `wc` must be shielded: find exits non-zero when the shard dir does
    # not exist yet (every drug not started), and `set -o pipefail` turns that into a script exit.
    shard_dir=$SHARD_ROOT/$drug/$COHORT
    if [ -d "$shard_dir" ]; then
        shards=$({ find "$shard_dir" -maxdepth 1 -name 'chunk_*.assoc' -size +0 2>/dev/null || true; } | wc -l)
        empty=$({ find "$shard_dir" -maxdepth 1 -name 'chunk_*.assoc' -size 0 2>/dev/null || true; } | wc -l)
    else
        shards=0; empty=0
    fi
    assoc=$([ -s "$d/gwas/$drug.assoc" ] && echo yes || echo -)
    hits=$([ -s "$d/gwas/${drug}_hits_annotated.tsv" ] || [ -s "$d/${drug}_hits_annotated.tsv" ] && echo yes || echo -)
    lr=$([ -s "$d/lr/results.json" ] && echo yes || echo -)
    note=""
    if [ "${empty:-0}" -gt 0 ]; then
        note="$empty EMPTY shard assoc -> rerun those shards, then combine"
    elif [ "$shards" -eq "$NSHARDS" ] && [ "$assoc" = "-" ]; then
        note="ALL SHARDS DONE BUT NO ASSOC -> resubmit combine"
    fi
    printf "%-32s %-9s %-8s %-8s %-7s %-6s %s\n" "$drug" \
        "$([ -s "$d/phenotype.tsv" ] && echo yes || echo -)" \
        "$shards/$NSHARDS" "$assoc" "$hits" "$lr" "$note"
done

held=$({ squeue -u "$USER" -h -o "%i %r" 2>/dev/null || true; } | grep -c "launch_failed_requeued_held" || true)
if [ "${held:-0}" -gt 0 ]; then
    echo
    echo "!! $held job(s) are PENDING(launch failed requeued held) and will never start. Release them:"
    squeue -u "$USER" -h -o "%i %r" | awk '/launch_failed_requeued_held/ {print "     scontrol release " $1}'
fi

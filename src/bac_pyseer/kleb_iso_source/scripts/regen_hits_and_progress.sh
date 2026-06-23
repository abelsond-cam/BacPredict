#!/bin/bash
# Re-postprocess the saved variant-LMM `.assoc` files with --effect-map, adding the per-hit `consequence`
# (synonymous / missense / LoF / noncoding) + `display_name` (gene-symbol→product→locus_tag) columns to the
# annotated hit tables. Pure pandas over the saved .assoc + the effect map → light (a couple of minutes,
# < a few GB): runs on the HPC LOGIN NODE, not SLURM. The .assoc inputs are untouched (only the derived
# *_hits_annotated.tsv + *_gwas_summary.json + figures regenerate; values are identical, two columns added).
#
# After this, scp the regenerated *_hits_annotated.tsv + *_gwas_summary.json into the repo docs dirs, then
# build the cross-axis table + progress figures LOCALLY (build_cross_axis_table.py / make_progress_figures.py).
# Usage (on HPC):  bash src/bac_pyseer/kleb_iso_source/scripts/regen_hits_and_progress.sh

set -euo pipefail
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/home/dca36/rds/hpc-work/.uv_cache
unset PYTHONPATH PYTHONHOME
REPO=/home/dca36/workspace/BacPredict
cd "$REPO"

DATA=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw
PYSEER=$DATA/david/processed/pyseer_iso_source
GFF=$DATA/david/raw/related_lr/gff/GCF_000016305.1.gff
EMAP=$PYSEER/source_hotspot/locus_effect_map.tsv.gz
[ -s "$EMAP" ] || { echo "ERROR: effect map $EMAP missing"; exit 1; }

regen () {  # <gwas_dir> <stem> <pos_label> <pair_title>
    local dir=$1 stem=$2 pos=$3 title=$4
    [ -s "$dir/$stem.assoc" ] || { echo "ERROR: missing $dir/$stem.assoc"; exit 1; }
    echo "=== re-postprocess $stem ($dir) ==="
    uv run python src/bac_pyseer/kleb_iso_source/pyseer_postprocess.py \
        --assoc "$dir/$stem.assoc" --patterns "$dir/patterns.txt" --gff "$GFF" --effect-map "$EMAP" \
        --feature-mode variants --out-fig-dir "$dir" \
        --out-table "$dir/${stem}_hits_annotated.tsv" \
        --summary-json "$dir/${stem}_gwas_summary.json" \
        --contig NC_009648 --pos-label "$pos" --neg-label faeces --pair-title "$title"
    echo "consequence column head:"; head -1 "$dir/${stem}_hits_annotated.tsv" | tr '\t' '\n' | grep -nE 'consequence|display_name' || true
}

regen "$PYSEER/blood_faeces/sampled_country_2_1_all/gwas_lmm"       blood_vs_faeces       "blood (invasion)"       "blood vs faeces"
regen "$PYSEER/faeces_respiratory/sampled_country_2_1_all/gwas_lmm" respiratory_vs_faeces "respiratory (invasion)" "faeces vs respiratory"
echo "=== done — scp the *_hits_annotated.tsv + *_gwas_summary.json into the repo docs dirs ==="

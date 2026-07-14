#!/usr/bin/env bash
# Build the single assembly list file and the two GFF list files
# used by add_paths_gff_fna_to_metadata.py.
#
# - assemblies_file_list.txt: all assembly FASTA paths
# - ncbi_gff.txt: all NCBI GFF paths (non-recursive)
# - klebsiella_gff.txt: all Klebsiella GFF paths (non-recursive)

set -euo pipefail

# Data root — one env var, cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
: "${BACPREDICT_DATA_ROOT:="$SCRATCHDIR"}"
D="$BACPREDICT_DATA_ROOT"
PY="$SCRATCHDIR/envs/bacpredict-gpu-venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"

# FLAG: the sibling 'seb' assembly/GFF tree is a CSD3-era external cohort with no confirmed
# Isambard home yet. Override with BACPREDICT_SEB_ROOT once that tree is staged on scratch.
SEB="${BACPREDICT_SEB_ROOT:-$D/seb}"

ASSEMBLIES_OUT="${D}/raw/assemblies_file_list.txt"
NCBI_GFF_DIR="${D}/raw/ncbi_gff3"
NCBI_GFF_OUT="${D}/raw/ncbi_gff.txt"
KLEB_GFF_DIR="${D}/raw/klebsiella_gff3"
KLEB_GFF_OUT="${D}/raw/klebsiella_gff.txt"

echo "Building assemblies_file_list.txt..."

# 1) assemblies_2: one line per compressed assembly (.gz) file in each
#    immediate subdir (overwrite). All assemblies (ncbi_*, NCTC, missing,
#    klebsiella_*, atb_*, etc.) now live as flat .fa.gz / .fna.gz files
#    directly under these subdirectories, so a simple *.gz match is enough.
: > "${ASSEMBLIES_OUT}"
for d in "${SEB}/assemblies_2"/*; do
  if [[ -d "${d}" ]]; then
    ls "${d}"/*.gz 2>/dev/null >> "${ASSEMBLIES_OUT}" || true
  fi
done

# 2) atb_david/kpsc and non_kpsc: .fa and .fa.gz
ls "${SEB}/assemblies/atb_david/kpsc"/*.fa* 2>/dev/null >> "${ASSEMBLIES_OUT}" || true
ls "${SEB}/assemblies/atb_david/non_kpsc"/*.fa* 2>/dev/null >> "${ASSEMBLIES_OUT}" || true

# 3) ncbi_dataset/data: one subdir per sample, each has a .fna (or .fna.gz)
for d in "${SEB}/assemblies_2/ncbi_03122025/ncbi_kpn/ncbi_dataset/data"/*; do
  if [[ -d "${d}" ]]; then
    ls "${d}"/*.fna* 2>/dev/null >> "${ASSEMBLIES_OUT}" || true
  fi
done

echo "Wrote $(wc -l < "${ASSEMBLIES_OUT}") assembly paths to ${ASSEMBLIES_OUT}"

echo
echo "Building ncbi_gff.txt..."
find "${NCBI_GFF_DIR}" -maxdepth 1 -type f -name "*.gff*" -print > "${NCBI_GFF_OUT}"
echo "Wrote $(wc -l < "${NCBI_GFF_OUT}") NCBI GFF paths to ${NCBI_GFF_OUT}"

echo
echo "Building klebsiella_gff.txt..."
find "${KLEB_GFF_DIR}" -maxdepth 1 -type f -name "*.gff*" -print > "${KLEB_GFF_OUT}"
echo "Wrote $(wc -l < "${KLEB_GFF_OUT}") Klebsiella GFF paths to ${KLEB_GFF_OUT}"

echo
echo "Running add_paths_gff_fna_to_metadata.py (parse path lists, write TSVs, add paths to metadata)..."
"$PY" -m bacpredict.apps.kleb.add_paths_gff_fna_to_metadata

# Having saved full paths in the data, now strip the base path from the metadata file.
METADATA_F="${D}/final/metadata_v2_all_samples_and_columns.tsv"

if [[ -f "$METADATA_F" ]]; then
  sed -i "s|${D}/||g" "$METADATA_F"
  sed -i "s|${SEB}/||g" "$METADATA_F"
  echo "Stripped ${D}/ and ${SEB}/ from paths in ${METADATA_F}"
else
  echo "Metadata file not found: ${METADATA_F}" >&2
fi

echo
echo "Done."

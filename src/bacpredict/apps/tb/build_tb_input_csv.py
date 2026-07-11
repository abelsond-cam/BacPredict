#!/usr/bin/env python3
"""Build the (Sample, sr_assembly_file, sr_gff_file) input CSV for the TB pipeline.

The Klebsiella path-builders (``find_missing_embeddings.py``,
``add_paths_gff_fna_to_metadata.py``) are tied to the curated KPSC metadata TSV
and cannot be reused for TB. This standalone builder derives the same
three-column CSV that ``preprocess_assemblies_to_protein_sequences.py`` expects
directly from the downloaded TB raw layout:

- assemblies: flat ``<assemblies_dir>/<BIOSAMPLE>.fa.gz``
- Bakta GFF3s: bucketed ``<gff_dir>/<bucket>/<BIOSAMPLE>/<BIOSAMPLE>.bakta.gff3.gz``

Only BioSamples that have *both* an assembly and a GFF on disk are written
(the extractor needs the FASTA + GFF pair). The sample universe is the unique
``phenotype-BioSample_ID`` values in the TB AMR records CSV.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

BIOSAMPLE_COL = "phenotype-BioSample_ID"
GFF_PATTERN = "*.bakta.gff3.gz"


def _gff_index(gff_dir: Path) -> dict[str, Path]:
    """Map BioSample -> Bakta GFF3 path.

    BakRep writes one subdir per BioSample, so the parent directory name is the
    BioSample ID regardless of the bucket prefix above it.
    """
    if not gff_dir.is_dir():
        return {}
    return {p.parent.name: p for p in gff_dir.rglob(GFF_PATTERN)}


def build(records_csv: Path, assemblies_dir: Path, gff_dir: Path, out_csv: Path) -> int:
    """Write the (Sample, sr_assembly_file, sr_gff_file) CSV; return the row count."""
    df = pd.read_csv(records_csv, low_memory=False)
    if BIOSAMPLE_COL not in df.columns:
        print(f"ERROR: '{BIOSAMPLE_COL}' not in {records_csv}", file=sys.stderr)
        sys.exit(1)

    biosamples = (
        df[BIOSAMPLE_COL].astype(str).str.strip()
    )
    sam = sorted(set(biosamples[biosamples.str.startswith("SAM")]))
    print(f"Unique SAM* BioSamples in records: {len(sam):,}", file=sys.stderr)

    gff_by_bs = _gff_index(gff_dir)
    print(f"Bakta GFF3 files indexed: {len(gff_by_bs):,}", file=sys.stderr)

    rows = []
    no_asm = no_gff = 0
    for bs in sam:
        asm = assemblies_dir / f"{bs}.fa.gz"
        gff = gff_by_bs.get(bs)
        if not asm.is_file():
            no_asm += 1
            continue
        if gff is None:
            no_gff += 1
            continue
        rows.append({"Sample": bs, "sr_assembly_file": str(asm), "sr_gff_file": str(gff)})

    print(
        f"Dropped: {no_asm:,} without assembly, {no_gff:,} without GFF "
        f"(of those with an assembly)",
        file=sys.stderr,
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows, columns=["Sample", "sr_assembly_file", "sr_gff_file"])
    out.to_csv(out_csv, index=False)
    print(f"Wrote {len(out):,} rows to {out_csv}", file=sys.stderr)
    return len(out)


def main() -> None:
    """Parse CLI args and build the TB protein-extraction input CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-csv", type=Path, required=True, help="TB AMR records CSV")
    parser.add_argument("--assemblies-dir", type=Path, required=True, help="Flat <BIOSAMPLE>.fa.gz dir")
    parser.add_argument("--gff-dir", type=Path, required=True, help="Bucketed BakRep GFF3 dir")
    parser.add_argument("--out-csv", type=Path, required=True, help="Output (Sample, sr_assembly_file, sr_gff_file) CSV")
    args = parser.parse_args()
    build(args.records_csv, args.assemblies_dir, args.gff_dir, args.out_csv)


if __name__ == "__main__":
    main()

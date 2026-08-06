"""Back-compat shim — the non-coding (IGR) extractor now lives in :mod:`genome_prep`.

The CDS/IGR interval logic + feature taxonomy were centralised into the shared ``genome_prep`` package
(one home instead of four duplicated engine parsers). This module keeps the historical import path
``bacpredict.engine.embedding.extract_intergenic_from_gff_fna.extract_intergenic_from_gff_fna`` working
for existing callers (``extract_intergenic_to_parquet``) and tests; new code should import from
``genome_prep`` directly.
"""

from __future__ import annotations

from genome_prep.annotation import extract_intergenic as extract_intergenic_from_gff_fna

__all__ = ["extract_intergenic_from_gff_fna"]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Extract non-coding DNA regions (3 views) from a GFF + FASTA pair.")
    ap.add_argument("--gff", required=True)
    ap.add_argument("--fna", required=True)
    ap.add_argument("--min-len", type=int, default=30)
    a = ap.parse_args()
    out = extract_intergenic_from_gff_fna(a.gff, a.fna, min_len=a.min_len)
    print(
        f"{len(out['noncoding_sequence'])} whole runs; {len(out['fragment_sequence'])} fragments; "
        f"{len(out['feature_sequence'])} named feature bodies"
    )

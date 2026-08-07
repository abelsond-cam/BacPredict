"""genome_prep — the shared genome-annotation model for BacPredict.

One GFF3/FASTA parser, one CDS/IGR feature taxonomy + interval math, the non-coding sequence
extractor (used by the engine's baclm re-embed), a ``(contig, start, end) → (coding_class, igr_type)``
classifier, and a genome-wide coding-fraction baseline. Consumed by both ``bacpredict`` (engine) and
``bac_pyseer`` so the coding-vs-intergenic definition is identical everywhere.
"""

from __future__ import annotations

from genome_prep.annotation import (
    CDS_CLASS,
    IGR_CLASS,
    CodingIndex,
    Feature,
    coding_fraction,
    contig_lengths,
    extract_intergenic,
    parse_gff_features,
)
from genome_prep.features import (
    FEATURE_TYPES,
    OCCUPYING_TYPE,
    UNCLASSIFIED_IGR,
    complement,
    merge_intervals,
    subtract,
)
from genome_prep.gff import (
    is_gbff_path,
    is_gff_path,
    load_fna,
    open_text,
    parse_gff_attributes,
)

__all__ = [
    "CDS_CLASS",
    "IGR_CLASS",
    "CodingIndex",
    "Feature",
    "FEATURE_TYPES",
    "OCCUPYING_TYPE",
    "UNCLASSIFIED_IGR",
    "coding_fraction",
    "complement",
    "contig_lengths",
    "extract_intergenic",
    "is_gbff_path",
    "is_gff_path",
    "load_fna",
    "merge_intervals",
    "open_text",
    "parse_gff_attributes",
    "parse_gff_features",
    "subtract",
]

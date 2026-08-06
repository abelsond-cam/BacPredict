"""Low-level GFF3 / FASTA IO primitives shared across genome-annotation code.

These were previously private helpers inside ``bacpredict.engine.embedding.extract_proteins_from_gff_fna``
(reached by cross-module underscore imports). They are canonicalised here so every consumer — the engine
extractors and the ``bac_pyseer`` mapping jobs — parses GFF/FASTA the same way.
"""

from __future__ import annotations

import gzip
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq

_GFF_SUFFIXES = (".gff", ".gff3", ".gff.gz", ".gff3.gz")
_GBFF_SUFFIXES = (".gbff", ".gbff.gz")


def is_gff_path(path: str | Path) -> bool:
    """Return True if ``path`` ends with a GFF/GFF3 extension (gzipped or not)."""
    return str(path).lower().endswith(_GFF_SUFFIXES)


def is_gbff_path(path: str | Path) -> bool:
    """Return True if ``path`` ends with a GenBank flat-file extension (gzipped or not)."""
    return str(path).lower().endswith(_GBFF_SUFFIXES)


def open_text(path: str | Path):
    """Open a text file transparently, decompressing ``.gz`` if needed."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


def load_fna(fna_path: str | Path) -> dict[str, Seq]:
    """Load contigs from a (gzipped) FASTA into a ``dict[seqid, Seq]``."""
    with open_text(fna_path) as handle:
        return {record.id: record.seq for record in SeqIO.parse(handle, "fasta")}


def parse_gff_attributes(field: str) -> dict[str, str]:
    """Parse a GFF3 column-9 attributes string into a dict (last value wins on duplicate keys)."""
    out: dict[str, str] = {}
    for attr in field.split(";"):
        if "=" in attr:
            key, value = attr.split("=", 1)
            out[key.strip()] = value.strip()
    return out

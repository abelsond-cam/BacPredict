"""Extract protein sequences from a Bakta/NCBI GFF3 + sibling FASTA pair.

Bacformer's `preprocess_genome_assembly` GFF code path returns annotation
metadata only (no translations). For samples with `.gff(.gff3)(.gz)` annotations
+ separate `.fna(.gz)` assemblies, we splice CDS regions from the FASTA and
translate them with the bacterial codon table.

Output shape matches the keys the downstream parquet writer expects.
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path
from typing import Any

from Bio import SeqIO
from Bio.Seq import Seq

logger = logging.getLogger(__name__)

_GFF_SUFFIXES = (".gff", ".gff3", ".gff.gz", ".gff3.gz")
_GBFF_SUFFIXES = (".gbff", ".gbff.gz")

# Bacterial initiator codons (table 11). BioPython's default translate() renders
# GTG/TTG as V/L because it treats every codon as an internal codon. CDSs whose
# nucleotide sequence begins with any of these should have their first amino acid
# rewritten as M to match initiator-codon biology and the BakRep/GBFF cohort,
# whose translations already start with M.
_BACTERIAL_INITIATORS = frozenset({"ATG", "GTG", "TTG", "CTG", "ATT", "ATC", "ATA"})


def is_gff_path(path: str | Path) -> bool:
    """Return True if `path` ends with a GFF/GFF3 extension (gzipped or not)."""
    s = str(path).lower()
    return s.endswith(_GFF_SUFFIXES)


def is_gbff_path(path: str | Path) -> bool:
    """Return True if `path` ends with a GenBank flat-file extension (gzipped or not)."""
    s = str(path).lower()
    return s.endswith(_GBFF_SUFFIXES)


def _open_text(path: Path):
    """Open a text file transparently, decompressing if needed."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


def _load_fna(fna_path: Path) -> dict[str, Seq]:
    """Load contigs from a (gzipped) FASTA into a dict[seqid, Seq]."""
    with _open_text(fna_path) as handle:
        return {record.id: record.seq for record in SeqIO.parse(handle, "fasta")}


def _parse_gff_attributes(field: str) -> dict[str, str]:
    """Parse a GFF3 attributes column into a dict (last value wins on duplicate keys)."""
    out: dict[str, str] = {}
    for attr in field.split(";"):
        if "=" in attr:
            key, value = attr.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def extract_proteins_from_gff_fna(
    gff_path: str | Path,
    fna_path: str | Path,
    *,
    translation_table: int = 11,
    keep_internal_stop: bool = False,
) -> dict[str, Any]:
    """Extract protein sequences for one genome from a GFF + FASTA pair.

    Parameters
    ----------
    gff_path : str or Path
        Path to a GFF3 file (plain or gzipped). Bakta-annotated files are
        preferred; NCBI PGAP GFFs also work.
    fna_path : str or Path
        Path to the genome FASTA (plain or gzipped) whose contig IDs match the
        GFF `seqid` column.
    translation_table : int, default 11
        NCBI translation table (11 = bacterial/archaeal/plant plastid).
    keep_internal_stop : bool, default False
        Keep CDS whose translation has an internal stop (the ``*`` is retained in
        the sequence) instead of dropping them. Off by default. Set this only to
        reproduce a *historical* protein order — e.g. regenerating the Kp
        protein-sequence parquets so their flat index still aligns 1:1 with
        embeddings generated before the internal-stop skip was added.

    Returns
    -------
    dict
        Keys mirror the structure produced by `preprocess_genome_assembly` for
        GBFF inputs (contig-grouped lists), at minimum containing
        ``protein_sequence`` (list of amino-acid strings).
    """
    gff_path = Path(gff_path)
    fna_path = Path(fna_path)

    contigs = _load_fna(fna_path)
    if not contigs:
        raise ValueError(f"No contigs parsed from FASTA: {fna_path}")

    # Per-contig accumulators (keyed by seqid encounter order for stable contig_idx).
    contig_order: list[str] = []
    per_contig: dict[str, dict[str, list]] = {}

    n_skipped_pseudo = 0
    n_skipped_internal_stop = 0
    n_skipped_missing_contig = 0

    with _open_text(gff_path) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                # Bakta appends a ##FASTA block at EOF; stop parsing features there.
                if line.startswith("##FASTA"):
                    break
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "CDS":
                continue

            seqid = parts[0]
            try:
                start = int(parts[3])
                end = int(parts[4])
            except ValueError:
                continue
            strand = parts[6]
            phase_raw = parts[7]
            try:
                phase = int(phase_raw) if phase_raw and phase_raw != "." else 0
            except ValueError:
                phase = 0
            attrs = _parse_gff_attributes(parts[8])

            if attrs.get("pseudo", "").lower() in {"true", "1"}:
                n_skipped_pseudo += 1
                continue

            contig_seq = contigs.get(seqid)
            if contig_seq is None:
                n_skipped_missing_contig += 1
                continue

            nt = contig_seq[start - 1 : end]
            if strand == "-":
                nt = nt.reverse_complement()

            # Phase column: drop this many bases from the start of the CDS to reach
            # the next codon boundary. Non-zero phase means the CDS is partial at
            # the 5' end (gene continues before this region), so there is no real
            # start codon to promote.
            if phase:
                nt = nt[phase:]
            # Trim trailing bases so the sequence is a clean multiple of 3
            # (prevents BiopythonWarning: "Partial codon").
            remainder = len(nt) % 3
            if remainder:
                nt = nt[:-remainder]
            if len(nt) < 3:
                continue

            is_partial_5p = phase != 0 or attrs.get("partial", "").lower() in {"true", "1", "10", "11"}

            protein = str(nt.translate(table=translation_table, to_stop=False))
            if protein.endswith("*"):
                protein = protein[:-1]
            if "*" in protein and not keep_internal_stop:
                n_skipped_internal_stop += 1
                continue
            if not protein:
                continue
            # Promote alternative start codons to M only when the CDS isn't
            # 5'-partial (a partial CDS does not begin at a real start codon).
            if (
                not is_partial_5p
                and len(nt) >= 3
                and str(nt[:3]).upper() in _BACTERIAL_INITIATORS
                and protein[0] != "M"
            ):
                protein = "M" + protein[1:]

            if seqid not in per_contig:
                contig_order.append(seqid)
                per_contig[seqid] = {
                    "gene_name": [],
                    "protein_name": [],
                    "start": [],
                    "end": [],
                    "protein_id": [],
                    "protein_sequence": [],
                }
            bucket = per_contig[seqid]
            locus_tag = attrs.get("locus_tag") or attrs.get("ID")
            bucket["gene_name"].append(attrs.get("gene") or locus_tag)
            bucket["protein_name"].append(locus_tag)
            bucket["start"].append(start)
            bucket["end"].append(end)
            bucket["protein_id"].append(attrs.get("protein_id"))
            bucket["protein_sequence"].append(protein)

    if n_skipped_pseudo or n_skipped_internal_stop or n_skipped_missing_contig:
        logger.info(
            "Skipped CDS records: pseudo=%d, internal_stop=%d, missing_contig=%d",
            n_skipped_pseudo,
            n_skipped_internal_stop,
            n_skipped_missing_contig,
        )

    # Flatten per-contig lists into the genome-level shape produced by
    # `preprocess_genome_assembly` for GBFF (lists-of-lists across contigs).
    contig_idx = list(range(len(contig_order)))
    out: dict[str, Any] = {
        "contig_idx": contig_idx,
        "gene_name": [per_contig[c]["gene_name"] for c in contig_order],
        "protein_name": [per_contig[c]["protein_name"] for c in contig_order],
        "start": [per_contig[c]["start"] for c in contig_order],
        "end": [per_contig[c]["end"] for c in contig_order],
        "protein_id": [per_contig[c]["protein_id"] for c in contig_order],
        "protein_sequence": [per_contig[c]["protein_sequence"] for c in contig_order],
    }
    return out

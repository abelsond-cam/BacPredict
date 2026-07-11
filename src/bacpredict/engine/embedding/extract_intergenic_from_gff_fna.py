"""Extract non-coding DNA regions (+ a named-RNA index) from a Bakta/NCBI GFF3 + sibling FASTA.

baclm-350m-masked is a mixed protein+DNA model: proteins are embedded UPPERCASE (the
`extract_proteins_from_gff_fna` output) and non-coding DNA is embedded **lowercase**. This module
produces the DNA half.

**Only ``CDS`` occupies (2d re-embed).** Earlier this module treated *every* annotated feature as
occupying, so an intergenic stretch abutting a tRNA/rRNA was split off from its neighbour and every
RNA *body* (``rrs``/``rrl``/``rrf``, tRNA, tmRNA, ncRNA) fell into no store at all. We now treat only
protein-coding ``CDS`` as occupying, so each non-coding region is the **maximal contiguous non-CDS
run** — promoter + any adjacent RNA embedded together, exactly the stretch baclm was trained on. RNA
bodies are additionally emitted **on their own** (``rna_*`` + their sequence) with a named index, so
``rrs`` is (a) locatable by name inside its run and (b) available as a standalone vector for the
"embed RNA separately from IGR?" architecture question.

Long regions are returned **whole** (no truncation here); the embedder windows + pools anything longer
than the model context. Strand is irrelevant for non-coding DNA, so the forward strand is used.

Returned keys (all parallel lists, contig-then-position order; 1-based inclusive forward coords):

* ``noncoding_sequence`` / ``noncoding_seqid`` / ``noncoding_start`` / ``noncoding_end`` — the maximal
  non-CDS runs (IGR ∪ any RNA they contain).
* ``rna_sequence`` / ``rna_gene_name`` / ``rna_type`` / ``rna_seqid`` / ``rna_start`` / ``rna_end`` —
  one entry per annotated RNA feature (best-effort ``gene``→``Name``→``product``→locus_tag label).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from bacpredict.engine.embedding.extract_proteins_from_gff_fna import _load_fna, _open_text, _parse_gff_attributes

logger = logging.getLogger(__name__)

# The only feature type that occupies sequence for the coding/non-coding split. Everything else
# (RNA genes, ``region``, ``gap``, misc features) is left inside the non-coding runs.
_OCCUPYING_TYPE = "CDS"

# GFF feature types whose bodies we index + embed as standalone RNA (lower-cased ``parts[2]``).
_RNA_TYPES = frozenset(
    {"rrna", "trna", "tmrna", "ncrna", "ncrna_gene", "antisense_rna", "rnase_p_rna", "srp_rna", "riboswitch"}
)


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping/adjacent 0-based half-open intervals; returns them sorted."""
    if not intervals:
        return []
    intervals.sort()
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _rna_label(attrs: dict[str, str]) -> str:
    """Best-effort RNA gene name: ``gene`` → ``Name`` → ``product`` → ``locus_tag``/``ID``.

    Bakta annotates rRNA with ``product`` (e.g. "16S ribosomal RNA") and often no ``gene``; tRNA/ncRNA
    usually carry ``gene``. Falling through in this order gives a searchable label for every RNA (so
    ``rrs`` is found by ``gene=rrs`` where present, else by the "16S ribosomal RNA" product string).
    """
    return attrs.get("gene") or attrs.get("Name") or attrs.get("product") or attrs.get("locus_tag") or attrs.get("ID") or ""


def extract_intergenic_from_gff_fna(
    gff_path: str | Path,
    fna_path: str | Path,
    *,
    min_len: int = 30,
) -> dict[str, Any]:
    """Extract non-coding DNA regions (+ RNA index) for one genome from a GFF + FASTA pair.

    Parameters
    ----------
    gff_path, fna_path : str or Path
        A GFF3 (plain/gzipped) and the genome FASTA whose contig IDs match its ``seqid`` column.
    min_len : int, default 30
        Minimum non-coding-run length (bp) to keep — drops the 1–few bp slivers between adjacent
        genes. RNA bodies are indexed regardless of ``min_len`` (a tRNA is ~76 bp but always wanted).

    Returns
    -------
    dict
        The ``noncoding_*`` and ``rna_*`` parallel-list columns documented in the module docstring.
    """
    gff_path, fna_path = Path(gff_path), Path(fna_path)
    contigs = _load_fna(fna_path)
    if not contigs:
        raise ValueError(f"No contigs parsed from FASTA: {fna_path}")

    # One pass: collect CDS intervals (occupying, 0-based half-open) + RNA features (indexed) per contig.
    occupied: dict[str, list[tuple[int, int]]] = {}
    rna_by_contig: dict[str, list[tuple[int, int, str, str]]] = {}  # seqid -> [(start1, end1, label, type)]
    with _open_text(gff_path) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                if line.startswith("##FASTA"):
                    break
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            ftype = parts[2]
            seqid = parts[0]
            try:
                start, end = int(parts[3]), int(parts[4])
            except ValueError:
                continue
            if ftype == _OCCUPYING_TYPE:
                occupied.setdefault(seqid, []).append((start - 1, end))  # 1-based incl -> 0-based half-open
            elif ftype.lower() in _RNA_TYPES:
                label = _rna_label(_parse_gff_attributes(parts[8]))
                rna_by_contig.setdefault(seqid, []).append((start, end, label, ftype.lower()))

    nc_seqs: list[str] = []
    nc_seqids: list[str] = []
    nc_starts: list[int] = []
    nc_ends: list[int] = []
    # Walk every contig (a contig with no CDS is wholly one non-coding run).
    for seqid in list(contigs.keys()):
        clen = len(contigs[seqid])
        merged = _merge_intervals(occupied.get(seqid, []))
        gaps: list[tuple[int, int]] = []
        prev_end = 0
        for s, e in merged:
            if s > prev_end:
                gaps.append((prev_end, s))
            prev_end = max(prev_end, e)
        if prev_end < clen:
            gaps.append((prev_end, clen))
        contig_seq = contigs[seqid]
        for g0, g1 in gaps:
            if g1 - g0 < min_len:
                continue
            nc_seqs.append(str(contig_seq[g0:g1]).lower())
            nc_seqids.append(seqid)
            nc_starts.append(g0 + 1)  # back to 1-based inclusive
            nc_ends.append(g1)

    # RNA bodies (their own sequence + named index), contig-then-position order.
    rna_seqs: list[str] = []
    rna_names: list[str] = []
    rna_types: list[str] = []
    rna_seqids: list[str] = []
    rna_starts: list[int] = []
    rna_ends: list[int] = []
    for seqid in list(contigs.keys()):
        contig_seq = contigs[seqid]
        for start1, end1, label, rtype in sorted(rna_by_contig.get(seqid, [])):
            rna_seqs.append(str(contig_seq[start1 - 1 : end1]).lower())
            rna_names.append(label)
            rna_types.append(rtype)
            rna_seqids.append(seqid)
            rna_starts.append(start1)
            rna_ends.append(end1)

    logger.info(
        "non-coding: %d runs >= %d bp, %d RNA bodies over %d contigs (%s)",
        len(nc_seqs), min_len, len(rna_seqs), len(contigs), fna_path.name,
    )
    return {
        "noncoding_sequence": nc_seqs,
        "noncoding_seqid": nc_seqids,
        "noncoding_start": nc_starts,
        "noncoding_end": nc_ends,
        "rna_sequence": rna_seqs,
        "rna_gene_name": rna_names,
        "rna_type": rna_types,
        "rna_seqid": rna_seqids,
        "rna_start": rna_starts,
        "rna_end": rna_ends,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Extract non-coding DNA regions (+ RNA index) from a GFF + FASTA pair.")
    ap.add_argument("--gff", required=True)
    ap.add_argument("--fna", required=True)
    ap.add_argument("--min-len", type=int, default=30)
    a = ap.parse_args()
    out = extract_intergenic_from_gff_fna(a.gff, a.fna, min_len=a.min_len)
    print(f"{len(out['noncoding_sequence'])} non-coding runs >= {a.min_len} bp; {len(out['rna_sequence'])} RNA bodies")

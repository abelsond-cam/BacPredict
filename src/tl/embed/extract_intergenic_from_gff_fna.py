"""Extract intergenic (non-coding) DNA regions from a Bakta/NCBI GFF3 + sibling FASTA.

baclm-350m-masked is a mixed protein+DNA model: proteins are embedded UPPERCASE (the
`extract_proteins_from_gff_fna` output) and non-coding DNA is embedded **lowercase**. This
module produces the DNA half — the stretches of each contig not covered by any annotated
feature (CDS, tRNA, rRNA, ncRNA, …). Merges all feature intervals per contig and returns the
complement (gaps ≥ ``min_len``), each as a lowercase forward-strand string with its coordinates.

Kept deliberately simple (per "don't over-engineer"): a region longer than the model context
(2048 chars) is returned whole — the embedder truncates + mean-pools it. Strand is irrelevant
for intergenic DNA, so the forward strand is used.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from tl.embed.extract_proteins_from_gff_fna import _load_fna, _open_text

logger = logging.getLogger(__name__)

# GFF feature types that do NOT occupy sequence for the coding/non-coding split.
_NON_OCCUPYING_TYPES = frozenset({"region", "databank_entry", "gap"})


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


def extract_intergenic_from_gff_fna(
    gff_path: str | Path,
    fna_path: str | Path,
    *,
    min_len: int = 30,
) -> dict[str, Any]:
    """Extract intergenic DNA regions for one genome from a GFF + FASTA pair.

    Parameters
    ----------
    gff_path, fna_path : str or Path
        A GFF3 (plain/gzipped) and the genome FASTA whose contig IDs match its ``seqid`` column.
    min_len : int, default 30
        Minimum region length (bp) to keep — drops the 1–few bp slivers between adjacent genes.

    Returns
    -------
    dict
        ``intergenic_sequence`` (list of lowercase DNA strings, in contig-then-position order),
        ``intergenic_seqid`` / ``intergenic_start`` / ``intergenic_end`` (parallel lists;
        1-based inclusive coordinates on the forward strand).
    """
    gff_path, fna_path = Path(gff_path), Path(fna_path)
    contigs = _load_fna(fna_path)
    if not contigs:
        raise ValueError(f"No contigs parsed from FASTA: {fna_path}")

    # Collect occupied (feature) intervals per contig, 0-based half-open.
    occupied: dict[str, list[tuple[int, int]]] = {}
    contig_order: list[str] = []
    with _open_text(gff_path) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                if line.startswith("##FASTA"):
                    break
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] in _NON_OCCUPYING_TYPES:
                continue
            seqid = parts[0]
            try:
                start, end = int(parts[3]), int(parts[4])
            except ValueError:
                continue
            if seqid not in occupied:
                occupied[seqid] = []
                contig_order.append(seqid)
            occupied[seqid].append((start - 1, end))  # 1-based incl -> 0-based half-open

    seqs: list[str] = []
    seqids: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    # Walk every contig (contigs with zero features are wholly intergenic).
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
            seqs.append(str(contig_seq[g0:g1]).lower())
            seqids.append(seqid)
            starts.append(g0 + 1)  # back to 1-based inclusive
            ends.append(g1)

    logger.info("intergenic: %d regions >= %d bp over %d contigs (%s)", len(seqs), min_len, len(contigs), fna_path.name)
    return {
        "intergenic_sequence": seqs,
        "intergenic_seqid": seqids,
        "intergenic_start": starts,
        "intergenic_end": ends,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Extract intergenic DNA regions from a GFF + FASTA pair.")
    ap.add_argument("--gff", required=True)
    ap.add_argument("--fna", required=True)
    ap.add_argument("--min-len", type=int, default=30)
    a = ap.parse_args()
    out = extract_intergenic_from_gff_fna(a.gff, a.fna, min_len=a.min_len)
    print(f"{len(out['intergenic_sequence'])} intergenic regions >= {a.min_len} bp")

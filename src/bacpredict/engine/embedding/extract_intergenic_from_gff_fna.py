"""Extract non-coding DNA regions from a Bakta/NCBI GFF3 + sibling FASTA — in THREE parallel views.

baclm-350m-masked is a mixed protein+DNA model: proteins are embedded UPPERCASE (the
`extract_proteins_from_gff_fna` output) and non-coding DNA is embedded **lowercase**. This module
produces the DNA half, emitting the same genome's non-coding space at two granularities plus a named
non-CDS feature index, so downstream LR screens can ask both "does the whole intergenic context predict?"
and "which specific element (this promoter fragment, this rRNA, this CRISPR) predicts?".

**Only ``CDS`` occupies.** Everything else (RNA genes, CRISPR, regulatory_region, oriC, ``region``,
``gap``, misc) is left inside the non-coding space. The three views:

* ``noncoding_*`` — **whole_igr**: each **maximal contiguous non-CDS run** (the region between two CDS,
  label-agnostic — any RNA/CRISPR *inside* it kept). One row per CDS-to-CDS gap.
* ``fragment_*`` — **per_unit (promoter fragments)**: each non-CDS run **split at every named non-CDS
  feature boundary** → the intergenic/regulatory fragments between features (a promoter isolated from an
  adjacent tRNA). One row per fragment.
* ``feature_*`` — **per_unit (named bodies)**: each **named non-CDS feature body** embedded standalone,
  with its ``type`` + ``name`` (rRNA ``rrs``/``rrl``/``rrf``, tRNA, tmRNA, ncRNA, **CRISPR**,
  **regulatory_region**, **oriC**), so ``rrs`` is locatable/probeable by name.

Long regions are returned **whole** (no truncation here); the embedder windows anything longer than the
model context into equal segments + pools. Strand is irrelevant for non-coding DNA, so the forward strand
is used. Coords are 1-based inclusive forward; sequences lowercased; contig-then-position order.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from bacpredict.engine.embedding.extract_proteins_from_gff_fna import _load_fna, _open_text, _parse_gff_attributes

logger = logging.getLogger(__name__)

# The only feature type that occupies sequence for the coding/non-coding split.
_OCCUPYING_TYPE = "CDS"

# Named non-CDS feature types (lower-cased ``parts[2]``): their bodies are indexed + embedded standalone,
# AND they fragment the non-coding run. RNA + CRISPR (the whole array, not crispr-repeat/spacer sub-features)
# + Bakta's explicit regulatory_region / oriC.
_FEATURE_TYPES = frozenset(
    {
        "rrna", "trna", "tmrna", "ncrna", "ncrna_gene", "antisense_rna", "rnase_p_rna", "srp_rna", "riboswitch",
        "crispr", "regulatory_region", "oric", "origin_of_replication",
    }
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


def _complement(occupied: list[tuple[int, int]], clen: int) -> list[tuple[int, int]]:
    """The gaps (0-based half-open) between merged occupied intervals over ``[0, clen)``."""
    merged = _merge_intervals(occupied)
    gaps: list[tuple[int, int]] = []
    prev_end = 0
    for s, e in merged:
        if s > prev_end:
            gaps.append((prev_end, s))
        prev_end = max(prev_end, e)
    if prev_end < clen:
        gaps.append((prev_end, clen))
    return gaps


def _subtract(run: tuple[int, int], cuts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sub-intervals of ``run`` (0-based half-open) left after removing ``cuts`` (clipped to the run)."""
    r0, r1 = run
    clipped = _merge_intervals([(max(s, r0), min(e, r1)) for s, e in cuts if e > r0 and s < r1])
    frags: list[tuple[int, int]] = []
    prev = r0
    for cs, ce in clipped:
        if cs > prev:
            frags.append((prev, cs))
        prev = max(prev, ce)
    if prev < r1:
        frags.append((prev, r1))
    return frags


def _feature_label(attrs: dict[str, str]) -> str:
    """Best-effort feature name: ``gene`` → ``Name`` → ``product`` → ``locus_tag``/``ID``.

    Bakta annotates rRNA with ``product`` (e.g. "16S ribosomal RNA") and often no ``gene``; tRNA/ncRNA
    usually carry ``gene``. Falling through in this order gives a searchable label for every feature.
    """
    return attrs.get("gene") or attrs.get("Name") or attrs.get("product") or attrs.get("locus_tag") or attrs.get("ID") or ""


def extract_intergenic_from_gff_fna(
    gff_path: str | Path,
    fna_path: str | Path,
    *,
    min_len: int = 30,
) -> dict[str, Any]:
    """Extract the three non-coding views for one genome from a GFF + FASTA pair.

    Parameters
    ----------
    gff_path, fna_path : str or Path
        A GFF3 (plain/gzipped) and the genome FASTA whose contig IDs match its ``seqid`` column.
    min_len : int, default 30
        Minimum length (bp) for a ``noncoding`` run and a ``fragment`` — drops the 1–few bp slivers.
        Named ``feature`` bodies are indexed regardless (a tRNA is ~76 bp but always wanted).

    Returns
    -------
    dict
        ``noncoding_*`` (whole runs), ``fragment_*`` (per-feature fragments), and
        ``feature_*`` (named bodies + ``feature_type``/``feature_name``) parallel-list columns.
    """
    gff_path, fna_path = Path(gff_path), Path(fna_path)
    contigs = _load_fna(fna_path)
    if not contigs:
        raise ValueError(f"No contigs parsed from FASTA: {fna_path}")

    # One pass: CDS intervals (occupying, 0-based half-open) + named non-CDS features (indexed) per contig.
    occupied: dict[str, list[tuple[int, int]]] = {}
    feat_by_contig: dict[str, list[tuple[int, int, str, str]]] = {}  # seqid -> [(start1, end1, label, type)]
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
            elif ftype.lower() in _FEATURE_TYPES:
                label = _feature_label(_parse_gff_attributes(parts[8]))
                feat_by_contig.setdefault(seqid, []).append((start, end, label, ftype.lower()))

    nc_seqs: list[str] = []
    nc_seqids: list[str] = []
    nc_starts: list[int] = []
    nc_ends: list[int] = []
    fr_seqs: list[str] = []
    fr_seqids: list[str] = []
    fr_starts: list[int] = []
    fr_ends: list[int] = []
    # Walk every contig (a contig with no CDS is wholly one non-coding run).
    for seqid in list(contigs.keys()):
        contig_seq = contigs[seqid]
        clen = len(contig_seq)
        # Named-feature cut intervals on this contig (0-based half-open), for fragmenting the runs.
        feat_cuts = [(s1 - 1, e1) for s1, e1, _lbl, _ft in feat_by_contig.get(seqid, [])]
        for g0, g1 in _complement(occupied.get(seqid, []), clen):
            # whole_igr: the maximal CDS-to-CDS run (label-agnostic).
            if g1 - g0 >= min_len:
                nc_seqs.append(str(contig_seq[g0:g1]).lower())
                nc_seqids.append(seqid)
                nc_starts.append(g0 + 1)  # back to 1-based inclusive
                nc_ends.append(g1)
            # per_unit fragments: the run split at every named-feature boundary.
            for f0, f1 in _subtract((g0, g1), feat_cuts):
                if f1 - f0 < min_len:
                    continue
                fr_seqs.append(str(contig_seq[f0:f1]).lower())
                fr_seqids.append(seqid)
                fr_starts.append(f0 + 1)
                fr_ends.append(f1)

    # Named non-CDS feature bodies (own sequence + type + name), contig-then-position order.
    feat_seqs: list[str] = []
    feat_names: list[str] = []
    feat_types: list[str] = []
    feat_seqids: list[str] = []
    feat_starts: list[int] = []
    feat_ends: list[int] = []
    for seqid in list(contigs.keys()):
        contig_seq = contigs[seqid]
        for start1, end1, label, ftype in sorted(feat_by_contig.get(seqid, [])):
            feat_seqs.append(str(contig_seq[start1 - 1 : end1]).lower())
            feat_names.append(label)
            feat_types.append(ftype)
            feat_seqids.append(seqid)
            feat_starts.append(start1)
            feat_ends.append(end1)

    logger.info(
        "non-coding: %d whole runs, %d fragments (>= %d bp), %d named feature bodies over %d contigs (%s)",
        len(nc_seqs), len(fr_seqs), min_len, len(feat_seqs), len(contigs), fna_path.name,
    )
    return {
        "noncoding_sequence": nc_seqs,
        "noncoding_seqid": nc_seqids,
        "noncoding_start": nc_starts,
        "noncoding_end": nc_ends,
        "fragment_sequence": fr_seqs,
        "fragment_seqid": fr_seqids,
        "fragment_start": fr_starts,
        "fragment_end": fr_ends,
        "feature_sequence": feat_seqs,
        "feature_name": feat_names,
        "feature_type": feat_types,
        "feature_seqid": feat_seqids,
        "feature_start": feat_starts,
        "feature_end": feat_ends,
    }


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

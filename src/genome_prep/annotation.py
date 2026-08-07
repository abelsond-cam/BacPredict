"""Genome-annotation model: GFF parser, CDS/IGR extractor, position classifier, coding-fraction baseline.

This is the shared home the codebase previously lacked — the CDS/IGR interval logic and the GFF
feature taxonomy were re-implemented four times across the engine. Both the engine (non-coding
re-embed) and the ``bac_pyseer`` unitig/variant IGR-mapping jobs build on the primitives here.

Conventions: GFF is 1-based inclusive on the wire; interval math runs 0-based half-open (see
:mod:`genome_prep.features`); ``extract_intergenic`` emits 1-based inclusive coordinates.
"""

from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from genome_prep.features import (
    FEATURE_TYPES,
    OCCUPYING_TYPE,
    UNCLASSIFIED_IGR,
    complement,
    merge_intervals,
    subtract,
)
from genome_prep.gff import load_fna, open_text, parse_gff_attributes

logger = logging.getLogger(__name__)

CDS_CLASS = "CDS"
IGR_CLASS = "IGR"


@dataclass(frozen=True)
class Feature:
    """One annotation feature kept for the CDS/IGR split (a ``CDS`` or a named non-CDS type).

    ``start``/``end`` are 1-based inclusive (GFF convention). ``ftype`` is the raw GFF type column
    (``"CDS"``, ``"rRNA"``, …); ``name`` is the best-effort label. Non-CDS / non-named GFF rows
    (``gene``, ``region``, ``gap``, …) are not represented — they neither occupy nor are indexed.
    """

    seqid: str
    start: int
    end: int
    ftype: str
    name: str

    @property
    def is_cds(self) -> bool:
        """True for the sole occupying type (``CDS``)."""
        return self.ftype == OCCUPYING_TYPE

    @property
    def igr_type(self) -> str:
        """Lower-cased type label used to name the IGR bucket for a named non-CDS feature."""
        return self.ftype.lower()


def _feature_label(attrs: dict[str, str]) -> str:
    """Best-effort feature name: ``gene`` → ``Name`` → ``product`` → ``locus_tag``/``ID``."""
    return (
        attrs.get("gene") or attrs.get("Name") or attrs.get("product")
        or attrs.get("locus_tag") or attrs.get("ID") or ""
    )


def parse_gff_features(gff_path: str | Path) -> dict[str, list[Feature]]:
    """Parse a GFF3 into per-contig :class:`Feature` lists, keeping CDS + named non-CDS types only.

    Stops at the ``##FASTA`` pragma (Bakta appends the genome FASTA to its GFF3). Returns a dict
    ``{seqid: [Feature, …]}`` in file order; only ``CDS`` and the :data:`~genome_prep.features.FEATURE_TYPES`
    are retained (``gene``/``region``/``gap``/misc are dropped — they neither occupy nor are named).
    """
    out: dict[str, list[Feature]] = {}
    with open_text(gff_path) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                if line.startswith("##FASTA"):
                    break
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            ftype, seqid = parts[2], parts[0]
            try:
                start, end = int(parts[3]), int(parts[4])
            except ValueError:
                continue
            if ftype == OCCUPYING_TYPE:
                out.setdefault(seqid, []).append(Feature(seqid, start, end, ftype, ""))
            elif ftype.lower() in FEATURE_TYPES:
                label = _feature_label(parse_gff_attributes(parts[8]))
                out.setdefault(seqid, []).append(Feature(seqid, start, end, ftype, label))
    return out


def extract_intergenic(
    gff_path: str | Path,
    fna_path: str | Path,
    *,
    min_len: int = 30,
) -> dict[str, Any]:
    """Extract the three non-coding views for one genome from a GFF + FASTA pair.

    **Only ``CDS`` occupies.** The three parallel-list views:

    * ``noncoding_*`` — each **maximal CDS-to-CDS run** (label-agnostic; RNA/CRISPR inside kept).
    * ``fragment_*`` — each run **split at every named non-CDS feature boundary** (promoter fragments).
    * ``feature_*`` — each **named non-CDS body** embedded standalone with ``feature_type``/``feature_name``.

    Parameters
    ----------
    gff_path, fna_path : str or Path
        A GFF3 (plain/gzipped) and the genome FASTA whose contig IDs match its ``seqid`` column.
    min_len : int, default 30
        Minimum length (bp) for a ``noncoding`` run and a ``fragment``. Named ``feature`` bodies are
        indexed regardless (a tRNA is ~76 bp but always wanted).

    Returns
    -------
    dict
        ``noncoding_*`` / ``fragment_*`` / ``feature_*`` parallel-list columns. Coords 1-based
        inclusive forward; sequences lowercased; contig-then-position order.
    """
    gff_path, fna_path = Path(gff_path), Path(fna_path)
    contigs = load_fna(fna_path)
    if not contigs:
        raise ValueError(f"No contigs parsed from FASTA: {fna_path}")
    feats_by_contig = parse_gff_features(gff_path)

    nc_seqs: list[str] = []
    nc_seqids: list[str] = []
    nc_starts: list[int] = []
    nc_ends: list[int] = []
    fr_seqs: list[str] = []
    fr_seqids: list[str] = []
    fr_starts: list[int] = []
    fr_ends: list[int] = []
    for seqid, contig_seq in contigs.items():
        clen = len(contig_seq)
        flist = feats_by_contig.get(seqid, [])
        occupied = [(f.start - 1, f.end) for f in flist if f.is_cds]  # 1-based incl -> 0-based half-open
        feat_cuts = [(f.start - 1, f.end) for f in flist if not f.is_cds]
        for g0, g1 in complement(occupied, clen):
            if g1 - g0 >= min_len:  # whole_igr: the maximal CDS-to-CDS run (label-agnostic)
                nc_seqs.append(str(contig_seq[g0:g1]).lower())
                nc_seqids.append(seqid)
                nc_starts.append(g0 + 1)  # back to 1-based inclusive
                nc_ends.append(g1)
            for f0, f1 in subtract((g0, g1), feat_cuts):  # per_unit: run split at named-feature bounds
                if f1 - f0 < min_len:
                    continue
                fr_seqs.append(str(contig_seq[f0:f1]).lower())
                fr_seqids.append(seqid)
                fr_starts.append(f0 + 1)
                fr_ends.append(f1)

    feat_seqs: list[str] = []
    feat_names: list[str] = []
    feat_types: list[str] = []
    feat_seqids: list[str] = []
    feat_starts: list[int] = []
    feat_ends: list[int] = []
    for seqid, contig_seq in contigs.items():
        named = sorted(
            (f.start, f.end, f.name, f.igr_type) for f in feats_by_contig.get(seqid, []) if not f.is_cds
        )
        for start1, end1, label, ftype in named:
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
        "noncoding_sequence": nc_seqs, "noncoding_seqid": nc_seqids,
        "noncoding_start": nc_starts, "noncoding_end": nc_ends,
        "fragment_sequence": fr_seqs, "fragment_seqid": fr_seqids,
        "fragment_start": fr_starts, "fragment_end": fr_ends,
        "feature_sequence": feat_seqs, "feature_name": feat_names, "feature_type": feat_types,
        "feature_seqid": feat_seqids, "feature_start": feat_starts, "feature_end": feat_ends,
    }


class CodingIndex:
    """Fast per-genome ``(contig, start, end) → (coding_class, igr_type)`` classifier.

    Classification priority: overlaps any **CDS** → ``("CDS", None)``; else overlaps a **named**
    non-CDS feature → ``("IGR", <type>)``; else (a gap between annotated features) →
    ``("IGR", "unclassified")`` — the promoter-candidate bucket. All coordinates 1-based inclusive.

    CDS is stored as **merged, non-overlapping** intervals (0-based half-open internally), so both the
    boolean overlap test and the exact **base-pair overlap** (:meth:`cds_overlap_bp` — how much of a
    span is coding, the input to the IGR-coverage analysis) are O(log n) bisect lookups. The
    named-feature scan is O(#named), tiny per genome. Build once per genome, classify many spans.
    """

    def __init__(self, per_contig: dict[str, tuple[list[tuple[int, int]], list[int], list[tuple[int, int, str]]]]):
        self._c = per_contig

    @classmethod
    def from_gff(cls, gff_path: str | Path) -> CodingIndex:
        """Build the index from a GFF3 (via :func:`parse_gff_features`)."""
        per_contig: dict[str, tuple[list[tuple[int, int]], list[int], list[tuple[int, int, str]]]] = {}
        for seqid, flist in parse_gff_features(gff_path).items():
            # 0-based half-open, merged so overlapping CDS (e.g. opposite strands) are not double-counted.
            cds = merge_intervals([(f.start - 1, f.end) for f in flist if f.is_cds])
            named = sorted((f.start - 1, f.end, f.igr_type) for f in flist if not f.is_cds)
            per_contig[seqid] = (cds, [s for s, _e in cds], named)
        return cls(per_contig)

    def classify_span(self, contig: str, start: int, end: int) -> tuple[str, str | None]:
        """Classify a 1-based inclusive span into ``(coding_class, igr_type)`` (CDS wins; see class docstring)."""
        entry = self._c.get(contig)
        if entry is None:  # contig has no CDS and no named features → wholly unclassified IGR
            return (IGR_CLASS, UNCLASSIFIED_IGR)
        cds, cds_starts, named = entry
        s0, e0 = start - 1, end  # 0-based half-open
        k = bisect.bisect_right(cds_starts, e0 - 1)  # merged+sorted → only the last start < e0 can overlap
        if k > 0 and cds[k - 1][1] > s0:
            return (CDS_CLASS, None)
        for fs, fe, ft in named:  # named sorted by start; scan while start < e0
            if fs >= e0:
                break
            if fe > s0:
                return (IGR_CLASS, ft)
        return (IGR_CLASS, UNCLASSIFIED_IGR)

    def cds_overlap_bp(self, contig: str, start: int, end: int) -> int:
        """Base pairs of the 1-based inclusive span ``[start, end]`` that lie within a CDS.

        ``igr_bp = (end - start + 1) - cds_overlap_bp`` is the intergenic coverage the IGR analysis
        weighs. Sums intersections with the merged CDS intervals (no double counting), including a
        span that straddles several genes.
        """
        entry = self._c.get(contig)
        if entry is None:
            return 0
        cds, cds_starts, _named = entry
        s0, e0 = start - 1, end
        total = 0
        j = bisect.bisect_right(cds_starts, e0 - 1) - 1  # last interval starting before e0
        while j >= 0 and cds[j][1] > s0:  # merged+sorted → walk back while ends still reach the span
            is_, ie = cds[j]
            total += min(e0, ie) - max(s0, is_)
            j -= 1
        return total


def contig_lengths(gff_path: str | Path, fna_path: str | Path | None = None) -> dict[str, int]:
    """Contig lengths from the FASTA if given, else from GFF ``##sequence-region`` pragmas."""
    if fna_path is not None:
        return {seqid: len(seq) for seqid, seq in load_fna(fna_path).items()}
    lengths: dict[str, int] = {}
    with open_text(gff_path) as handle:
        for line in handle:
            if line.startswith("##FASTA"):
                break
            if line.startswith("##sequence-region"):
                parts = line.split()
                if len(parts) >= 4:
                    lengths[parts[1]] = int(parts[3]) - int(parts[2]) + 1
            elif not line.startswith("#"):
                break  # pragmas precede feature lines; stop once features start
    if not lengths:
        raise ValueError(
            f"No contig lengths: {gff_path} has no ##sequence-region pragmas — pass fna_path."
        )
    return lengths


def coding_fraction(gff_path: str | Path, fna_path: str | Path | None = None) -> dict[str, Any]:
    """Genome-wide CDS vs IGR base-pair breakdown — the enrichment denominator.

    Returns ``total_bp``, ``cds_bp``, ``igr_bp``, ``named_igr_bp`` (IGR covered by a named non-CDS
    feature), ``unclassified_igr_bp`` (the rest — promoter candidates), ``per_type_bp`` (named IGR bp
    per feature type, clipped to non-CDS space), and ``n_contigs``. Contig lengths come from the FASTA
    if ``fna_path`` is given, else from the GFF ``##sequence-region`` pragmas.

    ``per_type_bp`` merges each type's intervals then clips to non-CDS space; a base covered by two
    named types (rare) counts toward both, so ``sum(per_type_bp)`` may marginally exceed
    ``named_igr_bp`` — ``named_igr_bp``/``unclassified_igr_bp`` (from the union) are the exact split.
    """
    feats = parse_gff_features(gff_path)
    lengths = contig_lengths(gff_path, fna_path)
    total_bp = sum(lengths.values())
    cds_bp = 0
    named_igr_bp = 0
    per_type_bp: dict[str, int] = {}
    for seqid, clen in lengths.items():
        flist = feats.get(seqid, [])
        cds = merge_intervals([(f.start - 1, f.end) for f in flist if f.is_cds])
        cds_bp += sum(e - s for s, e in cds)
        igr_runs = complement(cds, clen)  # non-CDS space (0-based half-open)
        by_type: dict[str, list[tuple[int, int]]] = {}
        for f in flist:
            if not f.is_cds:
                by_type.setdefault(f.igr_type, []).append((f.start - 1, f.end))
        # union of all named, clipped to IGR → named_igr_bp
        all_named = merge_intervals([iv for ivs in by_type.values() for iv in ivs])
        for run in igr_runs:
            # bp of `all_named` inside this run = run length minus the gaps left after subtracting named
            named_igr_bp += (run[1] - run[0]) - sum(e - s for s, e in subtract(run, all_named))
        for ftype, ivs in by_type.items():
            merged = merge_intervals(list(ivs))
            clipped = sum(
                (r[1] - r[0]) - sum(e - s for s, e in subtract(r, merged)) for r in igr_runs
            )
            per_type_bp[ftype] = per_type_bp.get(ftype, 0) + clipped
    igr_bp = total_bp - cds_bp
    return {
        "total_bp": total_bp,
        "cds_bp": cds_bp,
        "igr_bp": igr_bp,
        "named_igr_bp": named_igr_bp,
        "unclassified_igr_bp": igr_bp - named_igr_bp,
        "per_type_bp": per_type_bp,
        "n_contigs": len(lengths),
    }

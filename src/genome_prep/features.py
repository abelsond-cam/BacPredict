"""The single CDS / non-coding (IGR) feature taxonomy + interval math.

The coding-vs-non-coding split used everywhere in the codebase is: **only ``CDS`` occupies** sequence.
Everything else (RNA genes, CRISPR, regulatory_region, oriC, ``region``, ``gap``, misc) is non-coding
(IGR). A curated subset of non-CDS types is *named* — indexed and reported by type — while the gaps
between all named/coding features are the *unclassified* IGR (where promoters, which Bakta does not
annotate, live).

Interval math (``merge_intervals`` / ``complement`` / ``subtract``) operates on **0-based half-open**
intervals; callers convert to/from GFF's 1-based inclusive convention at the boundary.
"""

from __future__ import annotations

# The only feature type that occupies sequence for the coding/non-coding split.
OCCUPYING_TYPE = "CDS"

# Named non-CDS feature types (compared against a lower-cased GFF ``type`` column): their bodies are
# indexed + reported standalone, AND they fragment the non-coding runs. RNA + CRISPR (the whole array,
# not crispr-repeat/spacer sub-features) + Bakta's explicit regulatory_region / oriC.
FEATURE_TYPES = frozenset(
    {
        "rrna", "trna", "tmrna", "ncrna", "ncrna_gene", "antisense_rna", "rnase_p_rna", "srp_rna",
        "riboswitch", "crispr", "regulatory_region", "oric", "origin_of_replication",
    }
)

# The label for IGR that overlaps no named non-CDS feature (Bakta leaves promoters here).
UNCLASSIFIED_IGR = "unclassified"


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping/adjacent 0-based half-open intervals; returns them sorted ascending."""
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


def complement(occupied: list[tuple[int, int]], clen: int) -> list[tuple[int, int]]:
    """The gaps (0-based half-open) between merged occupied intervals over ``[0, clen)``."""
    merged = merge_intervals(occupied)
    gaps: list[tuple[int, int]] = []
    prev_end = 0
    for s, e in merged:
        if s > prev_end:
            gaps.append((prev_end, s))
        prev_end = max(prev_end, e)
    if prev_end < clen:
        gaps.append((prev_end, clen))
    return gaps


def subtract(run: tuple[int, int], cuts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sub-intervals of ``run`` (0-based half-open) left after removing ``cuts`` (clipped to the run)."""
    r0, r1 = run
    clipped = merge_intervals([(max(s, r0), min(e, r1)) for s, e in cuts if e > r0 and s < r1])
    frags: list[tuple[int, int]] = []
    prev = r0
    for cs, ce in clipped:
        if cs > prev:
            frags.append((prev, cs))
        prev = max(prev, ce)
    if prev < r1:
        frags.append((prev, r1))
    return frags

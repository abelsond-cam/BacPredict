"""Convert a GGCAT coloured de Bruijn graph build into a pyseer ``--kmers`` unitig matrix.

GGCAT (memory-capped, disk-based) replaces unitig-caller's Bifrost backend, which OOMs at the
~18k-genome union on a 502 GB himem node. A coloured ``ggcat build -c`` over the cohort
assemblies emits three artifacts this script joins into the line-oriented matrix pyseer reads
with ``--kmers``:

1. **unitig FASTA** (``ggcat build -c -o ….fa.gz``) — each record header is
   ``>{id} LN:i:{len} C:{subset_id}:{n_kmers}``; ``subset_id`` identifies the *colour subset*
   (the set of samples whose genomes contain that unitig), ``n_kmers`` = ``len - k + 1`` (ignored).
2. **colour names** (``ggcat dump-colors ….colors.dat names.jsonl``) — one JSON object per line,
   ``{"color_index": i, "color_name": "<SampleID>"}``. Because we build with
   ``-d`` (colored-input-lists, ``Sample<TAB>path``), the colour *name is the Sample ID directly*.
3. **colormap** (``ggcat dump-colormap --format ranges-csv ….colors.dat colormap.csv <ids…>``) —
   ``{subset_id},{tok},{tok},…`` where each token is a single colour index ``a`` or an inclusive
   range ``a-b``. Maps a subset id to its set of colour indices.

The pyseer ``--kmers`` line format (matching unitig-caller ``--pyseer``) is::

    <unitig_sequence> | <SampleA>:1 <SampleB>:1 …

so the join is: FASTA unitig → its ``subset_id`` → colour indices → Sample IDs → emit one line.
Pure stdlib (``gzip``/``json``) and fully streaming over the FASTA, so the only resident state is
the colour-name table (~18k entries) and a ``subset_id -> " ".join(SampleID:1)`` cache (one entry
per *distinct* subset, reused across every unitig sharing it). Output is gzipped for ``--kmers``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path


def _open_text(path: Path):
    """Open a plain or gzipped text file for reading, transparently by suffix."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


def load_color_names(path: Path) -> list[str]:
    """Read ``dump-colors`` JSONL into a dense ``color_index -> Sample ID`` list.

    Parameters
    ----------
    path
        ``color_names.jsonl`` from ``ggcat dump-colors`` — one ``{"color_index", "color_name"}``
        object per line.

    Returns
    -------
    list of str
        ``names[i]`` is the Sample ID for colour index ``i`` (indices are dense, ``0..C-1``).
    """
    pairs: list[tuple[int, str]] = []
    with _open_text(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pairs.append((int(obj["color_index"]), str(obj["color_name"])))
    if not pairs:
        raise ValueError(f"no colours parsed from {path}")
    n = max(i for i, _ in pairs) + 1
    names: list[str | None] = [None] * n
    for i, name in pairs:
        names[i] = name
    missing = [i for i, v in enumerate(names) if v is None]
    if missing:
        raise ValueError(f"colour indices with no name (gap in {path}): {missing[:10]}…")
    return names  # type: ignore[return-value]


def _expand_range_tokens(tokens: list[str]) -> list[int]:
    """Expand ranges-csv colour tokens (``a`` or ``a-b``) into a sorted list of colour indices."""
    out: list[int] = []
    for tok in tokens:
        if not tok:
            continue
        if "-" in tok:
            lo, hi = tok.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(tok))
    return out


def load_colormap(path: Path, names: list[str]) -> dict[int, str]:
    """Read ranges-csv colormap into ``subset_id -> "SampleA:1 SampleB:1 …"`` (pre-joined).

    Pre-joining once per *distinct* subset (rather than per unitig) keeps the hot loop a single
    dict lookup + f-string. The resident cost is one short string per subset.

    Parameters
    ----------
    path
        ``colormap_ranges.csv`` from ``ggcat dump-colormap --format ranges-csv``.
    names
        ``color_index -> Sample ID`` table from :func:`load_color_names`.

    Returns
    -------
    dict
        ``subset_id -> "<SampleID>:1 …"`` — the pyseer sample-presence field for that subset.
    """
    cmap: dict[int, str] = {}
    with _open_text(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            fields = line.split(",")
            subset_id = int(fields[0])
            color_ids = _expand_range_tokens(fields[1:])
            cmap[subset_id] = " ".join(f"{names[c]}:1" for c in color_ids)
    if not cmap:
        raise ValueError(f"no subsets parsed from {path}")
    return cmap


def _iter_fasta(path: Path):
    """Yield ``(header, sequence)`` for each record in a (optionally gzipped) FASTA, streaming."""
    header: str | None = None
    seq_parts: list[str] = []
    with _open_text(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts)
                header = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line.strip())
        if header is not None:
            yield header, "".join(seq_parts)


def _subset_id_from_header(header: str) -> int:
    """Parse the ``C:<subset_id>:<n_kmers>`` field out of a GGCAT unitig header."""
    for tok in header.split():
        if tok.startswith("C:"):
            return int(tok.split(":")[1])
    raise ValueError(f"no C:<subset>:<n> field in GGCAT header: {header!r}")


def convert(
    fasta: Path, color_names: Path, colormap: Path, out: Path, min_samples: int = 0,
) -> tuple[int, int]:
    """Join the three GGCAT artifacts into a gzipped pyseer ``--kmers`` matrix.

    Parameters
    ----------
    fasta, color_names, colormap
        The three GGCAT build outputs (see module docstring).
    out
        Output path; gzipped if it ends with ``.gz`` (pyseer reads gzipped ``--kmers``).
    min_samples
        Optional pre-filter — drop unitigs present in fewer than this many samples. Default 0
        (emit everything; let pyseer's ``--min-af`` do the frequency filtering).

    Returns
    -------
    (n_written, n_unitigs)
        Unitigs written and unitigs seen (differ only when ``min_samples`` drops some).
    """
    names = load_color_names(color_names)
    cmap = load_colormap(colormap, names)
    print(f"  colours: {len(names)} samples; colour subsets: {len(cmap)}", file=sys.stderr)

    # precompute per-subset sample counts only if we are filtering (count of ':' presence tokens)
    counts = {sid: (s.count(":1") if s else 0) for sid, s in cmap.items()} if min_samples else {}

    n_written = n_unitigs = 0
    opener = gzip.open(out, "wt") if str(out).endswith(".gz") else open(out, "w")
    with opener as ofh:
        for header, seq in _iter_fasta(fasta):
            n_unitigs += 1
            sid = _subset_id_from_header(header)
            if min_samples and counts.get(sid, 0) < min_samples:
                continue
            presence = cmap.get(sid)
            if not presence:  # subset id missing from colormap → cannot place; skip loudly later
                raise KeyError(f"unitig {header!r} references subset {sid} absent from {colormap}")
            ofh.write(f"{seq} | {presence}\n")
            n_written += 1
    return n_written, n_unitigs


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fasta", type=Path, required=True, help="GGCAT unitig FASTA (….fa.gz).")
    p.add_argument("--color-names", type=Path, required=True, help="ggcat dump-colors JSONL.")
    p.add_argument("--colormap", type=Path, required=True, help="ggcat dump-colormap ranges-csv.")
    p.add_argument("--out", type=Path, required=True, help="Output pyseer --kmers matrix (.gz).")
    p.add_argument("--min-samples", type=int, default=0,
                   help="Drop unitigs in fewer than N samples (default 0 — let pyseer --min-af filter).")
    args = p.parse_args(argv)
    n_written, n_unitigs = convert(
        args.fasta, args.color_names, args.colormap, args.out, min_samples=args.min_samples,
    )
    print(f"wrote {n_written}/{n_unitigs} unitigs -> {args.out}")


if __name__ == "__main__":
    main()

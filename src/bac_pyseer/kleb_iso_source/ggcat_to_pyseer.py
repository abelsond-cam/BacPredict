"""Convert a GGCAT coloured de Bruijn graph build into a pyseer ``--kmers`` unitig matrix.

GGCAT (memory-capped, disk-based) replaces unitig-caller's Bifrost backend, which OOMs at the
~18k-genome union on a 502 GB himem node. A coloured ``ggcat build -c`` over the cohort
assemblies emits three artifacts this script joins into the line-oriented matrix pyseer reads
with ``--kmers``:

1. **unitig FASTA** (``ggcat build -c -o ….fa.gz``) — each record header carries one or more
   ``C:{subset_id_HEX}:{n_kmers_DEC}`` segments, e.g. ``>4 LN:i:36 C:22f:2 C:333:3 C:22f:1``.
   ``subset_id`` (hexadecimal) identifies a *colour subset* (the set of samples sharing those
   k-mers); ``n_kmers`` (decimal) is that segment's k-mer count. GGCAT unitigs are *sequence*-
   maximal, so the colour can change along one unitig (~23% of ours have >1 segment). Each segment
   is therefore emitted as its own monochromatic feature — the contiguous sub-sequence
   ``seq[kmer_off : kmer_off + n_kmers + k - 1]`` present in exactly that subset's samples — which
   is the colour-consistent unitig definition the pyseer/DBGWAS workflow expects.
2. **colour names** (``ggcat dump-colors ….colors.dat names.jsonl``) — one JSON object per line,
   ``{"color_index": i, "color_name": "<SampleID>"}``. Because we build with
   ``-d`` (colored-input-lists, ``Sample<TAB>path``), the colour *name is the Sample ID directly*.
3. **colormap** (``ggcat dump-colormap --format ranges-csv ….colors.dat colormap.csv <ids…>``) —
   ``{subset_id_DEC},{tok},{tok},…`` where each token is a single colour index ``a`` or an inclusive
   range ``a-b``. Maps a (decimal) subset id to its set of colour indices. NB the FASTA writes
   subset ids in hex but the colormap keys them in decimal — they reconcile as the same integer.

The pyseer ``--kmers`` line format (matching unitig-caller ``--pyseer``) is::

    <unitig_sequence> | <SampleA>:1 <SampleB>:1 …

so the join is: FASTA unitig segment → its ``subset_id`` → colour indices → Sample IDs → one line.

A real cohort has millions of distinct colour subsets, many spanning thousands of samples, so
expanding them all into resident strings OOMs (a near-core subset is ~10 chars compact but hundreds
of KB expanded). Instead the join is done **disk-based, expanding a small amount at a time**: stream
the FASTA to a ``subset_id<TAB>subseq`` temp file, external-sort it by ``subset_id`` (GNU ``sort``,
spills to scratch), then merge against the subset-id-ordered colormap — each subset's ranges expand
to Sample IDs exactly once and only the *current* subset's presence string is ever resident, so RAM
stays ~constant (sort buffer + one subset) regardless of cohort size. Output is gzipped for ``--kmers``.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import shutil
import subprocess
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


def _count_range_tokens(tokens: list[str]) -> int:
    """Count colours encoded by ranges-csv tokens (``a`` or ``a-b``) without materialising them."""
    n = 0
    for tok in tokens:
        if not tok:
            continue
        if "-" in tok:
            lo, hi = tok.split("-", 1)
            n += int(hi) - int(lo) + 1
        else:
            n += 1
    return n


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


def _parse_color_segments(header: str) -> list[tuple[int, int]]:
    """Parse all ``C:<subset_hex>:<n_kmers_dec>`` segments from a GGCAT unitig header.

    GGCAT writes subset ids in hexadecimal and k-mer counts in decimal, and a single
    sequence-maximal unitig may carry several colour segments. Returns ``[(subset_int, n_kmers), …]``
    in 5'->3' order.
    """
    segs: list[tuple[int, int]] = []
    for tok in header.split():
        if tok.startswith("C:"):
            _, subset_hex, n_kmers = tok.split(":")
            segs.append((int(subset_hex, 16), int(n_kmers)))
    if not segs:
        raise ValueError(f"no C:<subset>:<n_kmers> segment in GGCAT header: {header!r}")
    return segs


def _fasta_to_segment_file(fasta: Path, seg_path: Path, k: int) -> tuple[int, int]:
    """Stream the FASTA → ``subset_id<TAB>subseq`` lines (one per colour segment) for sorting.

    Each colour segment of a unitig is the monochromatic sub-sequence
    ``seq[kmer_off : kmer_off + n_kmers + k - 1]`` (segments after the first overlap the previous by
    ``k-1`` bases — standard for colour-split de Bruijn unitigs). Returns ``(n_unitigs, n_segments)``.
    """
    n_unitigs = n_segments = 0
    with open(seg_path, "w") as sf:
        for header, seq in _iter_fasta(fasta):
            n_unitigs += 1
            kmer_off = 0
            for subset_id, n_kmers in _parse_color_segments(header):
                n_segments += 1
                sf.write(f"{subset_id}\t{seq[kmer_off:kmer_off + n_kmers + k - 1]}\n")
                kmer_off += n_kmers
    return n_unitigs, n_segments


def _open_matrix_writer(out: Path, threads: int):
    """Open a text writer for the matrix; close via the returned ``close_fn``.

    For ``.gz`` output, compress with multicore ``pigz`` (the matrix is large and single-thread
    gzip is the convert bottleneck), falling back to fast-level gzip if pigz is unavailable.
    """
    if str(out).endswith(".gz") and shutil.which("pigz"):
        raw = open(out, "wb")
        pz = subprocess.Popen(["pigz", "-p", str(threads), "-c"], stdin=subprocess.PIPE, stdout=raw)
        tw = io.TextIOWrapper(pz.stdin, encoding="utf-8")

        def _close() -> None:
            tw.flush()
            tw.close()  # closes pigz stdin → pigz drains and exits
            rc = pz.wait()
            raw.close()
            if rc:
                raise RuntimeError(f"pigz exited with code {rc}")

        return tw, _close
    if str(out).endswith(".gz"):
        fh = gzip.open(out, "wt", compresslevel=6)
        return fh, fh.close
    fh = open(out, "w")
    return fh, fh.close


def convert(
    fasta: Path, color_names: Path, colormap: Path, out: Path, kmer_length: int = 31,
    min_samples: int = 0, max_samples: int | None = None, tmp_dir: Path | None = None,
    sort_buffer: str = "2G", threads: int = 4,
) -> tuple[int, int, int]:
    """Disk-based sort-merge join of the GGCAT artifacts → a gzipped pyseer ``--kmers`` matrix.

    RAM stays ~constant (sort buffer + one subset's expanded sample list) regardless of cohort size
    — the bulk lives on disk (see module docstring):

    1. stream the FASTA to a ``subset_id<TAB>subseq`` temp file (one line per colour segment);
    2. external-sort it by ``subset_id`` (GNU ``sort``, spills to ``tmp_dir``);
    3. merge the sorted segments against the subset-id-ordered colormap, expanding each subset's
       ranges to Sample IDs exactly once (consecutive segments share it) and filtering by
       ``min_samples`` / ``max_samples`` (subsets outside the testable band are skipped, never
       expanded — which is also what keeps the output small, since a near-core subset expands to a
       presence string hundreds of KB long).

    Parameters
    ----------
    fasta, color_names, colormap
        The three GGCAT build outputs (see module docstring).
    out
        Output path; gzipped if it ends with ``.gz`` (pyseer reads gzipped ``--kmers``).
    kmer_length
        k used for the GGCAT build (segment lengths are ``n_kmers + k - 1`` bases).
    min_samples
        Drop segments whose colour subset spans fewer than this many samples (untestable below the
        pyseer MAF floor). Default 0 (emit everything; let pyseer's ``--min-af`` filter).
    max_samples
        Drop segments whose colour subset spans *more* than this many samples — the mirror of
        ``min_samples`` at the ``--max-af`` end. pyseer cannot test a near-universal unitig either,
        yet those are precisely the rows with the longest presence strings, so they dominate matrix
        size and every subsequent streaming pass over it. Matters most for a low-diversity cohort
        (e.g. TB), where the core genome is carried by nearly every sample. Default ``None``
        (no upper bound, preserving the previous behaviour).
    tmp_dir
        Directory for the temp segment file + external-sort spill (default: ``out``'s parent).
    sort_buffer
        GNU ``sort -S`` memory buffer (the dominant resident cost). Default ``2G``.

    Returns
    -------
    (n_written, n_unitigs, n_segments)
        Lines written, unitigs seen, and total colour segments (``n_written`` < ``n_segments`` when
        ``min_samples``/``max_samples`` drop some; the per-bound drop counts go to stderr).
    """
    if max_samples is not None and min_samples and max_samples < min_samples:
        raise ValueError(f"max_samples ({max_samples}) < min_samples ({min_samples}) — no unitig can pass")
    names = load_color_names(color_names)
    k = kmer_length
    tmp = Path(tmp_dir) if tmp_dir is not None else out.parent
    tmp.mkdir(parents=True, exist_ok=True)
    seg_path = tmp / f"{out.name}.segments.tsv"
    sorted_path = tmp / f"{out.name}.segments.sorted.tsv"

    n_unitigs, n_segments = _fasta_to_segment_file(fasta, seg_path, k)
    print(f"  colours: {len(names)} samples; k={k}; {n_unitigs} unitigs, {n_segments} segments "
          f"-> external sort by subset", file=sys.stderr)
    subprocess.run(
        ["sort", "-t", "\t", "-k1,1n", "-S", sort_buffer, "-T", str(tmp), "-o", str(sorted_path), str(seg_path)],
        check=True, env={**os.environ, "LC_ALL": "C"},
    )
    seg_path.unlink()

    # merge the sorted segments (ascending subset_id) with the subset-id-ordered colormap; expand
    # each subset's ranges to Sample IDs once and reuse for its consecutive segments.
    n_written = 0
    ofh, close_out = _open_matrix_writer(out, threads)
    with open(sorted_path) as segs, _open_text(colormap) as cm:
        cm_id = -1

        def _advance(target: int) -> str | None:
            """Read the colormap forward to ``target``; return its ranges string (or None if absent)."""
            nonlocal cm_id
            for line in cm:
                line = line.rstrip("\n")
                if not line:
                    continue
                cid, _, ranges = line.partition(",")
                cm_id = int(cid)
                if cm_id >= target:
                    return ranges if cm_id == target else None
            return None

        cur_sub: int | None = None
        cur_presence: str | None = None
        n_drop_rare = n_drop_common = 0
        for segline in segs:
            sub_s, _, subseq = segline.rstrip("\n").partition("\t")
            sub = int(sub_s)
            if sub != cur_sub:
                cur_sub = sub
                if cm_id > sub:  # both ascending & segments ⊆ colormap → must not overshoot
                    raise KeyError(f"colormap overshot subset {sub} (now at {cm_id}); ordering broken")
                ranges = _advance(sub)
                if ranges is None:
                    raise KeyError(f"subset {sub} absent from colormap {colormap}")
                toks = ranges.split(",")
                # Count first, expand only if testable — an out-of-band subset is never
                # materialised, which is the whole point of the bounds.
                n_carriers = _count_range_tokens(toks)
                if min_samples and n_carriers < min_samples:
                    cur_presence, n_drop_rare = None, n_drop_rare + 1
                elif max_samples is not None and n_carriers > max_samples:
                    cur_presence, n_drop_common = None, n_drop_common + 1
                else:
                    cur_presence = " ".join(f"{names[c]}:1" for c in _expand_range_tokens(toks))
            if cur_presence is None:
                continue
            ofh.write(f"{subseq} | {cur_presence}\n")
            n_written += 1
    close_out()
    sorted_path.unlink()
    print(f"  colour subsets dropped: {n_drop_rare} below min_samples={min_samples}, "
          f"{n_drop_common} above max_samples={max_samples}", file=sys.stderr)
    return n_written, n_unitigs, n_segments


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fasta", type=Path, required=True, help="GGCAT unitig FASTA (….fa.gz).")
    p.add_argument("--color-names", type=Path, required=True, help="ggcat dump-colors JSONL.")
    p.add_argument("--colormap", type=Path, required=True, help="ggcat dump-colormap ranges-csv.")
    p.add_argument("--out", type=Path, required=True, help="Output pyseer --kmers matrix (.gz).")
    p.add_argument("--kmer-length", type=int, default=31, help="k used for the GGCAT build (default 31).")
    p.add_argument("--min-samples", type=int, default=0,
                   help="Drop segments in fewer than N samples (default 0 — let pyseer --min-af filter).")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Drop segments in MORE than N samples (default: no cap). pyseer cannot test a "
                        "near-universal unitig, and those rows have the longest presence strings, so "
                        "capping at the --max-af equivalent is the main lever on matrix size.")
    p.add_argument("--tmp-dir", type=Path, default=None,
                   help="Dir for the temp segment file + external-sort spill (default: --out's parent).")
    p.add_argument("--sort-buffer", default="2G", help="GNU sort -S memory buffer (default 2G).")
    p.add_argument("--threads", type=int, default=4, help="pigz compression threads for .gz output (default 4).")
    args = p.parse_args(argv)
    n_written, n_unitigs, n_segments = convert(
        args.fasta, args.color_names, args.colormap, args.out,
        kmer_length=args.kmer_length, min_samples=args.min_samples, max_samples=args.max_samples,
        tmp_dir=args.tmp_dir, sort_buffer=args.sort_buffer, threads=args.threads,
    )
    print(f"wrote {n_written} features from {n_unitigs} unitigs ({n_segments} colour segments) -> {args.out}")


if __name__ == "__main__":
    main()

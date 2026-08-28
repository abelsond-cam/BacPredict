r"""Score unitig presence in a genome by **k-mer containment** — the rule GGCAT colouring encodes.

Why this exists
---------------
A train+validate-only unitig vocabulary has no holdout rows, so the holdout must be scored from
sequence. Two rules are available and they are *not* the same operator:

- **(A) k-mer containment** — every canonical ``k``-mer of the feature occurs somewhere in the genome.
- **(B) exact substring** — the feature occurs contiguously on one contig
  (:mod:`bac_pyseer.kleb_iso_source.unitig_placement`'s Aho-Corasick scan).

(B) implies (A) but not conversely: a feature whose k-mers are all present yet scattered — across a
contig break, or via a repeat — satisfies (A) and fails (B). Contig count tracks assembly quality,
which tracks lineage, which tracks resistance, so scoring train rows by one rule and holdout rows by
the other would inject an assembly-fragmentation confound into the test set alone.

**Rule (A) is the rule that produced the existing matrix.** ``ggcat_to_pyseer`` emits each feature as
``seq[kmer_off : kmer_off + n_kmers + k - 1]`` carrying one colour subset — by construction the set of
samples holding *every* k-mer of that span. So this module reproduces GGCAT's own definition, which
makes it self-verifying: run it over train+validate and it must reproduce ``hits_submatrix.tsv``
exactly, which proves the holdout rows were computed by the same rule as the training rows.

Method
------
Canonical 2-bit encoding, ``k <= 31`` (62 bits in a ``uint64``). The genome becomes one sorted unique
``uint64`` array (~44 MB for a 5.5 Mb *Klebsiella* assembly); every feature's k-mers are looked up with
``np.searchsorted`` and reduced per feature. Windows containing a non-ACGT base are dropped, matching
GGCAT, which never emits a k-mer spanning an ambiguity code.

The ``compare`` phase measures (A) against (B) against GGCAT's colouring on the same genomes, which is
what decides the scoring rule::

    python -m bac_pyseer.ast_gwas.unitig_kmer_presence compare \
        --design-dir .../pyseer_ast/kp/ertapenem/design \
        --reflist    .../pyseer_ast/kp/unitigs/assembly_refs.txt \
        --n-genomes  40 \
        --out        .../pyseer_ast/kp/ertapenem/design/rule_discordance.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:  # package import when bac_pyseer is on the path (editable install)
    from bac_pyseer.kleb_iso_source.unitig_placement import _load_contigs
except ImportError:  # invoked as a bare script
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kleb_iso_source"))
    from unitig_placement import _load_contigs  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

K = 31  # ggcat build -k 31; 2 bits x 31 = 62 bits, so a canonical k-mer fits a uint64

# base -> 2-bit code, everything else -> 255 (a window containing one is not a k-mer)
_CODE = np.full(256, 255, dtype=np.uint8)
for _b, _v in zip(b"ACGT", range(4), strict=True):
    _CODE[_b] = _v
    _CODE[_b + 32] = _v  # lowercase


def _codes(seq: str) -> np.ndarray:
    """DNA string -> ``uint8`` 2-bit codes, 255 for any non-ACGT base."""
    return _CODE[np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)]


def _roll(v: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling k-mer values over a code array -> ``(values, valid)``, both length ``len(v) - k + 1``."""
    if v.size < k:
        return np.empty(0, dtype=np.uint64), np.empty(0, dtype=bool)
    w = np.lib.stride_tricks.sliding_window_view(v, k)
    valid = (w != 255).all(axis=1)
    val = np.zeros(w.shape[0], dtype=np.uint64)
    for i in range(k):  # k passes, each vectorised over the whole sequence
        val = (val << np.uint64(2)) | np.where(w[:, i] == 255, 0, w[:, i]).astype(np.uint64)
    return val, valid


def canonical_kmers(seq: str, k: int = K) -> np.ndarray:
    """Canonical (``min(fwd, revcomp)``) k-mer values of one sequence, ambiguity windows dropped."""
    v = _codes(seq)
    rv = 3 - v.astype(np.int16)
    rv[v == 255] = 255
    rv = rv[::-1].astype(np.uint8)
    fwd, valid = _roll(v, k)
    rev, _ = _roll(rv, k)
    if fwd.size == 0:
        return fwd
    # the k-mer at position j reverse-complements to the rv k-mer at position m-1-j
    return np.minimum(fwd, rev[::-1])[valid]


def genome_kmer_index(contigs: dict[str, str], k: int = K) -> np.ndarray:
    """Sorted unique canonical k-mers of a whole assembly — the lookup table rule (A) queries."""
    parts = [canonical_kmers(s, k) for s in contigs.values()]
    parts = [p for p in parts if p.size]
    if not parts:
        return np.empty(0, dtype=np.uint64)
    return np.unique(np.concatenate(parts))


def feature_kmer_table(seqs: list[str], k: int = K) -> tuple[np.ndarray, np.ndarray]:
    """Feature sequences -> ``(flat canonical k-mers, per-feature start offsets)`` for ``reduceat``.

    Features shorter than ``k`` cannot be tested and are given an empty span; ``contains_all`` scores
    them absent. ``ggcat_to_pyseer`` never emits one (its minimum span is exactly ``k``).
    """
    flat: list[np.ndarray] = []
    offsets = np.zeros(len(seqs), dtype=np.int64)
    n = 0
    for i, s in enumerate(seqs):
        offsets[i] = n
        km = canonical_kmers(s, k)
        flat.append(km)
        n += km.size
    return (np.concatenate(flat) if flat else np.empty(0, dtype=np.uint64)), offsets


def contains_all(gkmers: np.ndarray, flat: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Rule (A): per feature, are *all* its canonical k-mers in the genome?"""
    n_feat = offsets.size
    if gkmers.size == 0 or flat.size == 0:
        return np.zeros(n_feat, dtype=bool)
    idx = np.searchsorted(gkmers, flat)
    hit = (idx < gkmers.size) & (gkmers[np.minimum(idx, gkmers.size - 1)] == flat)
    ends = np.append(offsets[1:], flat.size)
    empty = offsets >= ends  # a feature with no testable k-mer scores absent
    starts = np.where(empty, 0, offsets)
    out = np.logical_and.reduceat(hit, starts)
    out[empty] = False
    return out


def substring_presence(aut, contigs: dict[str, str], n_feat: int) -> np.ndarray:
    """Rule (B): exact-substring presence on either strand, via the shared Aho-Corasick automaton."""
    out = np.zeros(n_feat, dtype=bool)
    for cseq in contigs.values():
        for _end, idx in aut.iter(cseq):
            out[idx] = True
    return out


# --------------------------------------------------------------------------------------------------
# compare — rule (A) vs rule (B) vs GGCAT's own colouring, on the same genomes
# --------------------------------------------------------------------------------------------------
def _truth_rows(submatrix: Path, seq2idx: dict[str, int], want: set[str], n_feat: int) -> dict[str, np.ndarray]:
    """Stream ``hits_submatrix.tsv`` -> ``{sample: bool[n_feat]}`` for ``want`` only."""
    rows = {s: np.zeros(n_feat, dtype=bool) for s in want}
    with submatrix.open() as fh:
        for line in fh:
            seq, sep, rest = line.partition(" | ")
            if not sep:
                continue
            idx = seq2idx.get(seq)
            if idx is None:
                continue
            for tok in rest.split():
                s = tok.rpartition(":")[0]
                r = rows.get(s)
                if r is not None:
                    r[idx] = True
    return rows


def run_compare(args: argparse.Namespace) -> dict:
    """Score a sample of genomes by both rules and diff each against GGCAT's colouring."""
    import ahocorasick

    from bac_pyseer.kleb_iso_source.unitig_placement import build_automaton

    design = args.design_dir
    id_map = pd.read_csv(design / "id_map.tsv", sep="\t")
    seqs = [str(s).upper() for s in id_map["variant"]]
    n_feat = len(seqs)
    seq2idx = {s: int(i) for s, i in zip(seqs, id_map["unitig_idx"], strict=True)}
    lengths = np.array([len(s) for s in seqs])

    samples = [ln.strip() for ln in (design / "samples.txt").read_text().splitlines() if ln.strip()]
    step = max(1, len(samples) // args.n_genomes)
    picked = samples[::step][: args.n_genomes]  # deterministic stride, no RNG
    logger.info("features=%d  cohort samples=%d  scoring %d genomes", n_feat, len(samples), len(picked))

    refs = pd.read_csv(args.reflist, sep="\t", header=None, names=["Sample", "path"], dtype=str)
    path_of = dict(zip(refs["Sample"], refs["path"], strict=True))
    missing = [s for s in picked if s not in path_of]
    if missing:
        raise SystemExit(f"{len(missing)} sampled genomes absent from the reflist, e.g. {missing[:5]}")

    logger.info("streaming the GGCAT sub-matrix for the sampled rows ...")
    truth = _truth_rows(design / "hits_submatrix.tsv", seq2idx, set(picked), n_feat)

    logger.info("building the k-mer table and the Aho-Corasick automaton ...")
    flat, offsets = feature_kmer_table(seqs)
    aut = build_automaton(id_map)
    assert isinstance(aut, ahocorasick.Automaton)

    per_genome, disc_a, disc_b = [], [], []
    for n_done, s in enumerate(picked, 1):
        contigs = _load_contigs(Path(path_of[s]))
        gk = genome_kmer_index(contigs)
        a = contains_all(gk, flat, offsets)
        b = substring_presence(aut, contigs, n_feat)
        t = truth[s]
        rec = {
            "sample": s,
            "n_contigs": len(contigs),
            "assembly_bp": int(sum(len(c) for c in contigs.values())),
            "n_kmers": int(gk.size),
            "truth_present": int(t.sum()),
            "A_present": int(a.sum()),
            "B_present": int(b.sum()),
            "A_vs_truth_mismatch": int((a != t).sum()),
            "A_extra": int((a & ~t).sum()),
            "A_missing": int((~a & t).sum()),
            "B_vs_truth_mismatch": int((b != t).sum()),
            "B_extra": int((b & ~t).sum()),
            "B_missing": int((~b & t).sum()),
            "A_vs_B_mismatch": int((a != b).sum()),
        }
        per_genome.append(rec)
        for w in np.flatnonzero(a != t)[: args.max_examples]:
            disc_a.append({"sample": s, "unitig_idx": int(w), "len": int(lengths[w]),
                           "rule_A": bool(a[w]), "ggcat": bool(t[w]), "n_contigs": len(contigs)})
        for w in np.flatnonzero(b != t)[: args.max_examples]:
            disc_b.append({"sample": s, "unitig_idx": int(w), "len": int(lengths[w]),
                           "rule_B": bool(b[w]), "ggcat": bool(t[w]), "n_contigs": len(contigs)})
        logger.info("[%d/%d] %s contigs=%-5d A!=truth=%-6d B!=truth=%-6d A!=B=%d",
                    n_done, len(picked), s, rec["n_contigs"],
                    rec["A_vs_truth_mismatch"], rec["B_vs_truth_mismatch"], rec["A_vs_B_mismatch"])

    df = pd.DataFrame(per_genome)
    cells = int(len(picked)) * n_feat
    summary = {
        "design_dir": str(design), "k": K, "n_features": n_feat,
        "n_genomes": len(picked), "cells": cells,
        "feature_len": {"min": int(lengths.min()), "mean": float(lengths.mean()), "max": int(lengths.max())},
        "A_vs_truth": {"mismatch": int(df["A_vs_truth_mismatch"].sum()),
                       "extra": int(df["A_extra"].sum()), "missing": int(df["A_missing"].sum())},
        "B_vs_truth": {"mismatch": int(df["B_vs_truth_mismatch"].sum()),
                       "extra": int(df["B_extra"].sum()), "missing": int(df["B_missing"].sum())},
        "A_vs_B": {"mismatch": int(df["A_vs_B_mismatch"].sum())},
        "contig_count": {"min": int(df["n_contigs"].min()), "median": float(df["n_contigs"].median()),
                         "max": int(df["n_contigs"].max())},
        "per_genome": per_genome,
        "discordant_A_examples": disc_a[: args.max_examples],
        "discordant_B_examples": disc_b[: args.max_examples],
    }
    for name in ("A_vs_truth", "B_vs_truth"):
        summary[name]["rate"] = summary[name]["mismatch"] / cells if cells else 0.0
        if df[f"{name[0]}_vs_truth_mismatch"].sum():
            summary[name]["corr_with_contig_count"] = float(
                df["n_contigs"].corr(df[f"{name[0]}_vs_truth_mismatch"]))
    return summary


def _main_cli(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="phase", required=True)
    c = sub.add_parser("compare", help="Rule (A) vs rule (B) vs GGCAT colouring on a sample of genomes.")
    c.add_argument("--design-dir", type=Path, required=True, help="dir holding id_map.tsv, samples.txt, hits_submatrix.tsv")
    c.add_argument("--reflist", type=Path, required=True, help="Sample<TAB>assembly path")
    c.add_argument("--n-genomes", type=int, default=40)
    c.add_argument("--max-examples", type=int, default=200)
    c.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    summary = run_compare(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    brief = {k: summary[k] for k in ("cells", "n_genomes", "n_features", "A_vs_truth", "B_vs_truth", "A_vs_B")}
    print(json.dumps(brief, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    _main_cli()

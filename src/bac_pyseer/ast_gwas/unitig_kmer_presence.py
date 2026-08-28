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
from scipy import sparse

from bacpredict.engine.splits.load_splits import load_splits

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


# --------------------------------------------------------------------------------------------------
# score — the production scan: genomes x hit features, by rule (A)
# --------------------------------------------------------------------------------------------------
def score_samples(
    seqs: list[str], samples: list[str], path_of: dict[str, str], *, k: int = K, progress_every: int = 25,
) -> sparse.csr_matrix:
    """Score ``samples`` against ``seqs`` by rule (A) -> a ``(len(samples), len(seqs))`` binary CSR.

    Row order is ``samples`` order and column order is ``seqs`` order, which is ``id_map`` row order —
    the invariant that lets a fitted coefficient be traced back to its GWAS row. The feature k-mer
    table is built once and reused; per genome only the assembly index is rebuilt, so the cost is
    linear in assembly bases and effectively independent of how many features are being scored.
    """
    flat, offsets = feature_kmer_table(seqs, k)
    n_feat = len(seqs)
    rows: list[np.ndarray] = []
    indptr = np.zeros(len(samples) + 1, dtype=np.int64)
    for i, sample in enumerate(samples, 1):
        contigs = _load_contigs(Path(path_of[sample]))
        present = np.flatnonzero(contains_all(genome_kmer_index(contigs, k), flat, offsets))
        rows.append(present.astype(np.int32))
        indptr[i] = indptr[i - 1] + present.size
        if progress_every and (i % progress_every == 0 or i == len(samples)):
            logger.info("[%d/%d] %s: %d contigs, carries %d/%d features",
                        i, len(samples), sample, len(contigs), present.size, n_feat)
    indices = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int32)
    return sparse.csr_matrix(
        (np.ones(indices.size, dtype=np.int8), indices, indptr), shape=(len(samples), n_feat), dtype=np.int8,
    )


def _read_reflist(path: Path) -> dict[str, str]:
    """``Sample<TAB>assembly_path`` -> mapping."""
    refs = pd.read_csv(path, sep="\t", header=None, names=["Sample", "path"], dtype=str)
    return dict(zip(refs["Sample"], refs["path"], strict=True))


def run_score(args: argparse.Namespace) -> dict:
    """Scan one shard of genomes and persist the rows as a CSR beside its sample list."""
    id_map = pd.read_csv(args.id_map, sep="\t")
    seqs = [str(s).upper() for s in id_map["variant"]]

    _, train_ids, validate_ids, holdout_ids = load_splits(args.split_table)
    by_split = {"train": train_ids, "validate": validate_ids, "holdout": holdout_ids}
    wanted = [s.strip() for s in args.splits.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in by_split]
    if unknown:
        raise SystemExit(f"unknown split(s) {unknown}; expected a subset of {list(by_split)}")
    samples = [s for name in ("train", "validate", "holdout") if name in wanted for s in by_split[name]]

    path_of = _read_reflist(args.reflist)
    # A genome with no assembly cannot be scanned. It is dropped here rather than scored all-zero,
    # because an all-zero row is indistinguishable from a genuine non-carrier and would quietly
    # dilute the holdout. The merge reconciles the surviving ids against the split table.
    unresolved = [s for s in samples if s not in path_of]
    samples = [s for s in samples if s in path_of]
    shard = samples[args.shard_index :: args.n_shards]
    logger.info("shard %d/%d: %d of %d genomes, %d features (%d unresolved, dropped)",
                args.shard_index, args.n_shards, len(shard), len(samples), len(seqs), len(unresolved))

    matrix = score_samples(seqs, shard, path_of, progress_every=args.progress_every)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(args.out, matrix)
    args.out.with_suffix(".samples.txt").write_text("".join(f"{s}\n" for s in shard))
    summary = {
        "id_map": str(args.id_map.resolve()), "split_table": str(args.split_table), "splits": wanted,
        "shard_index": args.shard_index, "n_shards": args.n_shards,
        "n_features": len(seqs), "n_scored": len(shard), "n_unresolved": len(unresolved),
        "nnz": int(matrix.nnz), "out": str(args.out),
    }
    args.out.with_suffix(".scan.json").write_text(json.dumps(summary, indent=2))
    return summary


# --------------------------------------------------------------------------------------------------
# merge — GGCAT train+validate rows (+) scanned holdout rows, with the gates that make it trustworthy
# --------------------------------------------------------------------------------------------------
def _load_shards(paths: list[Path]) -> tuple[sparse.csr_matrix, list[str]]:
    """Concatenate scan shards in the order given -> ``(csr, sample ids)``."""
    mats, ids = [], []
    for p in paths:
        mats.append(sparse.load_npz(p).tocsr())
        ids.extend(p.with_suffix(".samples.txt").read_text().split())
    if not mats:
        raise SystemExit("no scan shards found")
    matrix = sparse.vstack(mats, format="csr")
    if matrix.shape[0] != len(ids):
        raise SystemExit(f"shards hold {matrix.shape[0]} rows but {len(ids)} sample ids")
    if len(set(ids)) != len(ids):
        raise SystemExit("a sample appears in more than one scan shard — check --shard-index/--n-shards")
    return matrix, ids


def verify_against_ggcat(
    scan: sparse.csr_matrix, scan_ids: list[str], ggcat: sparse.csr_matrix, ggcat_ids: list[str],
    *, max_examples: int = 50,
) -> dict:
    """Two-sided exact comparison of the scanner against GGCAT's own colouring, on shared genomes.

    This is the assertion that makes a mixed design matrix legitimate. The holdout rows come from
    this scanner and the train+validate rows come from GGCAT; if the two operators disagree, the
    holdout is being scored by a different rule from the data the model was fitted on, and any
    difference in performance is partly an artefact of that. Exact agreement on every shared genome
    is what rules it out — so the count is reported whether or not it is zero, and a skipped
    comparison is recorded as ``n_shared: 0`` rather than passing silently.
    """
    scan_row = {s: i for i, s in enumerate(scan_ids)}
    shared = [s for s in ggcat_ids if s in scan_row]
    if not shared:
        return {"n_shared": 0, "n_mismatch_cells": None, "n_mismatch_genomes": None, "examples": []}
    ggcat_row = {s: i for i, s in enumerate(ggcat_ids)}
    a = scan[[scan_row[s] for s in shared]].astype(bool)
    b = ggcat[[ggcat_row[s] for s in shared]].astype(bool)
    diff = (a != b).tocoo()
    per_genome = np.bincount(diff.row, minlength=len(shared))
    examples = [
        {"sample": shared[int(r)], "unitig_idx": int(c),
         "scanner": bool(a[int(r), int(c)]), "ggcat": bool(b[int(r), int(c)])}
        for r, c in list(zip(diff.row, diff.col, strict=True))[:max_examples]
    ]
    return {
        "n_shared": len(shared),
        "cells": len(shared) * scan.shape[1],
        "n_mismatch_cells": int(diff.nnz),
        "n_mismatch_genomes": int((per_genome > 0).sum()),
        "scanner_present": int(a.nnz),
        "ggcat_present": int(b.nnz),
        "examples": examples,
    }


def run_merge(args: argparse.Namespace) -> dict:
    """Assemble the final design: GGCAT train+validate rows, scanned holdout rows, gates on both."""
    from bac_pyseer.ast_gwas.unitig_design_matrix import check_holdout_coverage, load_design

    ggcat, ggcat_ids, id_map = load_design(args.design_dir)
    shards = sorted(args.shard_dir.glob(args.shard_glob))
    scan, scan_ids = _load_shards(shards)

    # Align the scan's columns to the design's by *sequence*, never by position. One scan then serves
    # both the full design and the ``--dedupe-patterns`` LD control, whose id_map is a subset of the
    # same features in the same order — re-scanning for it would double the cost of the whole stage
    # and would be a second chance to get the column order wrong.
    scan_id_map_path = args.scan_id_map or Path(json.loads(shards[0].with_suffix(".scan.json").read_text())["id_map"])
    scan_id_map = pd.read_csv(scan_id_map_path, sep="\t")
    col_of = {str(v).upper(): i for i, v in enumerate(scan_id_map["variant"])}
    wanted_cols = [col_of.get(str(v).upper()) for v in id_map["variant"]]
    absent = [i for i, c in enumerate(wanted_cols) if c is None]
    if absent:
        raise SystemExit(
            f"{len(absent)} design feature(s) were never scanned — the scan used {scan_id_map_path}, "
            f"which does not contain them. Re-run the scan against the design that produced them."
        )
    scan = scan[:, wanted_cols]
    if scan.shape[1] != ggcat.shape[1]:
        raise SystemExit(f"scan has {scan.shape[1]} features but the design has {ggcat.shape[1]}")

    verification = verify_against_ggcat(scan, scan_ids, ggcat, ggcat_ids)
    logger.info("verification: %s shared genomes, %s mismatched cells",
                verification["n_shared"], verification["n_mismatch_cells"])
    if verification["n_shared"] == 0:
        logger.warning("the scan covered no train/validate genome, so the scanner was never checked "
                       "against GGCAT's colouring — re-run the scan with --splits train,validate,holdout")
    elif verification["n_mismatch_cells"] > args.max_mismatch_cells:
        raise SystemExit(
            f"scanner disagrees with GGCAT on {verification['n_mismatch_cells']} cells across "
            f"{verification['n_mismatch_genomes']} genomes (limit {args.max_mismatch_cells}); "
            f"examples: {verification['examples'][:5]}"
        )

    _, train_ids, validate_ids, holdout_ids = load_splits(args.split_table)
    scan_row = {s: i for i, s in enumerate(scan_ids)}
    ggcat_row = {s: i for i, s in enumerate(ggcat_ids)}
    # Train+validate rows are GGCAT's; holdout rows are the scanner's. Both are rule (A), which the
    # verification above has just established on the genomes where the two overlap.
    kept_trainval = [s for s in [*train_ids, *validate_ids] if s in ggcat_row]
    kept_holdout = [s for s in holdout_ids if s in scan_row]
    if not kept_holdout:
        raise SystemExit("no holdout genome was scanned — the merged design would have nothing to score")
    sample_ids = [*kept_trainval, *kept_holdout]
    matrix = sparse.vstack(
        [ggcat[[ggcat_row[s] for s in kept_trainval]], scan[[scan_row[s] for s in kept_holdout]]],
        format="csr",
    ).astype(np.int8)

    coverage = check_holdout_coverage(
        matrix, sample_ids, kept_trainval, kept_holdout,
        min_ratio=args.min_holdout_carrier_ratio, min_holdout_genomes=args.min_holdout_genomes,
    )
    if not coverage["checked"]:
        logger.warning("holdout coverage not asserted: only %d holdout genomes", coverage["n_holdout"])

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(out_dir / "presence.npz", matrix)
    id_map.to_csv(out_dir / "id_map.tsv", sep="\t", index=False)
    (out_dir / "samples.txt").write_text("".join(f"{s}\n" for s in sample_ids))
    manifest = {
        "design_dir": str(args.design_dir), "shard_dir": str(args.shard_dir),
        "scan_id_map": str(scan_id_map_path), "n_scan_features": int(len(scan_id_map)),
        "split_table": str(args.split_table),
        "rows_from_ggcat": len(kept_trainval), "rows_from_scanner": len(kept_holdout),
        "n_dropped_trainval": len([s for s in [*train_ids, *validate_ids] if s not in ggcat_row]),
        "n_dropped_holdout": len([s for s in holdout_ids if s not in scan_row]),
        "n_features": int(matrix.shape[1]), "nnz": int(matrix.nnz),
        "verification": verification, "holdout_coverage": coverage,
    }
    (out_dir / "merge_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest

def _main_cli(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="phase", required=True)
    c = sub.add_parser("compare", help="Rule (A) vs rule (B) vs GGCAT colouring on a sample of genomes.")
    c.add_argument("--design-dir", type=Path, required=True, help="dir holding id_map.tsv, samples.txt, hits_submatrix.tsv")
    c.add_argument("--reflist", type=Path, required=True, help="Sample<TAB>assembly path")
    c.add_argument("--n-genomes", type=int, default=40)
    c.add_argument("--max-examples", type=int, default=200)
    c.add_argument("--out", type=Path, required=True)

    sc = sub.add_parser("score", help="Scan one shard of genomes for the hit features (rule A).")
    sc.add_argument("--id-map", type=Path, required=True, help="id_map.tsv — defines the features AND their column order.")
    sc.add_argument("--split-table", type=Path, required=True, help="<drug>_split.csv")
    sc.add_argument("--reflist", type=Path, required=True, help="Sample<TAB>assembly path — must cover every split")
    sc.add_argument("--splits", default="train,validate,holdout",
                    help="Which slices to scan. The default includes train+validate so the merge can "
                         "check the scanner against GGCAT's own colouring; holdout alone skips that gate.")
    sc.add_argument("--shard-index", type=int, default=0)
    sc.add_argument("--n-shards", type=int, default=1)
    sc.add_argument("--progress-every", type=int, default=25)
    sc.add_argument("--out", type=Path, required=True, help="Output .npz; .samples.txt and .scan.json sit beside it.")

    m = sub.add_parser("merge", help="GGCAT train+validate rows (+) scanned holdout rows -> one design.")
    m.add_argument("--design-dir", type=Path, required=True, help="The GGCAT design (train+validate rows).")
    m.add_argument("--shard-dir", type=Path, required=True, help="Directory holding the scan shards.")
    m.add_argument("--shard-glob", default="scan_*.npz")
    m.add_argument("--scan-id-map", type=Path, default=None,
                   help="id_map the scan was run against (default: read from the first shard's .scan.json). "
                        "Columns are matched by sequence, so a dedup design reuses the same scan.")
    m.add_argument("--split-table", type=Path, required=True)
    m.add_argument("--out-dir", type=Path, required=True, help="New directory — the GGCAT design is left intact.")
    m.add_argument("--max-mismatch-cells", type=int, default=0,
                   help="Scanner-vs-GGCAT disagreements tolerated on shared genomes. 0 is the gate.")
    m.add_argument("--min-holdout-carrier-ratio", type=float, default=0.5)
    m.add_argument("--min-holdout-genomes", type=int, default=30,
                   help="Below this many holdout rows the carrier ratio is noise, so it is recorded "
                        "but not asserted. Every real drug is far above it.")

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    if args.phase == "score":
        print(json.dumps(run_score(args), indent=2))
        return
    if args.phase == "merge":
        print(json.dumps(run_merge(args), indent=2))
        return

    summary = run_compare(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    brief = {k: summary[k] for k in ("cells", "n_genomes", "n_features", "A_vs_truth", "B_vs_truth", "A_vs_B")}
    print(json.dumps(brief, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    _main_cli()

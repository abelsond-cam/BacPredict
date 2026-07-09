"""Genome-wide audit of the baclm non-coding channel (Stage 2 step 4 — the architecture question).

Under the 2d rule (only ``CDS`` occupies), each non-coding region is the **maximal contiguous non-CDS
run**, so an RNA gene (rRNA/tRNA/tmRNA/ncRNA) and the intergenic DNA beside it are embedded as **one**
960-d vector. This module quantifies, across the whole cohort, two things that decide how we treat the
channel:

1. **Windowing load** — how many non-coding runs (and RNA bodies) exceed the model context (``MAX_LEN``
   = 2048 char), i.e. how many get windowed rather than fitting in one forward. This is the number to
   take to the baclm developers before fixing the window-pool scheme.
2. **IGR↔RNA fusion** — how often a non-coding run actually *contains* an RNA (so baclm fuses RNA+IGR),
   and how "contaminated" each RNA's run is by flanking IGR. If most RNA sit in runs that are almost
   entirely the RNA body, a separate RNA channel buys little; if they are routinely fused with long
   IGR, a separate channel (embedding the RNA body alone) is worth it.

Fast + label-free: parses the Bakta GFF only (CDS + RNA features + the ``##sequence-region`` contig
lengths) — no FASTA load — and mirrors :func:`tl.embed.extract_intergenic_from_gff_fna`'s run logic
(``min_len`` gap filter, only-CDS-occupying) so the run set is identical to what gets embedded.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import pandas as pd

from tl.embed.extract_proteins_from_gff_fna import _open_text

logger = logging.getLogger(__name__)

MAX_LEN = 2048  # baclm char-level context; a region longer than this must be windowed.
_RNA_TYPES = frozenset(
    {"rrna", "trna", "tmrna", "ncrna", "ncrna_gene", "antisense_rna", "rnase_p_rna", "srp_rna", "riboswitch"}
)
# Run-length histogram edges (bp). Last bin is the >MAX_LEN tail.
_LEN_BINS = [0, 100, 300, 1000, 2048, 4096, 8192, 1_000_000_000]


def _merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals.sort()
    out = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


# Feature types that describe the contig record itself, not a biological feature inside it.
_CONTIG_TYPES = frozenset({"region", "databank_entry"})


def _parse_gff_lengths_cds_annot(gff_path: Path):
    """Return ``(contig_len[seqid], cds[seqid]=[(s0,e)], annot[seqid]=[(start1,end1,type)])`` from a GFF.

    ``cds`` intervals are 0-based half-open (to match the extraction run logic). ``annot`` is **every
    non-CDS, non-contig-record feature** (RNA *and* anything else Bakta emits — CRISPR, oriC, ncRNA
    regions, …), 1-based inclusive, so we can measure both RNA-vs-IGR composition and what "other"
    features live in the runs. Contig length comes from ``##sequence-region`` pragmas (``region``
    feature as fallback).
    """
    contig_len: dict[str, int] = {}
    cds: dict[str, list[tuple[int, int]]] = {}
    annot: dict[str, list[tuple[int, int, str]]] = {}
    with _open_text(gff_path) as fh:
        for line in fh:
            if line.startswith("#"):
                if line.startswith("##sequence-region"):
                    p = line.split()
                    if len(p) >= 4:
                        try:
                            contig_len[p[1]] = int(p[3])
                        except ValueError:
                            pass
                elif line.startswith("##FASTA"):
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
            if ftype == "CDS":
                cds.setdefault(seqid, []).append((start - 1, end))
            elif ftype in _CONTIG_TYPES:
                if seqid not in contig_len:
                    contig_len[seqid] = end
            else:
                annot.setdefault(seqid, []).append((start, end, ftype.lower()))
    return contig_len, cds, annot


_IGR_MARGIN = 30  # a run counts as "IGR-fused" if this much unannotated DNA flanks its RNA/feature


def _audit_one(args_tuple: tuple) -> dict[str, Any] | None:
    """Audit one genome: non-coding runs, windowing, and per-RNA-type IGR/other-RNA adjacency."""
    sample_id, gff_path, min_len = args_tuple
    gpath = Path(gff_path)
    if not gpath.exists():
        return None
    try:
        contig_len, cds, annot = _parse_gff_lengths_cds_annot(gpath)
    except (OSError, ValueError):
        return None

    len_hist = Counter()
    n_runs = n_runs_gt_max = n_windows = n_runs_with_rna = max_run = 0
    # Per-RNA-type tallies: total, adjacent-to-IGR (run has >=_IGR_MARGIN unannotated bp),
    # adjacent-to-another-RNA (run holds >1 RNA), and solo (run is essentially just this RNA).
    rna_total, rna_adj_igr, rna_adj_rna, rna_solo = Counter(), Counter(), Counter(), Counter()
    feature_type_counts = Counter()           # every non-CDS feature type (RNA + "other")
    rna_len_all: list[int] = []

    for feats in annot.values():
        feature_type_counts.update(t for (_s, _e, t) in feats)

    seqids = set(contig_len) | set(cds) | set(annot)
    for seqid in seqids:
        clen = contig_len.get(seqid)
        feats = annot.get(seqid, [])
        if clen is None:
            coords = [e for _, e in cds.get(seqid, [])] + [e for _, e, _ in feats]
            if not coords:
                continue
            clen = max(coords)
        merged = _merge(cds.get(seqid, []))
        runs: list[tuple[int, int]] = []
        prev = 0
        for s, e in merged:
            if s > prev:
                runs.append((prev, s))
            prev = max(prev, e)
        if prev < clen:
            runs.append((prev, clen))

        for g0, g1 in runs:
            rlen = g1 - g0
            if rlen < min_len:
                continue
            n_runs += 1
            max_run = max(max_run, rlen)
            for i in range(len(_LEN_BINS) - 1):
                if _LEN_BINS[i] <= rlen < _LEN_BINS[i + 1]:
                    len_hist[i] += 1
                    break
            n_windows += -(-rlen // MAX_LEN)  # ceil
            if rlen > MAX_LEN:
                n_runs_gt_max += 1
            # Features inside this run (run 0-based half-open g0..g1; feats 1-based inclusive).
            in_run = [(s, e, t) for (s, e, t) in feats if (s - 1) < g1 and e > g0]
            rnas = [(s, e, t) for (s, e, t) in in_run if t in _RNA_TYPES]
            if rnas:
                n_runs_with_rna += 1
            # Unannotated (true IGR) DNA in the run = run length minus all annotated feature coverage.
            annotated = sum(min(g1, e) - max(g0, s - 1) for (s, e, _t) in in_run)
            igr_len = rlen - annotated
            has_igr = igr_len >= _IGR_MARGIN
            multi_rna = len(rnas) > 1
            for (s, e, t) in rnas:
                rna_total[t] += 1
                rna_len_all.append(e - s + 1)
                if has_igr:
                    rna_adj_igr[t] += 1
                if multi_rna:
                    rna_adj_rna[t] += 1
                if not has_igr and not multi_rna:
                    rna_solo[t] += 1

    return {
        "sample": sample_id,
        "n_runs": n_runs,
        "n_runs_gt_max": n_runs_gt_max,
        "n_windows": n_windows,
        "n_runs_with_rna": n_runs_with_rna,
        "max_run_len": max_run,
        "len_hist": dict(len_hist),
        "n_rna": sum(rna_total.values()),
        "n_rna_gt_max": sum(1 for x in rna_len_all if x > MAX_LEN),
        "max_rna_len": max(rna_len_all) if rna_len_all else 0,
        "rna_total": dict(rna_total),
        "rna_adj_igr": dict(rna_adj_igr),
        "rna_adj_rna": dict(rna_adj_rna),
        "rna_solo": dict(rna_solo),
        "feature_type_counts": dict(feature_type_counts),
    }


def _aggregate(per_genome: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll per-genome stats into cohort totals + the two headline fractions."""
    n_g = len(per_genome)
    tot_runs = sum(g["n_runs"] for g in per_genome)
    tot_runs_gt = sum(g["n_runs_gt_max"] for g in per_genome)
    tot_windows = sum(g["n_windows"] for g in per_genome)
    tot_runs_rna = sum(g["n_runs_with_rna"] for g in per_genome)
    tot_rna = sum(g["n_rna"] for g in per_genome)
    tot_rna_gt = sum(g["n_rna_gt_max"] for g in per_genome)
    len_hist = Counter()
    rna_total, rna_adj_igr, rna_adj_rna, rna_solo, feature_types = (
        Counter(), Counter(), Counter(), Counter(), Counter())
    for g in per_genome:
        for k, v in g["len_hist"].items():
            len_hist[int(k)] += v
        rna_total.update(g["rna_total"])
        rna_adj_igr.update(g["rna_adj_igr"])
        rna_adj_rna.update(g["rna_adj_rna"])
        rna_solo.update(g["rna_solo"])
        feature_types.update(g["feature_type_counts"])
    labels = [f"{_LEN_BINS[i]}-{_LEN_BINS[i + 1]}" for i in range(len(_LEN_BINS) - 1)]
    labels[-1] = f">{_LEN_BINS[-2]}"
    # Per-RNA-type: total, adjacent-to-IGR (+frac), adjacent-to-other-RNA, solo (CDS-flanked, alone).
    rna_breakdown = {
        t: {
            "total": rna_total[t],
            "adjacent_to_igr": rna_adj_igr[t],
            "adjacent_to_igr_frac": (rna_adj_igr[t] / rna_total[t]) if rna_total[t] else 0.0,
            "adjacent_to_other_rna": rna_adj_rna[t],
            "solo_cds_flanked": rna_solo[t],
        }
        for t in sorted(rna_total, key=lambda x: -rna_total[x])
    }
    return {
        "n_genomes": n_g,
        "total_noncoding_runs": tot_runs,
        "runs_over_maxlen": tot_runs_gt,
        "runs_over_maxlen_frac": (tot_runs_gt / tot_runs) if tot_runs else 0.0,
        "total_windows": tot_windows,
        "extra_windows_from_long_runs": tot_windows - tot_runs,  # forwards added purely by windowing
        "runs_containing_rna": tot_runs_rna,
        "runs_containing_rna_frac": (tot_runs_rna / tot_runs) if tot_runs else 0.0,
        "total_rna_bodies": tot_rna,
        "rna_over_maxlen": tot_rna_gt,
        "max_run_len_seen": max((g["max_run_len"] for g in per_genome), default=0),
        "max_rna_len_seen": max((g["max_rna_len"] for g in per_genome), default=0),
        "run_len_hist": {labels[int(k)]: v for k, v in sorted(len_hist.items())},
        "rna_breakdown": rna_breakdown,
        "feature_type_counts": dict(feature_types.most_common()),
        "mean_runs_per_genome": (tot_runs / n_g) if n_g else 0.0,
        "mean_rna_per_genome": (tot_rna / n_g) if n_g else 0.0,
    }


def run_audit(input_csv: Path, *, n: int | None, workers: int, min_len: int) -> dict[str, Any]:
    """Audit every genome in ``input_csv`` (cols ``Sample``/``sr_gff_file``); return aggregate stats."""
    df = pd.read_csv(input_csv).dropna(subset=["Sample", "sr_gff_file"])
    df["Sample"] = df["Sample"].astype(str)
    if n:
        df = df.head(n)
    tasks = [(r["Sample"], str(r["sr_gff_file"]), min_len) for _, r in df.iterrows()]
    nw = min(workers, cpu_count(), len(tasks)) or 1
    logger.info("auditing %d genomes with %d workers", len(tasks), nw)
    with Pool(processes=nw) as pool:
        per_genome = [g for g in pool.imap_unordered(_audit_one, tasks, chunksize=16) if g is not None]
    logger.info("audited %d/%d genomes (rest missing GFF)", len(per_genome), len(tasks))
    return _aggregate(per_genome)


def _print_summary(agg: dict[str, Any]) -> None:
    print("\n=== non-coding channel audit ===")
    print(f"genomes audited           : {agg['n_genomes']:,}")
    print(f"non-coding runs (total)   : {agg['total_noncoding_runs']:,}  ({agg['mean_runs_per_genome']:.0f}/genome)")
    print(f"runs > MAX_LEN (windowed) : {agg['runs_over_maxlen']:,}  ({agg['runs_over_maxlen_frac'] * 100:.3f}%)")
    print(f"extra forwards from windows: {agg['extra_windows_from_long_runs']:,} "
          f"(total windows {agg['total_windows']:,})")
    print(f"longest run seen          : {agg['max_run_len_seen']:,} bp")
    print(f"runs containing an RNA    : {agg['runs_containing_rna']:,}  "
          f"({agg['runs_containing_rna_frac'] * 100:.2f}% of runs)")
    print(f"RNA bodies (total)        : {agg['total_rna_bodies']:,}  ({agg['mean_rna_per_genome']:.0f}/genome)")
    print(f"RNA bodies > MAX_LEN      : {agg['rna_over_maxlen']:,}  (longest {agg['max_rna_len_seen']:,} bp)")
    print("run-length histogram (bp) :")
    for k, v in agg["run_len_hist"].items():
        print(f"    {k:>12} : {v:,}")
    print("per-RNA-type adjacency    :  (adj-IGR = run has >=30bp unannotated DNA beside the RNA)")
    print(f"    {'type':>10} {'total':>10} {'adj-IGR':>12} {'adj-RNA':>10} {'solo':>10}")
    for t, d in agg["rna_breakdown"].items():
        print(f"    {t:>10} {d['total']:>10,} {d['adjacent_to_igr']:>10,} "
              f"({d['adjacent_to_igr_frac'] * 100:4.1f}%) {d['adjacent_to_other_rna']:>10,} {d['solo_cds_flanked']:>10,}")
    print("all non-CDS feature types :")
    for k, v in agg["feature_type_counts"].items():
        print(f"    {k:>16} : {v:,}")


def main() -> None:
    """CLI: audit the non-coding channel over an embedding-input CSV and write a JSON + summary."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    ap = argparse.ArgumentParser(description="Audit baclm non-coding runs + RNA fusion (GFF-only, fast).")
    ap.add_argument("--input-csv", type=Path, required=True, help="CSV with Sample + sr_gff_file columns")
    ap.add_argument("--output", type=Path, required=True, help="output JSON path")
    ap.add_argument("--n", type=int, default=None, help="limit number of genomes (sampling)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--min-len", type=int, default=30, help="mirror the extraction gap filter")
    args = ap.parse_args()

    agg = run_audit(args.input_csv, n=args.n, workers=args.workers, min_len=args.min_len)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(agg, indent=2))
    _print_summary(agg)
    print(f"\nJSON -> {args.output}")


if __name__ == "__main__":
    main()

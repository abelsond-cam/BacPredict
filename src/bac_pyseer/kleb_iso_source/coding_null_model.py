r"""Uniform-placement null for the IGR-coverage thresholds — the comparator for the observed coverage.

The IGR-coverage result (``annotate_unitig_coding``) is only interpretable against the spatial null:
**if the GWAS signal were distributed uniformly across the genome, how would a hit unitig's IGR
coverage be distributed?** This computes, for every threshold the observed table reports
(entirely-CDS / touch / ≥0.25 / ≥0.5 / entirely-IGR), the fraction of *uniform-random* placements that
would meet it — by **sliding** each hit-unitig length L across every genome's coding/non-coding
architecture:

  per contig, mark a coding 0/1 array from the merged CDS intervals, take its cumsum, then for each L
  the IGR overlap of every length-L window is ``L − (cds_cum[i+L] − cds_cum[i])``; count the windows in
  each category over all start positions.

Pooled over a sample of carrier genomes and **weighted by the hit unitigs' own length distribution**
(overall + per direction). Because genes are long and unitigs short, the entirely-CDS null is high
(near the CDS bp fraction) — observed entirely-CDS well below it means the signal avoids pure coding
sequence. Sanity: the entirely-CDS null reproduces the analytic ``Σ max(0, cds_len−L+1)/(G−L+1)``.

CPU job (parallel across genomes).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from genome_prep import contig_lengths, merge_intervals, parse_gff_features

# Categories in the accumulator columns after the first (which is the start-position total).
_CATS = ["entirely_cds", "touch", "significant", "predominant", "entirely_igr"]
_SIGNIFICANT, _PREDOMINANT, _ENTIRELY = 0.25, 0.5, 0.999  # igr_frac thresholds, matching the observed job


def genome_threshold_counts(gff_path: str, lengths: np.ndarray) -> np.ndarray:
    """Per-length window counts for one genome: ``[n_starts, ent_cds, touch, sig, pred, ent_igr]`` × L.

    Slides every length in ``lengths`` across each contig's coding/non-coding architecture (built from
    the merged CDS intervals) and tallies each IGR-coverage category over all start positions.
    """
    feats = parse_gff_features(gff_path)
    clens = contig_lengths(gff_path)
    acc = np.zeros((len(lengths), 6), dtype=np.int64)
    for contig, clen in clens.items():
        cds = merge_intervals([(f.start - 1, f.end) for f in feats.get(contig, []) if f.is_cds])
        coding = np.zeros(clen, dtype=np.int32)
        for s, e in cds:
            coding[s:e] = 1
        cum = np.concatenate([[0], np.cumsum(coding)])  # prefix sum, len clen+1
        for k, L in enumerate(lengths):
            L = int(L)
            if clen < L:
                continue
            igr = L - (cum[L:clen + 1] - cum[0:clen - L + 1])  # IGR bp per length-L window
            acc[k, 0] += igr.size
            acc[k, 1] += int((igr == 0).sum())
            acc[k, 2] += int((igr > 0).sum())
            acc[k, 3] += int((igr >= _SIGNIFICANT * L).sum())
            acc[k, 4] += int((igr >= _PREDOMINANT * L).sum())
            acc[k, 5] += int((igr >= _ENTIRELY * L).sum())
    return acc


def _worker(gff_path: str, lengths: np.ndarray) -> np.ndarray | None:
    try:
        return genome_threshold_counts(gff_path, lengths)
    except (ValueError, OSError) as exc:  # unreadable/length-less GFF — drop it from the pool
        print(f"skip {gff_path}: {exc}", file=sys.stderr)
        return None


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bakta-lookup", type=Path, required=True, help="TSV Sample<TAB>path → Bakta GFF3.")
    p.add_argument("--id-map", type=Path, required=True, help="select id_map.tsv (unitig_len [, direction]).")
    p.add_argument("--n-sample", type=int, default=100, help="Number of genomes to average the null over.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", type=Path, required=True, help="Output JSON.")
    args = p.parse_args(argv)

    idm = pd.read_csv(args.id_map, sep="\t")
    lengths = np.array(sorted(idm["unitig_len"].unique()), dtype=np.int64)  # distinct L to evaluate
    gff = dict(zip(*[pd.read_csv(args.bakta_lookup, sep="\t", dtype=str)[c] for c in ("Sample", "path")], strict=True))

    samples = sorted(gff)
    if args.n_sample and args.n_sample < len(samples):
        rng = np.random.default_rng(args.seed)
        samples = sorted(rng.choice(samples, size=args.n_sample, replace=False).tolist())
    paths = [gff[s] for s in samples if gff.get(s) and Path(gff[s]).is_file()]

    with Pool(args.workers) as pool:
        results = pool.starmap(_worker, [(pth, lengths) for pth in paths])
    accs = [a for a in results if a is not None]
    if not accs:
        raise SystemExit("no usable GFFs")
    total = np.sum(accs, axis=0)  # (nL, 6) pooled over genomes
    starts = total[:, 0:1]
    pbar = np.divide(total[:, 1:], starts, out=np.zeros_like(total[:, 1:], dtype=float), where=starts > 0)
    lidx = {int(L): k for k, L in enumerate(lengths)}

    def _weighted(sub: pd.DataFrame) -> dict[str, float]:
        w = Counter(int(x) for x in sub["unitig_len"])
        tot = sum(w.values())
        return {cat: round(sum(cnt * pbar[lidx[L], j] for L, cnt in w.items()) / tot, 4)
                for j, cat in enumerate(_CATS)}

    result: dict[str, object] = {
        "n_genomes": len(accs),
        "n_unitigs": int(len(idm)),
        "unitig_len": {"min": int(lengths.min()), "median": int(np.median(idm["unitig_len"])),
                       "max": int(lengths.max()), "n_distinct": int(len(lengths))},
        "null_overall": _weighted(idm),
    }
    if "direction" in idm.columns:
        result["null_by_direction"] = {str(d): _weighted(sub) for d, sub in idm.groupby("direction")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()

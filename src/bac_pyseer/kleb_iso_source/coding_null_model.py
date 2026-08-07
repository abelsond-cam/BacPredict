r"""Uniform-placement null: what fraction of the hit unitigs would land entirely within a CDS by chance?

The IGR-coverage result (``annotate_unitig_coding``) is only interpretable against the spatial null:
**if the GWAS signal were distributed uniformly across the genome, how often would a hit unitig fall
entirely inside a coding sequence?** Because genes are long and unitigs short, that null is high (near
the CDS base-pair fraction). Observed ``entirely_cds`` well *below* the null means the signal avoids
pure coding sequence — i.e. IGR-touching unitigs are over-represented.

For an L-mer placed uniformly over a genome with contig lengths ``G_c`` and merged CDS intervals of
length ``ℓ_i``::

    P(entirely in a CDS | L) = Σ_i max(0, ℓ_i − L + 1) / Σ_c (G_c − L + 1)

computed per genome (merged CDS so overlapping genes are not double-counted), averaged over a sample of
carrier genomes, then weighted by the **hit unitigs' own length distribution** (overall and per
direction) — so the null uses exactly the lengths of the unitigs we are testing.

Light, single-process — fine on a login node (or a tiny CPU job).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from genome_prep import contig_lengths, merge_intervals, parse_gff_features


def _genome_cds_lengths(gff_path: str) -> tuple[np.ndarray, int]:
    """Merged CDS interval lengths (bp) and total genome length for one carrier GFF."""
    feats = parse_gff_features(gff_path)
    lens: list[int] = []
    for flist in feats.values():
        for s, e in merge_intervals([(f.start - 1, f.end) for f in flist if f.is_cds]):
            lens.append(e - s)
    total = sum(contig_lengths(gff_path).values())
    return np.asarray(lens, dtype=np.int64), total


def _p_entirely_cds(cds_lens: np.ndarray, genome_bp: int, lengths: np.ndarray) -> np.ndarray:
    """P(L-mer entirely within a CDS) for each L in ``lengths``, for one genome."""
    out = np.empty(len(lengths), dtype=float)
    for k, L in enumerate(lengths):
        starts_total = genome_bp - int(L) + 1  # approx: whole-genome start positions (contig edges negligible)
        num = np.maximum(0, cds_lens - int(L) + 1).sum()
        out[k] = num / starts_total if starts_total > 0 else 0.0
    return out


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bakta-lookup", type=Path, required=True, help="TSV Sample<TAB>path → Bakta GFF3.")
    p.add_argument("--id-map", type=Path, required=True, help="select id_map.tsv (unitig_len [, direction]).")
    p.add_argument("--n-sample", type=int, default=200, help="Number of genomes to average the null over.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True, help="Output JSON.")
    args = p.parse_args(argv)

    idm = pd.read_csv(args.id_map, sep="\t")
    lengths = np.array(sorted(idm["unitig_len"].unique()), dtype=np.int64)  # distinct L to evaluate
    gff = dict(zip(*[pd.read_csv(args.bakta_lookup, sep="\t", dtype=str)[c] for c in ("Sample", "path")], strict=True))

    samples = sorted(gff)
    if args.n_sample and args.n_sample < len(samples):
        rng = np.random.default_rng(args.seed)
        samples = sorted(rng.choice(samples, size=args.n_sample, replace=False).tolist())

    acc = np.zeros(len(lengths), dtype=float)
    n_ok = 0
    for s in samples:
        g = gff.get(s)
        if not g or not Path(g).is_file():
            continue
        try:
            cds_lens, genome_bp = _genome_cds_lengths(g)
        except (ValueError, OSError) as exc:
            print(f"skip {s}: {exc}", file=sys.stderr)
            continue
        acc += _p_entirely_cds(cds_lens, genome_bp, lengths)
        n_ok += 1
    if n_ok == 0:
        raise SystemExit("no usable GFFs")
    pbar = {int(L): acc[k] / n_ok for k, L in enumerate(lengths)}  # mean P(entirely-CDS) per length

    def _weighted(sub: pd.DataFrame) -> float:
        w = Counter(int(x) for x in sub["unitig_len"])
        num = sum(cnt * pbar[L] for L, cnt in w.items())
        return round(num / sum(w.values()), 4) if w else 0.0

    result = {
        "n_genomes": n_ok,
        "n_unitigs": int(len(idm)),
        "null_frac_entirely_cds_overall": _weighted(idm),
        "unitig_len_summary": {"min": int(lengths.min()), "median": int(np.median(idm["unitig_len"])),
                               "max": int(lengths.max())},
    }
    if "direction" in idm.columns:
        result["null_frac_entirely_cds_by_direction"] = {
            str(d): _weighted(sub) for d, sub in idm.groupby("direction")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()

r"""Sample an af-matched set of NON-significant unitigs — the control for the IGR-coverage analysis.

The significant hit unitigs are ~2× IGR-enriched vs a uniform-placement null. Is that a property of
*divergent unitigs in general* (so non-significant unitigs are enriched too) or *specific to the
phenotype-associated hits*? To separate the two we run the identical coverage pipeline on a matched set
of unitigs that did **not** pass GWAS significance. Significant hits concentrate at common af and
IGR-coverage varies with af, so the control must be **af-matched** — otherwise "significant vs not"
would confound with "common vs rare".

This reads the full unitig LMM ``.assoc`` (~6.28M unitigs), takes everything **less significant than the
least-significant hit** (and not itself a hit), and draws ``--n-target`` unitigs so their af histogram
matches the hit set's (per fine af bin). Output is a ``nonsig_hits.tsv`` with the ``variant``/``af``
columns the generic ``unitig_placement.py --phase select`` needs — no geNomad, no direction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assoc", type=Path, required=True, help="Full unitig LMM .assoc (variant, af, lrt-pvalue).")
    p.add_argument("--hits-tsv", type=Path, required=True, help="The significant hits TSV (variant, af, lrt-pvalue).")
    p.add_argument("--n-target", type=int, default=100_000, help="Target number of non-sig unitigs to sample.")
    p.add_argument("--af-bin-width", type=float, default=0.01, help="af histogram bin width for matching.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True, help="Output nonsig_hits.tsv.")
    args = p.parse_args(argv)

    hits = pd.read_csv(args.hits_tsv, sep="\t", usecols=["variant", "af", "lrt-pvalue"], low_memory=False)
    hit_variants = set(hits["variant"].astype(str))
    sig_threshold = float(hits["lrt-pvalue"].max())  # least-significant hit → non-sig is anything above it
    print(f"hits: {len(hits)}, least-significant hit lrt-pvalue = {sig_threshold:.3g}", file=sys.stderr)

    assoc = pd.read_csv(args.assoc, sep="\t", usecols=["variant", "af", "lrt-pvalue"], low_memory=False)
    assoc["variant"] = assoc["variant"].astype(str)
    nonsig = assoc[(assoc["lrt-pvalue"] > sig_threshold) & (assoc["af"] > 0) & (assoc["af"] <= 1)
                   & (~assoc["variant"].isin(hit_variants))].copy()
    print(f"assoc: {len(assoc)}, non-significant pool: {len(nonsig)}", file=sys.stderr)

    edges = np.arange(0.0, 1.0 + args.af_bin_width, args.af_bin_width)
    hit_bin = pd.cut(hits["af"], edges, include_lowest=True)
    non_bin = pd.cut(nonsig["af"], edges, include_lowest=True)
    hit_counts = hit_bin.value_counts()
    scale = args.n_target / len(hits)
    rng = np.random.default_rng(args.seed)

    picks: list[pd.DataFrame] = []
    short = 0
    for b, n_hit in hit_counts.items():
        want = int(round(n_hit * scale))
        if want == 0:
            continue
        pool = nonsig[non_bin == b]
        if len(pool) <= want:
            picks.append(pool)
            short += max(0, want - len(pool))
        else:
            picks.append(pool.sample(n=want, random_state=rng.integers(1 << 31)))
    sample = pd.concat(picks, ignore_index=True)[["variant", "af"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.out, sep="\t", index=False)
    print(f"wrote {args.out}: {len(sample)} af-matched non-sig unitigs "
          f"(short by {short} where the pool was thin)", file=sys.stderr)


if __name__ == "__main__":
    main()

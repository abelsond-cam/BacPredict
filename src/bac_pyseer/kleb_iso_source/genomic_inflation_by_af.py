r"""Genomic-inflation λ stratified by allele frequency — a GWAS calibration check.

The genomic-inflation factor λ = median(observed χ²) / median(null χ²) summarises whether the *bulk*
of test statistics is calibrated (λ≈1), conservative (λ<1) or inflated (λ>1). Because χ² = isf(p, 1)
is monotonic in p, λ = ``chi2.isf(median p, 1) / chi2.isf(0.5, 1)`` — one robust median per group.

Computing λ **within allele-frequency bins** diagnoses *where* any miscalibration lives, which a single
genome-wide λ hides:

* roughly uniform across af  → consistent with polygenicity (or a global mis-scaling);
* inflated specifically at **common** af  → residual population structure tagged by common
  (lineage-defining) markers that the kinship has not absorbed;
* **deflated** (λ<1)  → a conservative correction — e.g. an LMM kinship built from the *same* features
  being tested (the tested feature is partly absorbed by the random effect).

Runs on any pyseer ``.assoc`` (variant or unitig); pass several as ``label=path`` to tabulate them
together. Output: a tidy TSV (``contrast, af_bin, n, lambda``). See ``genomic_inflation_by_af.tsv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

DEFAULT_BINS: tuple[tuple[float, float], ...] = ((0.01, 0.05), (0.05, 0.20), (0.20, 0.50), (0.50, 1.0))
_NULL_MED = float(chi2.isf(0.5, 1))  # median of the 1-df χ² null ≈ 0.4549


def lambda_gc(pvals: np.ndarray) -> float:
    """Genomic-control λ from a set of p-values (1-df), via the monotonic median transform."""
    p = np.clip(np.asarray(pvals, dtype=float), 1e-300, 1.0)
    p = p[~np.isnan(p)]
    if len(p) == 0:
        return float("nan")
    return float(chi2.isf(np.median(p), 1) / _NULL_MED)


def analyse(path: Path, pval_col: str, bins: tuple[tuple[float, float], ...]) -> list[dict]:
    """Per-af-bin (and overall) λ for one ``.assoc`` file."""
    d = pd.read_csv(path, sep="\t", usecols=["af", pval_col])
    af = pd.to_numeric(d["af"], errors="coerce").to_numpy()
    p = pd.to_numeric(d[pval_col], errors="coerce").to_numpy()
    rows = [{"af_bin": "overall", "n": int(np.isfinite(p).sum()), "lambda": round(lambda_gc(p), 3)}]
    for lo, hi in bins:
        m = (af > lo) & (af <= hi)
        rows.append({"af_bin": f"{lo:.2f}-{hi:.2f}", "n": int(m.sum()), "lambda": round(lambda_gc(p[m]), 3)})
    return rows


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assoc", nargs="+", required=True, help="One or more pyseer .assoc as <label>=<path>.")
    p.add_argument("--pval-col", default="lrt-pvalue")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    out_rows: list[dict] = []
    for spec in args.assoc:
        label, _, path = str(spec).partition("=")
        if not path:
            raise SystemExit(f"--assoc entries must be <label>=<path>, got {spec!r}")
        for r in analyse(Path(path), args.pval_col, DEFAULT_BINS):
            out_rows.append({"contrast": label, **r})
    df = pd.DataFrame(out_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, sep="\t", index=False)
    print(df.to_string(index=False), file=sys.stderr)
    print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

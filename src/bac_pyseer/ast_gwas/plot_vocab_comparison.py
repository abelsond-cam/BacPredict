"""The two figures for the vocabulary rebuild: a paired scatter and a caterpillar of deltas.

Both read the CSV written by :mod:`bac_pyseer.ast_gwas.compare_vocab_arms`, so neither recomputes a
metric and neither can drift from the table it illustrates.

**Scatter — old vs new AUROC against y=x.** A table of 22 deltas says how big the shift is; the
scatter says *where* it falls. A uniform downward shift and one concentrated in the low-signal drugs
mean different things, and a mean delta cannot tell them apart. Points below the diagonal are drugs
the full-cohort arm scored higher.

**Caterpillar — delta with its paired bootstrap CI.** Ordered by delta, with a line at zero. This is
the figure that stops a point estimate being read as a result: with per-drug holdouts in the
hundreds, a delta of a few thousandths is routinely a tie, and only the interval shows that. Drugs
whose CI excludes zero are drawn filled; the rest are hollow, so "separates from zero" is visible
without reading numbers off the axis.

**Neither figure attributes the delta.** Three things move between the arms — representation
advantage, out-of-vocabulary penalty, and the ``MIN_SAMP`` rebase, which pushes the opposite way —
so the axis label says "difference", not "leakage". See ``compare_vocab_arms``'s module docstring.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FULL_COLOUR = "#7B3294"     # matches the ladder's purple unitig rung


def _separates(table: pd.DataFrame) -> np.ndarray:
    """``separates_from_zero`` as a real boolean array, treating missing as **not** separating.

    ``paired_delta_ci`` omits this key when every bootstrap resample was single-class, so the column
    can be absent or NaN. ``NaN`` is truthy, so a naive ``.to_numpy(bool)`` would draw a degenerate
    interval as a drug whose CI excludes zero — the strongest claim the figure can make, asserted
    from a CI that could not be computed.
    """
    if "separates_from_zero" not in table.columns:
        return np.zeros(len(table), dtype=bool)
    col = table["separates_from_zero"]
    if col.dtype == bool:
        return col.to_numpy()
    # A CSV round-trip yields "True"/"False" strings, or float NaN where the key was absent.
    return col.astype(str).str.strip().str.lower().isin(("true", "1", "1.0")).to_numpy()


def paired_scatter(table: pd.DataFrame, out_path: Path, *, organism: str = "kp") -> Path:
    """Old vs new holdout AUROC, one point per drug, against the y=x line."""
    x = table["full_cohort_auroc"].to_numpy(float)
    y = table["trainval_vocab_auroc"].to_numpy(float)
    sep = _separates(table)

    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    # Square, equal limits, y=x: without those a shift can be manufactured by the aspect ratio alone.
    lo = float(min(x.min(), y.min())) - 0.02
    hi = float(max(x.max(), y.max())) + 0.02
    ax.plot([lo, hi], [lo, hi], color="0.55", lw=1.0, ls="--", zorder=1, label="no change (y = x)")
    ax.scatter(x[~sep], y[~sep], s=52, facecolors="none", edgecolors=FULL_COLOUR, lw=1.4,
               zorder=3, label="CI includes 0")
    ax.scatter(x[sep], y[sep], s=52, color=FULL_COLOUR, zorder=3, label="CI excludes 0")
    for xi, yi, drug in zip(x, y, table["drug"], strict=True):
        ax.annotate(drug, (xi, yi), fontsize=6.0, color="0.35",
                    xytext=(3.5, 3.0), textcoords="offset points")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("holdout AUROC — full-cohort vocabulary")
    ax.set_ylabel("holdout AUROC — train+validate vocabulary")
    ax.set_title(f"{organism} unitig arm: does the result survive a holdout-free vocabulary?", fontsize=10.5)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def delta_caterpillar(table: pd.DataFrame, out_path: Path, *, organism: str = "kp") -> Path:
    """Per-drug AUROC difference with its paired bootstrap CI, ordered by delta."""
    t = table.sort_values("delta").reset_index(drop=True)
    y = np.arange(len(t))
    delta = t["delta"].to_numpy(float)
    lo = t["ci_lo"].to_numpy(float)
    hi = t["ci_hi"].to_numpy(float)
    sep = _separates(t)

    fig, ax = plt.subplots(figsize=(7.2, max(4.0, 0.32 * len(t) + 1.6)))
    ax.axvline(0.0, color="0.35", lw=1.0, ls="--", zorder=1)
    # Bars are drawn from the CI bounds, not from a symmetric +/- : the paired bootstrap interval is
    # not required to be symmetric about the observed delta, and forcing it to be would misreport it.
    ax.hlines(y, lo, hi, color="0.55", lw=1.6, zorder=2)
    ax.scatter(delta[~sep], y[~sep], s=42, facecolors="none", edgecolors=FULL_COLOUR, lw=1.4, zorder=3)
    ax.scatter(delta[sep], y[sep], s=42, color=FULL_COLOUR, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(t["drug"], fontsize=8)
    ax.set_ylim(-0.8, len(t) - 0.2)
    ax.set_xlabel("AUROC difference (full-cohort − train+validate), 95% paired bootstrap CI")
    ax.set_title(
        f"{organism}: per-drug difference between vocabularies\n"
        "positive = the full-cohort arm scored higher · filled = CI excludes 0",
        fontsize=10.5,
    )
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run(comparison_csv: Path, out_dir: Path, *, organism: str = "kp") -> int:
    """Render both figures from a ``compare_vocab_arms`` CSV."""
    table = pd.read_csv(comparison_csv)
    missing = [c for c in ("drug", "full_cohort_auroc", "trainval_vocab_auroc", "delta", "ci_lo", "ci_hi")
               if c not in table.columns]
    if missing:
        raise SystemExit(f"{comparison_csv} lacks {missing} — was it written by compare_vocab_arms?")
    if table.empty:
        raise SystemExit(f"{comparison_csv} has no rows")
    out_dir.mkdir(parents=True, exist_ok=True)
    a = paired_scatter(table, out_dir / f"vocab_paired_scatter_{organism}.png", organism=organism)
    b = delta_caterpillar(table, out_dir / f"vocab_delta_caterpillar_{organism}.png", organism=organism)
    n_sep = int(_separates(table).sum())
    print(f"wrote {a}\nwrote {b}")
    print(f"{len(table)} drug(s); {n_sep} with a CI excluding zero")
    return 0


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--comparison-csv", type=Path, required=True, help="output of compare_vocab_arms")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--organism", default="kp")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    sys.exit(run(args.comparison_csv, args.out_dir, organism=args.organism))


if __name__ == "__main__":
    main()

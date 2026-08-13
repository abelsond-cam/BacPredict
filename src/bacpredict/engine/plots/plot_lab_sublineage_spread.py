r"""Predicted-invasiveness spread within each sublineage, for picking isolates to test in vivo.

The point of the figure is the *within-clone* spread, not the between-clone ordering. An animal
experiment that compares a high-scoring SL258 isolate against a low-scoring SL258 isolate tests the
model; one that compares SL258 against SL107 mostly tests lineage. So genomes are drawn as
individual points per sublineage, and the target lineage gets its own panel with the extremes
labelled by ``LabID`` — the identifier that names a physical tube.

Usage
-----
    python -m bacpredict.engine.plots.plot_lab_sublineage_spread \
        --predictions lab_collection_invasion_predictions.csv \
        --prob-column bacformer_pooled_prob \
        --focus SL258 --out lab_sublineage_spread.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

POINT_COLOUR = "#2b6cb0"
FOCUS_COLOUR = "#c0392b"
LABELLED_COLOUR = "#2e7d32"   # genomes with a known blood/faeces label
UNKNOWN_SL = "unknown"


def _jitter(n: int, rng: np.random.Generator, width: float = 0.16) -> np.ndarray:
    """Vertical jitter so overlapping genomes stay countable."""
    return rng.uniform(-width, width, n)


def plot_spread(df: pd.DataFrame, prob_col: str, out_path: Path, *, focus: str | None = "SL258",
                min_n: int = 10, seed: int = 0, group_col: str = "Sublineage") -> dict:
    """Strip plot of per-genome predicted probability by sublineage, plus a focus panel."""
    rng = np.random.default_rng(seed)
    data = df[df[prob_col].notna() & df[group_col].notna()].copy()
    data = data[data[group_col] != UNKNOWN_SL]

    sizes = data[group_col].value_counts()
    keep = sizes[sizes >= min_n].index.tolist()
    kept = data[data[group_col].isin(keep)].copy()
    if kept.empty:
        raise SystemExit(f"no sublineage reaches n >= {min_n}")
    order = kept.groupby(group_col)[prob_col].median().sort_values().index.tolist()

    has_focus = focus is not None and focus in set(kept[group_col])
    n_panels = 2 if has_focus else 1
    fig, axes = plt.subplots(
        1, n_panels, figsize=(7.2 + 4.2 * (n_panels - 1), max(4.0, 0.34 * len(order) + 2.0)),
        squeeze=False, gridspec_kw={"width_ratios": [3, 2] if has_focus else [1]},
    )

    ax = axes[0][0]
    for i, sl in enumerate(order):
        g = kept[kept[group_col] == sl]
        colour = FOCUS_COLOUR if sl == focus else POINT_COLOUR
        ax.scatter(g[prob_col], i + _jitter(len(g), rng), s=14, c=colour, alpha=0.6,
                   edgecolors="none", zorder=3)
        ax.plot([g[prob_col].median()] * 2, [i - 0.3, i + 0.3], c="black", lw=1.6, zorder=4)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{sl}  (n={int(sizes[sl])})" for sl in order])
    ax.set_xlabel("predicted P(blood | genome)")
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color="#4a5568", ls=":", lw=1)
    ax.set_title(f"Predicted invasiveness within sublineage (n ≥ {min_n})\n"
                 "black bar = median; the SPREAD is what an in-vivo test exploits", fontsize=10)
    ax.grid(axis="x", alpha=0.25, ls=":")
    ax.spines[["top", "right"]].set_visible(False)

    summary: dict = {"n_sublineages": len(order), "n_genomes": int(len(kept))}
    if has_focus:
        fax = axes[0][1]
        g = kept[kept[group_col] == focus].sort_values(prob_col).reset_index(drop=True)
        labelled = g["true_label"].notna() if "true_label" in g.columns else pd.Series(False, index=g.index)
        fax.scatter(g.loc[~labelled, prob_col], np.arange(len(g))[~labelled.to_numpy()],
                    s=26, c=FOCUS_COLOUR, alpha=0.75, edgecolors="none", label="no lab label")
        if labelled.any():
            fax.scatter(g.loc[labelled, prob_col], np.arange(len(g))[labelled.to_numpy()],
                        s=34, c=LABELLED_COLOUR, edgecolors="black", linewidths=0.4,
                        label="known blood/faeces label", zorder=4)
        # Name the extremes — these are the isolates a lab would actually pull off the shelf.
        for pos in list(range(min(3, len(g)))) + list(range(max(0, len(g) - 3), len(g))):
            lab_id = str(g.loc[pos, "LabID"]) if "LabID" in g.columns else str(pos)
            fax.annotate(lab_id, (g.loc[pos, prob_col], pos), fontsize=7,
                         xytext=(4, 0), textcoords="offset points", va="center")
        fax.set_xlim(0, 1)
        fax.axvline(0.5, color="#4a5568", ls=":", lw=1)
        fax.set_xlabel("predicted P(blood | genome)")
        fax.set_ylabel(f"{focus} isolates, ranked")
        fax.set_title(f"{focus} (n={len(g)}): spread {g[prob_col].min():.2f}–{g[prob_col].max():.2f}",
                      fontsize=10)
        fax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
        fax.grid(axis="x", alpha=0.25, ls=":")
        fax.spines[["top", "right"]].set_visible(False)
        summary["focus"] = {
            "sublineage": focus, "n": int(len(g)),
            "min": float(g[prob_col].min()), "max": float(g[prob_col].max()),
            "median": float(g[prob_col].median()), "sd": float(g[prob_col].std(ddof=1)),
        }

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return summary


def _main_cli() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--prob-column", type=str, required=True)
    p.add_argument("--group-column", type=str, default="Sublineage")
    p.add_argument("--focus", type=str, default="SL258", help="Lineage to give its own panel.")
    p.add_argument("--min-n", type=int, default=10)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = pd.read_csv(args.predictions, low_memory=False)
    summary = plot_spread(df, args.prob_column, args.out, focus=args.focus, min_n=args.min_n,
                          group_col=args.group_column)
    print(summary)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    _main_cli()

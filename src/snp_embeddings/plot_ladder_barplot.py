"""Bar-plot the TB-rifampin AST localization ladder, coloured by method family.

Reads the assembled metrics table (`docs/rif_ladder_table.csv`) and renders a sorted bar chart of
AUROC, one bar per method, **coloured by family**: Bacformer = blue, ESM = maroon, one-hot = red,
mix (Bacformer ⊕ ESM) = purple. The headline read: the **concat** of the ESM-C rpoB vector with the
Bacformer genome-mean (purple) tops the ladder — above fine-tuned Bacformer (blue) and one-hot
mutation alone (red) — i.e. *injecting the causal-gene vector* recovers the signal pooling discards.

Login-node / local CPU only (pure matplotlib over a 7-row CSV). The small inter-method deltas at the
top of the ladder are not yet significance-tested — k-fold × seeds confirms them (see PROGRESS_REPORT).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Family → colour (the user's scheme).
FAMILY_COLOURS = {
    "Bacformer": "#1f77b4",  # blue
    "ESM": "#800000",        # maroon
    "one-hot": "#d62728",    # red
    "mix": "#7e3f9e",        # purple (Bacformer ⊕ ESM)
}
FAMILY_LABEL = {
    "Bacformer": "Bacformer embedding",
    "ESM": "ESM-C embedding",
    "one-hot": "one-hot mutation",
    "mix": "concat (Bacformer ⊕ ESM)",
}


def plot_ladder(csv_path: Path, out_path: Path, *, metric: str = "auroc", ymin: float = 0.75) -> None:
    """Render the ladder bar plot from the metrics CSV, sorted ascending by ``metric``."""
    df = pd.read_csv(csv_path).sort_values(metric).reset_index(drop=True)
    colours = [FAMILY_COLOURS.get(f, "#888888") for f in df["family"]]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = range(len(df))
    ax.bar(x, df[metric], color=colours, edgecolor="black", linewidth=0.7, width=0.72)
    for xi, v in zip(x, df[metric], strict=True):
        ax.text(xi, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["method"], rotation=28, ha="right", fontsize=9.5)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_ylim(ymin, 1.0)
    ax.set_title(
        "TB-rifampin AST — concatenating the causal-gene vector tops the read-out ladder\n"
        "(full eval, n≈6.9k; small top-end deltas pending k-fold × seeds)",
        fontsize=12.5,
    )
    # Family legend.
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, ec="black", lw=0.7) for c in FAMILY_COLOURS.values()]
    ax.legend(handles, [FAMILY_LABEL[k] for k in FAMILY_COLOURS], loc="upper left", fontsize=9.5, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """CLI entry point."""
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=here / "docs" / "rif_ladder_table.csv")
    parser.add_argument("--out", type=Path, default=here / "docs" / "visualisations" / "rif_ladder_barplot.png")
    parser.add_argument("--metric", type=str, default="auroc", choices=["auroc", "auprc", "sensitivity", "specificity"])
    parser.add_argument("--ymin", type=float, default=0.75)
    args = parser.parse_args()
    plot_ladder(args.csv, args.out, metric=args.metric, ymin=args.ymin)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

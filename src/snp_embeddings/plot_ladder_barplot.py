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

# How each method reads out the genome — drawn as bracketed groups separated by vertical dividers.
GROUP_LABEL = {
    "genome_pooled": "genome-pooled embeddings",
    "single_gene": "single-gene (rpoB) features",
    "concat": "concatenated",
}


def _group_boundaries(df: pd.DataFrame) -> list[int]:
    """Indices where the ``group`` column changes between consecutive (already-sorted) rows."""
    if "group" not in df.columns:
        return []
    g = df["group"].to_numpy()
    return [i for i in range(1, len(g)) if g[i] != g[i - 1]]


def _draw_metric_panel(
    ax, df: pd.DataFrame, metric: str, *, ymin: float, show_xticklabels: bool, boundaries: list[int]
) -> None:
    """Render one bar panel of ``metric`` onto ``ax`` (sorted order fixed by the caller).

    A ``<metric>_sd`` column (e.g. ``auroc_sd``), if present, is drawn as a black ±sd error bar over
    each bar — the k-fold × m-seed spread. Rows without a value get 0 (no visible whisker). ``boundaries``
    are bar indices where the read-out group changes; a dashed vertical divider is drawn before each.
    """
    colours = [FAMILY_COLOURS.get(f, "#888888") for f in df["family"]]
    x = range(len(df))
    sd_col = f"{metric}_sd"
    yerr = df[sd_col].fillna(0.0).to_numpy() if sd_col in df.columns else None
    ax.bar(
        x, df[metric], color=colours, edgecolor="black", linewidth=0.7, width=0.72,
        yerr=yerr, error_kw={"ecolor": "black", "elinewidth": 1.0, "capsize": 3.5},
    )
    for b in boundaries:
        ax.axvline(b - 0.5, color="0.35", linestyle="--", linewidth=1.1, alpha=0.8)
    for xi, v in zip(x, df[metric], strict=True):
        ax.text(xi, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_xticks(list(x))
    if show_xticklabels:
        ax.set_xticklabels(df["method"], rotation=28, ha="right", fontsize=9.5)
    else:
        ax.set_xticklabels([])
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_ylim(ymin, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def _annotate_groups(ax, df: pd.DataFrame, boundaries: list[int]) -> None:
    """Label each contiguous read-out group (genome-pooled / single-gene / concat) above its bars."""
    if "group" not in df.columns:
        return
    starts = [0, *boundaries]
    ends = [*boundaries, len(df)]
    # x in data coords, y in axes coords → a header band just above the panel, clear of the value labels.
    trans = ax.get_xaxis_transform()
    for s, e in zip(starts, ends, strict=True):
        key = df["group"].iloc[s]
        ax.text(
            (s + e - 1) / 2, 1.03, GROUP_LABEL.get(key, key), transform=trans, ha="center", va="bottom",
            clip_on=False, fontsize=9.5, fontstyle="italic", color="0.25",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.7", "alpha": 0.85},
        )


def plot_ladder(csv_path: Path, out_path: Path, *, sort_metric: str = "auroc") -> None:
    """Two-panel ladder bar plot — AUROC (top) and AUPRC (bottom), one bar per method, family-coloured.

    Both panels share the same method ordering (ascending by ``sort_metric``) so the bars line up. The
    headline read: the **concat** of the ESM-C rpoB vector with the Bacformer genome-mean (purple) tops
    both panels — above fine-tuned Bacformer (blue) and one-hot mutation alone (red).
    """
    df = pd.read_csv(csv_path).sort_values(sort_metric).reset_index(drop=True)
    boundaries = _group_boundaries(df)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11.5, 9.5), sharex=True)
    _draw_metric_panel(ax_top, df, "auroc", ymin=0.75, show_xticklabels=False, boundaries=boundaries)
    _draw_metric_panel(ax_bot, df, "auprc", ymin=0.60, show_xticklabels=True, boundaries=boundaries)
    _annotate_groups(ax_top, df, boundaries)

    # Family legend on the lower panel's empty upper-left, so the top panel's upper-left is free for the
    # genome-pooled group label.
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, ec="black", lw=0.7) for c in FAMILY_COLOURS.values()]
    ax_bot.legend(handles, [FAMILY_LABEL[k] for k in FAMILY_COLOURS], loc="upper left", fontsize=9.5, framealpha=0.95)
    fig.suptitle(
        "Comparing Bacformer predictions of Rif Resistance in TB with ESM and concatenated models",
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """CLI entry point."""
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=here / "docs" / "rif_ladder_table.csv")
    parser.add_argument("--out", type=Path, default=here / "docs" / "visualisations" / "rif_ladder_barplot.png")
    args = parser.parse_args()
    plot_ladder(args.csv, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

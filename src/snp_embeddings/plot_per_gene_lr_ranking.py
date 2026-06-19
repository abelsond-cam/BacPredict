"""Bar-plot the per-gene logistic-regression ranking — which single gene's ESM-C vector predicts AST.

Reads a per-gene LR ranking table (``per_gene_lr_<drug>.csv`` from ``build_per_gene_lr_store``) and
renders the top-N genes by out-of-fold train AUROC, descending (highest on the left). This is the
auto-discovery step that picks the causal-gene candidate we concat onto the Bacformer mean: for
rifampin the top gene is **rpoB** — highlighted as "our pick" — and the genes right behind it are the
*other* drugs' canonical resistance genes (embB, katG, pncA, gyrA, rpsL), which surface here through
TB's multi-drug co-resistance. One picture of "how we choose the gene to inject".

Login-node / local CPU only (pure matplotlib over a small CSV).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PICK_COLOUR = "#7e3f9e"   # purple — matches the concat bars on the ladder ("the gene we inject")
OTHER_COLOUR = "#9aa3ad"  # muted grey — the rest of the ranking

# AST column name (US) → display / directory name (the proper drug name used for tb_<drug>/ dirs).
DRUG_DISPLAY = {"rifampin": "rifampicin"}


def display_name(drug: str) -> str:
    """The proper drug name used in titles and the per-drug ``tb_<drug>/`` visualisation dir."""
    return DRUG_DISPLAY.get(drug, drug)


def plot_ranking(csv_path: Path, out_path: Path, *, drug: str | None = None, top_n: int = 10) -> None:
    """Top-``top_n`` genes by out-of-fold LR AUROC, descending; the top gene highlighted as our pick."""
    df = pd.read_csv(csv_path)
    auroc_cols = [c for c in df.columns if c.startswith("lr_auroc_")]
    if not auroc_cols:
        raise ValueError(f"{csv_path} has no lr_auroc_<drug> column — not a per-gene ranking table.")
    auroc_col = f"lr_auroc_{drug}" if drug else auroc_cols[0]
    drug_name = display_name(auroc_col.removeprefix("lr_auroc_"))

    top = df.sort_values(auroc_col, ascending=False).head(top_n).reset_index(drop=True)
    colours = [PICK_COLOUR if i == 0 else OTHER_COLOUR for i in range(len(top))]

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    x = range(len(top))
    ax.bar(x, top[auroc_col], color=colours, edgecolor="black", linewidth=0.7, width=0.74)
    for xi, v in zip(x, top[auroc_col], strict=True):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax.axhline(0.5, color="0.6", linestyle=":", linewidth=1.0)  # chance
    ax.text(len(top) - 0.5, 0.505, "chance", ha="right", va="bottom", fontsize=8, color="0.5")

    ax.set_xticks(list(x))
    ax.set_xticklabels(top["gene_name"], rotation=30, ha="right", fontsize=10, fontstyle="italic")
    ax.set_ylabel("out-of-fold train AUROC", fontsize=12)
    ax.set_ylim(0.45, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    pick = top["gene_name"].iloc[0]
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=PICK_COLOUR, ec="black", lw=0.7),
        plt.Rectangle((0, 0), 1, 1, color=OTHER_COLOUR, ec="black", lw=0.7),
    ]
    ax.legend(handles, [f"our pick: {pick} (injected gene)", "other ranked genes"],
              loc="upper right", fontsize=9.5, framealpha=0.95)
    ax.set_title(
        f"Per-gene LR ranking ({drug_name}): which single gene's ESM-C vector predicts resistance?",
        fontsize=12.5,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """CLI entry point."""
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=here / "docs" / "per_gene_lr_rifampin.csv")
    parser.add_argument("--drug", type=str, default="rifampin", help="AST drug column to rank by.")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--out", type=Path, default=None,
                        help="Output PNG (default: docs/visualisations/tb_<drug>/<drug>_per_gene_lr_ranking.png).")
    args = parser.parse_args()
    disp = display_name(args.drug)
    out = args.out or here / "docs" / "visualisations" / f"tb_{disp}" / f"{disp}_esm_lr_screen_histogram.png"
    plot_ranking(args.csv, out, drug=args.drug, top_n=args.top_n)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

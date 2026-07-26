"""Bar-plot the Kp per-gene ESM-C LR ranking — which single gene's ESM-C vector predicts AST.

The Kp analogue of ``bacpredict.engine.plots.plot_segment_ranking``. Reads a per-gene ranking table
(``per_gene_lr_<drug>.csv`` from ``bacpredict.engine.gene_lr.build_per_gene_lr_store`` driven on the Kp cohort) and
renders the top-N genes by out-of-fold train AUROC, ascending (highest on the right). This is the
auto-discovery step that picks the causal gene the concat probe injects: for the chromosomal/intrinsic
drugs we expect the known biology to surface (colistin → *pmrB*/*mgrB*/*phoQ*, azithromycin →
*mph*/efflux, tetracycline → *tet*/*ramR*/efflux, ciprofloxacin → *gyrA*/*parC*). The full Kleborate
determinant ceiling is drawn as a red reference line so the best single ESM gene reads against "all
Kleborate determinants combined". Login-node / local CPU only (matplotlib over a small CSV).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import to_rgba

from bacpredict.engine.config import visualisations_dir

logger = logging.getLogger(__name__)

ALL_KEY = "__ALL_Kleborate__"
PICK_COLOUR = "#7e3f9e"        # purple — ESM single-gene (family colour, consistent across plots)
CEILING_COLOUR = "#c0392b"     # red — the Kleborate determinant ceiling reference line


def plot_ranking(csv_path: Path, out_path: Path, *, drug: str, top_n: int = 12,
                 kleborate_ceiling: float | None = None, min_n_eval: int | None = None) -> None:
    """Top-``top_n`` genes by out-of-fold LR AUROC, ascending (highest on the right); top gene = our pick.

    ``min_n_eval`` gates the screen to genes carried by **more than** that many evaluate-set genomes
    (``n_eval``) — the present-embeddings-only, well-powered filter for the *non-imputed* carrier screen
    (needs an eval-holdout ranking). Skips the figure if the gate empties the table.
    """
    df = pd.read_csv(csv_path)
    if min_n_eval is not None and "n_eval" in df.columns:
        df = df[df["n_eval"] > min_n_eval].reset_index(drop=True)
        if df.empty:
            logger.warning("%s: no gene with n_eval > %d — skipping the gated non-imputed screen",
                           Path(csv_path).name, min_n_eval)
            return
    auroc_cols = [c for c in df.columns if c.startswith("lr_auroc_")]
    if not auroc_cols:
        raise ValueError(f"{csv_path} has no lr_auroc_<drug> column — not a per-gene ranking table.")
    auroc_col = f"lr_auroc_{drug}" if f"lr_auroc_{drug}" in df.columns else auroc_cols[0]

    top = (df.sort_values(auroc_col, ascending=False).head(top_n)
           .sort_values(auroc_col, ascending=True).reset_index(drop=True))
    pick_idx = len(top) - 1
    colours = [PICK_COLOUR if i == pick_idx else to_rgba(PICK_COLOUR, 0.5) for i in range(len(top))]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    x = range(len(top))
    ax.bar(x, top[auroc_col], color=colours, edgecolor="black", linewidth=0.7, width=0.74)
    for xi, v in zip(x, top[auroc_col], strict=True):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax.axhline(0.5, color="0.6", linestyle=":", linewidth=1.0)  # chance
    ax.text(-0.4, 0.505, "chance", ha="left", va="bottom", fontsize=8, color="0.5")
    if kleborate_ceiling is not None:
        ax.axhline(kleborate_ceiling, color=CEILING_COLOUR, linestyle="--", linewidth=1.4)
        ax.text((len(top) - 1) / 2, kleborate_ceiling + 0.006,
                f"Ceiling (all Kleborate determinants) = {kleborate_ceiling:.3f}",
                ha="center", va="bottom", fontsize=7.5, color=CEILING_COLOUR)

    ax.set_xticks(list(x))
    ax.set_xticklabels(top["gene_name"], rotation=30, ha="right", fontsize=10, fontstyle="italic")
    ax.set_ylabel("out-of-fold train AUROC", fontsize=12)
    ax.set_ylim(0.45, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    pick = top["gene_name"].iloc[pick_idx]
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=PICK_COLOUR, ec="black", lw=0.7),
        plt.Rectangle((0, 0), 1, 1, color=to_rgba(PICK_COLOUR, 0.5), ec="black", lw=0.7),
        plt.Line2D([0], [0], color=CEILING_COLOUR, linestyle="--", linewidth=1.4),
    ]
    ax.legend(handles, [f"our pick: {pick} — top ESM prediction, injected gene", "other ranked genes",
                        "all Kleborate determinants"], loc="upper left", bbox_to_anchor=(0.01, 0.82),
              fontsize=9.0, framealpha=0.95)
    gate_note = f"  ·  non-imputed screen (>{min_n_eval} eval carriers)" if min_n_eval is not None else ""
    ax.set_title(f"{drug}: Kp per-gene ESM-C LR ranking — which gene's embedding predicts resistance{gate_note}",
                 fontsize=12.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _kleborate_ceiling(kleborate_csv: Path) -> float | None:
    """Read the full ``__ALL_Kleborate__`` AUROC ceiling from a per-drug determinant CSV (or None)."""
    if not kleborate_csv.exists():
        return None
    kdf = pd.read_csv(kleborate_csv)
    row = kdf[kdf["gene_name"] == ALL_KEY]
    return float(row["mut_auroc"].iloc[0]) if not row.empty else None


def main() -> None:
    """CLI entry point."""
    vis = visualisations_dir("kp")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drug", type=str, required=True, help="AST drug column / <drug> dir name.")
    parser.add_argument("--csv", type=Path, required=True, help="per_gene_lr_<drug>.csv from the ranking job.")
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--kleborate-csv", type=Path, default=None,
                        help="Default: kp_<drug>/kleborate_determinant_lr_<drug>.csv (draws the ceiling line).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Default: kp_<drug>/<drug>_esm_per_gene_ranking.png.")
    parser.add_argument("--min-n-eval", type=int, default=None,
                        help="Gate the screen to genes with n_eval > this (the non-imputed carrier screen "
                             "over present-embeddings-only, well-powered genes; needs an eval-holdout ranking).")
    args = parser.parse_args()
    drug_dir = vis / args.drug
    kleborate_csv = args.kleborate_csv or drug_dir / f"kleborate_determinant_lr_{args.drug}.csv"
    out = args.out or drug_dir / f"{args.drug}_esm_per_gene_ranking.png"
    plot_ranking(args.csv, out, drug=args.drug, top_n=args.top_n,
                 kleborate_ceiling=_kleborate_ceiling(kleborate_csv), min_n_eval=args.min_n_eval)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

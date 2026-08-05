"""Bar-plot the per-gene logistic-regression ranking — which single gene's ESM-C vector predicts AST.

Reads a per-gene LR ranking table (``per_gene_lr_<drug>.csv`` from ``build_per_gene_lr_store``) and
renders the top-N genes by out-of-fold train AUROC, ascending (highest on the right, to match the ladder). This is the
auto-discovery step that picks the causal-gene candidate we concat onto the Bacformer mean: for
rifampin the top gene is **rpoB** — highlighted as "our pick" — and the genes right behind it are the
*other* drugs' canonical resistance genes (embB, katG, pncA, gyrA, rpsL), which surface here through
TB's multi-drug co-resistance. One picture of "how we choose the gene to inject".

Login-node / local CPU only (pure matplotlib over a small CSV).
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
from bacpredict.engine.plots.display_labels import display_name

logger = logging.getLogger(__name__)

PICK_COLOUR = "#7e3f9e"   # purple — ESM single-gene (the family colour, consistent across plots)
OTHER_COLOUR = "#9aa3ad"  # muted grey — the rest of the ranking
WHO_LINE_COLOUR = "#d62728"  # red — the WHO one-hot family (the reference line)

# AST column name (US) → display / directory name (the proper drug name used for tb_<drug>/ dirs).




def plot_ranking(csv_path: Path, out_path: Path, *, drug: str | None = None, top_n: int = 10,
                 who_onehot_auroc: float | None = None, min_n_eval: int | None = None,
                 source_label: str = "ESM", annotation_label: str = "Prokka",
                 ceiling_label: str = "all WHO mutations") -> None:
    """Top-``top_n`` genes by out-of-fold LR AUROC, ascending (highest on the right); top gene = our pick.

    ``who_onehot_auroc`` (the full catalogue one-hot ceiling for this drug) is drawn as a red reference
    line, so the best single embedded gene can be read against the combined catalogue. ``source_label``
    (embedding source: "ESM"/"baclm"), ``annotation_label`` (gene caller: "Prokka"/"Bakta") and
    ``ceiling_label`` ("all WHO mutations"/"all CARD determinants") make the labels organism-agnostic — TB
    keeps the WHO/ESM/Prokka defaults, Kp passes baclm/Bakta/CARD. ``min_n_eval`` gates the screen to genes
    carried by **more than** that many evaluate-set genomes (``n_eval``) — the present-embeddings-only,
    well-powered filter for the *non-imputed* carrier screen (needs an eval-holdout ranking, which carries
    ``n_eval``). Skips the figure if the gate empties the table.
    """
    df = pd.read_csv(csv_path)
    if min_n_eval is not None and "n_eval" in df.columns:
        df = df[df["n_eval"] > min_n_eval].reset_index(drop=True)
        if df.empty:
            logger.warning("%s: no gene with n_eval > %d — skipping the gated non-imputed screen",
                           Path(csv_path).name, min_n_eval)
            return
    sel_cols = [c for c in df.columns if c.startswith("lr_auroc_")]
    if not sel_cols:
        raise ValueError(f"{csv_path} has no lr_auroc_<drug> column — not a per-gene ranking table.")
    sel_col = f"lr_auroc_{drug}" if drug else sel_cols[0]     # SELECT/order on out-of-fold TRAIN (leakage-free)
    drug_key = sel_col.removeprefix("lr_auroc_")
    disp_col = f"eval_auroc_{drug_key}"                        # DISPLAY the deployment HOLDOUT
    drug_name = display_name(drug_key)
    if disp_col not in df.columns:
        raise ValueError(f"{csv_path} has no {disp_col} column — need the holdout AUROC to display.")
    df = df[df[disp_col].notna()].reset_index(drop=True)       # only genes with a held-out number

    # Ascending by train-OOF (pick = rightmost = the ladder's injected gene); bar HEIGHTS are the holdout,
    # so the bars need not be monotonic — that gap between selection and holdout is honest.
    top = (df.sort_values(sel_col, ascending=False).head(top_n)
           .sort_values(sel_col, ascending=True).reset_index(drop=True))
    pick_idx = len(top) - 1
    # All bars purple (embedding family): the pick solid, the rest the same purple at alpha 0.5.
    colours = [PICK_COLOUR if i == pick_idx else to_rgba(PICK_COLOUR, 0.5) for i in range(len(top))]

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    x = range(len(top))
    ax.bar(x, top[disp_col], color=colours, edgecolor="black", linewidth=0.7, width=0.74)
    for xi, v in zip(x, top[disp_col], strict=True):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax.axhline(0.5, color="0.6", linestyle=":", linewidth=1.0)  # chance
    ax.text(-0.4, 0.505, "chance", ha="left", va="bottom", fontsize=8, color="0.5")
    if who_onehot_auroc is not None:
        ax.axhline(who_onehot_auroc, color=WHO_LINE_COLOUR, linestyle="--", linewidth=1.4)
        ax.text((len(top) - 1) / 2, who_onehot_auroc + 0.006,
                f"Ceiling combining {ceiling_label} = {who_onehot_auroc:.3f}",
                ha="center", va="bottom", fontsize=7.5, color=WHO_LINE_COLOUR)

    ax.set_xticks(list(x))
    ax.set_xticklabels(top["gene_name"], rotation=30, ha="right", fontsize=10, fontstyle="italic")
    ax.set_ylabel("deployment-holdout AUROC (ordered by train-OOF)", fontsize=12)
    ax.set_ylim(0.45, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    pick = top["gene_name"].iloc[pick_idx]
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=PICK_COLOUR, ec="black", lw=0.7),
        plt.Rectangle((0, 0), 1, 1, color=to_rgba(PICK_COLOUR, 0.5), ec="black", lw=0.7),
        plt.Line2D([0], [0], color=WHO_LINE_COLOUR, linestyle="--", linewidth=1.4),
    ]
    ax.legend(handles, [f"our pick: {pick} — top {source_label} prediction, injected gene",
                        "other ranked genes", f"{ceiling_label} (one-hot)"], loc="upper left",
              bbox_to_anchor=(0.01, 0.82), fontsize=9.0, framealpha=0.95)
    gate_note = f"  ·  non-imputed screen (>{min_n_eval} eval carriers)" if min_n_eval is not None else ""
    ax.set_title(
        f"{drug_name}: {source_label} mean embedding predictions by LR, "
        f"for each gene by {annotation_label} annotation{gate_note}",
        fontsize=12.5,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drug", type=str, default="rifampin", help="AST drug column to rank by.")
    parser.add_argument("--csv", type=Path, default=None,
                        help="per_gene_lr_<drug>.csv (default: visualisations/tb/<drug>/per_gene_lr_<drug>.csv).")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--out", type=Path, default=None,
                        help="Output PNG (default: visualisations/tb/<drug>/<drug>_esm_lr_screen_histogram.png).")
    parser.add_argument("--who-onehot-csv", type=Path, default=None,
                        help="tbprofiler_gene_lr_<drug>.csv — draws its __ALL_WHO_one_hot__ AUROC as a red "
                             "reference line (default: in the same per-drug folder).")
    parser.add_argument("--min-n-eval", type=int, default=None,
                        help="Gate the screen to genes with n_eval > this (the non-imputed carrier screen "
                             "over present-embeddings-only, well-powered genes; needs an eval-holdout ranking).")
    parser.add_argument("--ceiling-key", default="__ALL_WHO_one_hot__",
                        help="row key of the combined-catalogue ceiling in --who-onehot-csv (Kp: __ALL_CARD__).")
    parser.add_argument("--source-label", default="ESM", help='embedding source label (TB "ESM", Kp "baclm").')
    parser.add_argument("--annotation-label", default="Prokka", help='gene caller label (TB "Prokka", Kp "Bakta").')
    parser.add_argument("--ceiling-label", default="all WHO mutations",
                        help='catalogue-ceiling legend label (Kp "all CARD determinants").')
    args = parser.parse_args()
    disp = display_name(args.drug)
    drug_dir = visualisations_dir("tb") / disp  # each drug's data + figures live together
    csv = args.csv or drug_dir / f"per_gene_lr_{args.drug}.csv"
    out = args.out or drug_dir / f"{disp}_esm_lr_screen_histogram.png"
    who_csv = args.who_onehot_csv or drug_dir / f"tbprofiler_gene_lr_{args.drug}.csv"
    who_auroc = None
    if who_csv.exists():
        wdf = pd.read_csv(who_csv)
        row = wdf[wdf["gene_name"] == args.ceiling_key]
        who_auroc = float(row["mut_auroc"].iloc[0]) if not row.empty else None
    plot_ranking(csv, out, drug=args.drug, top_n=args.top_n, who_onehot_auroc=who_auroc,
                 min_n_eval=args.min_n_eval, source_label=args.source_label,
                 annotation_label=args.annotation_label, ceiling_label=args.ceiling_label)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

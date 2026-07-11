"""Per-drug cause histogram: per-gene WHO-mutation LR with the rRNA sites included (hatched).

The companion to the ESM per-gene ranking. Where that plot asks "which gene's *embedding* predicts
resistance?" (protein genes only), this asks "which gene's *mutations* cause it?" — scored by
``tbprofiler_gene_lr`` over the WHO catalogue, so it includes the **rRNA causes (rrs/rrl)** that have no
embedding. Bars are coloured by what we can do with each gene:

- **our pick** (purple) — the top *embeddable* gene, i.e. the protein vector concat actually injects;
- other embeddable genes (grey) — protein causes we *could* inject;
- **rRNA / un-embeddable** (orange, hatched) — rrs/rrl (and any non-CDS cause): the mechanism is real
  and predictive but a protein-only model (ESM-C / current Bacformer) cannot represent it.

A dashed line marks the **full WHO one-hot** AUROC (all of the drug's mutations together — the catalogue
ceiling). For a drug like kanamycin you see rrs towering, hatched, well above our weak embeddable pick:
the gap a nucleotide-aware model would need to close. Login-node / local CPU only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from bacpredict.engine.plots.labels import display_name

# WHO one-hot is the red family (consistent across plots). The top embeddable gene is the "pick" we
# could inject; rRNA / un-embeddable causes are the same red but hatched (real cause, can't be embedded).
ROYAL_RED = "#c0392b"       # duller "royal" red — the WHO one-hot family colour
PICK_COLOUR = ROYAL_RED     # top embeddable WHO cause (the injectable pick)
EMBED_COLOUR = "#9aa3ad"    # grey — other embeddable (protein) genes
RRNA_COLOUR = ROYAL_RED     # hatched — un-embeddable causes (all red, distinguished by hatch)
# Distinct hatch per un-embeddable mechanism category, so the categories read clearly on the chart.
# (The new Bacformer encodes these non-coding regions, so they are the targets to fold in next.)
CATEGORY_HATCH = {"promoter": "xx", "rRNA": "//", "other": ".."}
CATEGORY_LABEL = {
    "promoter": "promoter / non-coding — un-embeddable",
    "rRNA": "rRNA — un-embeddable",
    "other": "coding, not in our embeddings",
}
ALL_KEY = "__ALL_WHO_one_hot__"




def _category(row) -> str:
    """Un-embeddable mechanism category of a site: rRNA / promoter / other-coding, else 'embeddable'."""
    if row["is_rrna"]:
        return "rRNA"
    if row.get("is_noncoding", False):
        return "promoter"
    if not row["embeddable"]:
        return "other"
    return "embeddable"


def _site_label(row) -> str:
    """X-axis label: rRNA is the gene itself (not a promoter); non-coding protein genes get '(promoter)'."""
    gene = str(row["gene_name"])
    if not row["is_rrna"] and row.get("is_noncoding", False):
        return f"{gene} (promoter)"
    return gene


def plot_cause(csv_path: Path, out_path: Path, *, drug: str, top_n: int = 20) -> None:
    """Top-``top_n`` WHO sites (gene × coding/promoter) by mutation LR AUROC; un-embeddable hatched."""
    df = pd.read_csv(csv_path)
    if "site" not in df.columns:        # back-compat with pre-split CSVs
        df["site"] = df["gene_name"]
    full = df[df["gene_name"] == ALL_KEY]
    # Ascending order (smallest left, largest right) to match the ladder; the pick (top embeddable) is rightmost.
    genes = (df[df["gene_name"] != ALL_KEY].sort_values("mut_auroc", ascending=False).head(top_n)
             .sort_values("mut_auroc", ascending=True).reset_index(drop=True))
    cats = [_category(r) for _, r in genes.iterrows()]
    pick = next((i for i in range(len(cats) - 1, -1, -1) if cats[i] == "embeddable"), -1)

    colours, hatches = [], []
    for i, cat in enumerate(cats):
        if cat == "embeddable":
            colour, hatch = (PICK_COLOUR if i == pick else EMBED_COLOUR), ""
        else:
            colour, hatch = RRNA_COLOUR, CATEGORY_HATCH[cat]
        colours.append(colour)
        hatches.append(hatch)

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    x = range(len(genes))
    bars = ax.bar(x, genes["mut_auroc"], yerr=genes["mut_auroc_sd"], capsize=3,
                  color=colours, edgecolor="black", linewidth=0.7, width=0.74,
                  error_kw={"ecolor": "black", "elinewidth": 1.0})
    for b, h in zip(bars, hatches, strict=True):
        if h:
            b.set_hatch(h)
    for i, (v, sd) in enumerate(zip(genes["mut_auroc"], genes["mut_auroc_sd"], strict=True)):
        ax.text(i, v + sd + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    if not full.empty:
        fa = float(full["mut_auroc"].iloc[0])
        ax.axhline(fa, color="black", linestyle="--", linewidth=1.2)
        ax.text((len(genes) - 1) / 2, fa + 0.004,
                f"Ceiling combining all WHO mutations = {fa:.3f}",
                ha="center", va="bottom", fontsize=7.5)
    ax.axhline(0.5, color="0.6", linestyle=":", linewidth=1.0)

    ax.set_xticks(list(x))
    ax.set_xticklabels([_site_label(r) for _, r in genes.iterrows()], rotation=30, ha="right",
                       fontsize=9.5, fontstyle="italic")
    ax.set_ylabel("WHO-mutation LR AUROC (k-fold)", fontsize=12)
    ax.set_ylim(0.45, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=PICK_COLOUR, ec="black", lw=0.7),
               plt.Rectangle((0, 0), 1, 1, color=EMBED_COLOUR, ec="black", lw=0.7)]
    labels = ["our pick - top ESM prediction, injected gene", "other embeddable (protein) gene"]
    for cat in ("promoter", "rRNA", "other"):   # only the un-embeddable categories actually present
        if cat in cats:
            handles.append(plt.Rectangle((0, 0), 1, 1, color=RRNA_COLOUR, ec="black", lw=0.7,
                                         hatch=CATEGORY_HATCH[cat]))
            labels.append(CATEGORY_LABEL[cat])
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.72), fontsize=9, framealpha=0.95)
    ax.set_title(
        f"{display_name(drug)}: WHO mutation prediction from single genes / rna regions, by one hot embedding",
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
    parser.add_argument("--drug", type=str, required=True, help="AST drug column (for the title / dir).")
    parser.add_argument("--csv", type=Path, default=None,
                        help="tbprofiler_gene_lr_<drug>.csv (default: docs/visualisations/tb_<drug>/...).")
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--out", type=Path, default=None,
                        help="Default: docs/visualisations/tb_<drug>/<drug>_WHO_one_hot_histogram.png.")
    args = parser.parse_args()
    disp = display_name(args.drug)
    drug_dir = here / "docs" / "visualisations" / f"tb_{disp}"
    csv = args.csv or drug_dir / f"tbprofiler_gene_lr_{args.drug}.csv"
    out = args.out or drug_dir / f"{disp}_WHO_one_hot_histogram.png"
    plot_cause(csv, out, drug=args.drug, top_n=args.top_n)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

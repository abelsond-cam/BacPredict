"""Per-drug Kp cause histogram: Kleborate determinant LR by mechanism class (HGT vs chromosomal).

The Kp analogue of ``snp_embeddings.plot_cause_histogram``. Reads a per-drug ceiling table
(``kp_<drug>/kleborate_determinant_lr_<drug>.csv`` from :mod:`kleb_ast.kleborate_determinant_lr`) and
draws one bar per Kleborate determinant column, ascending (highest on the right, to match the ladder),
coloured + hatched by **mechanism category** so the HGT-vs-chromosomal split — the programme's central
axis — reads at a glance:

- **acquired gene (HGT)** — teal, solid: a horizontally-acquired gene Bacformer/ESM can embed (the
  catalogue's and Bacformer's shared strength);
- **intrinsic chromosomal gene** — light teal, solid: a chromosomal gene (e.g. intrinsic ``Bla_chr``);
- **chromosomal point mutation / porin truncation / loss-of-function** — red, hatched: the curated but
  *narrow* chromosomal calls; the determinants Kleborate under-catalogues and a protein-mean model loses.

A dashed line marks the **``__ALL_Kleborate__``** ceiling (all of the drug's determinants together).
Where the hatched bars sit far below it (e.g. colistin, azithromycin) the catalogue is blind — and the
deployed Bacformer AUROC (drawn if ``--bacformer-auroc`` is given) sits *above* the ceiling. Login/CPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ALL_KEY = "__ALL_Kleborate__"
HGT = "#2a9d8f"        # teal — acquired (HGT), Bacformer's strength
HGT_LIGHT = "#83c5be"  # light teal — intrinsic chromosomal gene (still embeddable)
CHROM = "#c0392b"      # royal red — chromosomal mutation / truncation (the narrow curated calls)
BACFORMER_COLOUR = "#7e3f9e"  # purple — the deployed Bacformer reference line

# category → (colour, hatch, legend label). Hatched = chromosomal/un-embeddable; solid = embeddable gene.
CATEGORY_STYLE: dict[str, tuple[str, str, str]] = {
    "acquired_hgt": (HGT, "", "acquired gene (HGT)"),
    "chromosomal_coding": (HGT_LIGHT, "", "intrinsic chromosomal gene"),
    "chromosomal_mutation": (CHROM, "//", "chromosomal point mutation"),
    "porin_truncation": (CHROM, "xx", "porin truncation / loss"),
    "truncation_lof": (CHROM, "..", "truncation / loss-of-function"),
}


def plot_cause(csv_path: Path, out_path: Path, *, drug: str, bacformer_auroc: float | None = None) -> None:
    """Bars per Kleborate column by mechanism category, ascending, with the all-determinant ceiling line."""
    df = pd.read_csv(csv_path)
    full = df[df["gene_name"] == ALL_KEY]
    bars = df[df["gene_name"] != ALL_KEY].sort_values("mut_auroc", ascending=True).reset_index(drop=True)

    colours, hatches = [], []
    for cat in bars["category"]:
        colour, hatch, _ = CATEGORY_STYLE.get(cat, ("#9aa3ad", "", "other"))
        colours.append(colour)
        hatches.append(hatch)

    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    x = range(len(bars))
    rendered = ax.bar(x, bars["mut_auroc"], yerr=bars["mut_auroc_sd"], capsize=3,
                      color=colours, edgecolor="black", linewidth=0.7, width=0.74,
                      error_kw={"ecolor": "black", "elinewidth": 1.0})
    for b, h in zip(rendered, hatches, strict=True):
        if h:
            b.set_hatch(h)
    # Value label *inside* the top of each bar (white) — keeps it clear of the ceiling/Bacformer lines
    # even when a single tall bar reaches them.
    for i, v in enumerate(bars["mut_auroc"]):
        ax.text(i, v - 0.012, f"{v:.3f}", ha="center", va="top", fontsize=8.5, fontweight="bold", color="white")

    # Line labels at opposite ends (ceiling far-left, Bacformer far-right) so they never overlap each
    # other, via a blended transform (x = axes fraction, y = data).
    blend = ax.get_yaxis_transform()
    label_bbox = {"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.5}
    if not full.empty:
        fa = float(full["mut_auroc"].iloc[0])
        ax.axhline(fa, color="black", linestyle="--", linewidth=1.2)
        ax.text(0.01, fa + 0.005, f"Ceiling (all Kleborate determinants) = {fa:.3f}",
                transform=blend, ha="left", va="bottom", fontsize=7.5, bbox=label_bbox)
    if bacformer_auroc is not None:
        ax.axhline(bacformer_auroc, color=BACFORMER_COLOUR, linestyle="-.", linewidth=1.5)
        ax.text(0.99, bacformer_auroc + 0.005, f"Finetuned Bacformer mean = {bacformer_auroc:.3f}",
                transform=blend, ha="right", va="bottom", fontsize=7.5, color=BACFORMER_COLOUR, bbox=label_bbox)
    ax.axhline(0.5, color="0.6", linestyle=":", linewidth=1.0)

    ax.set_xticks(list(x))
    ax.set_xticklabels(bars["site"], rotation=30, ha="right", fontsize=9.5)
    ax.set_ylabel("Kleborate-determinant LR AUROC (k-fold)", fontsize=12)
    ax.set_ylim(0.45, 1.03)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    present = list(dict.fromkeys(bars["category"]))  # categories present, in first-seen order
    handles, labels = [], []
    for cat in present:
        if cat not in CATEGORY_STYLE:
            continue
        colour, hatch, label = CATEGORY_STYLE[cat]
        handles.append(plt.Rectangle((0, 0), 1, 1, color=colour, ec="black", lw=0.7, hatch=hatch))
        labels.append(label)
    if bacformer_auroc is not None:
        handles.append(plt.Line2D([0], [0], color=BACFORMER_COLOUR, linestyle="-.", linewidth=1.5))
        labels.append("Finetuned Bacformer mean")
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.99), fontsize=9, framealpha=0.95)
    ax.set_title(f"{drug}: Kp resistance from Kleborate determinants by class (HGT vs chromosomal)", fontsize=12.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _bacformer_auroc(eval_summary: Path | None, drug: str) -> float | None:
    """Look up the deployed Bacformer held-out AUROC for ``drug`` from eval_summary.csv (or None)."""
    if eval_summary is None or not eval_summary.exists():
        return None
    df = pd.read_csv(eval_summary)
    row = df[df["drug"] == drug]
    return float(row["auroc"].iloc[0]) if not row.empty else None


def main() -> None:
    """CLI entry point."""
    here = Path(__file__).resolve().parent
    vis = here / "docs" / "visualisations"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drug", type=str, required=True, help="AST drug column / kp_<drug> dir name.")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Default: docs/visualisations/kp_<drug>/kleborate_determinant_lr_<drug>.csv.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Default: docs/visualisations/kp_<drug>/<drug>_kleborate_cause_histogram.png.")
    parser.add_argument("--eval-summary", type=Path, default=vis / "eval" / "eval_summary.csv",
                        help="eval_summary.csv — draws the deployed Bacformer AUROC reference line.")
    args = parser.parse_args()
    drug_dir = vis / f"kp_{args.drug}"
    csv = args.csv or drug_dir / f"kleborate_determinant_lr_{args.drug}.csv"
    out = args.out or drug_dir / f"{args.drug}_kleborate_cause_histogram.png"
    plot_cause(csv, out, drug=args.drug, bacformer_auroc=_bacformer_auroc(args.eval_summary, args.drug))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

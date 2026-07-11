"""Per-drug Kp cause histogram: Kleborate determinant LR by mechanism class (HGT vs chromosomal).

The Kp analogue of ``bacpredict.engine.plots.plot_cause_histogram``. Reads a per-drug ceiling table
(``kp_<drug>/kleborate_determinant_lr_<drug>.csv`` from :mod:`bacpredict.apps.kleb.kleborate_determinant_lr`) and
draws one bar per Kleborate determinant column, ascending (highest on the right, to match the ladder),
all in **one CARD red**. The HGT-vs-chromosomal split — the programme's central axis — is carried by the
**x-label colour** (matching the per-gene esm-vs-ft plot):

- **red label — acquired / HGT**: a distinct horizontally-acquired gene Bacformer/ESM can embed;
- **blue label — chromosomal**: a SNP / truncation (gyrA/parC QRDR, ompK porin loss, mgrB, …) where the WT
  and mutant share an embedding — so the only question the histogram asks is *which the model picks up*.

A dashed red line marks the **``__ALL_Kleborate__``** ceiling (all determinants together). Where the bars
sit far below it (e.g. colistin, azithromycin) the catalogue is blind — and the best Bacformer read-out
(drawn if given) sits *above* the ceiling. Login/CPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ALL_KEY = "__ALL_Kleborate__"
# Every bar is a CARD/Kleborate determinant → all one red. The HGT-vs-chromosomal split is carried by the
# x-label colour (matching the per-gene esm-vs-ft plot): red = acquired/HGT (a distinct gene), blue =
# chromosomal (a SNP/truncation, where WT and mutant share an embedding). Porin loss folds into chromosomal.
BAR_RED = "#c0392b"             # all determinant bars — one red (matches the ceiling)
ACQUIRED_LABEL = "#c0392b"      # red x-label — acquired / HGT determinant
CHROM_LABEL = "#1f6f8b"         # blue x-label — chromosomal determinant (SNP / truncation)
BACFORMER_COLOUR = "#2e2a7a"    # dark purple — best-Bacformer reference line (matches esm-vs-ft Bacformer)
CEILING_COLOUR = "#c0392b"      # red — the CARD/Kleborate one-hot ceiling (uniform across all plots)
ACQUIRED_CATS = {"acquired_hgt"}  # everything else (coding / mutation / porin / truncation) is chromosomal


def _label_colour(category: str) -> str:
    """Red for acquired/HGT, blue for chromosomal (coding / mutation / porin / truncation)."""
    return ACQUIRED_LABEL if category in ACQUIRED_CATS else CHROM_LABEL


def plot_cause(csv_path: Path, out_path: Path, *, drug: str, bacformer_auroc: float | None = None,
               source_name: str = "Kleborate", all_key: str | None = None, grain: str = "family",
               bacformer_label: str = "Finetuned Bacformer mean") -> None:
    """Bars per determinant (all one red), ascending; x-label colour marks acquired (red) vs chromosomal (blue).

    ``source_name`` ("Kleborate" / "CARD") drives the ceiling-row key (``__ALL_<source_name>__``, unless
    ``all_key`` overrides), the title and the y-label — so the same plotter renders the CARD one-hot
    histogram (:mod:`bacpredict.apps.kleb.card_determinant_lr`) and the Kleborate one. ``bacformer_auroc`` /
    ``bacformer_label`` draw the **best** Bacformer read-out (FT mean, single FT gene, or concat) so the
    line shows Bacformer clearing the catalogue ceiling. On ``grain="allele"`` (many narrow bars) the
    in-bar value labels are dropped and the x-tick labels halved.
    """
    key = all_key or f"__ALL_{source_name}__"
    df = pd.read_csv(csv_path)
    full = df[df["gene_name"] == key]
    bars = df[df["gene_name"] != key].sort_values("mut_auroc", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    x = range(len(bars))
    ax.bar(x, bars["mut_auroc"], yerr=bars["mut_auroc_sd"], capsize=3,
           color=BAR_RED, edgecolor="black", linewidth=0.7, width=0.74,
           error_kw={"ecolor": "black", "elinewidth": 1.0})
    # Value label *inside* the top of each bar (white). Dropped on the dense allele grain — the bars get
    # too narrow for a legible number there (the y-axis carries it instead).
    if grain != "allele":
        val_fs = max(5.5, min(7.5, 75.0 / max(len(bars), 1)))
        for i, v in enumerate(bars["mut_auroc"]):
            ax.text(i, v - 0.012, f"{v:.3f}", ha="center", va="top", fontsize=val_fs,
                    fontweight="bold", color="white")

    # Line labels at opposite ends (ceiling far-left, Bacformer far-right) so they never overlap each
    # other, via a blended transform (x = axes fraction, y = data).
    blend = ax.get_yaxis_transform()
    label_bbox = {"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.5}
    if not full.empty:
        fa = float(full["mut_auroc"].iloc[0])
        ax.axhline(fa, color=CEILING_COLOUR, linestyle="--", linewidth=1.3)
        ax.text(0.5, fa + 0.005, f"Ceiling (all {source_name} determinants) = {fa:.3f}",
                transform=blend, ha="center", va="bottom", fontsize=7.5, color=CEILING_COLOUR, bbox=label_bbox)
    if bacformer_auroc is not None:
        ax.axhline(bacformer_auroc, color=BACFORMER_COLOUR, linestyle="-.", linewidth=1.5)
        ax.text(0.99, bacformer_auroc + 0.005, f"{bacformer_label} = {bacformer_auroc:.3f}",
                transform=blend, ha="right", va="bottom", fontsize=7.5, color=BACFORMER_COLOUR, bbox=label_bbox)
    ax.axhline(0.5, color="0.6", linestyle=":", linewidth=1.0)

    ax.set_xticks(list(x))
    ax.set_xticklabels(bars["site"], rotation=30, ha="right", fontsize=4.75 if grain == "allele" else 9.5)
    for tick, cat in zip(ax.get_xticklabels(), bars["category"], strict=True):
        tick.set_color(_label_colour(str(cat)))
    ax.set_ylabel(f"{source_name}-determinant LR AUROC (k-fold)", fontsize=12)
    ax.set_ylim(0.45, 1.03)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    # bars are all one red; the legend keys the x-label colour (mechanism) + the best-Bacformer line
    handles = [plt.Rectangle((0, 0), 1, 1, color=ACQUIRED_LABEL, ec="black", lw=0.7),
               plt.Rectangle((0, 0), 1, 1, color=CHROM_LABEL, ec="black", lw=0.7)]
    labels = ["label: acquired / HGT", "label: chromosomal (SNP/truncation)"]
    if bacformer_auroc is not None:
        handles.append(plt.Line2D([0], [0], color=BACFORMER_COLOUR, linestyle="-.", linewidth=1.5))
        labels.append(bacformer_label)
    # low-left: the ceiling + Bacformer lines (and their labels) live up at the top, so keep the legend clear
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.62), fontsize=9, framealpha=0.3)
    ax.set_title(f"{drug}: Kp resistance from {source_name} determinants by class (HGT vs chromosomal)",
                 fontsize=12.5)
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


def best_bacformer(summary_csv: Path | None, drug: str) -> tuple[float | None, str | None]:
    """Best *deployable* FT-Bacformer read-out for ``drug`` — ``(auroc, short label)`` or ``(None, None)``.

    Compares the FT genome-mean and the FT mean ⊕ FT gene concat — both genome-level models — and returns
    the stronger, so the reference line shows the best Bacformer clearing the catalogue ceiling. Bare
    single-gene FT probes are *excluded*: their AUROC can ride lineage (e.g. FT GyrA scoring high on
    colistin) rather than mechanism, which would overstate the line.
    """
    if summary_csv is None or not summary_csv.exists():
        return None, None
    summ = pd.read_csv(summary_csv)
    row = summ[summ["drug"] == drug]
    if row.empty:
        return None, None
    row = row.iloc[0]
    cands: list[tuple[float, str]] = []
    if pd.notna(row.get("ft_mean_only_auroc")):
        cands.append((float(row["ft_mean_only_auroc"]), "FT mean"))
    if pd.notna(row.get("ft_concat_best_ft_auroc")):
        cands.append((float(row["ft_concat_best_ft_auroc"]), f"FT mean ⊕ FT {row.get('best_ft_gene', '')}"))
    return max(cands, key=lambda t: t[0]) if cands else (None, None)


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
                        help="eval_summary.csv — fallback deployed Bacformer AUROC if no reliable summary.")
    parser.add_argument("--summary-csv", type=Path,
                        default=vis / "reliable_amr" / "kp_reliable_concat_summary.csv",
                        help="reliable concat summary — for the best-Bacformer (FT mean / concat) reference line.")
    parser.add_argument("--grain", type=str, default="family", choices=["family", "allele"],
                        help="CARD grain — allele drops the in-bar numbers and halves the x-tick labels.")
    parser.add_argument("--source-name", type=str, default="Kleborate",
                        help='Determinant source ("Kleborate" / "CARD") — drives the title, y-label and ceiling key.')
    parser.add_argument("--all-key", type=str, default=None,
                        help="Ceiling-row key override (default __ALL_<source-name>__, e.g. __ALL_CARD__).")
    args = parser.parse_args()
    drug_dir = vis / f"kp_{args.drug}"
    csv = args.csv or drug_dir / f"kleborate_determinant_lr_{args.drug}.csv"
    out = args.out or drug_dir / f"{args.drug}_kleborate_cause_histogram.png"
    bac_auroc, bac_variant = best_bacformer(args.summary_csv, args.drug)
    if bac_auroc is not None:
        bac_label = f"Best Bacformer ({bac_variant})"
    else:
        bac_auroc, bac_label = _bacformer_auroc(args.eval_summary, args.drug), "Finetuned Bacformer mean"
    plot_cause(csv, out, drug=args.drug, bacformer_auroc=bac_auroc, source_name=args.source_name,
               all_key=args.all_key, grain=args.grain, bacformer_label=bac_label)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

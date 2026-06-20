"""Headline Kp figure: the Kleborate determinant ceiling vs the deployed Bacformer, across all drugs.

For every drug with a per-drug ceiling table (``kp_<drug>/kleborate_determinant_lr_<drug>.csv``) this
pulls the ``__ALL_Kleborate__`` ceiling AUROC and the deployed-Bacformer held-out AUROC (eval_summary.csv)
and draws a **dumbbell** per drug — ceiling marker ● and Bacformer marker ◆ joined by a line — sorted by
the gap (Bacformer − ceiling). It makes the three-regime story immediate:

- **Bacformer ≫ catalogue** (gap > +0.05): the chromosomal/intrinsic weak drugs Kleborate is blind to
  (azithromycin, colistin, tetracycline) — Bacformer reads resistance no determinant catalogue captures;
- **ties** (|gap| ≤ 0.05): HGT-driven + well-catalogued drugs (β-lactams, aminoglycosides, FQ) where the
  catalogue and Bacformer agree;
- **catalogue > Bacformer** (gap < −0.05): the determinant one-hot is the stronger read-out.

Also writes ``kp_ceiling_vs_bacformer.csv`` (drug, ceiling, bacformer, gap, regime). Login/CPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ALL_KEY = "__ALL_Kleborate__"
CEILING_COLOUR = "#c0392b"    # red — Kleborate determinant ceiling (dumbbell)
BACFORMER_COLOUR = "#7e3f9e"  # purple — deployed Bacformer (dumbbell)
BEATS_COLOUR = "#2a9d8f"      # teal — Bacformer beats the catalogue (the headline regime)
TIE_COLOUR = "#9aa3ad"        # grey — tie / catalogue-ahead
GAP_THRESHOLD = 0.05

# Grouped-bar palette (the two-bars-per-drug panel): both in the purple family,
# Bacformer leaning blue so the pair reads as catalogue-vs-model at a glance.
BAR_CEILING_COLOUR = "#7e3f9e"    # purple — Kleborate determinant ceiling
BAR_BACFORMER_COLOUR = "#4f5bd5"  # blue-purple — finetuned Bacformer mean


def _regime(gap: float) -> str:
    """Three-regime label from the Bacformer − ceiling gap."""
    if gap > GAP_THRESHOLD:
        return "Bacformer >> catalogue"
    if gap < -GAP_THRESHOLD:
        return "catalogue > Bacformer"
    return "tie"


def collect(vis_dir: Path, eval_summary: Path) -> pd.DataFrame:
    """Build the per-drug (ceiling, bacformer, gap, regime) table from the kp_<drug> CSVs + eval summary."""
    evals = pd.read_csv(eval_summary).set_index("drug")["auroc"].to_dict()
    rows = []
    for csv in sorted(vis_dir.glob("kp_*/kleborate_determinant_lr_*.csv")):
        drug = csv.stem.removeprefix("kleborate_determinant_lr_")
        df = pd.read_csv(csv)
        ceiling_row = df[df["gene_name"] == ALL_KEY]
        if ceiling_row.empty or drug not in evals:
            continue
        ceiling = float(ceiling_row["mut_auroc"].iloc[0])
        bacformer = float(evals[drug])
        rows.append({"drug": drug, "ceiling": ceiling, "bacformer": bacformer,
                     "gap": bacformer - ceiling, "regime": _regime(bacformer - ceiling)})
    return pd.DataFrame(rows).sort_values("gap", ascending=True).reset_index(drop=True)


def plot_summary(table: pd.DataFrame, out_path: Path) -> None:
    """Dumbbell per drug (ceiling ● vs Bacformer ◆), sorted by the gap; the beats-catalogue rows in teal."""
    fig, ax = plt.subplots(figsize=(11.8, 9.0))
    y = range(len(table))
    for yi, row in zip(y, table.itertuples(), strict=True):
        line_colour = BEATS_COLOUR if row.gap > GAP_THRESHOLD else TIE_COLOUR
        ax.plot([row.ceiling, row.bacformer], [yi, yi], color=line_colour, linewidth=2.2, zorder=1)
    ax.scatter(table["ceiling"], list(y), color=CEILING_COLOUR, s=70, zorder=2, label="Kleborate ceiling", marker="o")
    ax.scatter(table["bacformer"], list(y), color=BACFORMER_COLOUR, s=70, zorder=2,
               label="Finetuned Bacformer mean", marker="D")

    for yi, row in zip(y, table.itertuples(), strict=True):
        if abs(row.gap) > GAP_THRESHOLD:
            xm = (row.ceiling + row.bacformer) / 2
            ax.text(xm, yi + 0.18, f"{row.gap:+.2f}", ha="center", va="bottom", fontsize=8,
                    color=BEATS_COLOUR if row.gap > 0 else TIE_COLOUR, fontweight="bold")

    ax.set_yticks(list(y))
    ax.set_yticklabels(table["drug"], fontsize=10)
    ax.set_xlabel("held-out AUROC", fontsize=12)
    ax.set_xlim(0.55, 1.0)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.95)
    ax.set_title("Kp AST: Kleborate determinant ceiling vs finetuned Bacformer mean\n"
                 "teal = Bacformer exceeds the catalogue (the drugs Kleborate is blind to)",
                 fontsize=12.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_grouped_bars(table: pd.DataFrame, out_path: Path) -> None:
    """Grouped horizontal bars per drug: Kleborate ceiling (purple) vs finetuned Bacformer mean (blue-purple).

    Sorted by the gap (Bacformer − ceiling) so the two bars sit level at the top (well-catalogued,
    HGT-driven drugs) and the Bacformer bar pulls clear of the ceiling toward the bottom (the
    chromosomal/intrinsic drugs the catalogue is blind to).
    """
    t = table.sort_values("gap", ascending=False).reset_index(drop=True)
    y = np.arange(len(t))
    h = 0.38  # half-height of each bar; the two bars in a group are offset by ±h/2

    fig, ax = plt.subplots(figsize=(11.5, max(7.0, 0.46 * len(t) + 1.6)))
    b_ceiling = ax.barh(y + h / 2, t["ceiling"], height=h, color=BAR_CEILING_COLOUR,
                        edgecolor="black", linewidth=0.5, label="Kleborate determinant ceiling", zorder=3)
    b_bac = ax.barh(y - h / 2, t["bacformer"], height=h, color=BAR_BACFORMER_COLOUR,
                    edgecolor="black", linewidth=0.5, label="Finetuned Bacformer mean", zorder=3)

    # Value label at the end of each bar (kept just clear of the bar tip).
    for rect in list(b_ceiling) + list(b_bac):
        w = rect.get_width()
        ax.text(w + 0.004, rect.get_y() + rect.get_height() / 2, f"{w:.3f}",
                va="center", ha="left", fontsize=7.5)

    ax.axvline(0.5, color="0.6", linestyle=":", linewidth=1.0, zorder=1)  # chance
    ax.text(0.5, len(t) - 0.4, "chance", color="0.5", fontsize=7.5, ha="center", va="bottom")

    ax.set_yticks(y)
    ax.set_yticklabels(t["drug"], fontsize=10)
    ax.invert_yaxis()  # biggest Bacformer-over-ceiling gap at the top
    ax.set_xlabel("held-out AUROC", fontsize=12)
    ax.set_xlim(0.5, 1.06)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    ax.set_title("Kp AST panel: Kleborate determinant ceiling vs finetuned Bacformer mean\n"
                 "two bars per drug — sorted by Bacformer − ceiling gap (largest at top)", fontsize=12.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """CLI entry point."""
    here = Path(__file__).resolve().parent
    vis = here / "docs" / "visualisations"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vis-dir", type=Path, default=vis, help="Dir holding the kp_<drug>/ ceiling CSVs.")
    parser.add_argument("--eval-summary", type=Path, default=vis / "eval" / "eval_summary.csv",
                        help="eval_summary.csv with the deployed Bacformer per-drug AUROCs.")
    parser.add_argument("--out", type=Path, default=vis / "kp_ceiling_vs_bacformer.png")
    parser.add_argument("--out-bars", type=Path, default=vis / "kp_ceiling_vs_bacformer_bars.png",
                        help="Grouped two-bars-per-drug panel (ceiling vs Bacformer).")
    parser.add_argument("--out-csv", type=Path, default=vis / "kp_ceiling_vs_bacformer.csv")
    args = parser.parse_args()
    table = collect(args.vis_dir, args.eval_summary)
    if table.empty:
        raise RuntimeError("No (ceiling, bacformer) pairs found — check vis-dir / eval-summary.")
    table.to_csv(args.out_csv, index=False)
    plot_summary(table, args.out)
    plot_grouped_bars(table, args.out_bars)
    beats = table[table["gap"] > GAP_THRESHOLD]
    print(f"Wrote {args.out}, {args.out_bars} and {args.out_csv} ({len(table)} drugs; "
          f"{len(beats)} where Bacformer beats the catalogue: {list(beats['drug'])})")


if __name__ == "__main__":
    main()

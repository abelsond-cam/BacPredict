"""Per-drug line plot: concat-panel AUROC vs panel size — FT-gene panel vs ESM-gene panel.

Reads ``concat_panel_<drug>.csv`` (from :mod:`bacpredict.apps.kleb.concat_gene_panel_kleb`) and draws AUROC against
the number of concatenated genes k, one line for the **FT panel** (top-k by ``ft_lr_auroc``, indigo) and
one for the **ESM panel** (top-k by ``esm_lr_auroc``, purple), both anchored at k=0 on the **mean-only**
baseline (a horizontal grey line). Where the indigo line sits above the purple, an FT-gene panel is the
better concat ingredient than the ESM-gene panel it would replace; where either clears the dashed
mean-only line, the genes add signal over the genome-mean alone. Optional Kleborate determinant ceiling.
Login/CPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ALL_KEY = "__ALL_Kleborate__"
FT_COLOUR = "#2e2a7a"       # deep indigo — FT-gene panel
ESM_COLOUR = "#7e3f9e"      # purple — ESM-gene panel
MEAN_COLOUR = "0.45"        # grey — mean-only baseline
CEILING_COLOUR = "#c0392b"  # red — Kleborate determinant ceiling


def _kleborate_ceiling(kleborate_csv: Path) -> float | None:
    """Full ``__ALL_Kleborate__`` AUROC ceiling from a per-drug determinant CSV (or None)."""
    if not kleborate_csv.exists():
        return None
    kdf = pd.read_csv(kleborate_csv)
    row = kdf[kdf["gene_name"] == ALL_KEY]
    return float(row["mut_auroc"].iloc[0]) if not row.empty else None


def plot_panel(csv_path: Path, out_path: Path, *, drug: str, ceiling: float | None = None) -> None:
    """Plot FT- vs ESM-panel AUROC against panel size k, anchored on the mean-only baseline."""
    df = pd.read_csv(csv_path)
    mean_row = df[df["config"] == "mean_only"]
    mean_au = float(mean_row["auroc"].iloc[0]) if not mean_row.empty else float("nan")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.axhline(mean_au, color=MEAN_COLOUR, linestyle="--", linewidth=1.4, zorder=1,
               label=f"genome-mean only ({mean_au:.3f})")
    for source, colour, label in (("ft", FT_COLOUR, "mean ⊕ top-k FT genes"),
                                  ("esm", ESM_COLOUR, "mean ⊕ top-k ESM genes")):
        sub = df[df["gene_source"] == source].sort_values("k")
        if sub.empty:
            continue
        ks = [0, *sub["k"].tolist()]
        aus = [mean_au, *sub["auroc"].tolist()]
        ax.plot(ks, aus, marker="o", color=colour, linewidth=2.0, markersize=6, zorder=3, label=label)
        for k, au in zip(sub["k"], sub["auroc"], strict=True):
            ax.annotate(f"{au:.3f}", (k, au), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=7.5, color=colour)
    if ceiling is not None:
        ax.axhline(ceiling, color=CEILING_COLOUR, linestyle=":", linewidth=1.3, zorder=1,
                   label=f"Kleborate ceiling ({ceiling:.3f})")

    ax.set_xlabel("number of concatenated genes (k)  —  k=0 is genome-mean only", fontsize=11)
    ax.set_ylabel("concat out-of-fold AUROC (eval holdout)", fontsize=11.5)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
    ax.set_title(f"{drug}: genome-mean ⊕ top-k gene panel — FT vs ESM ingredient\n"
                 f"indigo above purple = FT-learned gene tokens are the better concat ingredient",
                 fontsize=11.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """CLI entry point."""
    here = Path(__file__).resolve().parent
    vis = here / "docs" / "visualisations"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drug", type=str, required=True)
    parser.add_argument("--csv", type=Path, required=True, help="concat_panel_<drug>.csv.")
    parser.add_argument("--kleborate-csv", type=Path, default=None,
                        help="Default: kp_<drug>/kleborate_determinant_lr_<drug>.csv (draws the ceiling line).")
    parser.add_argument("--out", type=Path, default=None, help="Default: kp_<drug>/<drug>_concat_panel.png.")
    args = parser.parse_args()
    drug_dir = vis / f"kp_{args.drug}"
    kleborate_csv = args.kleborate_csv or drug_dir / f"kleborate_determinant_lr_{args.drug}.csv"
    out = args.out or drug_dir / f"{args.drug}_concat_panel.png"
    plot_panel(args.csv, out, drug=args.drug, ceiling=_kleborate_ceiling(kleborate_csv))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

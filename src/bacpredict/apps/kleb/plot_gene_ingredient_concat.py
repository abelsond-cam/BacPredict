"""Plot #5 figure — which gene block is the better concat ingredient, across the drug panel.

Reads the per-drug ``gene_ingredient_concat_<drug>.csv`` (from :mod:`bacpredict.apps.kleb.gene_ingredient_concat`) and
draws two views of the *ingredient* question (holding the genome-mean context fixed):

1. a **summary panel** — per drug, the ``ft_mean ⊕ <ingredient>`` AUROC for the three gene ingredients
   (frozen-ESM, frozen-Bacformer, FT-Bacformer), with the FT-mean-only baseline as a reference line — so
   whether the **ESM** gene or the **Bacformer** gene is the better block, and whether **fine-tuning** the
   Bacformer gene helps, reads at a glance across drugs;
2. a **delta strip** — the mean Δ(ingredient − its mean) for each (mean × ingredient) cell, the compact
   headline of which ingredient adds the most signal on average.

Pure matplotlib over the small per-drug CSVs — login/CPU.
"""

from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from bacpredict.engine.config import KP

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

INGREDIENT_COLOUR = {
    "esm": "#7e3f9e",         # purple — raw ESM-C gene
    "frozen_bac": "#3aa0a0",  # teal — frozen Bacformer gene
    "ft_bac": "#2e2a7a",      # indigo — fine-tuned Bacformer gene
}
INGREDIENT_LABEL = {"esm": "ESM-C gene", "frozen_bac": "frozen Bacformer gene",
                    "ft_bac": "FT Bacformer gene"}


def load_all(concat_root: Path) -> pd.DataFrame:
    """Concatenate every ``*/gene_ingredient_concat_<drug>.csv`` under ``concat_root`` (tagging the drug)."""
    frames = []
    for csv in sorted(glob.glob(str(concat_root / "*" / "gene_ingredient_concat_*.csv"))):
        drug = Path(csv).stem[len("gene_ingredient_concat_"):]
        df = pd.read_csv(csv)
        df["drug"] = drug
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no gene_ingredient_concat_*.csv under {concat_root}/*/")
    return pd.concat(frames, ignore_index=True)


def plot_summary(df: pd.DataFrame, out_path: Path, *, mean: str = "ft_mean") -> None:
    """Per-drug grouped bars of ``<mean> ⊕ ingredient`` AUROC for the 3 ingredients + the mean-only line."""
    sub = df[df["mean"] == mean]
    drugs = (sub[sub["ingredient"] == "none"].sort_values("auroc", ascending=False)["drug"].tolist())
    ings = ["esm", "frozen_bac", "ft_bac"]
    x = np.arange(len(drugs))
    w = 0.8 / len(ings)
    offsets = [(i - (len(ings) - 1) / 2) * w for i in range(len(ings))]

    fig, ax = plt.subplots(figsize=(max(10.0, 0.62 * len(drugs) + 3.0), 6.2))
    by = {(r["drug"], r["ingredient"]): r["auroc"] for _, r in sub.iterrows()}
    for off, ing in zip(offsets, ings, strict=True):
        ax.bar(x + off, [by.get((d, ing), np.nan) for d in drugs], width=w,
               color=INGREDIENT_COLOUR[ing], edgecolor="black", linewidth=0.4, zorder=3)
    base = {r["drug"]: r["auroc"] for _, r in sub[sub["ingredient"] == "none"].iterrows()}
    ax.plot(x, [base.get(d, np.nan) for d in drugs], "k_", markersize=14, markeredgewidth=2.0,
            label=f"{mean} only", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(drugs, rotation=45, ha="right", fontsize=9, fontstyle="italic")
    ax.set_ylabel(f"AUROC ({mean} ⊕ best gene)", fontsize=12)
    ax.set_ylim(0.45, 1.02)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [Patch(facecolor=INGREDIENT_COLOUR[i], edgecolor="black", label=INGREDIENT_LABEL[i]) for i in ings]
    handles.append(plt.Line2D([0], [0], color="black", marker="_", markersize=12, linestyle="None",
                              markeredgewidth=2.0, label=f"{mean} only (baseline)"))
    ax.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.95)
    ax.set_title(f"Klebsiella pneumoniae — gene concat ingredient ({mean} ⊕ best gene): "
                 f"ESM vs frozen-Bacformer vs FT-Bacformer", fontsize=11.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def delta_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean Δ(ingredient − its own mean) per (mean × ingredient) cell across drugs."""
    g = df[df["ingredient"] != "none"].groupby(["mean", "ingredient"])["delta_vs_its_mean"]
    # rename the agg "mean" before reset_index — else it collides with the "mean" groupby index level
    return g.agg(["mean", "std", "count"]).rename(columns={"mean": "delta_mean"}).reset_index()


def run(concat_root: Path, out_dir: Path) -> None:
    """Load all per-drug CSVs; write the summary panel + the delta table."""
    df = load_all(concat_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "gene_ingredient_concat_all.csv", index=False)
    for mean in ("ft_mean", "frozen_mean"):
        if (df["mean"] == mean).any():
            plot_summary(df, out_dir / f"gene_ingredient_concat_{mean}.png", mean=mean)
    delta = delta_summary(df)
    delta.to_csv(out_dir / "gene_ingredient_concat_delta_summary.csv", index=False)
    logger.info("ingredient delta summary:\n%s", delta.to_string(index=False))


def main() -> None:
    """CLI entry point."""
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--concat-root", type=Path, default=None,
                   help="Root holding per-drug gene_ingredient_concat CSVs (default: <data-root>/processed/"
                   "train_kleb_ast/pangena_predict/gene_ingredient_concat).")
    p.add_argument("--out-dir", type=Path, default=here / "docs" / "visualisations" / "amr_per_abx" / "ingredient")
    args = p.parse_args()
    concat_root = args.concat_root or KP.data_root() / "pangena_predict" / "gene_ingredient_concat"
    run(concat_root, args.out_dir)


if __name__ == "__main__":
    main()

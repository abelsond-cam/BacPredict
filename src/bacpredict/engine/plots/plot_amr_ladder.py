"""Render the AMR concat **ladder**: FT-mean → +baclm gene → +baclm IGR, against the catalogue ceiling.

The figure for :mod:`bacpredict.engine.concat.build_amr_ladder`'s ``<drug>_amr_ladder_table.csv`` — three
BLUE bars in *additive* order (never re-sorted; the ladder's order is its meaning) under the RED catalogue
one-hot ceiling:

    rung 1  FT genome-mean          (light blue)
    rung 2  + best baclm gene       (mid blue)
    rung 3  + best baclm IGR        (dark blue)

The read: for a coding-determinant drug (rifampin/rpoB) rung 3 should add ~nothing — the control. For the
**weak, non-coding-determinant drugs** (ethionamide, streptomycin, kanamycin) the question is whether rung 3
closes the gap to the red ceiling. Each bar is annotated with its value; the rung-3 lift over rung 1 and the
residual gap to the ceiling are called out. Pure matplotlib, CPU/login.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from bacpredict.engine.config import organism, visualisations_dir
from bacpredict.engine.plots.labels import display_name

# ft_mean baseline (light) + the three added-block configs; the non-coding block gets its own hue (green)
# since it is a different block type, not the next step of an additive chain; "+ both" is the darkest.
RUNG_COLOUR = {1: "#a6cee3", 2: "#4292c6", 3: "#41ab5d", 4: "#08519c"}
CEILING_COLOUR = "#c0392b"
_CHANCE = 0.5
_RUNG_ADD = {1: "FT genome-mean", 2: "+ baclm gene", 3: "+ baclm noncoding", 4: "+ gene + noncoding"}


def _rung_label(row: pd.Series) -> str:
    """x-tick label for one config — the added block named on a second line (e.g. "+ baclm gene" / "(rpoB)")."""
    r = int(row["rung"])
    base = _RUNG_ADD.get(r, str(row.get("config") or ""))
    if r == 1:
        return base
    block = str(row.get("block") or "").strip()
    return f"{base}\n({block})" if block else f"{base}\n(none)"


def plot_amr_ladder(table: pd.DataFrame, out_path: Path, *, species: str, drug: str, metric: str = "auroc") -> None:
    """Draw one drug's 3-rung additive ladder + the catalogue ceiling → ``out_path``."""
    df = table.sort_values("rung").reset_index(drop=True)
    if df.empty:
        return
    x = np.arange(len(df))
    colours = [RUNG_COLOUR.get(int(r), "#888888") for r in df["rung"]]

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.bar(x, df[metric], width=0.62, color=colours, edgecolor="black", linewidth=0.7)
    for xi, v in zip(x, df[metric], strict=True):
        if pd.notna(v):
            ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ceiling = float(df[f"ceiling_{metric}"].iloc[0]) if f"ceiling_{metric}" in df.columns else float("nan")
    if pd.notna(ceiling):
        ax.axhline(ceiling, color=CEILING_COLOUR, linestyle="--", linewidth=1.5)
        ax.text(len(df) - 1, ceiling + 0.004, f"catalogue ceiling = {ceiling:.3f}",
                ha="right", va="bottom", fontsize=8.5, color=CEILING_COLOUR)
    ax.axhline(_CHANCE, color="0.6", linestyle=":", linewidth=1.0)

    # Headline: what the IGR rung actually bought, and what is still missing to the ceiling.
    top = df.loc[df["rung"].idxmax()]
    base = df.loc[df["rung"].idxmin()]
    if pd.notna(top[metric]) and pd.notna(base[metric]):
        lift = float(top[metric]) - float(base[metric])
        bits = [f"lift (+both − FT) = {lift:+.3f}"]
        if pd.notna(ceiling):
            bits.append(f"gap to ceiling = {ceiling - float(top[metric]):+.3f}")
        ax.set_title(f"{species.upper()} {display_name(drug)} — FT ⊕ baclm gene ⊕ noncoding vs catalogue\n"
                     + "   ·   ".join(bits), fontsize=11)
    else:
        ax.set_title(f"{species.upper()} {display_name(drug)} — concat ladder", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels([_rung_label(r) for _, r in df.iterrows()], fontsize=9)
    ax.set_ylabel(f"{metric.upper()} (out-of-fold, FT eval-holdout)")
    lo = float(np.nanmin([df[metric].min(), ceiling if pd.notna(ceiling) else 1.0]))
    ax.set_ylim(min(0.45, max(0.0, lo - 0.05)), 1.02)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [Patch(facecolor=RUNG_COLOUR[r], edgecolor="black", label=lbl) for r, lbl in
               [(1, "FT genome-mean"), (2, "+ best baclm gene"), (3, "+ best baclm noncoding"),
                (4, "+ gene + noncoding")]]
    if pd.notna(ceiling):
        handles.append(Line2D([0], [0], ls="--", c=CEILING_COLOUR, lw=1.5, label="catalogue one-hot ceiling"))
    # Upper-left: the bars ascend left→right, so this corner stays clear of rung 3 and the ceiling label.
    ax.legend(handles=handles, fontsize=8, loc="upper left", framealpha=0.95)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(*, species: str, drug: str, table_csv: Path, out_path: Path, metric: str = "auroc") -> pd.DataFrame:
    """Read one ladder table and render its figure."""
    table = pd.read_csv(table_csv)
    plot_amr_ladder(table, Path(out_path), species=species, drug=drug, metric=metric)
    return table


def main() -> None:
    """CLI: render one drug's ladder figure into the visualisations tree."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--species", required=True, choices=["tb", "kp"])
    p.add_argument("--drug", required=True)
    p.add_argument("--table-csv", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--metric", default="auroc", choices=["auroc", "auprc"])
    args = p.parse_args()
    table_csv = args.table_csv or (organism(args.species).data_root() / "pangena_predict" / "amr_ladder"
                                   / args.drug / f"{args.drug}_amr_ladder_table.csv")
    out = args.out or visualisations_dir(args.species) / display_name(args.drug) / "amr_concat_ladder.png"
    run(species=args.species, drug=args.drug, table_csv=table_csv, out_path=out, metric=args.metric)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

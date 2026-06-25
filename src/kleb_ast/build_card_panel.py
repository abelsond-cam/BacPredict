"""Plot #4 — the headline "can Bacformer beat the CARD catalogue?" panel.

One bar pair per antibiotic, two rows (top AUROC, bottom AUPRC):

- **CARD one-hot ceiling** (red) — ``__ALL_CARD__`` from :mod:`kleb_ast.card_determinant_lr`, a linear head
  over a one-hot of *all* the drug's CARD determinants (the determinant-knowledge ceiling on our calls);
- **best Bacformer** (indigo) — the strongest deployable Bacformer read-out for the drug: the higher of the
  FT genome-mean and the FT mean ⊕ best-gene concats (usually FT mean ⊕ FT gene, occasionally another).

The plot deliberately does **not** name which Bacformer model won each drug — it just shows red (catalogue)
vs indigo (best Bacformer). Where indigo clears red, Bacformer reads resistance the catalogue misses; the
message is "we beat CARD once we find the right model — keep optimising". Drugs sorted by the indigo−red gap.
Reads the reliable-concat summary + the per-drug ``__ALL_CARD__`` ceilings (both carry AUROC + AUPRC). Login/CPU.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CARD_ALL_KEY = "__ALL_CARD__"
CEILING_COLOUR = "#c0392b"   # red — CARD one-hot ceiling
BACFORMER_COLOUR = "#2e2a7a"  # indigo — best Bacformer model


def _f(v) -> float:
    """Coerce a possibly-missing (None/NaN/absent) value to float, defaulting to NaN."""
    return float(v) if v is not None and pd.notna(v) else float("nan")


def _ceiling(card_csv: Path) -> tuple[float, float] | None:
    """``(AUROC, AUPRC)`` of the ``__ALL_CARD__`` ceiling row, or None."""
    if not card_csv.exists():
        return None
    df = pd.read_csv(card_csv)
    row = df[df["gene_name"] == CARD_ALL_KEY]
    if row.empty:
        return None
    return float(row["mut_auroc"].iloc[0]), float(row["mut_auprc"].iloc[0])


def _best_bacformer(s: pd.Series) -> tuple[float, float]:
    """``(AUROC, AUPRC)`` of the strongest deployable Bacformer read-out: FT mean or a FT⊕gene concat."""
    cand = [(s.get("ft_mean_only_auroc"), s.get("ft_mean_only_auprc")),
            (s.get("ft_concat_best_ft_auroc"), s.get("ft_concat_best_ft_auprc")),
            (s.get("ft_concat_best_esm_auroc"), s.get("ft_concat_best_esm_auprc")),
            (s.get("ft_concat_best_frozen_auroc"), s.get("ft_concat_best_frozen_auprc"))]
    cand = [(au, ap) for au, ap in cand if pd.notna(au)]
    return max(cand, key=lambda t: t[0]) if cand else (float("nan"), float("nan"))


def assemble_table(summary_csv: Path, card_dir: Path, grain: str) -> pd.DataFrame:
    """Per-drug CARD ceiling vs best-Bacformer AUROC + AUPRC; sorted by the (Bacformer − ceiling) gap."""
    summ = pd.read_csv(summary_csv)
    rows = []
    for _, s in summ.iterrows():
        drug = str(s["drug"])
        ceil = _ceiling(card_dir / f"kp_{drug}" / f"card_determinant_lr_{drug}_{grain}.csv")
        bac_auroc, bac_auprc = _best_bacformer(s)
        rows.append({
            "drug": drug,
            "ceiling_auroc": ceil[0] if ceil else float("nan"),
            "ceiling_auprc": ceil[1] if ceil else float("nan"),
            "bacformer_auroc": _f(bac_auroc), "bacformer_auprc": _f(bac_auprc),
        })
    df = pd.DataFrame(rows)
    df["gap"] = df["bacformer_auroc"] - df["ceiling_auroc"]
    df = df.sort_values("gap", ascending=False).reset_index(drop=True)
    logger.info("assembled %d drugs (%d with CARD ceiling)", len(df), int(df["ceiling_auroc"].notna().sum()))
    return df


def plot_panel(df: pd.DataFrame, out_path: Path, *, grain: str) -> None:
    """Two-row grouped-bar panel (AUROC top, AUPRC bottom); columns = antibiotics, red CARD vs indigo Bacformer."""
    series = [("ceiling", CEILING_COLOUR, "CARD one-hot ceiling"),
              ("bacformer", BACFORMER_COLOUR, "best Bacformer model")]
    x = np.arange(len(df))
    w = 0.8 / len(series)
    offsets = [(i - (len(series) - 1) / 2) * w for i in range(len(series))]

    fig, (ax_roc, ax_prc) = plt.subplots(2, 1, figsize=(max(9.5, 0.6 * len(df) + 3.0), 8.6), sharex=True)
    for ax, metric, ttl in ((ax_roc, "auroc", "AUROC"), (ax_prc, "auprc", "AUPRC")):
        for off, (key, colour, _lbl) in zip(offsets, series, strict=True):
            ax.bar(x + off, df[f"{key}_{metric}"], width=w, color=colour, edgecolor="black",
                   linewidth=0.4, zorder=3)
        ax.axhline(0.5, color="0.6", linestyle=":", linewidth=0.9, zorder=1)
        ax.set_ylabel(ttl, fontsize=12)
        ax.set_ylim(0.45, 1.02)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    ax_prc.set_xticks(x)
    ax_prc.set_xticklabels(df["drug"], rotation=45, ha="right", fontsize=9.5, fontstyle="italic")
    handles = [Patch(facecolor=c, edgecolor="black", label=lbl) for _k, c, lbl in series]
    ax_roc.legend(handles=handles, loc="lower left", fontsize=9.5, framealpha=0.3)
    ax_roc.set_title("Klebsiella pneumoniae — can Bacformer beat the CARD catalogue?\n"
                     "CARD one-hot ceiling (red) vs best Bacformer model (indigo) · "
                     f"{len(df)} drugs, sorted by the gap — where indigo clears red, keep optimising",
                     fontsize=11.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def run(summary_csv: Path, card_dir: Path, out_dir: Path, grain: str) -> pd.DataFrame:
    """Assemble the headline CARD-vs-best-Bacformer table and render it (one figure, the family-grain ceiling)."""
    df = assemble_table(summary_csv, card_dir, grain)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "kp_card_vs_best_bacformer.csv", index=False)
    plot_panel(df, out_dir / "kp_card_vs_best_bacformer.png", grain=grain)
    return df


def main() -> None:
    """CLI entry point."""
    here = Path(__file__).resolve().parent
    vis = here / "docs" / "visualisations"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary-csv", type=Path, default=vis / "reliable_amr" / "kp_reliable_concat_summary.csv")
    p.add_argument("--card-dir", type=Path, default=vis / "amr_per_abx")
    p.add_argument("--out-dir", type=Path, default=vis / "amr_per_abx")
    p.add_argument("--grain", type=str, default="family", choices=["family", "allele"],
                   help="Grain of the CARD ceiling to plot (Bacformer is grain-agnostic). Default family.")
    args = p.parse_args()
    run(args.summary_csv, args.card_dir, args.out_dir, args.grain)


if __name__ == "__main__":
    main()

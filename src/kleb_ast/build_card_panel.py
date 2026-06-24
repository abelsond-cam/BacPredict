"""Plot #4 — the combined Kp panel: CARD one-hot ceiling vs Bacformer (−ESM) vs Bacformer (+ESM).

The head-to-head the programme has not yet drawn for *Klebsiella* (the Kp analogue of the TB summary panel):
per antibiotic, three series on a two-row figure (top AUROC, bottom AUPRC), columns sorted by the deployed
fine-tuned AUROC:

- **CARD one-hot ceiling** — ``__ALL_CARD__`` from :mod:`kleb_ast.card_determinant_lr` (a linear head over a
  one-hot of *all* the drug's CARD determinants — the determinant-knowledge ceiling on **our** calls);
- **Bacformer fine-tuned (−ESM)** — the FT genome-mean alone (reliable-concat ``ft_mean_only``);
- **Bacformer + ESM (best concat)** — FT genome-mean ⊕ the best single gene (the higher of the best-ESM-gene
  and best-FT-gene reliable concats).

Where the Bacformer bars clear the CARD ceiling, Bacformer reads resistance the catalogue does not capture
(the Kp counterpart of TB pyrazinamide). Reads the reliable-concat summary
(:mod:`kleb_ast.aggregate_reliable_concat`) + the per-drug ``__ALL_CARD__`` ceilings — both carry AUROC and
AUPRC. A drug is drawn with whatever series are present (ceiling optional until card_determinant_lr lands).
Login/CPU.
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
FT_COLOUR = "#2e2a7a"        # indigo — Bacformer fine-tuned mean (−ESM)
CONCAT_COLOUR = "#1e8449"    # green — Bacformer FT ⊕ best gene (+ESM)


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


def assemble_table(summary_csv: Path, card_amr_dir: Path, grain: str) -> pd.DataFrame:
    """Per-drug ceiling / FT-mean / best-concat AUROC + AUPRC from the reliable summary + CARD ceilings."""
    summ = pd.read_csv(summary_csv)
    rows = []
    for _, s in summ.iterrows():
        drug = str(s["drug"])
        # best concat = whichever of the two reliable concats has the higher AUROC, with its own AUPRC.
        cand = [("ft", s.get("ft_concat_best_ft_auroc"), s.get("ft_concat_best_ft_auprc")),
                ("esm", s.get("ft_concat_best_esm_auroc"), s.get("ft_concat_best_esm_auprc"))]
        cand = [(k, au, ap) for k, au, ap in cand if pd.notna(au)]
        best = max(cand, key=lambda t: t[1]) if cand else (None, float("nan"), float("nan"))
        ceil = _ceiling(card_amr_dir / f"kp_{drug}" / f"card_determinant_lr_{drug}_{grain}.csv")
        rows.append({
            "drug": drug,
            "ceiling_auroc": ceil[0] if ceil else float("nan"),
            "ceiling_auprc": ceil[1] if ceil else float("nan"),
            "ft_auroc": _f(s["ft_mean_only_auroc"]), "ft_auprc": _f(s.get("ft_mean_only_auprc")),
            "concat_auroc": _f(best[1]), "concat_auprc": _f(best[2]),
        })
    df = pd.DataFrame(rows).sort_values("ft_auroc", ascending=False).reset_index(drop=True)
    logger.info("assembled %d drugs (%d with CARD ceiling)", len(df), int(df["ceiling_auroc"].notna().sum()))
    return df


def plot_panel(df: pd.DataFrame, out_path: Path, *, grain: str) -> None:
    """Two-row grouped-bar panel (AUROC top, AUPRC bottom); columns = antibiotics, 3 series."""
    series = [("ceiling", CEILING_COLOUR, "CARD one-hot ceiling"),
              ("ft", FT_COLOUR, "Bacformer fine-tuned (−ESM)"),
              ("concat", CONCAT_COLOUR, "Bacformer + ESM (best concat)")]
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
    ax_roc.legend(handles=handles, loc="lower left", fontsize=9.5, framealpha=0.95)
    ax_roc.set_title("Klebsiella pneumoniae — AMR across the panel: CARD ceiling vs Bacformer ± ESM "
                     f"({grain} grain)\n(top AUROC · bottom AUPRC · {len(df)} drugs, sorted by fine-tuned AUROC)",
                     fontsize=11.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def run(summary_csv: Path, card_amr_dir: Path, out_dir: Path, grain: str) -> pd.DataFrame:
    """Assemble the combined-panel table and render it for one grain."""
    df = assemble_table(summary_csv, card_amr_dir, grain)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"kp_card_summary_panel_{grain}.csv", index=False)
    plot_panel(df, out_dir / f"kp_card_summary_panel_{grain}.png", grain=grain)
    return df


def main() -> None:
    """CLI entry point."""
    here = Path(__file__).resolve().parent
    vis = here / "docs" / "visualisations"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary-csv", type=Path, default=vis / "reliable_amr" / "kp_reliable_concat_summary.csv")
    p.add_argument("--card-amr-dir", type=Path, default=vis / "amr_per_abx")
    p.add_argument("--out-dir", type=Path, default=vis / "amr_per_abx")
    p.add_argument("--grains", type=str, nargs="+", default=["family", "allele"], choices=["family", "allele"])
    args = p.parse_args()
    for grain in args.grains:
        run(args.summary_csv, args.card_amr_dir, args.out_dir, grain)


if __name__ == "__main__":
    main()

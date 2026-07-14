"""Cross-organism AMR summary panels for the progress report (TB and Kp).

For one organism, assemble — per antibiotic — three head-to-head series and draw a two-row figure
(**top: AUROC, bottom: AUPRC**) with one antibiotic per column (45° labels):

- **catalogue one-hot ceiling** — TB-Profiler/WHO (`__ALL_WHO_one_hot__`) for TB, Kleborate/CARD
  (`__ALL_Kleborate__`) for Kp: a logistic regression on a one-hot of *all* that drug's catalogued
  determinants — the determinant-knowledge ceiling;
- **Bacformer fine-tuned** — the deployed mean-pool model (ladder row ``fine-tuned Bacformer mean``);
- **Bacformer FT + concat best embedding** — FT genome-mean ⊕ the best single ESM gene vector
  (the best ``group == concat`` ladder row, preferring the FT-mean concat).

Everything is read from the already-computed per-drug ladder tables
(``visualisations/<org>/<drug>/<drug>_ladder_table.csv``) and the catalogue-ceiling CSVs
(``tbprofiler_gene_lr_*`` / ``kleborate_determinant_lr_*``); both carry AUROC and AUPRC. A drug is
included only if both a ladder table and a ceiling CSV exist for it. Also emits the simple
FT-only AUROC bar (the motivation figure, the TB analogue of ``kp_amr_panel_auroc.png``).

Pure matplotlib over small CSVs — login/CPU. Figures are written into the TB visualisation dir
(``visualisations/tb/``) so ``PROGRESS_REPORT.md`` can reference them.
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

from bacpredict.engine.config import visualisations_dir

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TB_VIZ = visualisations_dir("tb")
KP_VIZ = visualisations_dir("kp")
OUT_DIR = visualisations_dir("tb")  # the summary panel historically wrote next to the TB tree root

CEILING_COLOUR = "#c0392b"   # red — catalogue one-hot ceiling (WHO / Kleborate)
FT_COLOUR = "#2e2a7a"        # indigo — Bacformer fine-tuned mean-pool (deployed)
CONCAT_COLOUR = "#1e8449"    # green — FT mean ⊕ best ESM gene (concat)

ORGS = {
    "tb": {"viz": TB_VIZ, "ceiling_glob": "tbprofiler_gene_lr_*.csv",
           "ceiling_row": "__ALL_WHO_one_hot__", "ceiling_name": "TB-Profiler / WHO one-hot",
           "title": "M. tuberculosis"},
    "kp": {"viz": KP_VIZ, "ceiling_glob": "kleborate_determinant_lr_*.csv",
           "ceiling_row": "__ALL_Kleborate__", "ceiling_name": "Kleborate / CARD one-hot",
           "title": "Klebsiella pneumoniae"},
}


def _ceiling_metrics(ceiling_csv: Path, row_key: str) -> tuple[float, float] | None:
    """``(AUROC, AUPRC)`` of the full-catalogue one-hot row, or None if absent."""
    df = pd.read_csv(ceiling_csv)
    row = df[df["gene_name"] == row_key]
    if row.empty:
        return None
    return float(row["mut_auroc"].iloc[0]), float(row["mut_auprc"].iloc[0])


def _ft_metrics(ladder: pd.DataFrame) -> tuple[float, float] | None:
    """``(AUROC, AUPRC)`` of the fine-tuned Bacformer mean-pool ladder row."""
    row = ladder[ladder["method"].str.startswith("fine-tuned Bacformer mean")]
    if row.empty:
        return None
    return float(row["auroc"].iloc[0]), float(row["auprc"].iloc[0])


def _concat_metrics(ladder: pd.DataFrame) -> tuple[float, float] | None:
    """``(AUROC, AUPRC)`` of the best concat row, preferring the FT-mean concat."""
    concat = ladder[ladder["group"] == "concat"].copy()
    if concat.empty:
        return None
    ft = concat[concat["family"].eq("mix_ft") | concat["method"].str.contains("FT")]
    pick = (ft if not ft.empty else concat).sort_values("auroc", ascending=False).iloc[0]
    return float(pick["auroc"]), float(pick["auprc"])


def assemble_table(organism: str) -> pd.DataFrame:
    """Per-drug ceiling / FT / concat AUROC + AUPRC for one organism (drugs with all sources)."""
    cfg = ORGS[organism]
    rows = []
    for drug_dir in sorted(cfg["viz"].glob("*")):
        if not drug_dir.is_dir():
            continue
        drug = drug_dir.name
        ladders = glob.glob(str(drug_dir / "*_ladder_table.csv"))
        ceilings = glob.glob(str(drug_dir / cfg["ceiling_glob"]))
        if not ladders or not ceilings:
            continue
        ladder = pd.read_csv(ladders[0])
        ft = _ft_metrics(ladder)
        concat = _concat_metrics(ladder)
        ceil = _ceiling_metrics(Path(ceilings[0]), cfg["ceiling_row"])
        if ft is None or concat is None or ceil is None:
            logger.warning("%s/%s: missing a series (ft=%s concat=%s ceil=%s) — skipping",
                           organism, drug, ft is not None, concat is not None, ceil is not None)
            continue
        rows.append({
            "drug": drug,
            "ceiling_auroc": ceil[0], "ceiling_auprc": ceil[1],
            "ft_auroc": ft[0], "ft_auprc": ft[1],
            "concat_auroc": concat[0], "concat_auprc": concat[1],
        })
    df = pd.DataFrame(rows).sort_values("ft_auroc", ascending=False).reset_index(drop=True)
    logger.info("%s: assembled %d drugs", organism, len(df))
    return df


def plot_summary_panel(df: pd.DataFrame, organism: str, out_path: Path, *,
                       include_concat: bool) -> None:
    """Two-row grouped-bar panel (AUROC top, AUPRC bottom); columns = antibiotics.

    ``include_concat`` adds the FT + concat best-embedding series. It is off for Kp until that series
    is recomputed on the reliable Kleborate/CARD labels (the current Kp concat used auto-picked
    Bakta-labelled genes and is degenerate for several drugs).
    """
    cfg = ORGS[organism]
    series = [("ceiling", CEILING_COLOUR, cfg["ceiling_name"]),
              ("ft", FT_COLOUR, "Bacformer fine-tuned (deployed)")]
    if include_concat:
        series.append(("concat", CONCAT_COLOUR, "Bacformer FT + concat best embedding"))
    n_series = len(series)
    x = np.arange(len(df))
    w = 0.8 / n_series
    offsets = [(i - (n_series - 1) / 2) * w for i in range(n_series)]

    fig, (ax_roc, ax_prc) = plt.subplots(2, 1, figsize=(max(9.0, 0.55 * len(df) + 3.0), 8.4),
                                         sharex=True)
    for ax, metric, ttl in ((ax_roc, "auroc", "AUROC"), (ax_prc, "auprc", "AUPRC")):
        for off, (key, colour, _label) in zip(offsets, series, strict=True):
            ax.bar(x + off, df[f"{key}_{metric}"], width=w, color=colour,
                   edgecolor="black", linewidth=0.4, zorder=3)
        ax.axhline(0.5, color="0.6", linestyle=":", linewidth=0.9, zorder=1)
        ax.set_ylabel(ttl, fontsize=12)
        ax.set_ylim(0.45, 1.02)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    ax_prc.set_xticks(x)
    ax_prc.set_xticklabels(df["drug"], rotation=45, ha="right", fontsize=9.5, fontstyle="italic")
    handles = [Patch(facecolor=c, edgecolor="black", label=lbl) for _k, c, lbl in series]
    ax_roc.legend(handles=handles, loc="lower left", fontsize=9.5, framealpha=0.95, ncol=1)
    concat_note = "" if include_concat else "  (FT + concat series pending reliable-label recompute)"
    ax_roc.set_title(f"{cfg['title']} — AMR prediction across the panel: catalogue ceiling vs "
                     f"Bacformer{' vs FT + concat' if include_concat else ''}\n(top AUROC · "
                     f"bottom AUPRC · {len(df)} drugs, sorted by fine-tuned AUROC){concat_note}",
                     fontsize=11.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def plot_ft_bar(df: pd.DataFrame, organism: str, out_path: Path) -> None:
    """Single-panel fine-tuned-only AUROC bars (the motivation figure)."""
    cfg = ORGS[organism]
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(max(8.0, 0.5 * len(df) + 2.0), 5.2))
    bars = ax.bar(x, df["ft_auroc"], color=FT_COLOUR, edgecolor="black", linewidth=0.4, zorder=3)
    for rect, v in zip(bars, df["ft_auroc"], strict=True):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.004, f"{v:.3f}", rotation=90,
                ha="center", va="bottom", fontsize=7)
    ax.axhline(0.5, color="0.6", linestyle=":", linewidth=0.9, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(df["drug"], rotation=45, ha="right", fontsize=9.5, fontstyle="italic")
    ax.set_ylabel("held-out AUROC", fontsize=12)
    ax.set_ylim(0.45, 1.02)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{cfg['title']} — fine-tuned Bacformer AMR prediction (deployed model, held-out AUROC)",
                 fontsize=11.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--organisms", nargs="+", default=["tb", "kp"], choices=["tb", "kp"])
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()
    for org in args.organisms:
        df = assemble_table(org)
        if df.empty:
            logger.warning("%s: no drugs assembled — skipping", org)
            continue
        df.to_csv(args.out_dir / f"{org}_amr_summary_panel.csv", index=False)
        # TB concat used the causal gene (rpoB/katG/…) and is reliable; Kp concat used auto-picked
        # Bakta-labelled genes and is degenerate for several drugs, so its concat is held back until
        # the reliable-label recompute (Phase 2b).
        plot_summary_panel(df, org, args.out_dir / f"{org}_amr_summary_panel.png",
                           include_concat=(org == "tb"))
        if org == "tb":
            # The TB motivation figure (the analogue of the existing kp_amr_panel_auroc.png); the Kp
            # one already exists and is referenced by the report, so it is not regenerated here.
            plot_ft_bar(df, org, args.out_dir / "tb_amr_panel_auroc.png")


if __name__ == "__main__":
    main()

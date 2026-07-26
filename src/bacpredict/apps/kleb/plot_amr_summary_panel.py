"""Cross-drug AMR summary panels for the progress report (TB and Kp).

For one organism, draw a two-row grouped-bar figure (**top: AUROC, bottom: AUPRC**), one antibiotic per
column, comparing just **two** series per drug (deliberately simple — the concat head is consistently the
top-ranked model, so the panel reports it against the catalogue and nothing else):

- **catalogue ceiling** (red) — the all-determinant one-hot LR ceiling (``ceiling_auroc``/``ceiling_auprc``
  baked into every ladder table): TB-Profiler/WHO for TB, CARD for Kp;
- **Bacformer FT ⊕ gene ⊕ IGR** (dark blue) — the full additive concat head (ladder rung 4), the strongest
  Bacformer model.

Everything is read from the per-drug ladder tables
(``visualisations/<org>/<drug>/<drug>_amr_ladder_table.csv``) produced by
:mod:`bacpredict.engine.segment_amr_lr.concat.build_amr_ladder`; a drug is included if its ladder table exists. Pure
matplotlib over small CSVs — login/CPU. Figures + a CSV go into the TB visualisation dir root.
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

OUT_DIR = visualisations_dir("tb")  # the summary panel historically writes next to the TB tree root

CEILING_COLOUR = "#c0392b"   # red — catalogue all-determinant one-hot ceiling (consistent with the ladder)
CONCAT_COLOUR = "#08306b"    # deep royal blue — FT ⊕ gene ⊕ IGR concat head (consistent with the ladder)
_FULL_CONCAT_RUNG = 4        # ft_mean + baclm gene + baclm noncoding

ORGS = {
    "tb": {"viz": visualisations_dir("tb"), "title": "M. tuberculosis"},
    "kp": {"viz": visualisations_dir("kp"), "title": "Klebsiella pneumoniae"},
}


def _ladder_metrics(ladder_csv: Path) -> dict[str, float] | None:
    """``{ceiling_auroc, ceiling_auprc, concat_auroc, concat_auprc}`` from one drug's ladder table.

    ``concat`` is the full additive head (rung 4 = FT ⊕ gene ⊕ IGR); if a drug's ladder has no rung 4 (no
    qualifying non-coding block), the highest rung present is used. Returns None if the table is empty.
    """
    df = pd.read_csv(ladder_csv)
    if df.empty or "ceiling_auroc" not in df.columns:
        return None
    full = df[df["rung"] == _FULL_CONCAT_RUNG]
    if full.empty:
        full = df[df["rung"] == df["rung"].max()]
    if full.empty:
        return None
    top = df.iloc[0]
    row = full.iloc[0]
    return {
        "ceiling_auroc": float(top["ceiling_auroc"]), "ceiling_auprc": float(top["ceiling_auprc"]),
        "concat_auroc": float(row["auroc"]), "concat_auprc": float(row["auprc"]),
    }


def assemble_table(organism: str) -> pd.DataFrame:
    """Per-drug catalogue-ceiling + full-concat AUROC/AUPRC for one organism (drugs with a ladder table)."""
    cfg = ORGS[organism]
    rows = []
    for drug_dir in sorted(cfg["viz"].glob("*")):
        if not drug_dir.is_dir():
            continue
        tables = glob.glob(str(drug_dir / "*_amr_ladder_table.csv"))
        if not tables:
            continue
        m = _ladder_metrics(Path(tables[0]))
        if m is None:
            logger.warning("%s/%s: empty/invalid ladder table — skipping", organism, drug_dir.name)
            continue
        rows.append({"drug": drug_dir.name, **m})
    df = pd.DataFrame(rows).sort_values("concat_auroc", ascending=False).reset_index(drop=True)
    logger.info("%s: assembled %d drugs", organism, len(df))
    return df


def plot_summary_panel(df: pd.DataFrame, organism: str, out_path: Path) -> None:
    """Two-row grouped-bar panel (AUROC top, AUPRC bottom); columns = antibiotics; two series."""
    cfg = ORGS[organism]
    series = [("ceiling", CEILING_COLOUR, "catalogue ceiling (all determinants)"),
              ("concat", CONCAT_COLOUR, "Bacformer FT ⊕ gene ⊕ IGR (concat)")]
    n_series = len(series)
    x = np.arange(len(df))
    w = 0.8 / n_series
    offsets = [(i - (n_series - 1) / 2) * w for i in range(n_series)]

    fig, (ax_roc, ax_prc) = plt.subplots(2, 1, figsize=(max(9.0, 0.6 * len(df) + 3.0), 8.4), sharex=True)
    for ax, metric, ttl in ((ax_roc, "auroc", "AUROC"), (ax_prc, "auprc", "AUPRC")):
        for off, (key, colour, _label) in zip(offsets, series, strict=True):
            ax.bar(x + off, df[f"{key}_{metric}"], width=w, color=colour,
                   edgecolor="black", linewidth=0.5, zorder=3)
        ax.axhline(0.5, color="0.6", linestyle=":", linewidth=0.9, zorder=1)
        ax.set_ylabel(ttl, fontsize=12)
        ax.set_ylim(0.45, 1.02)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    ax_prc.set_xticks(x)
    ax_prc.set_xticklabels(df["drug"], rotation=45, ha="right", fontsize=9.5, fontstyle="italic")
    handles = [Patch(facecolor=c, edgecolor="black", label=lbl) for _k, c, lbl in series]
    ax_roc.legend(handles=handles, loc="lower left", fontsize=9.5, framealpha=0.95)
    ax_roc.set_title(f"{cfg['title']} — AMR prediction across the panel: catalogue ceiling vs Bacformer concat"
                     f"\n(top AUROC · bottom AUPRC · {len(df)} drugs, sorted by concat AUROC)", fontsize=11.5)
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
        plot_summary_panel(df, org, args.out_dir / f"{org}_amr_summary_panel.png")


if __name__ == "__main__":
    main()

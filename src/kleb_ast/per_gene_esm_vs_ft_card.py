"""Plot #1 — per-gene ESM-C LR vs fine-tuned Bacformer LR over the merged CARD/Bakta universe, causal-hatched.

The corrected, label-migrated version of :mod:`kleb_ast.plot_per_gene_esm_vs_ft`. Gene identity now comes
from **CARD** (acquired alleles + chromosomal QRDR/porin/MgrB-PmrB refs) where a minimap call qualifies, and
from **Bakta** ``gene_name`` otherwise — so every AMR gene's carrier set is reliable while the non-AMR
("lineage") genes keep their Bakta names. For one drug we draw, per gene, two bars: the raw **ESM-C**
per-gene LR (purple) and the **fine-tuned Bacformer** contextualised-token LR (indigo), both eval-holdout,
same zero-imputed out-of-fold k-fold.

The plot makes two things visible at once:

- **the head-to-head** (does FT beat raw ESM per gene — the concat-ingredient question), and
- **where the signal sits**: every gene *known to be causal for the drug* (:func:`kleb_ast.card_label.
  causal_genes_for_drug`) is **cross-hatched**, and x-labels are coloured by source (acquired = red,
  chromosomal = blue, non-AMR Bakta = grey). FT pulling *non-causal* lineage genes up alongside the hatched
  causal ones is the lasso motivation — the unsupervised "best gene" grabs context, not mechanism.

Data: reuses the committed reliable AMR per-gene CSV (CARD-correct carriers) and unions in the top-N
**non-AMR** Bakta genes from the old Bakta-keyed CSV (dropping any Bakta row that re-names a CARD AMR
family, so nothing is double-counted). Pure CSV manipulation — no forward pass; runs local / login.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from kleb_ast.card_label import causal_genes_for_drug

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ESM_COLOUR = "#7e3f9e"        # purple — ESM-C per-gene LR
FROZEN_COLOUR = "#3b6fd4"     # royal blue — frozen Bacformer per-gene LR (between ESM purple and FT indigo)
FT_COLOUR = "#2e2a7a"         # deep indigo — fine-tuned Bacformer per-gene LR
CEILING_COLOUR = "#c0392b"    # red — CARD determinant ceiling
RES_HATCH = "xxx"             # cross-hatch marks a gene known causal for the drug
SOURCE_LABEL_COLOUR = {       # x-tick label colour by gene source
    "acquired": "#c0392b",        # red — acquired / HGT
    "chromosomal": "#1f6f8b",     # blue — chromosomal coding / mutation site
    "bakta_nonamr": "#7a7a7a",    # grey — non-AMR lineage gene
}
_CARD_ALL_KEY = "__ALL_CARD__"
_NORM = re.compile(r"[^a-z0-9]")


def _norm(tok) -> str:
    """Lowercase + strip non-alphanumerics (so ``OmpK36`` and ``ompK36`` compare equal)."""
    return _NORM.sub("", str(tok).lower()) if tok is not None and not pd.isna(tok) else ""


def build_merged_per_gene(
    drug: str, *, grain: str, reliable_csv: Path, bakta_csv: Path | None,
    top_n_nonamr: int = 15, card_csv: Path | None = None,
) -> pd.DataFrame:
    """Merge reliable AMR per-gene rows with top-N non-AMR Bakta genes; tag ``source`` + ``is_causal``.

    Parameters
    ----------
    drug, grain
        Drug column and ``"family"``/``"allele"`` grain (selects the causal set + reliable CSV).
    reliable_csv
        ``reliable_esm_vs_ft_per_gene_<drug>.csv`` — AMR families/alleles with CARD-correct carriers.
    bakta_csv
        Old Bakta-keyed ``esm_vs_ft_per_gene_<drug>.csv`` (non-AMR lineage genes); ``None`` to skip.
    """
    causal = causal_genes_for_drug(drug, grain=grain, card_csv=card_csv)
    causal_norm = {_norm(g) for g in causal}

    rel = pd.read_csv(reliable_csv)
    rel = rel.rename(columns={"gene_family": "gene", "amr_source": "source"})
    rel["is_causal"] = rel["gene"].map(lambda g: _norm(g) in causal_norm)
    keep = ["gene", "source", "n_carriers", "esm_lr_auroc", "frozen_lr_auroc", "ft_lr_auroc", "is_causal"]
    if "frozen_lr_auroc" not in rel.columns:
        rel["frozen_lr_auroc"] = float("nan")   # older per-gene CSV (pre-frozen) — frozen bar simply absent
    frames = [rel[keep]]
    amr_norm = {_norm(g) for g in rel["gene"]}

    if bakta_csv is not None and Path(bakta_csv).exists():
        bak = pd.read_csv(bakta_csv)
        bak = bak[~bak["gene_name"].map(lambda g: _norm(g) in amr_norm)]  # drop AMR genes already covered
        bak = bak[bak["esm_lr_auroc"].notna()].sort_values("esm_lr_auroc", ascending=False).head(top_n_nonamr)
        n_col = "n_carriers_esm" if "n_carriers_esm" in bak.columns else "prevalence"
        bak = bak.rename(columns={"gene_name": "gene", n_col: "n_carriers"})
        bak["source"] = "bakta_nonamr"
        bak["is_causal"] = False
        frames.append(bak.reindex(columns=keep))
    else:
        logger.warning("%s: no Bakta non-AMR CSV at %s — AMR-only universe", drug, bakta_csv)

    merged = pd.concat(frames, ignore_index=True)
    merged["delta_ft_minus_esm"] = merged["ft_lr_auroc"] - merged["esm_lr_auroc"]
    return merged


def card_ceiling(card_lr_csv: Path | None) -> float | None:
    """``__ALL_CARD__`` AUROC ceiling from a per-drug CARD-determinant CSV (or None if absent)."""
    if card_lr_csv is None or not Path(card_lr_csv).exists():
        return None
    df = pd.read_csv(card_lr_csv)
    row = df[df["gene_name"] == _CARD_ALL_KEY]
    return float(row["mut_auroc"].iloc[0]) if not row.empty else None


def plot_merged(merged: pd.DataFrame, out_path: Path, *, drug: str, grain: str,
                top_n: int = 18, ceiling: float | None = None) -> None:
    """Grouped ESM-vs-FT bars over the merged universe; cross-hatch causal genes, colour labels by source.

    The plotted set is the top-``top_n`` genes by ESM-LR **unioned with every causal gene** (so the acquired
    causal genes that are rarer — lower ESM-LR — are never dropped from the causal-vs-lineage contrast).
    """
    df = merged[merged["esm_lr_auroc"].notna()].copy()
    top = df.sort_values("esm_lr_auroc", ascending=False).head(top_n)
    forced = df[df["is_causal"] & ~df["gene"].isin(top["gene"])]
    df = (pd.concat([top, forced]).drop_duplicates("gene")
          .sort_values("esm_lr_auroc", ascending=True).reset_index(drop=True))
    x = np.arange(len(df))
    w = 0.8 / 3

    fig, ax = plt.subplots(figsize=(max(10.0, 0.78 * len(df) + 2.0), 6.2))
    b_esm = ax.bar(x - w, df["esm_lr_auroc"], width=w, color=ESM_COLOUR, edgecolor="black",
                   linewidth=0.5, label="ESM-C per-gene LR", zorder=3)
    b_fr = ax.bar(x, df["frozen_lr_auroc"], width=w, color=FROZEN_COLOUR, edgecolor="black",
                  linewidth=0.5, label="frozen Bacformer per-gene LR", zorder=3)
    b_ft = ax.bar(x + w, df["ft_lr_auroc"], width=w, color=FT_COLOUR, edgecolor="black",
                  linewidth=0.5, label="fine-tuned Bacformer per-gene LR", zorder=3)
    for rects in (b_esm, b_fr, b_ft):
        for rect, causal in zip(rects, df["is_causal"], strict=True):
            if causal:
                rect.set_hatch(RES_HATCH)
            h = rect.get_height()
            if not np.isnan(h):
                ax.text(rect.get_x() + rect.get_width() / 2, h + 0.006, f"{h:.2f}",
                        rotation=90, ha="center", va="bottom", fontsize=5.5)

    ax.axhline(0.5, color="0.6", linestyle=":", linewidth=1.0, zorder=1)
    ax.text(len(df) - 0.5, 0.5, " chance", color="0.5", fontsize=7.5, ha="right", va="bottom")
    if ceiling is not None:
        ax.axhline(ceiling, color=CEILING_COLOUR, linestyle="--", linewidth=1.3, zorder=1)
        ax.text(0.0, ceiling + 0.006, f"CARD one-hot ceiling = {ceiling:.3f}",
                color=CEILING_COLOUR, fontsize=7.5, ha="left", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(df["gene"], rotation=45, ha="right", fontsize=8.5, fontstyle="italic")
    for tick, src in zip(ax.get_xticklabels(), df["source"], strict=True):
        tick.set_color(SOURCE_LABEL_COLOUR.get(str(src), "black"))
    ax.set_ylabel("per-gene LR out-of-fold AUROC (eval holdout)", fontsize=11.5)
    ax.set_ylim(0.45, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    legend = [
        Patch(facecolor=ESM_COLOUR, edgecolor="black", label="ESM-C per-gene LR"),
        Patch(facecolor=FROZEN_COLOUR, edgecolor="black", label="frozen Bacformer per-gene LR"),
        Patch(facecolor=FT_COLOUR, edgecolor="black", label="fine-tuned Bacformer per-gene LR"),
        Patch(facecolor="white", edgecolor="black", hatch=RES_HATCH, label=f"known causal for {drug}"),
        Patch(facecolor=SOURCE_LABEL_COLOUR["acquired"], label="label: acquired (CARD)"),
        Patch(facecolor=SOURCE_LABEL_COLOUR["chromosomal"], label="label: chromosomal (CARD)"),
        Patch(facecolor=SOURCE_LABEL_COLOUR["bakta_nonamr"], label="label: non-AMR (Bakta)"),
    ]
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.01, 0.7), fontsize=8.0,
              framealpha=0.3, ncol=1)
    ax.set_title(
        f"{drug} ({grain} grain): per-gene ESM-C vs frozen vs fine-tuned Bacformer LR\n"
        f"gene identity from CARD (acquired/chromosomal) else Bakta · cross-hatch = known causal for {drug}",
        fontsize=11.0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("%s (%s): wrote %s (%d genes; %d causal hatched)",
                drug, grain, out_path, len(df), int(df["is_causal"].sum()))


def run(*, drug: str, grain: str, reliable_csv: Path, bakta_csv: Path | None, out_dir: Path,
        card_lr_csv: Path | None = None, top_n_nonamr: int = 15, top_n: int = 18,
        card_csv: Path | None = None) -> pd.DataFrame:
    """Build the merged per-gene CSV and render the causal-hatched ESM-vs-FT plot."""
    merged = build_merged_per_gene(drug, grain=grain, reliable_csv=reliable_csv, bakta_csv=bakta_csv,
                                   top_n_nonamr=top_n_nonamr, card_csv=card_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_out = out_dir / f"card_esm_vs_ft_per_gene_{drug}_{grain}.csv"
    merged.sort_values("esm_lr_auroc", ascending=False).to_csv(csv_out, index=False)
    plot_merged(merged, out_dir / f"card_esm_vs_ft_per_gene_{drug}_{grain}.png", drug=drug, grain=grain,
                top_n=top_n, ceiling=card_ceiling(card_lr_csv))
    logger.info("%s (%s): %d genes (%d causal) -> %s", drug, grain, len(merged),
                int(merged["is_causal"].sum()), csv_out)
    return merged


def main() -> None:
    """CLI entry point."""
    here = Path(__file__).resolve().parent
    vis = here / "docs" / "visualisations"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--drug", type=str, required=True)
    p.add_argument("--grain", choices=["family", "allele"], default="family")
    p.add_argument("--reliable-csv", type=Path, default=None,
                   help="Default: reliable_amr/per_gene/reliable_esm_vs_ft_per_gene_<drug>.csv.")
    p.add_argument("--bakta-csv", type=Path, default=None,
                   help="Default: amr_per_abx/kp_<drug>/esm_vs_ft_per_gene_<drug>.csv (non-AMR lineage genes).")
    p.add_argument("--card-lr-csv", type=Path, default=None,
                   help="Default: amr_per_abx/kp_<drug>/card_determinant_lr_<drug>_<grain>.csv (ceiling line).")
    p.add_argument("--out-dir", type=Path, default=None, help="Default: docs/visualisations/amr_per_abx/kp_<drug>.")
    p.add_argument("--top-n-nonamr", type=int, default=15)
    p.add_argument("--top-n", type=int, default=18)
    args = p.parse_args()

    reliable = args.reliable_csv or vis / "reliable_amr" / "per_gene" / f"reliable_esm_vs_ft_per_gene_{args.drug}.csv"
    bakta = args.bakta_csv or vis / "amr_per_abx" / f"kp_{args.drug}" / f"esm_vs_ft_per_gene_{args.drug}.csv"
    card_lr = args.card_lr_csv or vis / "amr_per_abx" / f"kp_{args.drug}" / f"card_determinant_lr_{args.drug}_{args.grain}.csv"
    out_dir = args.out_dir or vis / "amr_per_abx" / f"kp_{args.drug}"
    run(drug=args.drug, grain=args.grain, reliable_csv=reliable, bakta_csv=bakta, out_dir=out_dir,
        card_lr_csv=card_lr, top_n_nonamr=args.top_n_nonamr, top_n=args.top_n)


if __name__ == "__main__":
    main()

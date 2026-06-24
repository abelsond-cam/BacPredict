"""Plot #3 — the CARD read-out ladder: mean → +best unsupervised gene → concat, vs the CARD one-hot ceiling.

The label-migrated, reliable-carrier version of :mod:`kleb_ast.build_kleb_ladder`. For one drug it draws a
single AUROC ladder (the reliable artifacts are AUROC-only) from the committed reliable-CARD numbers:

============================  ========  ===========================================================
rung                          family    source
============================  ========  ===========================================================
fine-tuned Bacformer mean     Bacformer kp_reliable_concat_summary ``ft_mean_only_auroc``
CARD top determinant          one-hot   card_determinant_lr_<drug>_<grain> top non-ALL gene
ESM <best gene>               ESM       reliable_esm_vs_ft_per_gene ``esm_lr_auroc`` of best_esm_gene
FT  <best gene>               FT_gene   reliable_esm_vs_ft_per_gene ``ft_lr_auroc`` of best_ft_gene
concat: FT mean ⊕ ESM gene    mix       summary ``ft_concat_best_esm_auroc``
concat: FT mean ⊕ FT gene     mix_ft    summary ``ft_concat_best_ft_auroc``
============================  ========  ===========================================================

with the **``__ALL_CARD__``** one-hot ceiling (Plot #2) drawn as a reference line. The unsupervised "best
gene" name is annotated and flagged ``causal`` / ``context`` via :func:`kleb_ast.card_label.
causal_genes_for_drug` — for several drugs the selector grabs a porin/lineage context proxy (OmpK/PmrB),
*not* the causal mechanism, which is the headline that motivates the sparse-lasso next step. Login/CPU.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from kleb_ast.card_label import causal_genes_for_drug

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CARD_ALL_KEY = "__ALL_CARD__"
FAMILY_COLOURS = {
    "Bacformer": "#1f77b4",   # blue — FT genome-mean
    "one-hot": "#c0392b",     # red — CARD top determinant (one-hot)
    "ESM": "#7e3f9e",         # purple — ESM single-gene
    "FT_gene": "#3aa0a0",     # teal — fine-tuned Bacformer single-gene token
    "mix": "#6a4fb3",         # purple-blue — concat FT mean ⊕ ESM gene
    "mix_ft": "#2e2a7a",      # deep indigo — concat FT mean ⊕ FT gene
}
FAMILY_LABEL = {
    "Bacformer": "FT Bacformer mean", "one-hot": "CARD top determinant", "ESM": "ESM-C single gene",
    "FT_gene": "FT Bacformer single gene", "mix": "concat (FT mean ⊕ ESM gene)",
    "mix_ft": "concat (FT mean ⊕ FT gene)",
}
CEILING_COLOUR = "#c0392b"


def _gene_auroc(per_gene_csv: Path, gene: str, col: str) -> float | None:
    """``col`` (esm_lr_auroc / ft_lr_auroc) for ``gene`` in the reliable per-gene CSV, or None."""
    if not per_gene_csv.exists():
        return None
    df = pd.read_csv(per_gene_csv)
    row = df[df["gene_family"].astype(str) == str(gene)]
    return float(row[col].iloc[0]) if not row.empty else None


def _card_ceiling_and_top(card_csv: Path | None) -> tuple[float | None, dict | None]:
    """``(__ALL_CARD__ auroc, top-determinant row)`` from a card_determinant_lr CSV (or (None, None))."""
    if card_csv is None or not Path(card_csv).exists():
        return None, None
    df = pd.read_csv(card_csv)
    ceil_row = df[df["gene_name"] == CARD_ALL_KEY]
    ceiling = float(ceil_row["mut_auroc"].iloc[0]) if not ceil_row.empty else None
    bars = df[df["gene_name"] != CARD_ALL_KEY].sort_values("mut_auroc", ascending=False)
    top = bars.iloc[0].to_dict() if not bars.empty else None
    return ceiling, top


def build_table(drug: str, *, grain: str, summary_csv: Path, per_gene_csv: Path,
                card_csv: Path | None, card_csv_path: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Assemble the CARD ladder rows + meta for ``drug`` from the reliable summary/per-gene/ceiling CSVs."""
    summ = pd.read_csv(summary_csv)
    srow = summ[summ["drug"] == drug]
    if srow.empty:
        raise ValueError(f"{drug} not in {summary_csv}")
    srow = srow.iloc[0]
    best_esm, best_ft = str(srow["best_esm_gene"]), str(srow["best_ft_gene"])
    causal = causal_genes_for_drug(drug, grain=grain, card_csv=card_csv_path)

    rows = [{"rung": "FT Bacformer mean", "family": "Bacformer",
             "auroc": float(srow["ft_mean_only_auroc"]), "gene": "", "causal": None}]

    ceiling, top = _card_ceiling_and_top(card_csv)
    if top is not None:
        rows.append({"rung": f"CARD top: {top['site']}", "family": "one-hot",
                     "auroc": float(top["mut_auroc"]), "gene": str(top["site"]),
                     "causal": bool(top.get("is_causal", False))})

    esm_au = _gene_auroc(per_gene_csv, best_esm, "esm_lr_auroc")
    if esm_au is not None:
        rows.append({"rung": f"ESM {best_esm}", "family": "ESM", "auroc": esm_au,
                     "gene": best_esm, "causal": best_esm in causal})
    ft_au = _gene_auroc(per_gene_csv, best_ft, "ft_lr_auroc")
    if ft_au is not None:
        rows.append({"rung": f"FT {best_ft}", "family": "FT_gene", "auroc": ft_au,
                     "gene": best_ft, "causal": best_ft in causal})

    rows.append({"rung": f"concat: FT mean ⊕ ESM {best_esm}", "family": "mix",
                 "auroc": float(srow["ft_concat_best_esm_auroc"]), "gene": best_esm,
                 "causal": best_esm in causal})
    rows.append({"rung": f"concat: FT mean ⊕ FT {best_ft}", "family": "mix_ft",
                 "auroc": float(srow["ft_concat_best_ft_auroc"]), "gene": best_ft,
                 "causal": best_ft in causal})

    info = {"best_esm_gene": best_esm, "best_ft_gene": best_ft, "ceiling": ceiling,
            "best_esm_causal": best_esm in causal, "best_ft_causal": best_ft in causal}
    return pd.DataFrame(rows), info


def plot_ladder(df: pd.DataFrame, out_path: Path, *, drug: str, grain: str, info: dict) -> None:
    """Single-panel AUROC ladder, bars ascending, family-coloured, with the __ALL_CARD__ ceiling line."""
    df = df.sort_values("auroc").reset_index(drop=True)
    colours = [FAMILY_COLOURS.get(f, "#888888") for f in df["family"]]
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    x = range(len(df))
    bars = ax.bar(x, df["auroc"], color=colours, edgecolor="black", linewidth=0.7, width=0.7)
    for b, causal in zip(bars, df["causal"], strict=True):
        if causal:
            b.set_hatch("xxx")  # cross-hatch a rung whose injected gene is a known causal determinant
    for xi, v in zip(x, df["auroc"], strict=True):
        ax.text(xi, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ceiling = info.get("ceiling")
    if ceiling is not None:
        ax.axhline(ceiling, color=CEILING_COLOUR, linestyle=":", linewidth=1.6)
        ax.text(0.3 * (len(df) - 1), ceiling + 0.004, f"CARD one-hot ceiling = {ceiling:.3f}",
                ha="center", va="bottom", fontsize=8.0, color=CEILING_COLOUR)

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["rung"], rotation=28, ha="right", fontsize=9.0)
    ax.set_ylabel("eval-holdout AUROC (k-fold)", fontsize=12)
    lo = df["auroc"].min()
    if ceiling is not None:
        lo = min(lo, ceiling)
    ax.set_ylim(max(0.0, lo - 0.05), 1.005)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, ec="black", lw=0.7) for c in FAMILY_COLOURS.values()]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white", ec="black", lw=0.7, hatch="xxx"))
    labels = list(FAMILY_LABEL.values()) + ["injected gene is causal"]
    ax.legend(handles, labels, loc="lower right", fontsize=8.5, framealpha=0.95)

    esm_tag = "causal" if info["best_esm_causal"] else "CONTEXT proxy"
    ft_tag = "causal" if info["best_ft_causal"] else "CONTEXT proxy"
    ax.set_title(f"{drug} ({grain} grain): CARD AST read-out ladder\n"
                 f"unsupervised best ESM gene = {info['best_esm_gene']} ({esm_tag}) · "
                 f"best FT gene = {info['best_ft_gene']} ({ft_tag})", fontsize=11.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("%s (%s): ladder -> %s (ceiling=%s)", drug, grain, out_path, info.get("ceiling"))


def run(*, drug: str, grain: str, summary_csv: Path, per_gene_csv: Path, card_csv: Path | None,
        out_dir: Path, card_csv_path: Path | None = None) -> pd.DataFrame:
    """Build the CARD ladder table + plot for one drug/grain."""
    table, info = build_table(drug, grain=grain, summary_csv=summary_csv, per_gene_csv=per_gene_csv,
                              card_csv=card_csv, card_csv_path=card_csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / f"{drug}_card_ladder_table_{grain}.csv", index=False)
    plot_ladder(table, out_dir / f"{drug}_card_ladder_{grain}.png", drug=drug, grain=grain, info=info)
    return table


def main() -> None:
    """CLI entry point."""
    here = Path(__file__).resolve().parent
    vis = here / "docs" / "visualisations"
    rel = vis / "reliable_amr"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--drug", type=str, required=True)
    p.add_argument("--grain", choices=["family", "allele"], default="family")
    p.add_argument("--summary-csv", type=Path, default=rel / "kp_reliable_concat_summary.csv")
    p.add_argument("--per-gene-csv", type=Path, default=None,
                   help="Default: reliable_amr/per_gene/reliable_esm_vs_ft_per_gene_<drug>.csv.")
    p.add_argument("--card-csv", type=Path, default=None,
                   help="Default: card_amr/kp_<drug>/card_determinant_lr_<drug>_<grain>.csv (ceiling + top).")
    p.add_argument("--out-dir", type=Path, default=None, help="Default: docs/visualisations/card_amr/kp_<drug>.")
    args = p.parse_args()
    per_gene = args.per_gene_csv or rel / "per_gene" / f"reliable_esm_vs_ft_per_gene_{args.drug}.csv"
    card_csv = args.card_csv or vis / "card_amr" / f"kp_{args.drug}" / f"card_determinant_lr_{args.drug}_{args.grain}.csv"
    out_dir = args.out_dir or vis / "card_amr" / f"kp_{args.drug}"
    run(drug=args.drug, grain=args.grain, summary_csv=args.summary_csv, per_gene_csv=per_gene,
        card_csv=card_csv, out_dir=out_dir)


if __name__ == "__main__":
    main()

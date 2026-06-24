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
# Three families only, matching the per-gene esm-vs-ft plot's palette: CARD red, ESM purple, Bacformer
# dark purple. Every Bacformer read-out (FT mean / single gene / concat) shares one colour — the specific
# variant is spelled out in the bar's x-label, not encoded by hue.
FAMILY_COLOURS = {
    "CARD": "#c0392b",        # red — CARD catalogue (top determinant + all-determinant ceiling)
    "ESM": "#7e3f9e",         # purple — ESM single gene
    "Bacformer": "#2e2a7a",   # dark purple — every Bacformer read-out
}
FAMILY_LABEL = {
    "CARD": "CARD catalogue", "ESM": "ESM single gene", "Bacformer": "Bacformer (BacF)",
}


def _gene_auroc(per_gene_csv: Path, gene: str, col: str) -> float | None:
    """``col`` (esm_lr_auroc / ft_lr_auroc / *_auprc) for ``gene`` in the reliable per-gene CSV, or None."""
    if not per_gene_csv.exists():
        return None
    df = pd.read_csv(per_gene_csv)
    if col not in df.columns:                       # older per-gene CSV without the AUPRC columns
        return None
    row = df[df["gene_family"].astype(str) == str(gene)]
    return float(row[col].iloc[0]) if not row.empty else None


def _card_ceiling_and_top(card_csv: Path | None) -> tuple[float | None, float | None, dict | None]:
    """``(__ALL_CARD__ auroc, __ALL_CARD__ auprc, top-determinant row)`` from a determinant CSV."""
    if card_csv is None or not Path(card_csv).exists():
        return None, None, None
    df = pd.read_csv(card_csv)
    ceil_row = df[df["gene_name"] == CARD_ALL_KEY]
    ceiling = float(ceil_row["mut_auroc"].iloc[0]) if not ceil_row.empty else None
    ceiling_pr = float(ceil_row["mut_auprc"].iloc[0]) if not ceil_row.empty else None
    bars = df[df["gene_name"] != CARD_ALL_KEY].sort_values("mut_auroc", ascending=False)
    top = bars.iloc[0].to_dict() if not bars.empty else None
    return ceiling, ceiling_pr, top


def _frozen_concat(ingredient_csv: Path | None) -> tuple[float | None, str | None]:
    """``(auroc, gene)`` of the FT mean ⊕ frozen-Bacformer-gene concat from the gene-ingredient CSV."""
    if ingredient_csv is None or not Path(ingredient_csv).exists():
        return None, None
    df = pd.read_csv(ingredient_csv)
    row = df[df["config"] == "ft_mean+frozen_bac_gene"]
    return (float(row["auroc"].iloc[0]), str(row["gene"].iloc[0])) if not row.empty else (None, None)


def build_table(drug: str, *, grain: str, summary_csv: Path, per_gene_csv: Path,
                card_csv: Path | None, card_csv_path: Path | None = None,
                ingredient_csv: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Assemble the CARD ladder rows + meta for ``drug`` from the reliable summary/per-gene/ceiling CSVs."""
    summ = pd.read_csv(summary_csv)
    srow = summ[summ["drug"] == drug]
    if srow.empty:
        raise ValueError(f"{drug} not in {summary_csv}")
    srow = srow.iloc[0]
    best_esm, best_ft = str(srow["best_esm_gene"]), str(srow["best_ft_gene"])
    causal = causal_genes_for_drug(drug, grain=grain, card_csv=card_csv_path)

    nan = float("nan")
    rows = [{"rung": "BacF FT mean", "family": "Bacformer", "auroc": float(srow["ft_mean_only_auroc"]),
             "auprc": float(srow["ft_mean_only_auprc"]), "gene": "", "causal": None}]

    # CARD catalogue — the top single determinant and the all-determinant ceiling, both as red bars
    ceiling, ceiling_pr, top = _card_ceiling_and_top(card_csv)
    if top is not None:
        rows.append({"rung": f"CARD top: {top['site']}", "family": "CARD",
                     "auroc": float(top["mut_auroc"]), "auprc": float(top.get("mut_auprc", nan)),
                     "gene": str(top["site"]), "causal": bool(top.get("is_causal", False))})
    if ceiling is not None:
        rows.append({"rung": "CARD all-determinant ceiling", "family": "CARD",
                     "auroc": ceiling, "auprc": ceiling_pr, "gene": "", "causal": None})

    # single-gene rungs — AUROC + AUPRC (auprc is None on older per-gene CSVs that predate the column)
    esm_au = _gene_auroc(per_gene_csv, best_esm, "esm_lr_auroc")
    if esm_au is not None:
        esm_ap = _gene_auroc(per_gene_csv, best_esm, "esm_lr_auprc")
        rows.append({"rung": f"ESM {best_esm}", "family": "ESM", "auroc": esm_au,
                     "auprc": esm_ap if esm_ap is not None else nan, "gene": best_esm, "causal": best_esm in causal})
    ft_au = _gene_auroc(per_gene_csv, best_ft, "ft_lr_auroc")
    if ft_au is not None:
        ft_ap = _gene_auroc(per_gene_csv, best_ft, "ft_lr_auprc")
        rows.append({"rung": f"BacF FT {best_ft}", "family": "Bacformer", "auroc": ft_au,
                     "auprc": ft_ap if ft_ap is not None else nan, "gene": best_ft, "causal": best_ft in causal})

    rows.append({"rung": f"BacF FT mean ⊕ ESM {best_esm}", "family": "Bacformer",
                 "auroc": float(srow["ft_concat_best_esm_auroc"]), "auprc": float(srow["ft_concat_best_esm_auprc"]),
                 "gene": best_esm, "causal": best_esm in causal})
    rows.append({"rung": f"BacF FT mean ⊕ BacF FT {best_ft}", "family": "Bacformer",
                 "auroc": float(srow["ft_concat_best_ft_auroc"]), "auprc": float(srow["ft_concat_best_ft_auprc"]),
                 "gene": best_ft, "causal": best_ft in causal})
    # FT mean ⊕ *frozen* BacF gene — isolates the value of fine-tuning the gene token (vs frozen). From the
    # gene-ingredient concat CSV (AUROC only; AUPRC n/a there).
    fr_au, fr_gene = _frozen_concat(ingredient_csv)
    if fr_au is not None:
        rows.append({"rung": f"BacF FT mean ⊕ frozen BacF {fr_gene}", "family": "Bacformer",
                     "auroc": fr_au, "auprc": nan, "gene": fr_gene, "causal": fr_gene in causal})

    info = {"best_esm_gene": best_esm, "best_ft_gene": best_ft, "ceiling": ceiling,
            "best_esm_causal": best_esm in causal, "best_ft_causal": best_ft in causal}
    return pd.DataFrame(rows), info


def _draw_metric_panel(ax, df: pd.DataFrame, colours: list[str], metric: str, ylabel: str) -> None:
    """One metric panel (auroc/auprc) over the shared bar order; bars absent + 'n/a' where the metric is NaN."""
    x = range(len(df))
    vals = df[metric]
    bars = ax.bar(x, vals.fillna(0.0), color=colours, edgecolor="black", linewidth=0.7, width=0.7)
    for b, v, causal in zip(bars, vals, df["causal"], strict=True):
        if pd.isna(v):
            b.set_visible(False)            # metric unavailable for this rung (single-gene AUPRC)
        elif causal:
            b.set_hatch("xxx")              # cross-hatch a rung whose injected gene is causal
    # headroom: lift the top to 1.01 when a bar clears 0.99 so its value label has room
    top = 1.01 if float(vals.max()) > 0.99 else 1.005
    ax.set_ylim(max(0.0, float(vals.min()) - 0.05), top)
    for xi, v in zip(x, vals, strict=True):
        if pd.notna(v):
            ax.text(xi, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        else:
            ax.text(xi, ax.get_ylim()[0] + 0.01, "n/a", ha="center", va="bottom", fontsize=8, color="0.55")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def plot_ladder(df: pd.DataFrame, out_path: Path, *, drug: str, grain: str, info: dict) -> None:
    """Two-panel ladder — AUROC (top) + AUPRC (bottom), same bar order/names; the CARD ceiling is a red bar."""
    df = df.sort_values("auroc").reset_index(drop=True)
    colours = [FAMILY_COLOURS.get(f, "#888888") for f in df["family"]]
    fig, (ax_roc, ax_pr) = plt.subplots(2, 1, figsize=(10.5, 9.6), sharex=True)
    _draw_metric_panel(ax_roc, df, colours, "auroc", "eval-holdout AUROC (k-fold)")
    _draw_metric_panel(ax_pr, df, colours, "auprc", "eval-holdout AUPRC (k-fold)")

    ax_pr.set_xticks(range(len(df)))
    ax_pr.set_xticklabels(df["rung"], rotation=28, ha="right", fontsize=9.0)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, ec="black", lw=0.7) for c in FAMILY_COLOURS.values()]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white", ec="black", lw=0.7, hatch="xxx"))
    labels = list(FAMILY_LABEL.values()) + ["injected gene is causal"]
    ax_roc.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.99), fontsize=8.5, framealpha=0.3)

    esm_tag = "causal" if info["best_esm_causal"] else "CONTEXT proxy"
    ft_tag = "causal" if info["best_ft_causal"] else "CONTEXT proxy"
    ax_roc.set_title(f"{drug} ({grain} grain): CARD AST read-out ladder\n"
                     f"unsupervised best ESM gene = {info['best_esm_gene']} ({esm_tag}) · "
                     f"best BacF FT gene = {info['best_ft_gene']} ({ft_tag})", fontsize=11.5)
    fig.text(0.5, 0.012, "BacF = Bacformer · AUPRC n/a for the frozen-concat rung",
             ha="center", va="bottom", fontsize=8.0, color="0.4")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("%s (%s): ladder -> %s (ceiling=%s)", drug, grain, out_path, info.get("ceiling"))


def run(*, drug: str, grain: str, summary_csv: Path, per_gene_csv: Path, card_csv: Path | None,
        out_dir: Path, card_csv_path: Path | None = None, ingredient_csv: Path | None = None) -> pd.DataFrame:
    """Build the CARD ladder table + plot for one drug/grain."""
    table, info = build_table(drug, grain=grain, summary_csv=summary_csv, per_gene_csv=per_gene_csv,
                              card_csv=card_csv, card_csv_path=card_csv_path, ingredient_csv=ingredient_csv)
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
                   help="Default: amr_per_abx/kp_<drug>/card_determinant_lr_<drug>_<grain>.csv (ceiling + top).")
    p.add_argument("--out-dir", type=Path, default=None, help="Default: docs/visualisations/amr_per_abx/kp_<drug>.")
    p.add_argument("--ingredient-csv", type=Path, default=None,
                   help="Default: amr_per_abx/ingredient/<drug>/gene_ingredient_concat_<drug>.csv (frozen concat).")
    args = p.parse_args()
    per_gene = args.per_gene_csv or rel / "per_gene" / f"reliable_esm_vs_ft_per_gene_{args.drug}.csv"
    card_csv = args.card_csv or vis / "amr_per_abx" / f"kp_{args.drug}" / f"card_determinant_lr_{args.drug}_{args.grain}.csv"
    out_dir = args.out_dir or vis / "amr_per_abx" / f"kp_{args.drug}"
    ingredient = args.ingredient_csv or vis / "amr_per_abx" / "ingredient" / args.drug / f"gene_ingredient_concat_{args.drug}.csv"
    run(drug=args.drug, grain=args.grain, summary_csv=args.summary_csv, per_gene_csv=per_gene,
        card_csv=card_csv, out_dir=out_dir, ingredient_csv=ingredient)


if __name__ == "__main__":
    main()

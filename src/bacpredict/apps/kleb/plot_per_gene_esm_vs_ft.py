"""Per-drug histogram: per-gene ESM-C LR vs fine-tuned Bacformer LR (the concat-ingredient head-to-head).

Reads ``esm_vs_ft_per_gene_<drug>.csv`` (from :mod:`bacpredict.engine.gene_lr.segment_vs_ft`) and draws, for the
top-N genes (by ESM-LR), two bars per gene — **ESM-LR (purple)** and **Bacformer-FT-LR (indigo)** — on the
same axes, with a chance line and (optionally) the Kleborate determinant ceiling. Where the indigo bar
clears the purple, the fine-tuned Bacformer per-gene representation predicts resistance better than the raw
ESM-C embedding — i.e. a better gene ingredient for the concat. Both AUROCs are eval-holdout, same samples,
same zero-imputed out-of-fold k-fold. Login/CPU.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bacpredict.engine.config import visualisations_dir

ALL_KEY = "__ALL_Kleborate__"
ESM_COLOUR = "#7e3f9e"     # purple — ESM-C per-gene LR
FT_COLOUR = "#2e2a7a"      # deep indigo — fine-tuned Bacformer per-gene LR
CEILING_COLOUR = "#c0392b"  # red — Kleborate determinant ceiling
ACQ_LABEL_COLOUR = "#c0392b"  # red label — acquired / HGT resistance gene
MUT_LABEL_COLOUR = "#1f6f8b"  # blue label — chromosomal mutation / loss
RES_HATCH = "xxx"             # cross-hatch marks a known resistance gene

# Resistance-gene name patterns (matched case-insensitively against the ranking's Bakta ``gene_name``),
# each tagged "mutation" (chromosomal point-mutation / loss) or "acquired" (HGT gene). Anchored so e.g.
# ``tet(`` catches tet(A)/tet(D), ``bla`` catches every blaXXX. Relocated here from the retired
# ``causal_gene_esm_scorecard`` (this legacy plot is its only consumer); the authoritative causal-gene
# source for the current CARD Plot #1 is :func:`bacpredict.apps.kleb.card_label.causal_genes_for_drug`.
_FQ = [("^gyrA$", "mutation"), ("^parC$", "mutation"), ("^gyrB$", "mutation"), ("^parE$", "mutation"),
       ("^qnr", "acquired"), ("^oqxA$", "acquired"), ("^oqxB$", "acquired"), (r"^aac\(6'\)-Ib", "acquired")]
_COLISTIN = [("^pmrA$", "mutation"), ("^pmrB$", "mutation"), ("^phoP$", "mutation"), ("^phoQ$", "mutation"),
             ("^mgrB$", "mutation"), ("^crrA$", "mutation"), ("^crrB$", "mutation"), ("^mcr", "acquired")]
_MACROLIDE = [("^mph", "acquired"), ("^ere", "acquired"), ("^erm", "acquired"), ("^msr", "acquired"),
              ("^mef", "acquired"), ("^oqxB$", "acquired"), ("^acrB$", "mutation")]
_TETRACYCLINE = [("^tet", "acquired"), ("^ramR$", "mutation"), ("^ramA$", "mutation"),
                 ("^acrA$", "mutation"), ("^acrB$", "mutation"), ("^oqxB$", "acquired"), ("^marR$", "mutation")]
_AMINOGLYCOSIDE = [(r"^aac\(", "acquired"), (r"^aph\(", "acquired"), (r"^ant\(", "acquired"),
                   ("^aad", "acquired"), ("^armA$", "acquired"), ("^rmt", "acquired"), ("^npmA$", "acquired")]
_BETALACTAM = [("^bla", "acquired"), ("^ampC$", "mutation"), ("^ampH$", "mutation"), ("^ompK3[567]$", "mutation")]
_SULFA_TRIM = [("^dfr", "acquired"), ("^sul[0-9]", "acquired")]

DRUG_CAUSAL: dict[str, list[tuple[str, str]]] = {
    "ciprofloxacin": _FQ, "levofloxacin": _FQ,
    "colistin": _COLISTIN,
    "azithromycin": _MACROLIDE,
    "tetracycline": _TETRACYCLINE,
    "gentamicin": _AMINOGLYCOSIDE, "tobramycin": _AMINOGLYCOSIDE, "amikacin": _AMINOGLYCOSIDE,
    "trimethoprim-sulfamethoxazole": _SULFA_TRIM,
    "cefotaxime": _BETALACTAM, "ceftriaxone": _BETALACTAM, "ceftazidime": _BETALACTAM,
    "cefepime": _BETALACTAM, "cefuroxime": _BETALACTAM, "cefazolin": _BETALACTAM,
    "cefoxitin": _BETALACTAM, "aztreonam": _BETALACTAM, "ampicillin-sulbactam": _BETALACTAM,
    "piperacillin-tazobactam": _BETALACTAM, "ertapenem": _BETALACTAM, "imipenem": _BETALACTAM,
    "meropenem": _BETALACTAM,
}


def _resistance_mechanism(gene: str, drug: str) -> str | None:
    """Mechanism tag ('mutation'/'acquired') if ``gene`` is a known resistance gene for ``drug``, else None."""
    for pat, mech in DRUG_CAUSAL.get(drug, []):
        if re.search(pat, gene, flags=re.IGNORECASE):
            return mech
    return None


def _kleborate_ceiling(kleborate_csv: Path) -> float | None:
    """Full ``__ALL_Kleborate__`` AUROC ceiling from a per-drug determinant CSV (or None)."""
    if not kleborate_csv.exists():
        return None
    kdf = pd.read_csv(kleborate_csv)
    row = kdf[kdf["gene_name"] == ALL_KEY]
    return float(row["mut_auroc"].iloc[0]) if not row.empty else None


def plot_esm_vs_ft(csv_path: Path, out_path: Path, *, drug: str, top_n: int = 15,
                   ceiling: float | None = None) -> None:
    """Grouped bars (ESM-LR vs FT-LR) for the top-``top_n`` genes by ESM-LR, ascending (highest right)."""
    df = pd.read_csv(csv_path)
    df = df[df["esm_lr_auroc"].notna()]
    top = (df.sort_values("esm_lr_auroc", ascending=False).head(top_n)
           .sort_values("esm_lr_auroc", ascending=True).reset_index(drop=True))
    x = np.arange(len(top))
    w = 0.4

    fig, ax = plt.subplots(figsize=(max(9.0, 0.62 * len(top) + 2.0), 6.0))
    b_esm = ax.bar(x - w / 2, top["esm_lr_auroc"], width=w, color=ESM_COLOUR, edgecolor="black",
                   linewidth=0.5, label="ESM-C per-gene LR", zorder=3)
    b_ft = ax.bar(x + w / 2, top["ft_lr_auroc"], width=w, color=FT_COLOUR, edgecolor="black",
                  linewidth=0.5, label="fine-tuned Bacformer per-gene LR", zorder=3)
    for rect in list(b_esm) + list(b_ft):
        h = rect.get_height()
        if not np.isnan(h):
            ax.text(rect.get_x() + rect.get_width() / 2, h + 0.006, f"{h:.2f}",
                    rotation=90, ha="center", va="bottom", fontsize=6.5)

    ax.axhline(0.5, color="0.6", linestyle=":", linewidth=1.0, zorder=1)
    ax.text(len(top) - 0.5, 0.5, " chance", color="0.5", fontsize=7.5, ha="right", va="bottom")
    if ceiling is not None:
        ax.axhline(ceiling, color=CEILING_COLOUR, linestyle="--", linewidth=1.3, zorder=1)
        ax.text(0.0, ceiling + 0.006, f"Kleborate ceiling = {ceiling:.3f}",
                color=CEILING_COLOUR, fontsize=7.5, ha="left", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(top["gene_name"], rotation=45, ha="right", fontsize=9, fontstyle="italic")
    ax.set_ylabel("per-gene LR out-of-fold AUROC (eval holdout)", fontsize=11.5)
    ax.set_ylim(0.45, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.95)
    ax.set_title(f"{drug}: per-gene ESM-C LR vs fine-tuned Bacformer LR — top {len(top)} genes by ESM-LR\n"
                 f"purple = raw ESM-C embedding · indigo = fine-tuned Bacformer contextualised token "
                 f"(same samples, zero-imputed out-of-fold k-fold)",
                 fontsize=10.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """CLI entry point."""
    vis = visualisations_dir("kp")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drug", type=str, required=True)
    parser.add_argument("--csv", type=Path, required=True, help="esm_vs_ft_per_gene_<drug>.csv.")
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--kleborate-csv", type=Path, default=None,
                        help="Default: kp_<drug>/kleborate_determinant_lr_<drug>.csv (draws the ceiling line).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Default: kp_<drug>/<drug>_esm_vs_ft_per_gene.png.")
    args = parser.parse_args()
    drug_dir = vis / args.drug
    kleborate_csv = args.kleborate_csv or drug_dir / f"kleborate_determinant_lr_{args.drug}.csv"
    out = args.out or drug_dir / f"{args.drug}_esm_vs_ft_per_gene.png"
    plot_esm_vs_ft(args.csv, out, drug=args.drug, top_n=args.top_n,
                   ceiling=_kleborate_ceiling(kleborate_csv))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

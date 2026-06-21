"""Causal-gene ESM-LR scorecard — where the *known* resistance genes land in the per-gene ESM ranking.

The per-gene ESM-LR ranking (``snp_embeddings.build_per_gene_lr_store``) ranks every gene by "does this
gene's own mean-pooled ESM-C protein vector predict resistance". For most Kp drugs the top of that
ranking is **lineage-correlated accessory genes**, not the causal determinant — the lineage shortcut.

This module asks the sharper question: for each drug, take the genes that *are* the resistance
mechanism (the ones inside the Kleborate one-hot — gyrA/parC for FQ, pmrB/phoPQ/mgrB for colistin, the
bla genes for β-lactams, …) and report **what each scored on the ESM-LR and where it ranks**, against the
Kleborate one-hot ceiling. It exposes three failure modes of the mean-pooled per-gene read-out:

1. **SNP genes** (gyrA) — the point mutation survives mean-pooling only partly: high-ish AUROC, near the
   top of the ranking, but below the one-hot SNP (mean-pooling dilution).
2. **Subtle regulators** (pmrB/phoPQ) — washed out to ≈chance, buried deep in the ranking.
3. **Acquired genes** (mph(A), tet(A), bla*) — ≈chance, because the per-gene LR *conditions on the gene
   being present* and so is blind to the presence/absence signal the one-hot uses.

Inputs (per drug): the ranking ``per_gene_lr_<drug>.csv`` + the ceiling
``kp_<drug>/kleborate_determinant_lr_<drug>.csv``. Login/CPU. Writes a per-drug scorecard CSV + bar
chart and a combined panel CSV + summary figure. Drug→causal-gene patterns are a curated clinical-
pharmacology map (the analogue of ``kleborate_determinant_lr.DRUG_COLUMNS``).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ALL_KEY = "__ALL_Kleborate__"
CAUSAL_COLOUR = "#7e3f9e"     # purple — ESM single-gene (family colour)
CEILING_COLOUR = "#c0392b"    # red — Kleborate one-hot ceiling
LINEAGE_COLOUR = "#9aa3ad"    # grey — the top lineage marker (the shortcut)

# Resistance-gene name patterns by drug class (matched against the ranking's Bakta `gene_name`,
# case-insensitively). Anchored so e.g. `tet(` catches tet(A)/tet(D), `bla` catches every blaXXX.
_FQ = [r"^gyrA$", r"^parC$", r"^gyrB$", r"^parE$", r"^qnr", r"^oqxA$", r"^oqxB$", r"^aac\(6'\)-Ib"]
_COLISTIN = [r"^pmrA$", r"^pmrB$", r"^phoP$", r"^phoQ$", r"^mgrB$", r"^crrA$", r"^crrB$", r"^mcr"]
_MACROLIDE = [r"^mph", r"^ere", r"^erm", r"^msr", r"^mef", r"^oqxB$", r"^acrB$"]
_TETRACYCLINE = [r"^tet", r"^ramR$", r"^ramA$", r"^acrA$", r"^acrB$", r"^oqxA$", r"^oqxB$", r"^marR$"]
_AMINOGLYCOSIDE = [r"^aac\(", r"^aph\(", r"^ant\(", r"^aad", r"^armA$", r"^rmt", r"^npmA$"]
_BETALACTAM = [r"^bla", r"^ampC$", r"^ampH$", r"^ompK3[567]$"]
_SULFA_TRIM = [r"^dfr", r"^sul[0-9]"]

DRUG_CAUSAL: dict[str, list[str]] = {
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


def _match(gene: str, patterns: list[str]) -> bool:
    """True if ``gene`` matches any causal pattern (case-insensitive)."""
    return any(re.search(p, gene, flags=re.IGNORECASE) for p in patterns)


def build_scorecard(drug: str, ranking_csv: Path, kleborate_csv: Path) -> tuple[pd.DataFrame, dict]:
    """Locate ``drug``'s causal genes in the ESM-LR ranking; return (scorecard df, context dict).

    The ranking CSV is sorted by AUROC descending, so a gene's row position is its rank. Returns one row
    per matched causal gene (auroc, rank, percentile, prevalence, n_pos) and a context dict with the
    Kleborate ceiling, the top-ranked (lineage) gene, and the total gene count.
    """
    patterns = DRUG_CAUSAL.get(drug)
    if patterns is None:
        raise ValueError(f"No causal-gene patterns defined for {drug!r}")
    rank_df = pd.read_csv(ranking_csv)
    auroc_col = next(c for c in rank_df.columns if c.startswith("lr_auroc_"))
    rank_df = rank_df.sort_values(auroc_col, ascending=False).reset_index(drop=True)
    rank_df["rank"] = rank_df.index + 1
    total = len(rank_df)

    hits = rank_df[rank_df["gene_name"].astype(str).apply(lambda g: _match(g, patterns))].copy()
    hits = hits.sort_values(auroc_col, ascending=False)
    rows = [{
        "drug": drug, "gene_name": r["gene_name"], "esm_lr_auroc": float(r[auroc_col]),
        "rank": int(r["rank"]), "total_genes": total, "percentile": 100.0 * (1 - (r["rank"] - 1) / total),
        "prevalence": float(r["prevalence"]), "n_pos": int(r["n_pos"]),
    } for _, r in hits.iterrows()]
    scorecard = pd.DataFrame(rows)

    ceiling = None
    if kleborate_csv.exists():
        kdf = pd.read_csv(kleborate_csv)
        crow = kdf[kdf["gene_name"] == ALL_KEY]
        ceiling = float(crow["mut_auroc"].iloc[0]) if not crow.empty else None
    top = rank_df.iloc[0]
    context = {
        "drug": drug, "kleborate_ceiling": ceiling, "total_genes": total,
        "top_gene": str(top["gene_name"]), "top_gene_auroc": float(top[auroc_col]),
        "best_causal_auroc": float(scorecard["esm_lr_auroc"].max()) if not scorecard.empty else None,
        "best_causal_gene": scorecard.iloc[0]["gene_name"] if not scorecard.empty else None,
    }
    return scorecard, context


def plot_scorecard(scorecard: pd.DataFrame, context: dict, out_path: Path) -> None:
    """Per-drug bars: each causal gene's ESM-LR AUROC (rank annotated), vs the Kleborate ceiling + chance."""
    drug = context["drug"]
    sc = scorecard.sort_values("esm_lr_auroc", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.5, max(4.2, 0.5 * len(sc) + 2.2)))
    y = range(len(sc))
    ax.barh(y, sc["esm_lr_auroc"], color=CAUSAL_COLOUR, edgecolor="black", linewidth=0.6, height=0.66)
    for yi, r in zip(y, sc.itertuples(), strict=True):
        ax.text(r.esm_lr_auroc + 0.006, yi, f"{r.esm_lr_auroc:.3f}  (rank {r.rank}/{r.total_genes})",
                va="center", ha="left", fontsize=8.5)

    ax.axvline(0.5, color="0.6", linestyle=":", linewidth=1.0)
    ax.text(0.5, len(sc) - 0.45, "chance", color="0.5", fontsize=8, ha="center", va="bottom")
    ceiling = context.get("kleborate_ceiling")
    if ceiling is not None:
        ax.axvline(ceiling, color=CEILING_COLOUR, linestyle="--", linewidth=1.5)
        ax.text(ceiling, -0.7, f"Kleborate one-hot ceiling = {ceiling:.3f}",
                color=CEILING_COLOUR, fontsize=8.5, ha="center", va="top")
    # The top-ranked (lineage) gene, for contrast — the shortcut that out-scores the causal genes.
    tg, ta = context["top_gene"], context["top_gene_auroc"]
    ax.axvline(ta, color=LINEAGE_COLOUR, linestyle="-.", linewidth=1.3)
    ax.text(ta, len(sc) - 0.45, f"top ESM gene: {tg} = {ta:.3f}", color="0.4",
            fontsize=8, ha="center", va="bottom", rotation=0)

    ax.set_yticks(list(y))
    ax.set_yticklabels(sc["gene_name"], fontsize=10, fontstyle="italic")
    ax.set_xlabel("per-gene ESM-LR out-of-fold AUROC", fontsize=11.5)
    ax.set_xlim(0.4, 1.02)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{drug}: where the causal genes score on the ESM-LR (vs the Kleborate one-hot)",
                 fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_panel_summary(contexts: list[dict], out_path: Path) -> None:
    """Across drugs: Kleborate one-hot ceiling vs best causal-gene ESM-LR vs top (lineage) ESM gene."""
    rows = [c for c in contexts if c.get("kleborate_ceiling") is not None and c.get("best_causal_auroc")]
    df = pd.DataFrame(rows).sort_values("kleborate_ceiling", ascending=True).reset_index(drop=True)
    x = range(len(df))
    fig, ax = plt.subplots(figsize=(15.0, 7.0))
    ax.plot(df["kleborate_ceiling"], list(x), "D", color=CEILING_COLOUR, markersize=8,
            label="Kleborate one-hot ceiling")
    ax.plot(df["best_causal_auroc"], list(x), "o", color=CAUSAL_COLOUR, markersize=8,
            label="best causal-gene ESM-LR")
    ax.plot(df["top_gene_auroc"], list(x), "s", color=LINEAGE_COLOUR, markersize=7,
            label="top ESM gene (lineage marker)")
    for xi, r in zip(x, df.itertuples(), strict=True):
        ax.plot([r.best_causal_auroc, r.kleborate_ceiling], [xi, xi], color="0.8", lw=1.0, zorder=0)

    ax.axvline(0.5, color="0.6", linestyle=":", linewidth=1.0)
    ax.set_yticks(list(x))
    ax.set_yticklabels(df["drug"], fontsize=9.5)
    ax.set_xlabel("AUROC", fontsize=12)
    ax.set_xlim(0.45, 1.02)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    ax.set_title("Kp panel: the causal gene's mean-pooled ESM-LR vs its one-hot — and the lineage shortcut\n"
                 "where the grey square sits above the purple circle, an accessory lineage marker beats the "
                 "actual resistance gene", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """CLI entry point."""
    here = Path(__file__).resolve().parent
    vis = here / "docs" / "visualisations"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ranking-dir", type=Path, required=True,
                        help="Dir of per_gene_lr_ranking/<drug>/per_gene_lr_<drug>.csv (fetched from RDS).")
    parser.add_argument("--drugs", type=str, nargs="*", default=None,
                        help="Drugs to score (default: every drug with both a ranking CSV and causal patterns).")
    parser.add_argument("--out-dir", type=Path, default=vis / "causal_gene_scorecard")
    args = parser.parse_args()

    drugs = args.drugs or sorted(
        d for d in DRUG_CAUSAL if (args.ranking_dir / d / f"per_gene_lr_{d}.csv").exists()
    )
    all_scorecards, contexts = [], []
    for drug in drugs:
        ranking_csv = args.ranking_dir / drug / f"per_gene_lr_{drug}.csv"
        if not ranking_csv.exists():
            print(f"skip {drug}: no ranking CSV")
            continue
        kleborate_csv = vis / f"kp_{drug}" / f"kleborate_determinant_lr_{drug}.csv"
        scorecard, context = build_scorecard(drug, ranking_csv, kleborate_csv)
        if scorecard.empty:
            print(f"skip {drug}: no causal genes matched in the ranking")
            continue
        all_scorecards.append(scorecard)
        contexts.append(context)
        plot_scorecard(scorecard, context, args.out_dir / f"{drug}_causal_gene_scorecard.png")
        print(f"{drug}: best causal {context['best_causal_gene']}={context['best_causal_auroc']:.3f} "
              f"vs ceiling {context['kleborate_ceiling']} vs top {context['top_gene']}={context['top_gene_auroc']:.3f}")

    if all_scorecards:
        combined = pd.concat(all_scorecards, ignore_index=True)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        combined.to_csv(args.out_dir / "causal_gene_scorecard_all.csv", index=False)
        pd.DataFrame(contexts).to_csv(args.out_dir / "causal_gene_scorecard_summary.csv", index=False)
        plot_panel_summary(contexts, args.out_dir / "causal_gene_scorecard_panel.png")
        print(f"Wrote {len(all_scorecards)} per-drug scorecards + combined CSV + panel to {args.out_dir}")


if __name__ == "__main__":
    main()

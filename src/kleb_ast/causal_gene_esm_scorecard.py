"""Causal-gene ESM-LR scorecard — does the mean-pooled per-gene ESM read-out match the one-hot?

The per-gene ESM-LR ranking (``snp_embeddings.build_per_gene_lr_store``) ranks every gene by "does this
gene's own mean-pooled ESM-C protein vector predict resistance". For most Kp drugs the *top* of that
ranking is lineage-correlated accessory genes — the lineage shortcut. This module asks the sharper,
correct question: for each drug, take the genes that *are* the resistance mechanism (the ones inside the
Kleborate one-hot — gyrA/parC for FQ, pmrB/phoPQ/mgrB for colistin, the bla genes for β-lactams, …) and
compare **what each scored on the ESM-LR against the one-hot for its *own mechanism***:

- a **chromosomal-mutation** gene (gyrA) is compared to the **mutations** determinant one-hot
  (``Flq_mutations``), NOT the combined ``__ALL_Kleborate__`` ceiling (which also adds the acquired
  mechanisms — a different, additive signal the single gene should not be measured against);
- an **acquired** gene (qnr, bla*, mph) is compared to the **acquired** determinant one-hot.

The result (cipro): ESM-LR(gyrA)=0.911 == one-hot(Flq_mutations)=0.911 — the mean-pooled embedding is an
*exact* substitute for the one-hot mutation, **no mean-pooling penalty**. Where the ESM-LR falls short is
the **acquired** genes: the per-gene LR conditions on the gene being present, so it is structurally blind
to the presence/absence signal the acquired one-hot uses (qnr 0.49 vs Flq_acquired 0.68).

Inputs (per drug): the ranking ``per_gene_lr_<drug>.csv`` + the ceiling
``kp_<drug>/kleborate_determinant_lr_<drug>.csv``. Login/CPU. Writes a per-drug scorecard CSV + bar chart
and a combined panel CSV + summary figure. Drug→causal-gene patterns (with a mutation/acquired tag) are a
curated clinical-pharmacology map (the analogue of ``kleborate_determinant_lr.DRUG_COLUMNS``).
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
ESM_COLOUR = "#7e3f9e"        # purple — the drop-absent per-gene ESM-LR bar
IMPUTED_COLOUR = "#2a9d8f"    # teal — the zero-imputed ESM-LR (presence/absence recovered)
ONEHOT_COLOUR = "#c0392b"     # red — the matching-mechanism Kleborate one-hot marker
LINEAGE_COLOUR = "#9aa3ad"    # grey — the top-ranked (lineage) gene, for contrast

# Kleborate determinant categories that are a *mutation/loss* mechanism vs an *acquired* gene. A
# chromosomal-mutation causal gene is compared to the strongest mutation-category determinant; an acquired
# gene to the acquired-category determinant. (Kleborate tags Flq_mutations as chromosomal_mutation but
# Col_mutations as truncation_lof, so we match on the *set* of mutation-type categories, not one label.)
_MUT_CATS = {"chromosomal_mutation", "truncation_lof", "porin_truncation", "chromosomal_coding"}
_ACQ_CATS = {"acquired_hgt"}

# Resistance-gene name patterns (matched case-insensitively against the ranking's Bakta `gene_name`),
# each tagged "mutation" (chromosomal point-mutation / loss) or "acquired" (HGT gene). Anchored so e.g.
# `tet(` catches tet(A)/tet(D), `bla` catches every blaXXX.
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
MECH_LABEL = {"mutation": "chromosomal mutation / loss", "acquired": "acquired gene (HGT)"}


def _match(gene: str, patterns: list[tuple[str, str]]) -> str | None:
    """Return the mechanism tag of the first pattern ``gene`` matches, or None."""
    for pat, mech in patterns:
        if re.search(pat, gene, flags=re.IGNORECASE):
            return mech
    return None


def _mechanism_onehot(kdf: pd.DataFrame, mechanism: str) -> tuple[float | None, str | None]:
    """Strongest matching-mechanism one-hot (auroc, determinant name) from a drug's Kleborate CSV."""
    cats = _MUT_CATS if mechanism == "mutation" else _ACQ_CATS
    rows = kdf[(kdf["gene_name"] != ALL_KEY) & (kdf["category"].isin(cats))]
    if rows.empty:
        return None, None
    best = rows.sort_values("mut_auroc", ascending=False).iloc[0]
    return float(best["mut_auroc"]), str(best["site"])


def _ranking_lookup(csv: Path) -> dict[str, tuple[float, int]]:
    """Map gene_name -> (auroc, rank) from a per-gene ranking CSV (rank = row position, sorted desc)."""
    df = pd.read_csv(csv)
    col = next(c for c in df.columns if c.startswith("lr_auroc_"))
    df = df.sort_values(col, ascending=False).reset_index(drop=True)
    return {str(g): (float(a), i + 1) for i, (g, a) in enumerate(zip(df["gene_name"], df[col], strict=True))}


def build_scorecard(
    drug: str, ranking_csv: Path, kleborate_csv: Path, imputed_ranking_csv: Path | None = None
) -> tuple[pd.DataFrame, dict]:
    """Locate ``drug``'s causal genes and pair each with its own-mechanism one-hot (and imputed ESM-LR).

    Locates the genes in the drop-absent ESM-LR ranking; if ``imputed_ranking_csv`` is given, also attaches
    each gene's zero-imputed ESM-LR (so each shows drop-absent vs imputed vs one-hot — the acquired-gene
    recovery test).
    """
    patterns = DRUG_CAUSAL.get(drug)
    if patterns is None:
        raise ValueError(f"No causal-gene patterns defined for {drug!r}")
    rank_df = pd.read_csv(ranking_csv)
    auroc_col = next(c for c in rank_df.columns if c.startswith("lr_auroc_"))
    rank_df = rank_df.sort_values(auroc_col, ascending=False).reset_index(drop=True)
    rank_df["rank"] = rank_df.index + 1
    total = len(rank_df)

    kdf = pd.read_csv(kleborate_csv) if kleborate_csv.exists() else pd.DataFrame()
    onehot_cache = {m: _mechanism_onehot(kdf, m) for m in ("mutation", "acquired")} if not kdf.empty else {}
    imp = _ranking_lookup(imputed_ranking_csv) if imputed_ranking_csv and imputed_ranking_csv.exists() else {}
    imp_total = len(imp) if imp else None

    rows = []
    for _, r in rank_df.iterrows():
        mech = _match(str(r["gene_name"]), patterns)
        if mech is None:
            continue
        oh_auroc, oh_name = onehot_cache.get(mech, (None, None))
        imp_auroc, imp_rank = imp.get(str(r["gene_name"]), (None, None))
        rows.append({
            "drug": drug, "gene_name": r["gene_name"], "mechanism": mech,
            "esm_lr_auroc": float(r[auroc_col]), "rank": int(r["rank"]), "total_genes": total,
            "percentile": 100.0 * (1 - (r["rank"] - 1) / total),
            "imputed_esm_lr_auroc": imp_auroc, "imputed_rank": imp_rank, "imputed_total_genes": imp_total,
            "onehot_determinant": oh_name, "onehot_auroc": oh_auroc,
            "esm_minus_onehot": (float(r[auroc_col]) - oh_auroc) if oh_auroc is not None else None,
            "imputed_minus_onehot": (imp_auroc - oh_auroc) if (imp_auroc is not None and oh_auroc is not None) else None,
            "prevalence": float(r["prevalence"]), "n_pos": int(r["n_pos"]),
        })
    scorecard = pd.DataFrame(rows).sort_values("esm_lr_auroc", ascending=False).reset_index(drop=True)

    top = rank_df.iloc[0]
    ceiling = None
    if not kdf.empty:
        crow = kdf[kdf["gene_name"] == ALL_KEY]
        ceiling = float(crow["mut_auroc"].iloc[0]) if not crow.empty else None
    # Headline gene per drug = the one with the highest read-out the per-gene method achieves. Use the
    # imputed AUROC to pick it when available (an acquired gene can leap past the chromosomal ones on
    # imputation — gentamicin aac(3), meropenem blaKPC), else the drop-absent AUROC. Report that single
    # gene's drop / imputed / own-mechanism one-hot so the three panel markers are the SAME gene.
    best = None
    if not scorecard.empty:
        pick_col = "imputed_esm_lr_auroc" if scorecard["imputed_esm_lr_auroc"].notna().any() else "esm_lr_auroc"
        best = scorecard.loc[scorecard[pick_col].idxmax()]
    context = {
        "drug": drug, "total_genes": total, "combined_ceiling": ceiling,
        "top_gene": str(top["gene_name"]), "top_gene_auroc": float(top[auroc_col]),
        "best_causal_gene": None if best is None else best["gene_name"],
        "best_causal_mechanism": None if best is None else best["mechanism"],
        "best_causal_esm": None if best is None else float(best["esm_lr_auroc"]),
        "best_causal_imputed": None if best is None or pd.isna(best["imputed_esm_lr_auroc"])
                               else float(best["imputed_esm_lr_auroc"]),
        "best_causal_onehot": None if best is None else best["onehot_auroc"],
    }
    return scorecard, context


def plot_scorecard(scorecard: pd.DataFrame, context: dict, out_path: Path) -> None:
    """Per-drug chart: drop-absent ESM-LR bar, zero-imputed ESM-LR (●), and own-mechanism one-hot (◆).

    For acquired genes the teal ● leaps from the bar toward the red ◆ — the presence/absence recovery.
    """
    drug = context["drug"]
    sc = scorecard.sort_values("esm_lr_auroc", ascending=True).reset_index(drop=True)
    has_imputed = sc["imputed_esm_lr_auroc"].notna().any()
    fig, ax = plt.subplots(figsize=(11.5, max(4.2, 0.52 * len(sc) + 2.2)))
    y = list(range(len(sc)))
    ax.barh(y, sc["esm_lr_auroc"], color=ESM_COLOUR, edgecolor="black", linewidth=0.6, height=0.6,
            zorder=2, label="drop-absent ESM-LR")
    for yi, r in zip(y, sc.itertuples(), strict=True):
        pts = [r.esm_lr_auroc]
        if has_imputed and r.imputed_esm_lr_auroc is not None and not pd.isna(r.imputed_esm_lr_auroc):
            pts.append(r.imputed_esm_lr_auroc)
        if r.onehot_auroc is not None:
            pts.append(r.onehot_auroc)
        ax.plot([min(pts), max(pts)], [yi, yi], color="0.65", lw=1.0, zorder=1)  # connector
        if has_imputed and r.imputed_esm_lr_auroc is not None and not pd.isna(r.imputed_esm_lr_auroc):
            ax.plot(r.imputed_esm_lr_auroc, yi, "o", color=IMPUTED_COLOUR, markersize=8.5, zorder=3,
                    markeredgecolor="black", markeredgewidth=0.5)
        if r.onehot_auroc is not None:
            ax.plot(r.onehot_auroc, yi, "D", color=ONEHOT_COLOUR, markersize=8, zorder=4)
        tag = "chr" if r.mechanism == "mutation" else "acq"
        ax.text(0.405, yi, f"({tag})", va="center", ha="left", fontsize=7.5, color="0.45")
        lab = f"drop {r.esm_lr_auroc:.3f}"
        if has_imputed and r.imputed_esm_lr_auroc is not None and not pd.isna(r.imputed_esm_lr_auroc):
            lab += f" → imp {r.imputed_esm_lr_auroc:.3f}"
        ax.text(max(pts) + 0.008, yi, lab, va="center", ha="left", fontsize=8)

    ax.axvline(0.5, color="0.6", linestyle=":", linewidth=1.0, zorder=0)
    ax.text(0.5, len(sc) - 0.4, "chance", color="0.5", fontsize=8, ha="center", va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels(sc["gene_name"], fontsize=10, fontstyle="italic")
    ax.set_xlabel("AUROC — drop-absent ESM-LR (bar) · zero-imputed ESM-LR (●) · own-mechanism one-hot (◆)",
                  fontsize=10.5)
    ax.set_xlim(0.4, 1.10)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ESM_COLOUR, ec="black", lw=0.6),
               plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=IMPUTED_COLOUR, markersize=9,
                          markeredgecolor="black"),
               plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=ONEHOT_COLOUR, markersize=9)]
    labels = ["drop-absent ESM-LR", "zero-imputed ESM-LR", "one-hot (same mechanism)"]
    if not has_imputed:
        handles, labels = [handles[0], handles[2]], [labels[0], labels[2]]
    ax.legend(handles, labels, loc="lower right", fontsize=9, framealpha=0.95)
    title = (f"{drug}: drop-absent vs zero-imputed ESM-LR vs the one-hot (per causal gene)"
             if has_imputed else f"{drug}: does the ESM-LR of each causal gene match its one-hot?")
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_panel_summary(contexts: list[dict], out_path: Path) -> None:
    """Across drugs: best causal gene's drop-absent ESM-LR (○), zero-imputed ESM-LR (●), own-mechanism one-hot (◆).

    The headline: imputation moves the ESM point onto the one-hot for the acquired-gene drugs.
    """
    rows = [c for c in contexts if c.get("best_causal_esm") is not None and c.get("best_causal_onehot") is not None]
    df = pd.DataFrame(rows).sort_values("best_causal_onehot", ascending=True).reset_index(drop=True)
    has_imp = "best_causal_imputed" in df.columns and df["best_causal_imputed"].notna().any()
    y = list(range(len(df)))
    fig, ax = plt.subplots(figsize=(13.5, 7.8))
    for yi, r in zip(y, df.itertuples(), strict=True):
        xs = [r.best_causal_esm, r.best_causal_onehot]
        if has_imp and not pd.isna(getattr(r, "best_causal_imputed", float("nan"))):
            xs.append(r.best_causal_imputed)
        ax.plot([min(xs), max(xs)], [yi, yi], color="0.8", lw=1.4, zorder=0)
    ax.plot(df["best_causal_onehot"], y, "D", color=ONEHOT_COLOUR, markersize=9, zorder=3,
            label="own-mechanism one-hot")
    ax.plot(df["best_causal_esm"], y, "o", color="w", markeredgecolor=ESM_COLOUR, markeredgewidth=1.6,
            markersize=9, zorder=3, label="drop-absent ESM-LR")
    if has_imp:
        ax.plot(df["best_causal_imputed"], y, "o", color=IMPUTED_COLOUR, markersize=9, zorder=4,
                markeredgecolor="black", markeredgewidth=0.5, label="zero-imputed ESM-LR")

    ax.axvline(0.5, color="0.6", linestyle=":", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(df["drug"], fontsize=9.5)
    ax.set_xlabel("AUROC", fontsize=12)
    ax.set_xlim(0.4, 1.04)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    sub = ("zero-imputing (teal ●) lifts the per-gene ESM-LR sharply toward the one-hot — a single gene "
           "still trails the combined determinant; chromosomal-mutation drugs (gyrA) already coincide" if has_imp
           else "chromosomal drugs coincide (mean-pool = one-hot); acquired drugs — ESM falls short")
    ax.set_title(f"Kp panel: best causal gene's ESM-LR vs the one-hot for its own mechanism\n{sub}", fontsize=11.5)
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
                        help="Dir of per_gene_lr_ranking/<drug>/per_gene_lr_<drug>.csv (drop-absent; fetched from RDS).")
    parser.add_argument("--imputed-ranking-dir", type=Path, default=None,
                        help="Optional dir of the zero-imputed rankings (per_gene_lr_ranking_imputed/<drug>/...); "
                             "when given, each gene also shows its imputed ESM-LR (the acquired-gene recovery).")
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
        imputed_csv = (args.imputed_ranking_dir / drug / f"per_gene_lr_{drug}.csv"
                       if args.imputed_ranking_dir else None)
        kleborate_csv = vis / f"kp_{drug}" / f"kleborate_determinant_lr_{drug}.csv"
        scorecard, context = build_scorecard(drug, ranking_csv, kleborate_csv, imputed_ranking_csv=imputed_csv)
        if scorecard.empty:
            print(f"skip {drug}: no causal genes matched in the ranking")
            continue
        all_scorecards.append(scorecard)
        contexts.append(context)
        plot_scorecard(scorecard, context, args.out_dir / f"{drug}_causal_gene_scorecard.png")
        bc_oh = context["best_causal_onehot"]
        print(f"{drug}: best causal {context['best_causal_gene']} ESM={context['best_causal_esm']:.3f} "
              f"vs own-mechanism one-hot {bc_oh:.3f}" if bc_oh is not None else f"{drug}: (no one-hot)")

    if all_scorecards:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        pd.concat(all_scorecards, ignore_index=True).to_csv(args.out_dir / "causal_gene_scorecard_all.csv", index=False)
        pd.DataFrame(contexts).to_csv(args.out_dir / "causal_gene_scorecard_summary.csv", index=False)
        plot_panel_summary(contexts, args.out_dir / "causal_gene_scorecard_panel.png")
        print(f"Wrote {len(all_scorecards)} per-drug scorecards + combined CSV + panel to {args.out_dir}")


if __name__ == "__main__":
    main()

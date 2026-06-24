"""Plot #2 data — per-CARD-gene one-hot LR + the ``__ALL_CARD__`` ceiling (our calls, not metadata columns).

The CARD analogue of :mod:`kleb_ast.kleborate_determinant_lr`. Where that one-hot-encodes Kleborate's
determinant *columns* from ``metadata_v2``, this one-hot-encodes **our own CARD gene/allele presence** from
the ``{Sample}_amr.parquet`` sidecars and scores ``presence → drug label`` through the same k-fold × m-seed
harness (:func:`snp_embeddings.kfold_probe.run_kfold_probe`). Per drug it emits, at family **and** allele
grain:

- one **bar per CARD gene** (carried by ≥ ``MIN_DETERMINANT_GENOMES`` genomes), tagged by mechanism
  ``category`` (acquired_hgt / chromosomal_coding / chromosomal_mutation / porin_truncation / truncation_lof
  — the HGT-vs-chromosomal axis) and flagged ``is_causal``; and
- the full per-drug one-hot together — row ``__ALL_CARD__``, the catalogue ceiling.

This is the *one-hot* (presence/absence) counterpart of Plot #1's per-gene *ESM-embedding* LR: same gene
universe, different feature. Renders with the shared cause histogram (:mod:`kleb_ast.
plot_kleborate_cause_histogram`, ``source_name="CARD"``). sklearn over a sparse binary matrix — a short CPU
job (reads the combined sidecar store, no GPU).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from kleb_ast.build_amr_calls_store import load_calls
from kleb_ast.card_label import causal_genes_for_drug, determinant_genes_for_drug
from kleb_ast.kleborate_determinant_lr import (
    MIN_DETERMINANT_GENOMES,
    _score,
    load_labels,
)
from kleb_ast.validate_amr_annotation import DEFAULT_SIDECAR_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ALL_KEY = "__ALL_CARD__"
# How many per-gene bars to keep for the histogram (most-prevalent genes), always unioned with causal genes.
MAX_BARS = 16

# Chromosomal display gene → cause-histogram category (everything else is an acquired HGT gene).
_CHROM_CATEGORY: dict[str, str] = {
    "GyrA": "chromosomal_mutation", "ParC": "chromosomal_mutation",
    "OmpK35": "porin_truncation", "OmpK36": "porin_truncation",
    "MgrB": "truncation_lof", "PmrB": "truncation_lof",
    "SHV-OKP-LEN": "chromosomal_coding",
}
_EMBEDDABLE = frozenset({"acquired_hgt", "chromosomal_coding"})


def _category(gene: str) -> str:
    """Mechanism category for a CARD gene (chromosomal display genes mapped, else acquired HGT)."""
    return _CHROM_CATEGORY.get(gene, "acquired_hgt")


def build_card_onehot(calls: pd.DataFrame, genes: set[str], universe: list[str], grain: str) -> pd.DataFrame:
    """Genomes × CARD-gene binary presence frame over ``universe`` for the drug's determinant ``genes``."""
    label_col = "amr_gene_family" if grain == "family" else "amr_allele"
    sub = calls[calls["amr_source"].isin(["acquired", "chromosomal"])]
    sub = sub[sub[label_col].isin(genes)].copy()
    sub["Sample"] = sub["Sample"].astype(str)
    sub = sub[sub["Sample"].isin(universe)]
    if sub.empty:
        return pd.DataFrame(index=universe)
    oh = pd.crosstab(sub["Sample"], sub[label_col]).clip(upper=1)
    return oh.reindex(universe).fillna(0).astype(int)


def score_drug(calls: pd.DataFrame, ast_sheet: Path, drug: str, *, grain: str,
               seeds: tuple[int, ...]) -> pd.DataFrame | None:
    """Per-CARD-gene one-hot LR + ``__ALL_CARD__`` ceiling for one drug/grain → the cause-histogram frame."""
    label_map = load_labels(ast_sheet, drug)
    universe = sorted(label_map)
    determ = determinant_genes_for_drug(drug, grain=grain)
    causal = causal_genes_for_drug(drug, grain=grain)
    oh = build_card_onehot(calls, determ, universe, grain)
    if oh.shape[1] == 0:
        logger.warning("%s (%s): no CARD determinants present — skipping", drug, grain)
        return None

    carriers = (oh > 0).sum(axis=0)
    keep = carriers[carriers >= MIN_DETERMINANT_GENOMES].sort_values(ascending=False)
    # display set: most-prevalent up to MAX_BARS, always unioned with causal genes that clear MIN
    display_genes = list(dict.fromkeys(list(keep.head(MAX_BARS).index)
                                       + [g for g in keep.index if g in causal]))
    rows = []
    for gene in display_genes:
        agg = _score(oh[[gene]], label_map, seeds)
        if agg is None:
            continue
        cat = _category(gene)
        rows.append({
            "gene_name": gene, "site": gene, "category": cat,
            "mut_auroc": agg["auroc"]["mean"], "mut_auroc_sd": agg["auroc"]["sd"],
            "mut_auprc": agg["auprc"]["mean"], "mut_auprc_sd": agg["auprc"]["sd"],
            "n_determinants": 1, "n_genomes_with_determinant": int(carriers[gene]),
            "embeddable": cat in _EMBEDDABLE, "is_causal": gene in causal,
            "is_rrna": False, "is_noncoding": cat not in _EMBEDDABLE,
        })
    full = _score(oh, label_map, seeds)
    if full is not None:
        rows.append({
            "gene_name": ALL_KEY, "site": ALL_KEY, "category": "all",
            "mut_auroc": full["auroc"]["mean"], "mut_auroc_sd": full["auroc"]["sd"],
            "mut_auprc": full["auprc"]["mean"], "mut_auprc_sd": full["auprc"]["sd"],
            "n_determinants": oh.shape[1], "n_genomes_with_determinant": int((oh.sum(axis=1) > 0).sum()),
            "embeddable": False, "is_causal": False, "is_rrna": False, "is_noncoding": False,
        })
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values("mut_auroc", ascending=False).reset_index(drop=True)


def run(calls_dir: Path, ast_sheet: Path, out_dir: Path, drugs: list[str], grains: list[str],
        seeds: tuple[int, ...] = (1, 2, 3)) -> None:
    """Score every (drug, grain) and write ``card_amr/kp_<drug>/card_determinant_lr_<drug>_<grain>.csv``."""
    calls = load_calls(calls_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for drug in drugs:
        for grain in grains:
            df = score_drug(calls, ast_sheet, drug, grain=grain, seeds=seeds)
            if df is None:
                continue
            drug_dir = out_dir / f"kp_{drug}"
            drug_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(drug_dir / f"card_determinant_lr_{drug}_{grain}.csv", index=False)
            ceil = df[df["gene_name"] == ALL_KEY]
            top = df[(df["gene_name"] != ALL_KEY)].head(3)
            logger.info("%s (%s): %d gene bars | top %s | ceiling %.3f", drug, grain, len(df) - len(ceil),
                        ", ".join(f"{r.site}={r.mut_auroc:.3f}" for r in top.itertuples()),
                        float(ceil["mut_auroc"].iloc[0]) if not ceil.empty else float("nan"))


def main() -> None:
    """CLI entry point."""
    rds = Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david")
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calls-dir", type=Path, default=DEFAULT_SIDECAR_DIR,
                   help="Sidecar dir holding amr_calls_all.parquet (else crawls the sidecars).")
    p.add_argument("--ast-sheet", type=Path,
                   default=rds / "processed" / "train_kleb_ast" / "binary_ast_with_split.csv")
    p.add_argument("--out-dir", type=Path, default=here / "docs" / "visualisations" / "card_amr")
    p.add_argument("--drugs", type=str, nargs="+", required=True)
    p.add_argument("--grains", type=str, nargs="+", default=["family", "allele"],
                   choices=["family", "allele"])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = p.parse_args()
    run(args.calls_dir, args.ast_sheet, args.out_dir, args.drugs, args.grains, tuple(args.seeds))


if __name__ == "__main__":
    main()

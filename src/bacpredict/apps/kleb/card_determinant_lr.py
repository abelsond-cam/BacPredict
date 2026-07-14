"""Plot #2 data — per-CARD-gene one-hot LR + the ``__ALL_CARD__`` ceiling (our calls, not metadata columns).

The CARD analogue of :mod:`bacpredict.apps.kleb.kleborate_determinant_lr`. Where that one-hot-encodes Kleborate's
determinant *columns* from ``metadata_v2``, this one-hot-encodes **our own CARD gene/allele presence** from
the ``{Sample}_amr.parquet`` sidecars and scores ``presence → drug label`` through the same k-fold × m-seed
harness (:func:`bacpredict.engine.gene_lr.kfold_probe.run_kfold_probe`). Per drug it emits, at family **and** allele
grain:

- one **bar per CARD gene** (carried by ≥ ``MIN_DETERMINANT_GENOMES`` genomes), tagged by mechanism
  ``category`` (acquired_hgt / chromosomal_coding / chromosomal_mutation / porin_truncation / truncation_lof
  — the HGT-vs-chromosomal axis) and flagged ``is_causal``; and
- the full per-drug one-hot together — row ``__ALL_CARD__``, the catalogue ceiling.

This is the *one-hot* (presence/absence) counterpart of Plot #1's per-gene *ESM-embedding* LR: same gene
universe, different feature. Renders with the shared cause histogram (:mod:`bacpredict.apps.kleb.
plot_kleborate_cause_histogram`, ``source_name="CARD"``). sklearn over a sparse binary matrix — a short CPU
job (reads the combined sidecar store, no GPU).
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from bacpredict.apps.kleb.build_amr_calls_store import load_calls
from bacpredict.apps.kleb.card_label import causal_genes_for_drug, determinant_genes_for_drug
from bacpredict.apps.kleb.kleborate_determinant_lr import (
    MIN_DETERMINANT_GENOMES,
    load_labels,
    tokenize_cell,
)
from bacpredict.apps.kleb.validate_amr_annotation import default_metadata, default_sidecar_dir
from bacpredict.engine.catalogue.base import score_onehot_frame
from bacpredict.engine.config import KP, visualisations_dir

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ALL_KEY = "__ALL_CARD__"
# How many per-gene bars to keep for the histogram (most-prevalent genes), always unioned with causal genes.
MAX_BARS = 18

# Chromosomal display gene → cause-histogram category (everything else is an acquired HGT gene).
_CHROM_CATEGORY: dict[str, str] = {
    "GyrA": "chromosomal_mutation", "ParC": "chromosomal_mutation",
    "OmpK35": "porin_truncation", "OmpK36": "porin_truncation",
    "MgrB": "truncation_lof", "PmrB": "truncation_lof",
    "SHV-OKP-LEN": "chromosomal_coding",
}
_EMBEDDABLE = frozenset({"acquired_hgt", "chromosomal_coding"})

# Chromosomal point-mutation / truncation genes whose *presence* one-hot is degenerate (the gene is
# intrinsic, ≈present in every genome) so it can't predict resistance — only the **mutant** does. Our
# minimap annotation can't call the codon, so the mutant status is sourced from Kleborate's mutation
# columns in metadata_v2 (user decision): each such gene is split into '<G> (mut)' (Kleborate calls a
# mutation) and '<G> (WT)' (present, no mutation), so the one-hot is no longer blind to chromosomal
# resistance. Maps display gene → (metadata mutation column, normalised token prefix).
_CHROM_MUT_COLUMN: dict[str, tuple[str, str]] = {
    "GyrA": ("Flq_mutations", "gyra"), "ParC": ("Flq_mutations", "parc"),
    "OmpK35": ("Omp_mutations", "ompk35"), "OmpK36": ("Omp_mutations", "ompk36"),
    "MgrB": ("Col_mutations", "mgrb"), "PmrB": ("Col_mutations", "pmrb"),
    "SHV-OKP-LEN": ("SHV_mutations", "shv"),
}
_NORM = re.compile(r"[^a-z0-9]")


def _norm(tok: str) -> str:
    """Lowercase + strip non-alphanumerics (so ``GyrA-83L`` → ``gyra83l`` matches the ``gyra`` prefix)."""
    return _NORM.sub("", str(tok).lower())


def _base_gene(name: str) -> str:
    """Strip a ``' (mut)'`` / ``' (WT)'`` suffix to the base gene name."""
    return name.split(" (", 1)[0]


def _category(gene: str) -> str:
    """Mechanism category; a WT split is intrinsic-coding, a mut split keeps the base gene's category."""
    if gene.endswith("(WT)"):
        return "chromosomal_coding"
    return _CHROM_CATEGORY.get(_base_gene(gene), "acquired_hgt")


def _is_causal(gene: str, causal: set[str]) -> bool:
    """A determinant is causal if its base gene is causal — but a chromosomal WT split never is."""
    if gene.endswith("(WT)"):
        return False
    return _base_gene(gene) in causal


def mutation_carriers(metadata: Path, genes: set[str], samples: set[str]) -> dict[str, set[str]]:
    """``{chromosomal gene → set(samples Kleborate calls mutated)}`` from the metadata_v2 mutation columns."""
    want = {g for g in genes if g in _CHROM_MUT_COLUMN}
    if not want:
        return {}
    cols = sorted({_CHROM_MUT_COLUMN[g][0] for g in want})
    meta = pd.read_csv(metadata, sep="\t", usecols=["Sample", *cols], low_memory=False)
    meta["Sample"] = meta["Sample"].astype(str)
    meta = meta[meta["Sample"].isin(samples)]
    out: dict[str, set[str]] = {g: set() for g in want}
    for _, row in meta.iterrows():
        toks = {c: [_norm(t) for t in tokenize_cell(row[c])] for c in cols}
        for g in want:
            col, prefix = _CHROM_MUT_COLUMN[g]
            if any(t.startswith(prefix) for t in toks[col]):
                out[g].add(row["Sample"])
    return out


def _present_by_gene(calls: pd.DataFrame, genes: set[str], universe: set[str], grain: str) -> dict[str, set]:
    """``{gene → set(samples in universe carrying it)}`` from the sidecar presence calls."""
    label_col = "amr_gene_family" if grain == "family" else "amr_allele"
    sub = calls[calls["amr_source"].isin(["acquired", "chromosomal"])]
    sub = sub[sub[label_col].isin(genes)].copy()
    sub["Sample"] = sub["Sample"].astype(str)
    sub = sub[sub["Sample"].isin(universe)]
    if sub.empty:
        return {}
    return {g: set(s) for g, s in sub.groupby(label_col)["Sample"]}


def build_card_onehot(calls: pd.DataFrame, genes: set[str], universe: list[str], grain: str,
                      *, chrom_mut: dict[str, set[str]] | None = None) -> pd.DataFrame:
    """Genomes × determinant one-hot, with chromosomal point-mutation genes split into mut/WT.

    Chromosomal genes in ``chrom_mut`` become ``'<G> (mut)'`` (Kleborate-called mutant) and ``'<G> (WT)'``
    (present, not mutant) so the feature is no longer the degenerate gene-presence; acquired genes stay
    simple presence.
    """
    chrom_mut = chrom_mut or {}
    uni = list(universe)
    uni_set = set(uni)
    present = _present_by_gene(calls, genes, uni_set, grain)
    cols: dict[str, set] = {}
    for g in genes:
        if g in chrom_mut:
            mut = chrom_mut[g] & uni_set
            wt = (present.get(g, set()) & uni_set) - mut
            if mut:
                cols[f"{g} (mut)"] = mut
            if wt:
                cols[f"{g} (WT)"] = wt
        else:
            p = present.get(g, set()) & uni_set
            if p:
                cols[g] = p
    if not cols:
        return pd.DataFrame(index=uni)
    data = {c: np.fromiter((1 if s in members else 0 for s in uni), dtype=int, count=len(uni))
            for c, members in cols.items()}
    return pd.DataFrame(data, index=uni)


def score_drug(calls: pd.DataFrame, ast_sheet: Path, drug: str, *, grain: str,
               seeds: tuple[int, ...], metadata: Path) -> pd.DataFrame | None:
    """Per-CARD-gene one-hot LR + ``__ALL_CARD__`` ceiling for one drug/grain → the cause-histogram frame."""
    label_map = load_labels(ast_sheet, drug)
    universe = sorted(label_map)
    determ = determinant_genes_for_drug(drug, grain=grain)
    causal = causal_genes_for_drug(drug, grain=grain)
    chrom_mut = mutation_carriers(metadata, determ, set(universe))
    oh = build_card_onehot(calls, determ, universe, grain, chrom_mut=chrom_mut)
    if oh.shape[1] == 0:
        logger.warning("%s (%s): no CARD determinants present — skipping", drug, grain)
        return None

    carriers = (oh > 0).sum(axis=0)
    keep = carriers[carriers >= MIN_DETERMINANT_GENOMES].sort_values(ascending=False)
    # display set: most-prevalent up to MAX_BARS, always unioned with causal determinants that clear MIN
    display_genes = list(dict.fromkeys(list(keep.head(MAX_BARS).index)
                                       + [g for g in keep.index if _is_causal(g, causal)]))
    rows = []
    for gene in display_genes:
        agg = score_onehot_frame(oh[[gene]], label_map, seeds)
        if agg is None:
            continue
        cat = _category(gene)
        rows.append({
            "gene_name": gene, "site": gene, "category": cat,
            "mut_auroc": agg["auroc"]["mean"], "mut_auroc_sd": agg["auroc"]["sd"],
            "mut_auprc": agg["auprc"]["mean"], "mut_auprc_sd": agg["auprc"]["sd"],
            "n_determinants": 1, "n_genomes_with_determinant": int(carriers[gene]),
            "embeddable": cat in _EMBEDDABLE, "is_causal": _is_causal(gene, causal),
            "is_rrna": False, "is_noncoding": cat not in _EMBEDDABLE,
        })
    full = score_onehot_frame(oh, label_map, seeds)
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
        metadata: Path, seeds: tuple[int, ...] = (1, 2, 3)) -> None:
    """Score every (drug, grain) and write ``visualisations/kp/<drug>/card_determinant_lr_<drug>_<grain>.csv``."""
    calls = load_calls(calls_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for drug in drugs:
        for grain in grains:
            df = score_drug(calls, ast_sheet, drug, grain=grain, seeds=seeds, metadata=metadata)
            if df is None:
                continue
            drug_dir = out_dir / drug
            drug_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(drug_dir / f"card_determinant_lr_{drug}_{grain}.csv", index=False)
            ceil = df[df["gene_name"] == ALL_KEY]
            top = df[(df["gene_name"] != ALL_KEY)].head(3)
            logger.info("%s (%s): %d gene bars | top %s | ceiling %.3f", drug, grain, len(df) - len(ceil),
                        ", ".join(f"{r.site}={r.mut_auroc:.3f}" for r in top.itertuples()),
                        float(ceil["mut_auroc"].iloc[0]) if not ceil.empty else float("nan"))


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calls-dir", type=Path, default=None,
                   help="Sidecar dir holding amr_calls_all.parquet (default: <data-root>/processed/"
                   "train_kleb_ast/amr_annotation; else crawls the sidecars).")
    p.add_argument("--ast-sheet", type=Path, default=None,
                   help="AST split sheet (default: <data-root>/processed/train_kleb_ast/binary_ast_with_split.csv).")
    p.add_argument("--out-dir", type=Path, default=visualisations_dir("kp"))
    p.add_argument("--metadata", type=Path, default=None,
                   help="metadata_v2 TSV (default: <data-root>/final/...) — Kleborate mutation columns "
                   "for the chromosomal mut/WT split.")
    p.add_argument("--drugs", type=str, nargs="+", required=True)
    p.add_argument("--grains", type=str, nargs="+", default=["family", "allele"],
                   choices=["family", "allele"])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = p.parse_args()
    calls_dir = args.calls_dir or default_sidecar_dir()
    ast_sheet = args.ast_sheet or KP.data_root() / "binary_ast_with_split.csv"
    metadata = args.metadata or default_metadata()
    run(calls_dir, ast_sheet, args.out_dir, args.drugs, args.grains, metadata, tuple(args.seeds))


if __name__ == "__main__":
    main()

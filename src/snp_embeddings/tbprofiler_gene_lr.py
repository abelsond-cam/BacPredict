"""Per-gene WHO-mutation LR — the *cause* ranking, including the non-embeddable rRNA sites (rrs/rrl).

The ESM per-gene ranking (``build_per_gene_lr_store``) can only score *protein-coding* genes — rRNA has
no protein to embed, so rrs/rrl (the aminoglycoside / linezolid causes) are physically absent from it.
This module fills that gap from the **mutation** side: for each drug × gene it builds a one-hot of that
gene's WHO-catalogue variants (from ``tbprofiler_variants.parquet``) and scores ``variants → drug label``
through the same k-fold × m-seed harness. Because it is mutation-based it scores *every* WHO gene,
rRNA included — so rrs/rrl finally get a predictive bar.

It also scores the **full per-drug one-hot** (all of that drug's variants together) — the WHO-catalogue
ceiling that captures *all* mechanisms (incl. the non-coding inhA/fabG1 promoter and rrs/rrl), the
comparator the protein-only concat is measured against (``one-hot − concat`` = the un-embeddable gap).

Output per drug: ``tbprofiler_gene_lr_<drug>.csv`` (gene, mut_auroc±sd, n_variants, embeddable) + a row
for the full one-hot. ``embeddable`` is cross-referenced against the ESM ranking's gene set. sklearn over
a sparse binary matrix — light; runs as a short CPU job.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path

import pandas as pd

from snp_embeddings.kfold_probe import FeatureSpec, run_kfold_probe

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# rRNA / non-coding causes that can never be ESM-embedded — flagged so the plot can hatch them.
RRNA_GENES = {"rrs", "rrl", "rrf"}
MIN_VARIANT_GENOMES = 10  # a gene needs ≥ this many genomes carrying a variant to be worth scoring


def load_labels(ast_sheet: Path, drug: str) -> dict[str, int]:
    """``Sample → 0/1`` for one drug from the AST sheet (drop NaN / ambiguous)."""
    df = pd.read_csv(ast_sheet, usecols=["Sample", drug]).dropna(subset=[drug])
    df = df[df[drug].isin([0, 1, 0.0, 1.0])]
    return {s: int(v) for s, v in zip(df["Sample"], df[drug], strict=True)}


def _gene_onehot(sub: pd.DataFrame, labelled: list[str]) -> pd.DataFrame:
    """Genomes × variant_id binary frame over ``labelled`` (genomes with no variant → all-zero rows)."""
    oh = pd.crosstab(sub["Sample"], sub["variant_id"]).clip(upper=1)
    return oh.reindex(labelled).fillna(0).astype(int)


def _score(frame: pd.DataFrame, label_map: dict[str, int], seeds: tuple[int, ...]) -> dict | None:
    """k-fold AUROC mean/sd for one binary feature frame (or None if degenerate)."""
    if frame.shape[1] == 0 or frame.sum().sum() == 0:
        return None
    kf = run_kfold_probe({"f": FeatureSpec(frame, kind="numeric", standardise=False)},
                         label_map, n_folds=5, seeds=seeds, evaluate_seed=1, evaluate_fraction=0.20)
    return kf["frames"]["f"]["aggregate"]["auroc"]


def run(variants_parquet: Path, ast_sheet: Path, esm_rank_dir: Path, out_dir: Path,
        drugs: list[str], seeds: tuple[int, ...] = (1, 2, 3)) -> None:
    """Per drug: per-gene WHO-mutation LR (incl. rRNA) + the full one-hot, vs the ESM-embeddable set."""
    variants = pd.read_parquet(variants_parquet)
    variants["drug_set"] = variants["drugs"].fillna("").str.split(";")
    out_dir.mkdir(parents=True, exist_ok=True)

    for drug in drugs:
        label_map = load_labels(ast_sheet, drug)
        labelled = sorted(label_map)
        dv = variants[variants["drug_set"].apply(lambda s: drug in s)]
        dv = dv[dv["Sample"].isin(label_map)]
        if dv.empty:
            logger.warning("%s: no WHO variants — skipping", drug)
            continue

        # Which genes are ESM-embeddable (present in the protein-LR ranking for this drug)?
        rank_csv = esm_rank_dir / drug / f"per_gene_lr_{drug}.csv"
        embeddable = set(pd.read_csv(rank_csv)["gene_name"]) if rank_csv.exists() else set()

        rows = []
        for gene, g in dv.groupby("gene_name"):
            n_genomes = g["Sample"].nunique()
            if n_genomes < MIN_VARIANT_GENOMES:
                continue
            auroc = _score(_gene_onehot(g, labelled), label_map, seeds)
            if auroc is None:
                continue
            rows.append({
                "gene_name": gene, "mut_auroc": auroc["mean"], "mut_auroc_sd": auroc["sd"],
                "n_variants": int(g["variant_id"].nunique()), "n_genomes_with_variant": int(n_genomes),
                "embeddable": gene in embeddable, "is_rrna": gene in RRNA_GENES,
            })

        full = _score(_gene_onehot(dv, labelled), label_map, seeds)
        if full is not None:
            rows.append({"gene_name": "__ALL_WHO_one_hot__", "mut_auroc": full["mean"],
                         "mut_auroc_sd": full["sd"], "n_variants": int(dv["variant_id"].nunique()),
                         "n_genomes_with_variant": int(dv["Sample"].nunique()),
                         "embeddable": False, "is_rrna": False})

        df = pd.DataFrame(rows).sort_values("mut_auroc", ascending=False)
        df.to_csv(out_dir / f"tbprofiler_gene_lr_{drug}.csv", index=False)
        top = df[df.gene_name != "__ALL_WHO_one_hot__"].head(3)
        logger.info("%s: %d genes scored | top: %s | full one-hot %.3f", drug, len(df) - 1,
                    ", ".join(f"{r.gene_name}{'*' if r.is_rrna else ''}={r.mut_auroc:.3f}" for r in top.itertuples()),
                    full["mean"] if full else float("nan"))

    # tiny manifest so the plot step knows what's there
    done = sorted(p.name for p in out_dir.glob("tbprofiler_gene_lr_*.csv"))
    (out_dir / "tbprofiler_gene_lr_manifest.json").write_text(json.dumps({"files": done}, indent=2))


def main() -> None:
    """CLI entry point."""
    default_drugs = ["rifampin", "isoniazid", "ethambutol", "pyrazinamide", "moxifloxacin",
                     "levofloxacin", "streptomycin", "ethionamide", "rifabutin", "kanamycin"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", type=Path, required=True, help="tbprofiler_variants.parquet.")
    parser.add_argument("--ast-sheet", type=Path, required=True, help="binary_ast_with_split.csv.")
    parser.add_argument("--esm-rank-dir", type=Path, required=True, help="per_gene_lr_ranking/ (per-drug subdirs).")
    parser.add_argument("--out-dir", type=Path, required=True, help="Where to write tbprofiler_gene_lr_<drug>.csv.")
    parser.add_argument("--drugs", type=str, nargs="+", default=default_drugs)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()
    run(args.variants, args.ast_sheet, args.esm_rank_dir, args.out_dir, args.drugs, tuple(args.seeds))


if __name__ == "__main__":
    main()

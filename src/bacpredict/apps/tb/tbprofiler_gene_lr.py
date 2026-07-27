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
import json
import logging
from pathlib import Path

import pandas as pd

from bacpredict.engine.config import TB
from bacpredict.engine.ref_catalogues.base import score_onehot_frame
from bacpredict.engine.splits.load_splits import load_splits

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# rRNA / non-coding causes that can never be ESM-embedded — flagged so the plot can hatch them.
RRNA_GENES = {"rrs", "rrl", "rrf"}
MIN_VARIANT_GENOMES = 10  # a (gene, region) needs ≥ this many genomes carrying a variant to be scored
# Variant types that do NOT change the protein sequence → un-embeddable like rRNA (e.g. the inhA promoter).
NONCODING_TOKENS = ("upstream", "downstream", "utr", "non_coding", "intergenic", "intragenic")


def _region(variant_type: str) -> str:
    """``"non-coding"`` for promoter/UTR/intergenic variants (protein stays WT), else ``"coding"``."""
    t = str(variant_type).lower()
    return "non-coding" if any(tok in t for tok in NONCODING_TOKENS) else "coding"


def _who_gene_onehot(sub: pd.DataFrame, labelled: list[str]) -> pd.DataFrame:
    """Genomes × variant_id binary frame over ``labelled`` (genomes with no variant → all-zero rows)."""
    oh = pd.crosstab(sub["Sample"], sub["variant_id"]).clip(upper=1)
    return oh.reindex(labelled).fillna(0).astype(int)


def run(variants_parquet: Path, splits_dir: Path, esm_rank_dir: Path, out_dir: Path,
        drugs: list[str]) -> None:
    """Per drug: per-gene WHO-mutation LR (incl. rRNA) + the full one-hot, vs the ESM-embeddable set.

    Each drug's train/holdout partition is read from its deployed ``<drug>_split.csv`` in ``splits_dir``
    (:func:`bacpredict.engine.splits.load_splits.load_splits`); a drug with no split table is skipped.
    """
    variants = pd.read_parquet(variants_parquet)
    variants["drug_set"] = variants["drugs"].fillna("").str.split(";")
    out_dir.mkdir(parents=True, exist_ok=True)

    for drug in drugs:
        split_table = Path(splits_dir) / f"{drug}_split.csv"
        if not split_table.exists():
            logger.warning("%s: no split table at %s — skipping", drug, split_table)
            continue
        label_map, train_ids, _validate_ids, holdout_ids = load_splits(split_table)
        labelled = sorted(label_map)
        dv = variants[variants["drug_set"].apply(lambda s, d=drug: d in s)]
        dv = dv[dv["Sample"].isin(label_map)]
        if dv.empty:
            logger.warning("%s: no WHO variants — skipping", drug)
            continue

        # Which genes are ESM-embeddable (present in the protein-LR ranking for this drug)?
        rank_csv = esm_rank_dir / drug / f"per_gene_lr_{drug}.csv"
        embeddable = set(pd.read_csv(rank_csv)["gene_name"]) if rank_csv.exists() else set()

        # Split each gene into coding vs non-coding (promoter) sites, so the promoter shows as its own bar
        # (and is flagged un-embeddable — a non-coding change leaves the protein WT, so ESM can't see it).
        dv = dv.assign(region=dv["type"].map(_region))
        rows = []
        for (gene, region), g in dv.groupby(["gene_name", "region"]):
            n_genomes = g["Sample"].nunique()
            if n_genomes < MIN_VARIANT_GENOMES:
                continue
            agg = score_onehot_frame(_who_gene_onehot(g, labelled), label_map, train_ids, holdout_ids)
            if agg is None:
                continue
            noncoding = region == "non-coding"
            # rRNA variants live in the rRNA itself (not a promoter), so don't tag them '(promoter)'.
            site = f"{gene} (promoter)" if (noncoding and gene not in RRNA_GENES) else gene
            rows.append({
                "gene_name": gene, "region": region,
                "site": site,
                "mut_auroc": agg["auroc"]["mean"], "mut_auroc_sd": agg["auroc"]["sd"],
                "mut_auprc": agg["auprc"]["mean"], "mut_auprc_sd": agg["auprc"]["sd"],
                "n_variants": int(g["variant_id"].nunique()), "n_genomes_with_variant": int(n_genomes),
                "embeddable": (gene in embeddable) and not noncoding,
                "is_rrna": gene in RRNA_GENES, "is_noncoding": noncoding,
            })

        full = score_onehot_frame(_who_gene_onehot(dv, labelled), label_map, train_ids, holdout_ids)
        if full is not None:
            rows.append({"gene_name": "__ALL_WHO_one_hot__", "region": "all", "site": "__ALL_WHO_one_hot__",
                         "mut_auroc": full["auroc"]["mean"], "mut_auroc_sd": full["auroc"]["sd"],
                         "mut_auprc": full["auprc"]["mean"], "mut_auprc_sd": full["auprc"]["sd"],
                         "n_variants": int(dv["variant_id"].nunique()),
                         "n_genomes_with_variant": int(dv["Sample"].nunique()),
                         "embeddable": False, "is_rrna": False, "is_noncoding": False})

        df = pd.DataFrame(rows).sort_values("mut_auroc", ascending=False)
        df.to_csv(out_dir / f"tbprofiler_gene_lr_{drug}.csv", index=False)
        top = df[df.gene_name != "__ALL_WHO_one_hot__"].head(3)
        logger.info("%s: %d sites scored | top: %s | full one-hot %.3f", drug, len(df) - 1,
                    ", ".join(f"{r.site}{'*' if (r.is_rrna or r.is_noncoding) else ''}={r.mut_auroc:.3f}"
                              for r in top.itertuples()),
                    full["auroc"]["mean"] if full else float("nan"))

    # tiny manifest so the plot step knows what's there
    done = sorted(p.name for p in out_dir.glob("tbprofiler_gene_lr_*.csv"))
    (out_dir / "tbprofiler_gene_lr_manifest.json").write_text(json.dumps({"files": done}, indent=2))


def main() -> None:
    """CLI entry point."""
    default_drugs = ["rifampin", "isoniazid", "ethambutol", "pyrazinamide", "moxifloxacin",
                     "levofloxacin", "streptomycin", "ethionamide", "rifabutin", "kanamycin"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", type=Path, required=True, help="tbprofiler_variants.parquet.")
    parser.add_argument("--splits-dir", type=Path, default=None,
                        help="Dir of deployed <drug>_split.csv tables (default: <data-root>/processed/train_tb_ast/splits).")
    parser.add_argument("--esm-rank-dir", type=Path, required=True, help="per_gene_lr_ranking/ (per-drug subdirs).")
    parser.add_argument("--out-dir", type=Path, required=True, help="Where to write tbprofiler_gene_lr_<drug>.csv.")
    parser.add_argument("--drugs", type=str, nargs="+", default=default_drugs)
    args = parser.parse_args()
    splits_dir = args.splits_dir or TB.data_root() / "splits"
    run(args.variants, splits_dir, args.esm_rank_dir, args.out_dir, args.drugs)


if __name__ == "__main__":
    main()

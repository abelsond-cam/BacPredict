r"""Carve a train+validate-only pyseer cohort out of a full-cohort one — the leakage-free selection.

The deployed unitig hit set (33,039 unitigs) was chosen by an LMM fitted over the **whole** cohort,
holdout rows included, so any downstream model built on those features has already seen the test
labels. That inflates the unitig comparator in its own favour. This builds the inputs for the honest
version: re-run the identical LMM pipeline with the ``evaluate`` genomes removed entirely, take the
hit set it produces, and only then score on the holdout the fine-tune was scored on.

Removing the holdout from the **kinship and distance matrices too**, not just the phenotype, is
deliberate. Those are unsupervised, so leaving them whole is a much weaker leak than label-based
selection — but they still describe the test genomes, and the point of this run is that the feature
set is derived from data the holdout had no part in. Cheap to do properly; do it properly.

The pyseer pipeline is already cohort-parameterised, so the output is simply a new cohort directory
that ``run_unitig_lmm_sharded.sh`` can be pointed at with ``COHORT=<name>``. The 77 GB unitig matrix
is **not** subset: pyseer keeps only the samples present in the phenotype file, so restricting the
phenotype restricts the analysis, and the per-unitig allele-frequency filter is then computed over
the train+validate subset — which is what we want.

Usage::

    python -m bac_pyseer.kleb_iso_source.subset_cohort_trainval \
        --src-cohort-dir .../pyseer_iso_source/blood_faeces/sampled_country_2_1_all \
        --split-csv      .../train_iso_source/.../binary_blood_vs_faeces_with_split.csv \
        --out-cohort-dir .../pyseer_iso_source/blood_faeces/sampled_country_2_1_all_trainval \
        --label-column   blood_vs_faeces_label
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

KEEP_SPLITS = ("train", "validate")


def read_keep_samples(split_csv: Path, label_column: str, keep_splits: tuple[str, ...]) -> set[str]:
    """Sample ids in the kept splits with a clean 0/1 label."""
    df = pd.read_csv(split_csv, low_memory=False, usecols=["Sample", label_column, "train_val_eval"])
    df["Sample"] = df["Sample"].astype(str)
    df = df[df[label_column].isin([0, 1])]
    keep = set(df.loc[df["train_val_eval"].isin(keep_splits), "Sample"])
    dropped = set(df.loc[~df["train_val_eval"].isin(keep_splits), "Sample"])
    logger.info("split CSV: keeping %d (%s), excluding %d", len(keep), "+".join(keep_splits), len(dropped))
    return keep


def subset_phenotype(src: Path, dst: Path, keep: set[str]) -> list[str]:
    """Write the phenotype rows for kept samples; returns the retained sample order."""
    df = pd.read_csv(src, sep="\t")
    id_col = df.columns[0]
    df[id_col] = df[id_col].astype(str)
    out = df[df[id_col].isin(keep)]
    out.to_csv(dst, sep="\t", index=False)
    logger.info("phenotype: %d -> %d rows (%s)", len(df), len(out), dst.name)
    return out[id_col].tolist()


def subset_square_tsv(src: Path, dst: Path, keep_order: list[str]) -> None:
    """Subset a square, symmetrically-labelled TSV (kinship or distances) to ``keep_order``.

    Rows and columns are reindexed to the *same* order so the matrix stays aligned with the
    phenotype; pyseer matches on labels, but an inconsistent order is the kind of silent corruption
    that produces a plausible-looking wrong answer.
    """
    m = pd.read_csv(src, sep="\t", index_col=0, low_memory=False)
    m.index = m.index.astype(str)
    m.columns = m.columns.astype(str)
    missing = [s for s in keep_order if s not in m.index]
    if missing:
        raise ValueError(f"{src.name}: {len(missing)} kept samples absent from the matrix, e.g. {missing[:5]}")
    sub = m.loc[keep_order, keep_order]
    if sub.shape[0] != sub.shape[1] != len(keep_order):
        raise ValueError(f"{src.name}: subset is not square ({sub.shape}) for {len(keep_order)} samples")
    sub.to_csv(dst, sep="\t")
    logger.info("%s: %s -> %s", src.name, m.shape, sub.shape)


def _main_cli() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src-cohort-dir", type=Path, required=True)
    p.add_argument("--out-cohort-dir", type=Path, required=True)
    p.add_argument("--split-csv", type=Path, required=True)
    p.add_argument("--label-column", type=str, default="blood_vs_faeces_label")
    p.add_argument("--keep-splits", nargs="+", default=list(KEEP_SPLITS))
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    src, dst = args.src_cohort_dir, args.out_cohort_dir
    dst.mkdir(parents=True, exist_ok=True)

    keep = read_keep_samples(args.split_csv, args.label_column, tuple(args.keep_splits))
    order = subset_phenotype(src / "phenotype.tsv", dst / "phenotype.tsv", keep)

    for name in ("similarity.tsv", "jaccard_distances.tsv"):
        s = src / name
        if not s.exists():
            raise SystemExit(f"missing required matrix {s}")
        subset_square_tsv(s, dst / name, order)

    pheno = pd.read_csv(dst / "phenotype.tsv", sep="\t")
    label = pheno[args.label_column].astype(int)
    manifest = {
        "src_cohort_dir": str(src),
        "split_csv": str(args.split_csv),
        "keep_splits": list(args.keep_splits),
        "n_samples": int(len(pheno)),
        "n_positive": int(label.sum()),
        "n_negative": int((label == 0).sum()),
        "note": "Holdout genomes are absent from phenotype, kinship AND distances, so any hit set "
                "derived from this cohort is independent of the evaluate labels.",
    }
    (dst / "subset_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"\nNow run:  COHORT={dst.name} bash src/bac_pyseer/kleb_iso_source/scripts/run_unitig_lmm_sharded.sh")


if __name__ == "__main__":
    _main_cli()

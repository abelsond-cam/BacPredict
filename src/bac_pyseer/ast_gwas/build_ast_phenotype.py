"""Turn a drug's split table into the pyseer ``--phenotypes`` TSV, holding the holdout back.

This is the leakage guard of the whole comparison. The Bacformer fine-tune is scored on the
``holdout`` slice of ``<drug>_split.csv``; if the GWAS that *selects* the unitigs has seen those
same genomes' labels, the downstream logistic regression is fitted on features chosen with
knowledge of its own test set and its AUROC is not comparable to the fine-tune's.

So the phenotype file written here carries **train + validate only** by default. Everything the
GWAS derives from it is therefore holdout-blind: the allele-frequency filter (pyseer computes af
over the phenotyped samples), the unique-pattern count behind the Bonferroni threshold, the betas,
and hence the hit set. The holdout genomes re-enter only in
:mod:`bac_pyseer.ast_gwas.unitig_design_matrix`, which reads feature *values* (unsupervised
presence/absence, built over the whole cohort) and never a holdout label.

``--splits`` makes that a switch rather than a rule: passing ``train,validate,holdout`` reproduces
the leaky "classic GWAS" framing, which is worth running once alongside the honest one to quantify
the selection-leakage gap. The manifest records which splits went in, so a results table can never
silently mix the two.

Fractional ``ast_label`` values (a sample whose repeat DSTs disagreed) are dropped by
:func:`~bacpredict.engine.splits.load_splits.load_splits` and never reach pyseer.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from bacpredict.engine.splits.load_splits import load_splits

logger = logging.getLogger(__name__)

SAMPLE_COL = "samples"  # pyseer requires this literal name as the first column
DEFAULT_SPLITS = ("train", "validate")


def label_column(drug: str) -> str:
    """Phenotype column name for a drug — pass this to ``pyseer --phenotype-column``."""
    return f"{drug}_label"


def build_phenotype(
    split_table: Path, drug: str, splits: tuple[str, ...] = DEFAULT_SPLITS
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Select the requested splits from a drug's split table → ``(phenotype frame, manifest)``.

    Parameters
    ----------
    split_table
        ``<drug>_split.csv`` (columns ``Sample``, ``ast_label``, ``split``).
    drug
        AST column name, used for the phenotype column. Note TB uses ``rifampin``, not
        ``rifampicin``.
    splits
        Which split slices to include. Defaults to ``("train", "validate")`` — see module docstring.

    Returns
    -------
    (pandas.DataFrame, dict)
        The two-column pyseer phenotype frame, and a manifest recording the split composition and
        the holdout-exclusion check.

    Raises
    ------
    SystemExit
        If an unknown split is requested, or the selection is empty or single-class (pyseer cannot
        test a phenotype with no contrast).
    """
    known = {"train", "validate", "holdout"}
    unknown = sorted(set(splits) - known)
    if unknown:
        raise SystemExit(f"unknown split(s) {unknown}; expected a subset of {sorted(known)}")

    label_map, train_ids, validate_ids, holdout_ids = load_splits(split_table)
    by_split = {"train": train_ids, "validate": validate_ids, "holdout": holdout_ids}
    selected: list[str] = []
    for name in splits:
        selected.extend(by_split[name])

    if not selected:
        raise SystemExit(f"{split_table}: splits {list(splits)} selected 0 samples")
    labels = [label_map[s] for s in selected]
    n_pos = sum(labels)
    if n_pos in (0, len(labels)):
        raise SystemExit(
            f"{split_table}: splits {list(splits)} are single-class (n={len(labels)}, n_resistant={n_pos}) "
            "— pyseer needs both classes"
        )

    frame = pd.DataFrame({SAMPLE_COL: selected, label_column(drug): labels})

    excluded_holdout = set(holdout_ids) - set(selected)
    manifest = {
        "split_table": str(split_table),
        "drug": drug,
        "phenotype_column": label_column(drug),
        "splits_included": list(splits),
        "n_samples": len(selected),
        "n_resistant": int(n_pos),
        "n_susceptible": int(len(labels) - n_pos),
        "prevalence": n_pos / len(labels),
        "pheno_var": (n_pos / len(labels)) * (1 - n_pos / len(labels)),
        "n_by_split": {name: len(by_split[name]) for name in sorted(known)},
        "holdout_excluded": "holdout" not in splits,
        "n_holdout_excluded": len(excluded_holdout),
        "leakage_note": (
            "holdout labels withheld from the GWAS; unitig selection is holdout-blind"
            if "holdout" not in splits
            else "WARNING: holdout included — hit selection has seen the test labels; "
                 "downstream AUROC is optimistically biased and NOT comparable to the fine-tune"
        ),
    }
    return frame, manifest


def write_phenotype(
    split_table: Path, drug: str, out_tsv: Path, splits: tuple[str, ...] = DEFAULT_SPLITS
) -> dict[str, object]:
    """Write the pyseer phenotype TSV plus its sidecar manifest; return the manifest."""
    frame, manifest = build_phenotype(split_table, drug, splits)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_tsv, sep="\t", index=False)
    manifest["output"] = str(out_tsv)
    manifest_path = out_tsv.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Belt and braces: re-read what we wrote and prove no holdout sample is in it. A silent leak
    # here would invalidate every number downstream, so it is worth the second pass.
    if "holdout" not in splits:
        _, _, _, holdout_ids = load_splits(split_table)
        written = set(pd.read_csv(out_tsv, sep="\t")[SAMPLE_COL].astype(str))
        overlap = written & set(holdout_ids)
        if overlap:
            raise SystemExit(
                f"LEAK: {len(overlap)} holdout sample(s) reached {out_tsv}, e.g. {sorted(overlap)[:5]}"
            )
    logger.info(
        "wrote %s: n=%d (%d R / %d S, prevalence %.3f, pheno_var %.4f), splits=%s",
        out_tsv, manifest["n_samples"], manifest["n_resistant"], manifest["n_susceptible"],
        manifest["prevalence"], manifest["pheno_var"], ",".join(splits),
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split-table", type=Path, required=True, help="<drug>_split.csv for this drug.")
    p.add_argument("--drug", required=True, help="AST column name (TB uses 'rifampin', not 'rifampicin').")
    p.add_argument("--out-tsv", type=Path, required=True, help="Output pyseer --phenotypes TSV.")
    p.add_argument("--splits", default=",".join(DEFAULT_SPLITS),
                   help="Comma-separated splits to include. Default 'train,validate' (holdout-blind). "
                        "Adding 'holdout' reproduces the leaky classic-GWAS framing — see module docstring.")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    manifest = write_phenotype(
        args.split_table, args.drug, args.out_tsv,
        tuple(s.strip() for s in args.splits.split(",") if s.strip()),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

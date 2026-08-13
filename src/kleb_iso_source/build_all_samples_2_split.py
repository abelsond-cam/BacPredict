r"""Build the ``all_samples_2`` split: train on everything, test on a frozen country-controlled set.

**The question this settles.** ``all_samples`` beats the country-controlled model within every major
sublineage on its *own* holdout (+0.03 to +0.08) but ties exactly on genomes held out by both
(Δ −0.004, CI spanning zero). Two readings fit: the extra training data genuinely helps, or its
holdout is simply an easier test set — country is a strong proxy for isolation source, and the
country-clustered isolates that balancing removed are precisely what ``all_samples`` retains in both
its train and its test.

The confound is that the two models were tested on *different genomes*, and the common-genome subset
cannot fix it: that subset can only contain genomes present in the country-balanced cohort, so it
structurally cannot ask whether the extra data helps on clustered collections.

**The design.** Freeze the country-controlled cohort's ``evaluate`` split as the test set for a new
model, and train that model on every other labelled genome. Both models are then measured on the
same genomes, neither has seen them, and the only difference is the training set. If the
within-lineage advantage collapses, it was test-set composition.

**What each outcome licenses.** A collapse is clean evidence. A *win* is not conclusive: the larger
training pool may contain near-identical siblings (outbreak clusters) of the frozen test genomes,
so a win would still be ambiguous between "more data helps" and clonal leakage, and would need a
near-duplicate audit to separate them.

Usage
-----
    python -m kleb_iso_source.build_all_samples_2_split \
        --frozen-test-cohort sampled_country_2_1_all \
        --source-cohort      all_samples \
        --out-cohort         all_samples_2
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LABEL_COL = "blood_vs_faeces_label"
SPLIT_COL = "train_val_eval"
SPLIT_CSV = "binary_blood_vs_faeces_with_split.csv"


def build_split(frozen_csv: Path, source_csv: Path, *, validate_frac: float = 0.10,
                seed: int = 1, keep_out_of_train: set[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Frozen evaluate + a fresh train/validate split over everything else.

    The frozen genomes are removed from the training pool by ``Sample``, so no genome can appear in
    both — the property the whole comparison rests on.

    ``keep_out_of_train`` forces additional genomes into **validate** (unless they are already in the
    frozen test set). Used for the lab-collection isolates: any of them that the new model trained on
    would have a memorised, not predicted, score, and those are exactly the genomes whose predictions
    the collaborator will act on. Validate is not perfectly clean — it drives early stopping, so it
    carries mild model-selection optimism — but it is a world away from being fitted on.
    """
    frozen = pd.read_csv(frozen_csv, low_memory=False)
    frozen = frozen[frozen[LABEL_COL].isin([0, 1])]
    test_ids = set(frozen.loc[frozen[SPLIT_COL] == "evaluate", "Sample"].astype(str))
    if not test_ids:
        raise SystemExit(f"{frozen_csv} has no evaluate rows")

    source = pd.read_csv(source_csv, low_memory=False)
    source = source[source[LABEL_COL].isin([0, 1])].copy()
    source["Sample"] = source["Sample"].astype(str)

    in_source = test_ids & set(source["Sample"])
    missing = test_ids - in_source
    if missing:
        logger.warning("%d frozen test genomes are absent from %s and cannot be scored by the new "
                       "model; the comparison will run on the %d that are present",
                       len(missing), source_csv.name, len(in_source))

    is_test = source["Sample"].isin(in_source)
    pool = source[~is_test].copy()

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pool))
    n_val = int(round(validate_frac * len(pool)))
    val_pos = set(order[:n_val].tolist())
    pool = pool.reset_index(drop=True)
    pool[SPLIT_COL] = ["validate" if i in val_pos else "train" for i in range(len(pool))]

    forced = set(keep_out_of_train or ()) & set(pool["Sample"])
    if forced:
        pool.loc[pool["Sample"].isin(forced), SPLIT_COL] = "validate"

    test_rows = source[is_test].copy()
    test_rows[SPLIT_COL] = "evaluate"
    out = pd.concat([pool, test_rows], ignore_index=True)

    counts = out[SPLIT_COL].value_counts().to_dict()
    overlap = set(out.loc[out[SPLIT_COL] == "evaluate", "Sample"]) & set(
        out.loc[out[SPLIT_COL] != "evaluate", "Sample"])
    if overlap:
        raise SystemExit(f"{len(overlap)} genomes are in both the test set and the training pool")

    kept_out = set(keep_out_of_train or ())
    in_train = kept_out & set(out.loc[out[SPLIT_COL] == "train", "Sample"])
    if in_train:
        raise SystemExit(f"{len(in_train)} genomes asked to be kept out of train are in it")

    manifest = {
        "frozen_test_cohort": str(frozen_csv),
        "source_cohort": str(source_csv),
        "n_total": int(len(out)),
        "split_counts": {k: int(v) for k, v in counts.items()},
        "n_frozen_test_requested": len(test_ids),
        "n_frozen_test_present": len(in_source),
        "n_frozen_test_missing_from_source": len(missing),
        "n_kept_out_of_train_requested": len(kept_out),
        "n_kept_out_of_train_in_cohort": len(kept_out & set(out["Sample"])),
        "kept_out_split_counts": {
            k: int(v) for k, v in
            out[out["Sample"].isin(kept_out)][SPLIT_COL].value_counts().to_dict().items()
        },
        "validate_frac": validate_frac,
        "seed": seed,
        "prevalence_by_split": {
            k: float(out.loc[out[SPLIT_COL] == k, LABEL_COL].mean()) for k in counts
        },
    }
    return out, manifest


def _main_cli() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    base = Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_iso_source/"
                "blood_faeces")
    p.add_argument("--train-root", type=Path, default=base)
    p.add_argument("--flavor", type=str, default="kpsc_human")
    p.add_argument("--frozen-test-cohort", type=str, default="sampled_country_2_1_all")
    p.add_argument("--source-cohort", type=str, default="all_samples")
    p.add_argument("--out-cohort", type=str, default="all_samples_2")
    p.add_argument("--validate-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--keep-out-of-train-csv", type=Path, default=None,
                   help="CSV of genomes that must never be trained on (forced to validate unless "
                        "already in the frozen test set) — the lab-collection manifest.")
    p.add_argument("--keep-out-column", type=str, default="sample_accession")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    frozen_csv = args.train_root / args.frozen_test_cohort / args.flavor / SPLIT_CSV
    source_csv = args.train_root / args.source_cohort / args.flavor / SPLIT_CSV
    out_dir = args.train_root / args.out_cohort / args.flavor
    out_dir.mkdir(parents=True, exist_ok=True)

    keep_out: set[str] | None = None
    if args.keep_out_of_train_csv is not None:
        kdf = pd.read_csv(args.keep_out_of_train_csv, low_memory=False)
        keep_out = set(kdf[args.keep_out_column].dropna().astype(str))
        logger.info("keeping %d genomes out of train (from %s)", len(keep_out),
                    args.keep_out_of_train_csv.name)

    out, manifest = build_split(frozen_csv, source_csv, validate_frac=args.validate_frac,
                                seed=args.seed, keep_out_of_train=keep_out)
    out.to_csv(out_dir / SPLIT_CSV, index=False)
    (out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"\nWrote {out_dir / SPLIT_CSV}")


if __name__ == "__main__":
    _main_cli()

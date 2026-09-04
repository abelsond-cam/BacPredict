r"""Per-sublineage composition of the country-balanced pooled cohort, observed against predicted.

The model-comparison report says how well the models do; it says nothing about **the population they
were trained and evaluated on**. This module produces that: for each of the commonest sublineages,
how many genomes it holds, what fraction of them are blood, and what fraction the fine-tune *calls*
blood at its deployed operating point.

Two things about the numbers are easy to over-read, so they are computed and reported explicitly
rather than left for the reader to infer:

**The predicted rate sits below the observed rate almost everywhere, and that is the threshold, not
the lineage.** The Youden point maximises ``sensitivity + specificity``, which does not preserve
prevalence; for this model sensitivity (0.633) is well below specificity (0.790), so it under-calls
the positive class by roughly nine points *globally*. A per-sublineage gap is only interesting to the
extent it departs from that offset, so :func:`compose` returns the scope-level offset alongside.

**Scope is not cosmetic.** At ``all`` scope 70% of the cohort is training data the model memorised
(train AUROC 0.959 against 0.786 on the holdout), so its "predictions" there are largely recall.
``heldout`` (validate + evaluate) is the honest scope for anything predicted; ``all`` is the true
cohort composition. Both are emitted because the difference between them is itself informative.

Usage::

    python -m kleb_iso_source.sublineage_composition \
        --split-csv   .../sampled_country_2_1_all/kpsc_human/binary_blood_vs_faeces_with_split.csv \
        --scores-npz  .../sampled_country_2_1_all/kpsc_human/models/cohort_scores.npz \
        --thresholds  .../lab_collection/model_comparison_thresholds.json \
        --out-dir     .../lab_collection
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ID_COL = "Sample"
SL_COL = "Sublineage"
LABEL_COL = "blood_vs_faeces_label"
SPLIT_COL = "train_val_eval"

#: Sublineage values that carry no call. The cohort is built to have none of these (the sampler
#: requires ``Sublineage.notna()``), so a non-zero count here means the cohort definition changed.
MISSING_SL = frozenset({"", "nan", "NaN", "NA", "N/A", "None", "none", "unknown", "-"})

#: ``heldout`` is validate + evaluate: nothing fitted on. Named to match the convention already used
#: by the per-sublineage metrics tables, where it exists precisely for within-clone questions.
SCOPES: dict[str, tuple[str, ...]] = {
    "all": ("train", "validate", "evaluate"),
    "heldout": ("validate", "evaluate"),
    "evaluate": ("evaluate",),
}

OTHER_LABEL = "other"


def load_cohort(split_csv: Path, scores_npz: Path) -> pd.DataFrame:
    """Join the cohort split table to the scored archive, refusing anything but an exact match.

    The two files are produced by different jobs at different times. The guard that they describe the
    same cohort is not that their row counts agree — it is that the join is total *and* that the
    label column equals the archive's own ``y_true`` for every genome.
    """
    df = pd.read_csv(split_csv, low_memory=False)
    for col in (ID_COL, SL_COL, LABEL_COL, SPLIT_COL):
        if col not in df.columns:
            raise ValueError(f"{split_csv} is missing {col!r}")
    df[ID_COL] = df[ID_COL].astype(str)

    d = np.load(scores_npz, allow_pickle=False)
    for key in ("sample_ids", "y_prob", "y_true", "split"):
        if key not in d.files:
            raise ValueError(f"{scores_npz} has no {key!r} — needs a score_cohort.py archive")
    scores = pd.DataFrame({
        ID_COL: [str(s) for s in d["sample_ids"]],
        "prob": d["y_prob"].astype(float),
        "y_true": d["y_true"].astype(int),
        "split_npz": [str(s) for s in d["split"]],
    })

    merged = df[[ID_COL, SL_COL, LABEL_COL, SPLIT_COL]].merge(scores, on=ID_COL, how="inner")
    if len(merged) != len(df) or len(merged) != len(scores):
        raise ValueError(
            f"join is not total: split CSV {len(df)}, scores {len(scores)}, joined {len(merged)}. "
            "These two files do not describe the same cohort."
        )
    mismatched = int((merged[LABEL_COL].astype(int) != merged["y_true"]).sum())
    if mismatched:
        raise ValueError(
            f"{mismatched} genomes disagree between the split CSV's {LABEL_COL!r} and the archive's "
            "y_true. One of the two is from a different cohort or a different label definition."
        )
    if (merged[SPLIT_COL].astype(str) != merged["split_npz"]).any():
        raise ValueError("split assignment differs between the CSV and the archive")
    return merged.drop(columns=["y_true", "split_npz"])


def compose(
    cohort: pd.DataFrame,
    threshold: float,
    scope: str = "all",
    top_n: int = 15,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Per-sublineage observed vs predicted blood counts, top ``top_n`` by size plus an ``other`` row.

    Percentages are derived from the summed counts, never averaged across sublineages — averaging
    would weight a lineage of 26 equally with one of 2,416.

    Returns
    -------
    tuple of (pandas.DataFrame, dict)
        One row per sublineage (plus ``other``) ordered by ``n`` descending with ``other`` pinned
        last, and the scope-level totals including the global predicted-minus-observed offset that
        every per-lineage gap should be read against.
    """
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope!r} — expected one of {sorted(SCOPES)}")
    sub = cohort[cohort[SPLIT_COL].astype(str).isin(SCOPES[scope])].copy()
    if sub.empty:
        raise ValueError(f"scope {scope!r} selected no genomes")

    sl = sub[SL_COL].fillna("").astype(str).str.strip()
    n_no_call = int(sl.isin(MISSING_SL).sum())
    sub = sub.assign(**{SL_COL: sl})[~sl.isin(MISSING_SL)]

    sub["is_blood"] = sub[LABEL_COL].astype(int) == 1
    sub["pred_blood"] = sub["prob"] >= threshold

    g = (sub.groupby(SL_COL)
            .agg(n=("is_blood", "size"), n_blood=("is_blood", "sum"), n_pred_blood=("pred_blood", "sum"))
            .sort_values("n", ascending=False))
    top, rest = g.head(top_n), g.iloc[top_n:]

    rows = top.reset_index().assign(n_groups=1, is_other=False)
    if not rest.empty:
        rows = pd.concat([rows, pd.DataFrame([{
            SL_COL: OTHER_LABEL, "n": int(rest.n.sum()), "n_blood": int(rest.n_blood.sum()),
            "n_pred_blood": int(rest.n_pred_blood.sum()), "n_groups": int(len(rest)), "is_other": True,
        }])], ignore_index=True)

    for col in ("n", "n_blood", "n_pred_blood", "n_groups"):
        rows[col] = rows[col].astype(int)
    rows["pct_blood"] = rows.n_blood / rows.n * 100
    rows["pct_pred_blood"] = rows.n_pred_blood / rows.n * 100

    totals = {
        "scope": scope, "splits": list(SCOPES[scope]), "threshold": float(threshold),
        "n": int(rows.n.sum()), "n_blood": int(rows.n_blood.sum()),
        "n_pred_blood": int(rows.n_pred_blood.sum()),
        "n_sublineages": int(len(g)), "n_named": int(len(top)),
        "n_other_groups": int(len(rest)), "n_no_sublineage_call": n_no_call,
    }
    totals["pct_blood"] = totals["n_blood"] / totals["n"] * 100
    totals["pct_pred_blood"] = totals["n_pred_blood"] / totals["n"] * 100
    # The number every per-lineage gap must be read against: a threshold property, not biology.
    totals["global_offset_pp"] = totals["pct_pred_blood"] - totals["pct_blood"]
    return rows, totals


def _main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split-csv", type=Path, required=True)
    p.add_argument("--scores-npz", type=Path, required=True)
    p.add_argument("--thresholds", type=Path, required=True,
                   help="model_comparison_thresholds.json — the deployed Youden operating point")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--top-n", type=int, default=15)
    p.add_argument("--scopes", nargs="+", default=["all", "heldout"], choices=sorted(SCOPES))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    threshold = float(json.loads(args.thresholds.read_text())["models"]["bacformer_pooled"]["threshold"])
    cohort = load_cohort(args.split_csv, args.scores_npz)
    logger.info("cohort joined: %d genomes, threshold %.4f", len(cohort), threshold)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"schema_version": "1.0", "source": {
        "split_csv": str(args.split_csv), "scores_npz": str(args.scores_npz),
        "thresholds": str(args.thresholds)}, "scopes": {}}
    for scope in args.scopes:
        rows, totals = compose(cohort, threshold, scope=scope, top_n=args.top_n)
        rows.to_csv(args.out_dir / f"sublineage_composition_{scope}.csv", index=False)
        payload["scopes"][scope] = {"totals": totals, "rows": rows.to_dict(orient="records")}
        logger.info("scope %-8s n=%-6d %%blood %.1f  %%pred %.1f  offset %+.1f pp  (%d named + other of %d)",
                    scope, totals["n"], totals["pct_blood"], totals["pct_pred_blood"],
                    totals["global_offset_pp"], totals["n_named"], totals["n_sublineages"])
    (args.out_dir / "sublineage_composition.json").write_text(json.dumps(payload, indent=2))
    logger.info("wrote sublineage_composition.json + %d CSVs to %s", len(args.scopes), args.out_dir)


if __name__ == "__main__":
    _main()

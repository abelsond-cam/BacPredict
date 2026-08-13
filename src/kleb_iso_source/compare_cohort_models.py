r"""Which fine-tuned model predicts invasion better *within a lineage*: pooled or all_samples?

``all_samples`` scores higher overall (0.827 vs 0.786) but is country-confounded, which is why it was
never quotable as evidence that Bacformer learns biology. For choosing isolates to test in vivo the
criterion is different — the best ranking wins, and phylogenetic signal is legitimate if it predicts.

**The discriminating test is within-lineage.** Inside a sublineage, lineage is held constant, so if
``all_samples`` were winning only through country/lineage confounding its advantage should collapse
there. If it still wins within SL258, that is real signal bought with ~5,100 more training genomes.

Two views, both reported:

*Headline* — each model on its **own** holdout. Both numbers are honest (every genome is held out
from the model scoring it) and the per-SL counts are generous. The caveat is that the two holdouts
are different genome sets, so a gap reflects model *and* test-set composition; per-SL ``n`` and
prevalence are printed side by side so any composition difference is visible next to the AUROCs.

*Confirmatory* — genomes held out under **both** models, so the comparison is on identical data with
both models' folds respected. n is much smaller, so it is read for direction, not magnitude.

Usage
-----
    python -m kleb_iso_source.compare_cohort_models \
        --pooled-scores      <pooled>/models/cohort_scores.npz \
        --all-samples-scores <all_samples>/models/cohort_scores.npz \
        --metadata           metadata_v2_all_samples_and_columns.tsv \
        --out-dir            <lab_collection>/model_choice
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from bac_pyseer.kleb_iso_source.unitig_presence_model import paired_delta_ci
from bacpredict.engine.finetune.stratified_metrics import bootstrap_auroc_ci

logger = logging.getLogger(__name__)

MISSING_SL = {"", "nan", "NA", "None", "unknown", "-"}
HELDOUT = ("validate", "evaluate")


def load_cohort_scores(path: Path, tag: str) -> pd.DataFrame:
    """Read a ``cohort_scores.npz`` into ``Sample / prob / y_true / split`` columns tagged by model."""
    d = np.load(path, allow_pickle=False)
    for key in ("sample_ids", "y_prob", "y_true", "split"):
        if key not in d.files:
            raise ValueError(f"{path} has no {key!r} — needs a score_cohort.py archive")
    return pd.DataFrame({
        "Sample": [str(s) for s in d["sample_ids"]],
        f"prob_{tag}": d["y_prob"],
        f"y_true_{tag}": d["y_true"].astype(int),
        f"split_{tag}": [str(s) for s in d["split"]],
    })


def attach_sublineage(df: pd.DataFrame, metadata_tsv: Path) -> pd.DataFrame:
    """Join ``Sublineage`` from metadata_v2, dropping placeholder labels."""
    meta = pd.read_csv(metadata_tsv, sep="\t", usecols=["Sample", "Sublineage"], dtype=str,
                       low_memory=False).drop_duplicates("Sample")
    meta["Sublineage"] = meta["Sublineage"].fillna("").str.strip()
    meta = meta[~meta["Sublineage"].isin(MISSING_SL)]
    return df.merge(meta, on="Sample", how="left")


def per_sl_own_holdout(df: pd.DataFrame, tag: str, min_n: int) -> pd.DataFrame:
    """Per-sublineage AUROC for one model on the genomes IT held out."""
    sub = df[df[f"split_{tag}"] == "evaluate"].dropna(subset=["Sublineage"])
    rows = []
    for sl, g in sub.groupby("Sublineage"):
        if len(g) < min_n or g[f"y_true_{tag}"].nunique() < 2:
            continue
        y = g[f"y_true_{tag}"].to_numpy()
        p = g[f"prob_{tag}"].to_numpy()
        lo, hi, _ = bootstrap_auroc_ci(y, p)
        rows.append({"Sublineage": sl, f"n_{tag}": len(g), f"prev_{tag}": float(y.mean()),
                     f"auroc_{tag}": float(roc_auc_score(y, p)),
                     f"ci_lo_{tag}": lo, f"ci_hi_{tag}": hi})
    return pd.DataFrame(rows)


def per_sl_common(df: pd.DataFrame, scope: str, min_n: int, seed: int = 1) -> pd.DataFrame:
    """Per-sublineage AUROC for both models on genomes held out under BOTH, with a paired delta."""
    if scope == "evaluate":
        mask = (df["split_pooled"] == "evaluate") & (df["split_all_samples"] == "evaluate")
    else:
        mask = df["split_pooled"].isin(HELDOUT) & df["split_all_samples"].isin(HELDOUT)
    sub = df[mask].dropna(subset=["Sublineage"])
    rows = []
    for sl, g in list(sub.groupby("Sublineage")) + [("__pooled__", sub)]:
        if len(g) < min_n or g["y_true_pooled"].nunique() < 2:
            continue
        y = g["y_true_pooled"].to_numpy()
        pp, pa = g["prob_pooled"].to_numpy(), g["prob_all_samples"].to_numpy()
        delta = paired_delta_ci(y, pa, pp, seed=seed)  # all_samples minus pooled
        rows.append({
            "Sublineage": sl, "n": len(g), "prevalence": float(y.mean()),
            "auroc_pooled": float(roc_auc_score(y, pp)),
            "auroc_all_samples": float(roc_auc_score(y, pa)),
            "delta_all_minus_pooled": delta["delta"],
            "delta_ci_lo": delta["ci_lo"], "delta_ci_hi": delta["ci_hi"],
            "favours_all_samples": bool(delta["ci_lo"] > 0),
            "favours_pooled": bool(delta["ci_hi"] < 0),
        })
    return pd.DataFrame(rows)


def _main_cli() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pooled-scores", type=Path, required=True)
    p.add_argument("--all-samples-scores", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--min-group-n", type=int, default=100)
    p.add_argument("--min-common-n", type=int, default=30,
                   help="Lower floor for the common-set subtest, which is intrinsically smaller.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pooled = load_cohort_scores(args.pooled_scores, "pooled")
    allsamp = load_cohort_scores(args.all_samples_scores, "all_samples")
    merged = pooled.merge(allsamp, on="Sample", how="outer")
    merged = attach_sublineage(merged, args.metadata)
    logger.info("pooled %d genomes, all_samples %d, union %d, with a Sublineage %d",
                len(pooled), len(allsamp), len(merged), merged["Sublineage"].notna().sum())

    head_p = per_sl_own_holdout(merged, "pooled", args.min_group_n)
    head_a = per_sl_own_holdout(merged, "all_samples", args.min_group_n)
    headline = head_p.merge(head_a, on="Sublineage", how="outer")
    headline["delta_own_holdouts"] = headline["auroc_all_samples"] - headline["auroc_pooled"]
    headline = headline.sort_values("n_pooled", ascending=False, na_position="last")

    common_eval = per_sl_common(merged, "evaluate", args.min_common_n)
    common_held = per_sl_common(merged, "heldout", args.min_common_n)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    headline.to_csv(args.out_dir / "per_sl_own_holdout.csv", index=False)
    common_eval.to_csv(args.out_dir / "per_sl_common_evaluate.csv", index=False)
    common_held.to_csv(args.out_dir / "per_sl_common_heldout.csv", index=False)

    print("\n=== HEADLINE: each model on its OWN holdout (different genome sets) ===")
    cols = ["Sublineage", "n_pooled", "prev_pooled", "auroc_pooled", "n_all_samples",
            "prev_all_samples", "auroc_all_samples", "delta_own_holdouts"]
    print(headline[[c for c in cols if c in headline.columns]].to_string(index=False, float_format="%.3f"))

    for name, tbl in (("evaluate∩evaluate", common_eval), ("heldout∩heldout", common_held)):
        print(f"\n=== CONFIRMATORY: genomes held out under BOTH models ({name}) ===")
        if tbl.empty:
            print("  (no group reaches the minimum n)")
            continue
        print(tbl[["Sublineage", "n", "auroc_pooled", "auroc_all_samples",
                   "delta_all_minus_pooled", "delta_ci_lo", "delta_ci_hi"]]
              .to_string(index=False, float_format="%.3f"))

    summary = {
        "min_group_n": args.min_group_n, "min_common_n": args.min_common_n,
        "n_sublineages_headline": int(len(headline)),
        "headline_all_samples_wins": int((headline["delta_own_holdouts"] > 0).sum()),
        "headline_pooled_wins": int((headline["delta_own_holdouts"] < 0).sum()),
        "n_common_evaluate": int(common_eval["n"].max()) if len(common_eval) else 0,
        "n_common_heldout": int(common_held["n"].max()) if len(common_held) else 0,
    }
    (args.out_dir / "model_choice_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps(summary, indent=2))
    print("\nNOTE: the two holdouts are different genome sets, so a headline gap reflects model AND "
          "test-set composition — read the per-SL n and prevalence columns alongside the AUROCs.")


if __name__ == "__main__":
    _main_cli()

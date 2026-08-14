"""Collate unitig-LR results against the Bacformer fine-tune and the catalogue ceiling.

The deliverable of the whole exercise is one table per organism answering: for each drug, how does a
mechanism-agnostic unitig screen compare to the fine-tuned model, and to the ceiling set by known
resistance determinants?

Reads each drug's unitig-LR ``results.json`` (schema v1.2) and joins it to
``src/bacpredict/visualisations/<organism>/catalogue_ceiling_panel.csv``, which carries
``ceiling_auroc``/``ceiling_auprc`` (CARD for Kp, WHO/TB-Profiler for TB) plus the per-row provenance
that says how each ceiling was estimated. The output is deliberately a plain CSV with one row per
drug so it drops into the existing ladder plots without touching them.

That panel deliberately replaced the older ``amr_summary_panel.csv``, now in
``visualisations/_superseded/``. The old one carried an ``ft_auroc`` column from a superseded set of
fine-tunes — reading it is how colistin came to be quoted as 0.8072 when it is 0.9094 — and its
ceiling column was stale as well.

**A point estimate per drug is not enough.** With four drugs and deltas that may be a few
thousandths, a bare table invites reading a tie as a win — the sibling invasion comparator hit
exactly this (+0.0055 in Bacformer's favour). So when the fine-tune's ``eval_scores.npz`` is
available, this also computes a **paired** bootstrap CI on the AUROC difference, reusing
:func:`~bac_pyseer.kleb_iso_source.unitig_presence_model.paired_delta_ci`. Paired matters: both
models score the identical genomes, and resampling them independently would widen the interval by
ignoring that both face the same easy and hard cases. Predictions are aligned by **sample id**, not
position.

Two caveats worth carrying into any write-up, both inherited rather than introduced here:

* **The TB ceiling is provisional** and the output says so per row, via ``ceiling_estimator`` /
  ``ceiling_status``. It came from the retired whole-cohort k-fold probe rather than the
  deployment-holdout scorer Kp uses, so a TB ceiling-vs-unitig gap is not yet like-for-like. It is
  also missing rifabutin. See ``visualisations/PROVENANCE.md``.
* The ladder's "same holdout" property holds because every arm was built from the same
  ``<drug>_split.csv`` — a convention, not an enforced cross-check. ``n_holdout`` is carried into
  the output so a mismatch is at least visible.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from bac_pyseer.kleb_iso_source.unitig_presence_model import paired_delta_ci
from bacpredict.engine.finetune.metrics import compute_full_metrics, youden_threshold

logger = logging.getLogger(__name__)

# The panel supplies the CATALOGUE CEILING only — fine-tune numbers come from its own eval_scores.npz.
# The provenance columns travel with the number on purpose: TB's ceiling is a different estimator on a
# different evaluation set, and a bare ceiling_auroc gives a reader no way to know that.
PANEL_COLUMNS = (
    "drug", "ceiling_auroc", "ceiling_auprc",
    "ceiling_catalogue", "ceiling_estimator", "ceiling_status",
)

# The TB AST column is `rifampin` (US); the figure panels key on `rifampicin` (UK). Merging without
# this alias silently drops the headline TB drug — it matches nothing and the row comes back NaN.
PANEL_DRUG_ALIASES = {"rifampin": "rifampicin"}


def _load_scores(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Read an ``eval_scores.npz`` → ``(sample_ids, y_true, y_prob)``, or None if unpairable.

    ``sample_ids`` is required: without it the two models' predictions can only be aligned by
    position, which is exactly how two models silently get compared on different genomes.
    """
    scores = np.load(path, allow_pickle=False)
    if "sample_ids" not in scores:
        logger.warning("%s has no sample_ids — cannot pair predictions, skipping its CI", path)
        return None
    return (
        scores["sample_ids"].astype(str),
        np.asarray(scores["y_true"]).astype(int),
        np.asarray(scores["y_prob"]).astype(float),
    )


def paired_ci_against_ft(unitig_scores: Path, ft_scores: Path, seed: int = 1) -> dict[str, object] | None:
    """Paired bootstrap CI for ``AUROC(unitig) - AUROC(fine-tune)`` on their common genomes."""
    a, b = _load_scores(unitig_scores), _load_scores(ft_scores)
    if a is None or b is None:
        return None
    (ids_u, y_u, p_u), (ids_f, _y_f, p_f) = a, b

    by_ft = {s: i for i, s in enumerate(ids_f)}
    common = [(i, by_ft[s]) for i, s in enumerate(ids_u) if s in by_ft]
    if len(common) < 2:
        logger.warning("only %d genome(s) common to %s and %s — no CI", len(common), unitig_scores, ft_scores)
        return None
    ui = np.array([i for i, _ in common])
    fi = np.array([j for _, j in common])
    y = y_u[ui]
    if np.unique(y).size < 2:
        logger.warning("common holdout subset is single-class — no CI")
        return None

    delta = paired_delta_ci(y, p_u[ui], p_f[fi], seed=seed)
    return {
        "n_common_genomes": len(common),
        "n_unitig_holdout": int(len(ids_u)),
        "n_ft_holdout": int(len(ids_f)),
        "unitig_auroc_on_common": float(roc_auc_score(y, p_u[ui])),
        "ft_auroc_on_common": float(roc_auc_score(y, p_f[fi])),
        "delta_unitig_minus_ft": delta["delta"],
        "delta_ci_lo": delta["ci_lo"],
        "delta_ci_hi": delta["ci_hi"],
        "separates_from_zero": delta["separates_from_zero"],
    }


def operating_point(scores_npz: Path, prefix: str) -> dict[str, object]:
    """Sensitivity/specificity/balanced accuracy at the Youden-optimal point **on the holdout**.

    Computed here, from each arm's own ``eval_scores.npz``, rather than read from whatever block a
    given arm happened to store — that is what guarantees both arms are summarised under an
    identical convention. AUROC/AUPRC are threshold-free and unaffected.

    ⚠ **The threshold is chosen on the same genomes it is scored on**, so these three numbers are
    the model's *best achievable* operating point, not a held-out estimate of deployment
    performance — they are optimistically biased, by roughly the amount the ROC curve is noisy at
    that point. Report them as "at the optimal operating point" and never as expected field
    sensitivity. Selecting on validate instead is the unbiased alternative, but it transfers poorly
    at these split sizes (Kp ertapenem: balanced accuracy 0.925 on a 340-genome validate-chosen
    threshold vs 0.953 here), which is why this convention was chosen deliberately.
    """
    scores = np.load(scores_npz, allow_pickle=False)
    y, p = scores["y_true"], scores["y_prob"]
    thr = youden_threshold(y, p)
    m = compute_full_metrics(y, p, threshold=float(thr))
    return {
        # AUROC/AUPRC come from the same scored genomes as everything else, deliberately: taking a
        # fine-tune's headline from a summary panel instead is how a stale or partial table silently
        # becomes the comparator (see the panel notes on PANEL_COLUMNS).
        f"{prefix}_auroc": m["auroc"],
        f"{prefix}_auprc": m["auprc"],
        f"{prefix}_sensitivity": m["sensitivity"],
        f"{prefix}_specificity": m["specificity"],
        f"{prefix}_balanced_accuracy": m["balanced_accuracy"],
        f"{prefix}_f1": m["f1"],
        f"{prefix}_operating_threshold": float(thr),
        f"{prefix}_n_holdout": int(y.size),
    }


def read_unitig_results(results_json: Path) -> dict[str, object]:
    """Flatten one unitig-LR ``results.json`` into a row.

    Threshold-dependent metrics are deliberately **not** taken from ``metrics`` (which is computed
    at 0.5): :func:`operating_point` recomputes them for both arms alike. See its docstring.
    """
    payload = json.loads(results_json.read_text())
    metrics = payload["metrics"]
    extra = payload.get("extra", {})
    gwas = extra.get("gwas_summary", {})
    return {
        "drug": payload["drug"],
        "task": payload["task"],
        "unitig_auroc": metrics["auroc"],
        "unitig_auprc": metrics["auprc"],
        "n_holdout": payload["split"].get("n_evaluate"),
        "n_train": extra.get("n_train"),
        "n_unitigs": extra.get("n_unitigs"),
        "n_unique_patterns": gwas.get("n_unique_patterns"),
        "bonferroni_threshold": gwas.get("bonferroni_threshold"),
        "lambda_gc": gwas.get("genomic_inflation_lambda"),
        "pheno_var": gwas.get("pheno_var"),
        "results_json": str(results_json),
    }


def collect(
    results_jsons: list[Path], panel_csv: Path | None = None,
    ft_scores: dict[str, Path] | None = None, seed: int = 1,
) -> pd.DataFrame:
    """Join unitig-LR rows to the recorded fine-tune and catalogue numbers, one row per drug.

    ``ft_scores`` maps a drug to its fine-tune ``eval_scores.npz``; where given, a paired bootstrap
    CI on the unitig−fine-tune AUROC delta is added, computed on the genomes the two share.
    """
    rows = [read_unitig_results(p) for p in results_jsons]
    if not rows:
        raise SystemExit("no results.json files given")

    for row, results_json in zip(rows, results_jsons, strict=True):
        unitig_scores = Path(results_json).parent / "eval_scores.npz"
        if unitig_scores.is_file():
            row.update(operating_point(unitig_scores, "unitig"))
        else:
            logger.warning("no %s — no operating point for %s", unitig_scores, row["drug"])

        ft = (ft_scores or {}).get(row["drug"])
        if ft is None:
            continue
        # The same convention on the fine-tune's own scores, so the two arms' sens/spec are
        # comparable rather than one being at Youden and the other at whatever 0.5 gave.
        row.update(operating_point(Path(ft), "ft"))
        if not unitig_scores.is_file():
            logger.warning("no %s — skipping the CI for %s", unitig_scores, row["drug"])
            continue
        ci = paired_ci_against_ft(unitig_scores, ft, seed=seed)
        if ci is not None:
            row.update(ci)

    table = pd.DataFrame(rows).sort_values("drug").reset_index(drop=True)

    # Positive deltas mean the unitig screen found signal the other arm did not. ft_auroc comes from
    # the fine-tune's own scores (above), so this exists whenever --ft-scores was supplied.
    if "ft_auroc" in table.columns:
        table["delta_vs_ft_auroc"] = table["unitig_auroc"] - table["ft_auroc"]

    if panel_csv is not None and panel_csv.is_file():
        panel = pd.read_csv(panel_csv)
        available = [c for c in PANEL_COLUMNS if c in panel.columns]
        if "drug" not in available:
            raise SystemExit(f"{panel_csv} has no 'drug' column (has {list(panel.columns)[:10]})")
        if "ceiling_auroc" not in available:
            logger.warning(
                "%s has no ceiling_auroc (has %s) — no catalogue ceiling to compare against",
                panel_csv, list(panel.columns)[:8],
            )
        # Alias before merging, or `rifampin` matches nothing and comes back NaN without complaint.
        table["_panel_key"] = table["drug"].replace(PANEL_DRUG_ALIASES)
        merged = table.merge(
            panel[available].rename(columns={"drug": "_panel_key"}), on="_panel_key", how="left"
        )
        missing = merged.loc[merged.get("ceiling_auroc", pd.Series(dtype=float)).isna(), "drug"].tolist()
        if missing:
            # A silent NaN reads as "no ceiling exists" when it means "this drug is not in the panel".
            # TB is the live case: the panel has 9 of 10 drugs, rifabutin having never been built.
            logger.warning(
                "%d drug(s) absent from %s, so they have no ceiling: %s",
                len(missing), panel_csv.name, ", ".join(sorted(missing)),
            )
        provisional = (
            merged.loc[merged["ceiling_status"] == "provisional", "drug"]
            if "ceiling_status" in merged.columns
            else pd.Series(dtype=str)
        )
        if len(provisional):
            # Say it once, loudly, rather than trusting whoever reads the CSV to check the column.
            logger.warning(
                "%d drug(s) carry a PROVISIONAL ceiling — not a like-for-like comparison: %s",
                len(provisional), ", ".join(sorted(provisional)),
            )
        table = merged.drop(columns="_panel_key")
        if "ceiling_auroc" in table.columns:
            table["delta_vs_ceiling_auroc"] = table["unitig_auroc"] - table["ceiling_auroc"]
    else:
        logger.warning("no panel CSV — no catalogue ceiling column")
    return table


def run(
    *, results_jsons: list[Path], out_csv: Path, panel_csv: Path | None = None,
    ft_scores: dict[str, Path] | None = None, seed: int = 1,
) -> pd.DataFrame:
    """Build and write the comparison table."""
    table = collect(results_jsons, panel_csv, ft_scores=ft_scores, seed=seed)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    logger.info("wrote %s (%d drugs)", out_csv, len(table))
    for row in table.itertuples(index=False):
        ft = getattr(row, "ft_auroc", float("nan"))
        ceiling = getattr(row, "ceiling_auroc", float("nan"))
        line = (f"  {row.drug:<32} unitig-LR {row.unitig_auroc:.4f} | FT {ft:.4f} | "
                f"ceiling {ceiling:.4f}")
        lo, hi = getattr(row, "delta_ci_lo", None), getattr(row, "delta_ci_hi", None)
        if lo is not None and not pd.isna(lo):
            verdict = "separates from 0" if row.separates_from_zero else "CI spans 0 — a tie"
            line += f"  | delta {row.delta_unitig_minus_ft:+.4f} [{lo:+.4f}, {hi:+.4f}] {verdict}"
        logger.info(line)
    return table


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, nargs="+", required=True,
                   help="One or more unitig-LR results.json files.")
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--panel-csv", type=Path, default=None,
                   help="visualisations/<organism>/catalogue_ceiling_panel.csv — the catalogue ceiling "
                        "and its provenance. Never a summary panel: those carry a stale ft_auroc.")
    p.add_argument("--ft-scores", nargs="+", default=None, metavar="DRUG=PATH",
                   help="Fine-tune eval_scores.npz per drug, e.g. ertapenem=/path/eval_scores.npz. "
                        "Adds a paired bootstrap CI on the unitig-minus-FT AUROC delta.")
    p.add_argument("--seed", type=int, default=1, help="Seed for the paired bootstrap.")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ft_scores = None
    if args.ft_scores:
        ft_scores = {}
        for item in args.ft_scores:
            drug, sep, path = item.partition("=")
            if not sep:
                raise SystemExit(f"--ft-scores entry {item!r} is not DRUG=PATH")
            ft_scores[drug] = Path(path)

    run(results_jsons=args.results, out_csv=args.out_csv, panel_csv=args.panel_csv,
        ft_scores=ft_scores, seed=args.seed)


if __name__ == "__main__":
    main()

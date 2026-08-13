"""Collate unitig-LR results against the Bacformer fine-tune and the catalogue ceiling.

The deliverable of the whole exercise is one table per organism answering: for each drug, how does a
mechanism-agnostic unitig screen compare to the fine-tuned model, and to the ceiling set by known
resistance determinants?

Reads each drug's unitig-LR ``results.json`` (schema v1.2) and joins it to the numbers already
checked into ``src/bacpredict/visualisations/<organism>/amr_summary_panel.csv``, which carries
``ceiling_auroc``/``ceiling_auprc`` (CARD for Kp, WHO/TB-Profiler for TB) and ``ft_auroc``/
``ft_auprc``. The output is deliberately a plain CSV with one row per drug so it drops into the
existing ladder plots without touching them.

**A point estimate per drug is not enough.** With four drugs and deltas that may be a few
thousandths, a bare table invites reading a tie as a win — the sibling invasion comparator hit
exactly this (+0.0055 in Bacformer's favour). So when the fine-tune's ``eval_scores.npz`` is
available, this also computes a **paired** bootstrap CI on the AUROC difference, reusing
:func:`~bac_pyseer.kleb_iso_source.unitig_presence_model.paired_delta_ci`. Paired matters: both
models score the identical genomes, and resampling them independently would widen the interval by
ignoring that both face the same easy and hard cases. Predictions are aligned by **sample id**, not
position.

Two caveats worth carrying into any write-up, both inherited rather than introduced here:

* The Kp ``concat_*`` column in that panel is known-unreliable (``Bacformer_FT_DEFICITS.md`` §8);
  this module reads ``ft_*`` and ``ceiling_*`` only.
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

logger = logging.getLogger(__name__)

PANEL_COLUMNS = ("drug", "ceiling_auroc", "ceiling_auprc", "ft_auroc", "ft_auprc")


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


def read_unitig_results(results_json: Path) -> dict[str, object]:
    """Flatten one unitig-LR ``results.json`` into a row."""
    payload = json.loads(results_json.read_text())
    metrics = payload["metrics"]
    extra = payload.get("extra", {})
    gwas = extra.get("gwas_summary", {})
    return {
        "drug": payload["drug"],
        "task": payload["task"],
        "unitig_auroc": metrics["auroc"],
        "unitig_auprc": metrics["auprc"],
        "unitig_sensitivity": metrics["sensitivity"],
        "unitig_specificity": metrics["specificity"],
        "unitig_balanced_accuracy": metrics["balanced_accuracy"],
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
        ft = (ft_scores or {}).get(row["drug"])
        if ft is None:
            continue
        unitig_scores = Path(results_json).parent / "eval_scores.npz"
        if not unitig_scores.is_file():
            logger.warning("no %s — skipping the CI for %s", unitig_scores, row["drug"])
            continue
        ci = paired_ci_against_ft(unitig_scores, ft, seed=seed)
        if ci is not None:
            row.update(ci)

    table = pd.DataFrame(rows).sort_values("drug").reset_index(drop=True)

    if panel_csv is not None and panel_csv.is_file():
        panel = pd.read_csv(panel_csv)
        available = [c for c in PANEL_COLUMNS if c in panel.columns]
        if "drug" not in available:
            raise SystemExit(f"{panel_csv} has no 'drug' column (has {list(panel.columns)[:10]})")
        table = table.merge(panel[available], on="drug", how="left")
        # Positive deltas mean the unitig screen found signal the other arm did not.
        if "ft_auroc" in table.columns:
            table["delta_vs_ft_auroc"] = table["unitig_auroc"] - table["ft_auroc"]
        if "ceiling_auroc" in table.columns:
            table["delta_vs_ceiling_auroc"] = table["unitig_auroc"] - table["ceiling_auroc"]
    else:
        logger.warning("no panel CSV — emitting unitig-LR columns only, with nothing to compare against")
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
                   help="visualisations/<organism>/amr_summary_panel.csv for the FT + ceiling columns.")
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

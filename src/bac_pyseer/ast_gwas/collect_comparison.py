"""Collate unitig-LR results against the Bacformer fine-tune and the catalogue ceiling.

The deliverable of the whole exercise is one table per organism answering: for each drug, how does a
mechanism-agnostic unitig screen compare to the fine-tuned model, and to the ceiling set by known
resistance determinants?

Reads each drug's unitig-LR ``results.json`` (schema v1.2) and joins it to the numbers already
checked into ``src/bacpredict/visualisations/<organism>/amr_summary_panel.csv``, which carries
``ceiling_auroc``/``ceiling_auprc`` (CARD for Kp, WHO/TB-Profiler for TB) and ``ft_auroc``/
``ft_auprc``. The output is deliberately a plain CSV with one row per drug so it drops into the
existing ladder plots without touching them.

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

import pandas as pd

logger = logging.getLogger(__name__)

PANEL_COLUMNS = ("drug", "ceiling_auroc", "ceiling_auprc", "ft_auroc", "ft_auprc")


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


def collect(results_jsons: list[Path], panel_csv: Path | None = None) -> pd.DataFrame:
    """Join unitig-LR rows to the recorded fine-tune and catalogue numbers, one row per drug."""
    rows = [read_unitig_results(p) for p in results_jsons]
    if not rows:
        raise SystemExit("no results.json files given")
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


def run(*, results_jsons: list[Path], out_csv: Path, panel_csv: Path | None = None) -> pd.DataFrame:
    """Build and write the comparison table."""
    table = collect(results_jsons, panel_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    logger.info("wrote %s (%d drugs)", out_csv, len(table))
    for row in table.itertuples(index=False):
        ft = getattr(row, "ft_auroc", float("nan"))
        ceiling = getattr(row, "ceiling_auroc", float("nan"))
        logger.info("  %-32s unitig-LR %.4f | FT %s | ceiling %s",
                    row.drug, row.unitig_auroc, f"{ft:.4f}", f"{ceiling:.4f}")
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
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(results_jsons=args.results, out_csv=args.out_csv, panel_csv=args.panel_csv)


if __name__ == "__main__":
    main()

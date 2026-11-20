r"""Collect split-table fine-tunes into one table: a row per drug × holdout (× species), from each ``results.json``.

Reads ``<models-dir>/<drug>/results.json`` as written by ``finetune_amr --split-table`` (its ``holdouts`` block:
``holdout``, ``holdout_st``, …, each with per-species blocks when the table held more than one species). Every drug
asked for gets at least one row: a drug with no ``results.json`` is a ``missing`` row, never an absent one, so a
half-finished array cannot read as complete. A ``results.json`` without a ``holdouts`` block (an AST-sheet run)
or from the smoke path is flagged in ``status``.

Columns: ``drug, holdout, species, status, n_samples, n_resistant, n_susceptible, auroc, auprc, sensitivity,
specificity, balanced_accuracy, f1, prevalence, split_source, split_table_md5, git_sha, results_json, scores``.
``species`` is ``all`` for the whole holdout and the species name for a per-species row.

Usage::

    python -m bacpredict.engine.finetune.collect_holdout_metrics --models-dir $P/models \
        --drugs ciprofloxacin ceftazidime ... --out $P/models/holdout_metrics.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

METRICS = ("auroc", "auprc", "sensitivity", "specificity", "balanced_accuracy", "f1", "prevalence")
COLUMNS = ("drug", "holdout", "species", "status", "n_samples", "n_resistant", "n_susceptible", *METRICS,
           "split_source", "split_table_md5", "git_sha", "results_json", "scores")


def _row(drug: str, holdout: str, species: str, metrics: dict, base: dict) -> dict:
    per_class = metrics.get("per_class") or {}
    return {
        **base,
        "drug": drug,
        "holdout": holdout,
        "species": species,
        "n_samples": metrics.get("n_samples"),
        "n_resistant": (per_class.get("1") or {}).get("n"),
        "n_susceptible": (per_class.get("0") or {}).get("n"),
        **{m: metrics.get(m) for m in METRICS},
    }


def collect(models_dir: str | Path, drugs: list[str]) -> pd.DataFrame:
    """One row per drug × holdout × (all + each species); a ``missing`` row for a drug with no ``results.json``."""
    rows = []
    for drug in drugs:
        path = Path(models_dir) / drug / "results.json"
        if not path.exists():
            rows.append({"drug": drug, "status": "missing", "results_json": str(path)})
            continue
        res = json.loads(path.read_text())
        split = res.get("split") or {}
        source = split.get("source")
        base = {
            "status": "smoke" if source == "smoke" else "ok",
            "split_source": source,
            "split_table_md5": split.get("table_md5"),
            "git_sha": (res.get("model") or {}).get("git_sha"),
            "results_json": str(path),
        }
        holdouts = res.get("holdouts")
        if not holdouts:  # an AST-sheet run: only its holdout's metrics exist
            rows.append(_row(drug, "holdout", "all", res["metrics"], {**base, "status": "no_holdouts_block"}))
            continue
        for name, block in holdouts.items():
            scores = {"scores": str(path.parent / block["scores"])} if block.get("scores") else {}
            rows.append(_row(drug, name, "all", block["metrics"], {**base, **scores}))
            for species, metrics in (block.get("by_species") or {}).items():
                rows.append(_row(drug, name, species, metrics, {**base, **scores}))
    table = pd.DataFrame(rows).reindex(columns=list(COLUMNS))
    for col in ("n_samples", "n_resistant", "n_susceptible"):  # counts stay integers beside a missing row's NaN
        table[col] = table[col].astype("Int64")
    return table


def main() -> None:
    """CLI: write the collected table and print it; exit 1 if any drug is missing."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", type=Path, required=True, help="holds <drug>/results.json")
    ap.add_argument("--drugs", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True, help="output CSV")
    args = ap.parse_args()
    table = collect(args.models_dir, args.drugs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    shown = ["drug", "holdout", "species", "status", "n_samples", "n_resistant", "auroc", "auprc", "balanced_accuracy"]
    with pd.option_context("display.width", 200, "display.max_rows", 500, "display.float_format", "{:.3f}".format):
        print(table[shown].to_string(index=False))
    print(f"wrote {args.out}")
    missing = table.loc[table["status"] == "missing", "drug"].tolist()
    if missing:
        raise SystemExit(f"{len(missing)} of {len(args.drugs)} drugs have no results.json: {missing}")


if __name__ == "__main__":
    main()

"""Linear-model baselines for binary phenotype prediction.

Fits ``sklearn.LogisticRegression`` on one-hot metadata features (country,
Sublineage, or both) on the SAME train/eval split the deep model was scored
on. Use this to establish how much of the deep model's AUROC can be explained
by a simple linear model on country / Sublineage — the reviewer's first
challenge: "is the model just learning country / Sublineage?"

Three feature sets out of the box: ``country``, ``sublineage``,
``country+sublineage``. Reuses ``tl.train.metrics.compute_full_metrics`` for
§0.4 metrics (AUROC, AUPRC, sensitivity, specificity, balanced accuracy, F1,
confusion matrix, calibration) so the output is directly comparable with
``results.json`` / ``eval_results.json`` produced by the deep-model pipeline.

Output: a single JSON file (one entry per feature set), optionally appends a
Markdown table to an existing ``stratification_report.md`` (idempotently —
re-runs replace any prior baseline section rather than double-appending).

Usage::

    uv run python -m tl.train.linear_baselines \\
        --sheet-path <cohort>/binary_<pair>_with_split.csv \\
        --label-column blood_vs_faeces_label \\
        --metadata-file /home/.../metadata_v2_all_samples_and_columns.tsv \\
        --feature-sets country sublineage country+sublineage \\
        --out <cohort>/linear_baselines.json \\
        [--also-score-validate] \\
        [--update-report <cohort>/stratification_report.md] \\
        [--task kleb_iso_source]
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from tl.train.metrics import compute_full_metrics

DEFAULT_METADATA = Path(
    "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/final/metadata_v2_all_samples_and_columns.tsv"
)

FEATURE_SET_COLUMNS: dict[str, list[str]] = {
    "country": ["country_parsed"],
    "sublineage": ["Sublineage"],
    "country+sublineage": ["country_parsed", "Sublineage"],
}

_NA_TOKEN = "__NA__"


def _load_split(sheet_path: Path, label_column: str) -> pd.DataFrame:
    df = pd.read_csv(sheet_path, low_memory=False)
    for col in ("Sample", "train_val_eval", label_column):
        if col not in df.columns:
            raise ValueError(f"Split CSV {sheet_path} is missing required column {col!r}")
    df["Sample"] = df["Sample"].astype(str)
    return df


def _load_metadata(metadata_file: Path, columns: list[str]) -> pd.DataFrame:
    meta = pd.read_csv(metadata_file, sep="\t", low_memory=False)
    if "sample_accession" not in meta.columns:
        raise ValueError(f"Metadata {metadata_file} lacks the sample_accession column")
    missing = set(columns) - set(meta.columns)
    if missing:
        raise ValueError(f"Metadata is missing required feature columns: {sorted(missing)}")
    # v2 has ~115 duplicate sample_accessions; keep first.
    meta = meta.drop_duplicates("sample_accession", keep="first")
    meta = meta.set_index(meta["sample_accession"].astype(str))
    return meta[columns]


def _fit_and_score(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list[str],
    label_column: str,
) -> tuple[dict[str, Any], int, str]:
    """Fit OneHotEncoder + LR on TRAIN, score on TEST, return (§0.4 metrics, n_features, model_repr)."""
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    X_train = enc.fit_transform(df_train[feature_cols].astype(str).fillna(_NA_TOKEN))
    X_test = enc.transform(df_test[feature_cols].astype(str).fillna(_NA_TOKEN))
    y_train = df_train[label_column].astype(int).to_numpy()
    y_test = df_test[label_column].astype(int).to_numpy()
    model = LogisticRegression(max_iter=2000, solver="lbfgs")
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    return compute_full_metrics(y_test, y_prob), X_train.shape[1], repr(model)


def run_baselines(
    sheet_path: Path,
    label_column: str,
    metadata_file: Path,
    feature_sets: list[str],
    also_score_validate: bool = False,
    task: str = "linear_baseline",
) -> dict[str, Any]:
    """Fit + score every requested feature set; return a JSON-ready payload."""
    split_df = _load_split(sheet_path, label_column)
    needed_cols = sorted({c for fs in feature_sets for c in FEATURE_SET_COLUMNS[fs]})
    joined = split_df.set_index("Sample")
    missing_in_split = [c for c in needed_cols if c not in joined.columns]
    if missing_in_split:
        # Fall back to v2 metadata for any feature columns the split CSV doesn't carry.
        meta_df = _load_metadata(metadata_file, missing_in_split)
        joined = joined.join(meta_df, how="left")
        logging.info("  joined %s from %s", missing_in_split, metadata_file.name)
    else:
        logging.info("  using %s straight from the split CSV (no metadata join needed)", needed_cols)

    n_missing_all = int(joined[needed_cols].isna().all(axis=1).sum())
    if n_missing_all:
        logging.warning(
            "  %d split rows have no value across {%s} — dropping",
            n_missing_all, ", ".join(needed_cols),
        )
    joined = joined.dropna(subset=needed_cols, how="all")
    joined = joined.dropna(subset=[label_column])

    train_df = joined[joined["train_val_eval"] == "train"].copy()
    eval_df = joined[joined["train_val_eval"] == "evaluate"].copy()
    val_df = joined[joined["train_val_eval"] == "validate"].copy()
    logging.info("  splits: train=%d  validate=%d  evaluate=%d", len(train_df), len(val_df), len(eval_df))
    if train_df.empty or eval_df.empty:
        raise ValueError(
            f"After join+filter, train or evaluate split is empty (train={len(train_df)}, eval={len(eval_df)})."
        )

    baselines: dict[str, Any] = {}
    for fs in feature_sets:
        cols = FEATURE_SET_COLUMNS[fs]
        logging.info("  fitting %s (features %s)", fs, cols)
        eval_metrics, n_feat, model_repr = _fit_and_score(train_df, eval_df, cols, label_column)
        entry: dict[str, Any] = {"model_repr": model_repr, "n_features": int(n_feat), "metrics": eval_metrics}
        if also_score_validate and not val_df.empty:
            val_metrics, _, _ = _fit_and_score(train_df, val_df, cols, label_column)
            entry["metrics_validate"] = val_metrics
        baselines[fs] = entry
        logging.info("    eval AUROC=%.4f  AUPRC=%.4f  bal-acc=%.4f",
                     eval_metrics["auroc"], eval_metrics["auprc"], eval_metrics["balanced_accuracy"])

    return {
        "schema_version": "1.0",
        "task": task,
        "label_column": label_column,
        "sheet_path": str(sheet_path),
        "metadata_file": str(metadata_file),
        "n_train": int(len(train_df)),
        "n_evaluate": int(len(eval_df)),
        "n_validate": int(len(val_df)),
        "n_dropped_no_metadata": n_missing_all,
        "baselines": baselines,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
    }


_BASELINE_HEADING = "## Linear-model baseline (LogisticRegression on country / Sublineage one-hot features)"
_PRIOR_HEADINGS = (
    _BASELINE_HEADING,
    "## Metadata-only baseline (LogisticRegression on one-hot features)",  # legacy name
)


def _strip_prior_section(report_text: str) -> str:
    """Remove any prior baseline section (delimited by the heading and the next '## ' heading or EOF).

    Idempotent re-runs: re-appending replaces the prior section rather than double-writing.
    """
    lines = report_text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")
        if line in _PRIOR_HEADINGS:
            # Skip this heading and everything until the next '## ' (same-level) heading or EOF.
            i += 1
            while i < len(lines) and not lines[i].startswith("## "):
                i += 1
            continue
        out.append(lines[i])
        i += 1
    # Trim trailing blank lines so the appended section sits cleanly.
    text = "".join(out)
    while text.endswith("\n\n"):
        text = text[:-1]
    return text


def append_report_section(report_path: Path, payload: dict[str, Any]) -> None:
    """Append the linear-model baseline table to a stratification_report.md (idempotent)."""
    if not report_path.exists():
        logging.warning("  --update-report: %s does not exist; skipping", report_path)
        return
    existing = report_path.read_text()
    stripped = _strip_prior_section(existing)
    if stripped != existing:
        logging.info("  --update-report: removed prior baseline section before re-appending")

    label = payload["label_column"]
    lines = [
        "",
        _BASELINE_HEADING,
        "",
        "Fit on the TRAIN split, scored on the held-out EVALUATE split. Pure sanity check: "
        "shows the AUROC achievable from country / Sublineage alone (no genomic features). "
        "If this is near random and the deep model is much higher, the deep model is learning "
        "signal beyond what country + Sublineage encode.",
        "",
        "| Feature set | n_features | AUROC | AUPRC | bal-acc | F1 |",
        "|---|---|---|---|---|---|",
    ]
    for name, b in payload["baselines"].items():
        m = b["metrics"]
        lines.append(
            f"| {name} | {b['n_features']} | {m['auroc']:.3f} | {m['auprc']:.3f} "
            f"| {m['balanced_accuracy']:.3f} | {m['f1']:.3f} |"
        )
    lines += [
        "",
        f"_n_train={payload['n_train']:,}, n_evaluate={payload['n_evaluate']:,}, label={label}._",
        "",
    ]
    report_path.write_text(stripped + "\n".join(lines))


def _main_cli() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sheet-path", type=Path, required=True,
                   help="Split CSV (Sample, <label>, train_val_eval).")
    p.add_argument("--label-column", required=True,
                   help="Binary label column name in the split CSV (e.g. blood_vs_faeces_label).")
    p.add_argument("--metadata-file", type=Path, default=DEFAULT_METADATA,
                   help="v2 metadata TSV with country_parsed + Sublineage.")
    p.add_argument("--feature-sets", nargs="+",
                   default=["country", "sublineage", "country+sublineage"],
                   choices=list(FEATURE_SET_COLUMNS),
                   help="Which feature sets to fit (any combination).")
    p.add_argument("--out", type=Path, required=True,
                   help="Path to write linear_baselines.json.")
    p.add_argument("--also-score-validate", action="store_true",
                   help="Also score on the validate split (handy for direct val-curve comparison).")
    p.add_argument("--update-report", type=Path, default=None,
                   help="If set, append (idempotently) a 'Linear-model baseline' section to this stratification_report.md.")
    p.add_argument("--task", default="linear_baseline",
                   help="Free-form task label stored in the JSON (e.g. kleb_iso_source).")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    payload = run_baselines(
        sheet_path=args.sheet_path,
        label_column=args.label_column,
        metadata_file=args.metadata_file,
        feature_sets=list(args.feature_sets),
        also_score_validate=args.also_score_validate,
        task=args.task,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    logging.info("wrote %s", args.out)
    if args.update_report is not None:
        append_report_section(args.update_report, payload)
        logging.info("appended baseline section to %s", args.update_report)


if __name__ == "__main__":
    _main_cli()

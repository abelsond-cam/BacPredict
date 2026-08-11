"""Fit the logistic regression on significant unitigs and score the deployed holdout.

The read-out end of the comparison. Features are the binary presence/absence of the GWAS-significant
unitigs (:mod:`bac_pyseer.ast_gwas.unitig_design_matrix`); the model is the repo's pinned estimator,
``LOGREG_KW`` from :mod:`bacpredict.engine.segment_amr_lr.fit_lr` — C=1.0, L2, lbfgs, no class
weight — imported rather than restated so this baseline tracks the CARD/WHO catalogue ceilings if
that pin ever moves. The metric block and results schema are the engine's too, so a unitig-LR
``results.json`` sits alongside a fine-tune's and a catalogue's without translation.

Departures from :func:`~bacpredict.engine.segment_amr_lr.fit_lr.fit_score_step`, and why:

* **The design matrix stays sparse.** ``fit_score_step`` densifies via ``to_numpy()``, which is fine
  for a handful of gene columns and fatal at 10⁴–10⁶ unitigs. We keep CSR all the way into
  scikit-learn (lbfgs accepts it), following the ``sp.hstack`` pattern in
  :mod:`bacpredict.engine.finetune.linear_baselines`.
* **No standardisation.** The features are already 0/1, so there is no scaler — and therefore no
  scaler that could be accidentally fitted across a split boundary.

Splits come from the same ``<drug>_split.csv`` the fine-tuned checkpoint was evaluated on: fit on
``train``, choose the operating threshold by Youden's J on ``validate``, and touch ``holdout``
exactly once, at scoring time.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from bac_pyseer.ast_gwas.unitig_design_matrix import load_design
from bacpredict.engine.finetune.metrics import (
    build_results_payload,
    compute_full_metrics,
    write_results_json,
    youden_threshold,
)
from bacpredict.engine.segment_amr_lr.fit_lr import LOGREG_KW
from bacpredict.engine.splits.load_splits import load_splits

logger = logging.getLogger(__name__)

MODEL_NAME = "unitig_lr"
SPLIT_SOURCE = "split_table"
_TASK_BY_ORGANISM = {"kp": "kleb_ast", "tb": "tb_ast"}


def _rows_for(sample_ids: list[str], wanted: list[str], label_map: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """Row indices into the design matrix for ``wanted``, plus their labels, skipping absentees."""
    row_of = {s: i for i, s in enumerate(sample_ids)}
    rows = [row_of[s] for s in wanted if s in row_of and s in label_map]
    labels = [label_map[s] for s in wanted if s in row_of and s in label_map]
    return np.asarray(rows, dtype=np.int64), np.asarray(labels, dtype=np.int64)


def fit_unitig_lr(
    matrix: sparse.csr_matrix, sample_ids: list[str], split_table: Path
) -> dict[str, object]:
    """Fit on train, threshold on validate, score holdout → metrics + per-sample holdout scores.

    Returns
    -------
    dict
        ``metrics`` (the §0.4 block at 0.5), ``operating_point`` (Youden's J chosen on validate and
        reported on holdout), ``coef``, the split sizes, and the holdout ``y_true``/``y_prob``.
    """
    label_map, train_ids, validate_ids, holdout_ids = load_splits(split_table)
    tr_rows, y_tr = _rows_for(sample_ids, train_ids, label_map)
    va_rows, y_va = _rows_for(sample_ids, validate_ids, label_map)
    ho_rows, y_ho = _rows_for(sample_ids, holdout_ids, label_map)

    if tr_rows.size == 0 or ho_rows.size == 0:
        raise SystemExit(f"empty train ({tr_rows.size}) or holdout ({ho_rows.size}) after joining to the design")
    if np.unique(y_tr).size < 2:
        raise SystemExit(f"train split is single-class (n={y_tr.size}) — cannot fit")
    if matrix.shape[1] == 0:
        raise SystemExit("design matrix has no unitig columns")

    clf = LogisticRegression(**LOGREG_KW)
    clf.fit(matrix[tr_rows], y_tr)
    p_ho = clf.predict_proba(matrix[ho_rows])[:, 1]

    # Threshold from validate only; holdout is scored once, below. A single-class or empty validate
    # split leaves youden_threshold at its 0.5 default rather than borrowing the holdout.
    thr = youden_threshold(y_va, clf.predict_proba(matrix[va_rows])[:, 1]) if va_rows.size else 0.5

    metrics = compute_full_metrics(y_ho, p_ho)
    at_thr = compute_full_metrics(y_ho, p_ho, threshold=thr)
    operating_point = {
        "objective": "youden_j",
        "selected_on": "validation",
        "threshold": float(thr),
        "sensitivity": at_thr["sensitivity"],
        "specificity": at_thr["specificity"],
        "balanced_accuracy": at_thr["balanced_accuracy"],
        "f1": at_thr["f1"],
        "confusion_matrix": at_thr["confusion_matrix"],
    }
    return {
        "metrics": metrics,
        "operating_point": operating_point,
        "coef": clf.coef_.ravel(),
        "intercept": float(clf.intercept_[0]),
        "n_train": int(tr_rows.size),
        "n_validate": int(va_rows.size),
        "n_holdout": int(ho_rows.size),
        "n_train_resistant": int(y_tr.sum()),
        "y_true": y_ho,
        "y_prob": p_ho,
    }


def run(
    *, design_dir: Path, split_table: Path, drug: str, organism: str, out_dir: Path,
    gwas_summary: Path | None = None,
) -> dict[str, object]:
    """Fit, score, and write ``results.json`` + ``eval_scores.npz`` + ``coefficients.tsv``."""
    matrix, sample_ids, id_map = load_design(design_dir)
    fit = fit_unitig_lr(matrix, sample_ids, split_table)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "eval_scores.npz",
        y_true=fit["y_true"], y_prob=fit["y_prob"],
        drug=np.array(drug), operating_threshold=np.array(fit["operating_point"]["threshold"]),
    )
    coefficients = id_map.copy()
    coefficients["lr_coef"] = fit["coef"]
    coefficients.sort_values("lr_coef", key=abs, ascending=False).to_csv(
        out_dir / "coefficients.tsv", sep="\t", index=False
    )

    # Carry the GWAS provenance into extra{} so a results row can be traced to the run that
    # selected its features without opening a second file.
    extra: dict[str, object] = {
        "n_unitigs": int(matrix.shape[1]),
        "n_train": fit["n_train"],
        "n_validate": fit["n_validate"],
        "n_train_resistant": fit["n_train_resistant"],
        "design_dir": str(design_dir),
        "features": "significant_unitig_presence",
        "estimator": f"LogisticRegression(**{LOGREG_KW})",
        "standardised": False,
    }
    design_manifest = design_dir / "design_manifest.json"
    if design_manifest.is_file():
        extra["design_manifest"] = json.loads(design_manifest.read_text())
    if gwas_summary is not None and gwas_summary.is_file():
        extra["gwas_summary"] = json.loads(gwas_summary.read_text())

    payload = build_results_payload(
        task=_TASK_BY_ORGANISM.get(organism, organism),
        drug=drug,
        model_name_or_path=MODEL_NAME,
        checkpoint_dir=str(out_dir),
        split_source=SPLIT_SOURCE,
        metrics=fit["metrics"],
        n_evaluate=fit["n_holdout"],
        operating_point=fit["operating_point"],
        extra=extra,
    )
    write_results_json(out_dir / "results.json", payload)
    logger.info(
        "%s %s: holdout n=%d  AUROC=%.4f  AUPRC=%.4f  (%d unitigs, %d train genomes)",
        organism, drug, fit["n_holdout"], fit["metrics"]["auroc"], fit["metrics"]["auprc"],
        matrix.shape[1], fit["n_train"],
    )
    return payload


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--design-dir", type=Path, required=True, help="Output dir of unitig_design_matrix.")
    p.add_argument("--split-table", type=Path, required=True, help="<drug>_split.csv — the deployed splits.")
    p.add_argument("--drug", required=True, help="AST column name (TB uses 'rifampin').")
    p.add_argument("--organism", choices=sorted(_TASK_BY_ORGANISM), required=True)
    p.add_argument("--out-dir", type=Path, required=True, help="Where results.json + eval_scores.npz go.")
    p.add_argument("--gwas-summary", type=Path, default=None,
                   help="<drug>_gwas_summary.json, recorded into extra{} for provenance.")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    payload = run(
        design_dir=args.design_dir, split_table=args.split_table, drug=args.drug,
        organism=args.organism, out_dir=args.out_dir, gwas_summary=args.gwas_summary,
    )
    print(json.dumps({k: payload[k] for k in ("task", "drug", "metrics")}, indent=2, default=str))


if __name__ == "__main__":
    main()

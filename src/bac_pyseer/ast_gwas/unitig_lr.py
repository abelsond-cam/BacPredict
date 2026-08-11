"""Fit the logistic regression on significant unitigs and score the deployed holdout.

The read-out end of the comparison. Features are the binary presence/absence of the GWAS-significant
unitigs (:mod:`bac_pyseer.ast_gwas.unitig_design_matrix`); the model is the repo's estimator,
``LOGREG_KW`` from :mod:`bacpredict.engine.segment_amr_lr.fit_lr` — L2, lbfgs, no class weight —
imported rather than restated so this baseline tracks the CARD/WHO catalogue ceilings if that pin
ever moves, with only its ``C`` swept (see below). The metric block and results schema are the
engine's too, so a unitig-LR ``results.json`` sits alongside a fine-tune's and a catalogue's without
translation.

Departures from :func:`~bacpredict.engine.segment_amr_lr.fit_lr.fit_score_step`, and why:

* **The design matrix stays sparse.** ``fit_score_step`` densifies via ``to_numpy()``, which is fine
  for a handful of gene columns and fatal at 10⁴–10⁶ unitigs. We keep CSR all the way into
  scikit-learn (lbfgs accepts it), following the ``sp.hstack`` pattern in
  :mod:`bacpredict.engine.finetune.linear_baselines`.
* **No standardisation.** The features are already 0/1, so there is no scaler — and therefore no
  scaler that could be accidentally fitted across a split boundary.
* **``C`` is swept on validate rather than pinned.** The repo pins ``C=1.0`` for every read-out, and
  that is right for a catalogue one-hot of a few dozen determinant columns. It is not right here:
  the sibling invasion comparator
  (:mod:`bac_pyseer.kleb_iso_source.unitig_presence_model`) measured that ~33k correlated binary
  unitig columns against ~9.5k training rows *overfit badly* at ``C=1.0``, and swept several decades
  stronger. Unitig features are massively LD-redundant — one megaplasmid contributes thousands of
  co-inherited columns — so the same applies here. We reuse that module's ``DEFAULT_C_GRID`` so the
  two comparators stay consistent, select on ``validate`` only, and **also** fit the pinned
  ``C=1.0`` model as a secondary so the number remains directly comparable to the catalogue
  ceilings. Both land in the results JSON; the swept model is the headline.

Splits come from the same ``<drug>_split.csv`` the fine-tuned checkpoint was evaluated on: fit on
``train``, choose the operating threshold by Youden's J on ``validate``, and touch ``holdout``
exactly once, at scoring time.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from bac_pyseer.ast_gwas.unitig_design_matrix import load_design
from bac_pyseer.kleb_iso_source.unitig_presence_model import DEFAULT_C_GRID
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


def _rows_for(
    sample_ids: list[str], wanted: list[str], label_map: dict[str, int]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Row indices into the design matrix for ``wanted``, their labels, and the ids kept.

    The ids come back so the holdout predictions can be paired against another model's by sample
    rather than by position — assuming a shared order is how two models silently get compared on
    different genomes.
    """
    row_of = {s: i for i, s in enumerate(sample_ids)}
    kept = [s for s in wanted if s in row_of and s in label_map]
    rows = [row_of[s] for s in kept]
    labels = [label_map[s] for s in kept]
    return np.asarray(rows, dtype=np.int64), np.asarray(labels, dtype=np.int64), kept


def sweep_c(
    matrix: sparse.csr_matrix, tr_rows: np.ndarray, y_tr: np.ndarray,
    va_rows: np.ndarray, y_va: np.ndarray, c_grid: Sequence[float],
) -> tuple[float, list[dict[str, float]]]:
    """Choose ``C`` by validate AUROC → ``(best_C, sweep)``.

    Selection touches ``validate`` only, never the holdout, so the reported holdout AUROC is not
    tuned. A degenerate validate split (empty or single-class) cannot select anything, so the
    pinned ``C`` is used and the sweep comes back empty — recorded rather than hidden.
    """
    if va_rows.size == 0 or np.unique(y_va).size < 2:
        logger.warning(
            "validate split is empty or single-class — cannot sweep C without peeking at the "
            "holdout; falling back to the pinned C=%g", LOGREG_KW["C"],
        )
        return float(LOGREG_KW["C"]), []

    sweep: list[dict[str, float]] = []
    best: tuple[float, float] | None = None
    for c in c_grid:
        model = LogisticRegression(**{**LOGREG_KW, "C": c})
        model.fit(matrix[tr_rows], y_tr)
        auroc = float(roc_auc_score(y_va, model.predict_proba(matrix[va_rows])[:, 1]))
        sweep.append({"C": float(c), "validate_auroc": auroc})
        logger.info("  C=%-8g validate AUROC %.4f", c, auroc)
        if best is None or auroc > best[1]:
            best = (float(c), auroc)
    return best[0], sweep


def fit_unitig_lr(
    matrix: sparse.csr_matrix, sample_ids: list[str], split_table: Path,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
) -> dict[str, object]:
    """Fit on train, threshold on validate, score holdout → metrics + per-sample holdout scores.

    ``C`` is selected on validate from ``c_grid`` (see the module docstring for why the repo's
    pinned 1.0 is wrong for unitig features). The pinned model is also fitted and scored, so the
    catalogue-comparable number is never lost.

    Returns
    -------
    dict
        ``metrics`` (the §0.4 block at 0.5), ``operating_point`` (Youden's J chosen on validate and
        reported on holdout), ``coef``, the chosen ``C`` and its sweep, the pinned-``C`` metrics,
        the split sizes, and the holdout ``y_true``/``y_prob``.
    """
    label_map, train_ids, validate_ids, holdout_ids = load_splits(split_table)
    tr_rows, y_tr, _ = _rows_for(sample_ids, train_ids, label_map)
    va_rows, y_va, _ = _rows_for(sample_ids, validate_ids, label_map)
    ho_rows, y_ho, ho_ids = _rows_for(sample_ids, holdout_ids, label_map)

    if tr_rows.size == 0 or ho_rows.size == 0:
        raise SystemExit(f"empty train ({tr_rows.size}) or holdout ({ho_rows.size}) after joining to the design")
    if np.unique(y_tr).size < 2:
        raise SystemExit(f"train split is single-class (n={y_tr.size}) — cannot fit")
    if matrix.shape[1] == 0:
        raise SystemExit("design matrix has no unitig columns")

    best_c, sweep = sweep_c(matrix, tr_rows, y_tr, va_rows, y_va, c_grid)
    clf = LogisticRegression(**{**LOGREG_KW, "C": best_c})
    clf.fit(matrix[tr_rows], y_tr)
    p_ho = clf.predict_proba(matrix[ho_rows])[:, 1]

    # The pinned C=1.0 model, scored the same way. The catalogue ceilings this is compared against
    # are fitted at that C, so keeping it makes the ladder comparison like-for-like — and the gap
    # between the two is itself the evidence for how much the pin costs on unitig features.
    if best_c == float(LOGREG_KW["C"]):
        pinned_metrics = None
    else:
        pinned = LogisticRegression(**LOGREG_KW).fit(matrix[tr_rows], y_tr)
        pinned_metrics = compute_full_metrics(y_ho, pinned.predict_proba(matrix[ho_rows])[:, 1])

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
        "C": best_c,
        "c_grid": [float(c) for c in c_grid],
        "c_sweep": sweep,
        "pinned_metrics": pinned_metrics,
        "n_nonzero_coef": int((clf.coef_ != 0).sum()),
        "n_train": int(tr_rows.size),
        "n_validate": int(va_rows.size),
        "n_holdout": int(ho_rows.size),
        "n_train_resistant": int(y_tr.sum()),
        "y_true": y_ho,
        "y_prob": p_ho,
        "eval_sample_ids": ho_ids,
    }


def run(
    *, design_dir: Path, split_table: Path, drug: str, organism: str, out_dir: Path,
    gwas_summary: Path | None = None,
) -> dict[str, object]:
    """Fit, score, and write ``results.json`` + ``eval_scores.npz`` + ``coefficients.tsv``."""
    matrix, sample_ids, id_map = load_design(design_dir)
    fit = fit_unitig_lr(matrix, sample_ids, split_table)

    out_dir.mkdir(parents=True, exist_ok=True)
    # sample_ids is not in evaluate.py's original schema but is in the sibling comparator's, and
    # without it the paired bootstrap in collect_comparison cannot align two models' predictions.
    np.savez(
        out_dir / "eval_scores.npz",
        y_true=fit["y_true"], y_prob=fit["y_prob"],
        sample_ids=np.asarray(fit["eval_sample_ids"], dtype=np.str_),
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
        "estimator": f"LogisticRegression(**{ {**LOGREG_KW, 'C': fit['C']} })",
        "standardised": False,
        "C": fit["C"],
        "C_selected_on": "validate",
        "c_grid": fit["c_grid"],
        "c_sweep": fit["c_sweep"],
        "n_nonzero_coef": fit["n_nonzero_coef"],
        # The repo-pinned C=1.0 fit, kept so the ladder comparison against the catalogue ceilings
        # (which are fitted at that C) stays like-for-like. None when the sweep chose 1.0 anyway.
        "pinned_C_metrics": fit["pinned_metrics"],
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

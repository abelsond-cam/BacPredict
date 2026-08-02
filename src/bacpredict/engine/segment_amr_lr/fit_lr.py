"""The one logistic-regression fit/score engine every read-out step shares.

Two entry points over the SAME pinned estimator (``C=1.0`` L2 lbfgs, no class weight — user-confirmed), so
steps differ only in their features, never in the model:

* :func:`fit_score_step` — fit on TRAIN, score on EVALUATE (Youden operating point on VALIDATE). The
  split-based probe used by the concat ladder rungs.
* :func:`fit_one_segment` / :func:`fit_one_segment_imputed` / :func:`fit_per_segment` — the per-segment
  screen: out-of-fold CV AUROC on the fit set (the train-side **selection** metric) plus, when ``eval_ids``
  is given, a held-out ``eval_auroc`` on the deployed holdout (the **reported** number). Selection uses the
  OOF AUROC so the holdout is touched once, at evaluation.

Callers pass the split ids from :func:`bacpredict.engine.splits.load_splits.load_splits` (fit on ``train``,
evaluate on ``holdout``) — for a fine-tuned feature that is the only honest scope.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bacpredict.engine.finetune.metrics import compute_full_metrics, youden_threshold

# The locus-restricted probe head, fixed across every step (user-confirmed): C=1.0 L2 lbfgs, no
# class_weight. Pinned so the steps differ only in features. (L2 is lbfgs's default; passing penalty="l2"
# explicitly is deprecated in sklearn 1.8, so we rely on the default — same regularisation, no warning.)
LOGREG_KW = {"C": 1.0, "solver": "lbfgs", "max_iter": 2000}


def fit_score_step(
    feat_df: pd.DataFrame,
    *,
    kind: str,
    standardise: bool,
    label_map: dict[str, int],
    train_ids: list[str],
    validate_ids: list[str],
    evaluate_ids: list[str],
) -> dict:
    """Fit one LogisticRegression probe on TRAIN, score on EVALUATE.

    Parameters
    ----------
    feat_df
        Raw per-sample features indexed by Sample. Categorical (one-hot) for ``kind == "categorical"``,
        numeric otherwise.
    kind
        ``"categorical"`` (one-hot encode, encoder fit on train) or ``"numeric"``.
    standardise
        Fit a ``StandardScaler`` on train and apply to all splits (numeric only).
    label_map, train_ids, validate_ids, evaluate_ids
        The canonical clean splits + 0/1 labels.

    Returns
    -------
    dict
        ``metrics`` (§0.4 on this step's full evaluate subset), ``operating_point`` (Youden on validate),
        per-split kept counts, ``n_features``, ``model_repr``, and ``eval_probs``/``eval_labels`` (per
        evaluate sample — for the intersection headline + the plotting sidecar).
    """
    avail = set(feat_df.index)
    tr = [s for s in train_ids if s in avail]
    va = [s for s in validate_ids if s in avail]
    ev = [s for s in evaluate_ids if s in avail]
    if not tr or not ev:
        return {"error": f"empty split after feature alignment (train={len(tr)}, evaluate={len(ev)})"}

    y_tr = np.array([label_map[s] for s in tr], dtype=int)
    y_va = np.array([label_map[s] for s in va], dtype=int)
    y_ev = np.array([label_map[s] for s in ev], dtype=int)

    # A single-class training fold is unfittable (LogisticRegression needs ≥2 classes). This happens for a
    # near-fully-penetrant determinant scored on its carriers (all resistant) — return an error so
    # run_kfold_probe skips just this fold rather than crashing the whole probe.
    if len(np.unique(y_tr)) < 2:
        return {"error": f"single-class train fold (class {int(y_tr[0])}, n_train={len(y_tr)})"}

    if kind == "categorical":
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        x_tr = enc.fit_transform(feat_df.loc[tr].astype(str))
        x_va = enc.transform(feat_df.loc[va].astype(str)) if va else np.empty((0, x_tr.shape[1]))
        x_ev = enc.transform(feat_df.loc[ev].astype(str))
    else:
        x_tr = feat_df.loc[tr].to_numpy(dtype=float)
        x_va = feat_df.loc[va].to_numpy(dtype=float) if va else np.empty((0, x_tr.shape[1]))
        x_ev = feat_df.loc[ev].to_numpy(dtype=float)
        if standardise:
            scaler = StandardScaler().fit(x_tr)
            x_tr = scaler.transform(x_tr)
            x_va = scaler.transform(x_va) if va else x_va
            x_ev = scaler.transform(x_ev)

    clf = LogisticRegression(**LOGREG_KW)
    clf.fit(x_tr, y_tr)
    ev_prob = clf.predict_proba(x_ev)[:, 1]
    metrics = compute_full_metrics(y_ev, ev_prob)

    operating_point = None
    if va:
        va_prob = clf.predict_proba(x_va)[:, 1]
        thr = youden_threshold(y_va, va_prob)
        op = compute_full_metrics(y_ev, ev_prob, threshold=thr)
        operating_point = {
            "objective": "youden_j",
            "selected_on": "validate",
            "threshold": thr,
            "sensitivity": op["sensitivity"],
            "specificity": op["specificity"],
            "balanced_accuracy": op["balanced_accuracy"],
            "f1": op["f1"],
            "confusion_matrix": op["confusion_matrix"],
        }

    return {
        "n_features": int(x_tr.shape[1]),
        "n_train": len(tr),
        "n_validate": len(va),
        "n_evaluate": len(ev),
        "model_repr": repr(clf),
        "metrics": metrics,
        "operating_point": operating_point,
        "eval_probs": {s: float(p) for s, p in zip(ev, ev_prob, strict=True)},
        "eval_labels": {s: int(y) for s, y in zip(ev, y_ev, strict=True)},
    }


def fit_one_segment(
    ids: list[str], x: np.ndarray, y: np.ndarray, *, n_folds: int, seed: int,
    eval_ids: set[str] | None = None,
) -> dict | None:
    """Fit one segment's out-of-fold + full LR; ``None`` if its (fit) labels are single-class.

    With ``eval_ids`` the segment's genomes are split into a **fit** set (the ids *not* in ``eval_ids`` —
    train+validate) and a held-out **evaluate** set (the ids in ``eval_ids``). The out-of-fold CV + the
    full-fit LR are estimated on the fit set only, and ``eval_auroc`` is that full-fit model scored on the
    evaluate genomes — a real held-out-test number (present-conditioned, exactly like the OOF metric). The
    default (``eval_ids=None``) reproduces the original OOF-only behaviour bit-for-bit: every genome is a
    fit genome, ``oof_prob`` is keyed by all ``ids``, and the eval fields are empty.
    """
    ids = list(ids)
    x = np.asarray(x, dtype=np.float32)  # storage may be float16 (memory); fit in float32 (StandardScaler → f64)
    is_eval = np.array([s in eval_ids for s in ids], dtype=bool) if eval_ids else np.zeros(len(ids), bool)
    fit_sel = ~is_eval
    x_fit, y_fit = x[fit_sel], y[fit_sel]
    fit_ids = [s for s, e in zip(ids, is_eval, strict=True) if not e]
    n_pos = int(y_fit.sum())
    if n_pos == 0 or n_pos == len(y_fit):
        return None  # single-class fit set — no resistance contrast for this segment
    k = min(n_folds, n_pos, len(y_fit) - n_pos)
    if k < 2:
        return None
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    oof = np.full(len(y_fit), np.nan, dtype=float)
    for tr_idx, te_idx in skf.split(x_fit, y_fit):
        scaler = StandardScaler().fit(x_fit[tr_idx])
        clf = LogisticRegression(**LOGREG_KW).fit(scaler.transform(x_fit[tr_idx]), y_fit[tr_idx])
        oof[te_idx] = clf.predict_proba(scaler.transform(x_fit[te_idx]))[:, 1]
    full_scaler = StandardScaler().fit(x_fit)
    full_clf = LogisticRegression(**LOGREG_KW).fit(full_scaler.transform(x_fit), y_fit)
    result = {
        "auroc": float(roc_auc_score(y_fit, oof)),
        "oof_prob": {s: float(p) for s, p in zip(fit_ids, oof, strict=True)},
        "scaler": full_scaler,
        "clf": full_clf,
        "n_train": len(y_fit),
        "n_pos": n_pos,
        "eval_auroc": float("nan"),
        "eval_prob": {},
        "n_eval": 0,
        "n_eval_pos": 0,
    }
    if is_eval.any():
        x_ev, y_ev = x[is_eval], y[is_eval]
        eval_ids_list = [s for s, e in zip(ids, is_eval, strict=True) if e]
        n_ev_pos = int(y_ev.sum())
        result["n_eval"], result["n_eval_pos"] = int(len(y_ev)), n_ev_pos
        # Full-fit probabilities on the held-out evaluate genomes (present-conditioned, exactly like the
        # OOF metric) — keyed by Sample so a caller can compute AUPRC / any metric on the same holdout.
        p_ev = full_clf.predict_proba(full_scaler.transform(x_ev))[:, 1]
        result["eval_prob"] = {s: float(p) for s, p in zip(eval_ids_list, p_ev, strict=True)}
        if 0 < n_ev_pos < len(y_ev):  # need both classes for a held-out AUROC
            result["eval_auroc"] = float(roc_auc_score(y_ev, p_ev))
    return result


def fit_one_segment_imputed(
    present_ids: list[str], x_present: np.ndarray, all_ids: list[str], y_all: np.ndarray, dim: int,
    *, n_folds: int, seed: int, eval_ids: set[str] | None = None,
) -> dict | None:
    """Fit one segment over the **full** read universe, zero-imputing genomes where the segment is absent.

    Builds the ``[len(all_ids), dim]`` design matrix — the segment's real embedding for genomes that carry
    it single-copy, a 0-vector for the rest — so the LR sees the **presence/absence** signal (absent
    genomes are no longer dropped). For a universal segment (gyrA) this is ~identical to the drop-absent
    fit; for an accessory/acquired segment it lets the LR key on the absence pattern the one-hot uses.
    """
    pos = {s: i for i, s in enumerate(all_ids)}
    x = np.zeros((len(all_ids), dim), dtype=np.float32)
    rows = [pos[s] for s in present_ids if s in pos]
    if rows:
        x[rows] = x_present[: len(rows)]
    return fit_one_segment(list(all_ids), x, y_all, n_folds=n_folds, seed=seed, eval_ids=eval_ids)


def fit_per_segment(
    segment_matrices: dict[str, tuple[list[str], np.ndarray]],
    label_map: dict[str, int],
    *,
    n_folds: int,
    seed: int,
    n_jobs: int = 1,
    all_ids: list[str] | None = None,
    impute_absent_zero: bool = False,
    eval_ids: set[str] | None = None,
) -> dict[str, dict]:
    """Fit one LR per segment (out-of-fold train probs + full-train fit), segments in parallel.

    Each segment is independent, so the per-segment fits fan out over ``n_jobs`` worker processes
    (joblib). Returns ``{segment: {auroc, oof_prob: {sample: p}, scaler, clf, n_train, n_pos, eval_auroc,
    n_eval, n_eval_pos}}``; segments whose fit labels are single-class (no AUROC defined) are dropped.

    With ``impute_absent_zero`` the fit universe is ``all_ids`` (the full read set) and genomes lacking
    the segment get a 0-vector instead of being dropped — so the AUROC reflects presence/absence + the
    embedding, directly comparable to the determinant one-hot. Default off keeps the drop-absent
    (present-only) fit, conditioned on the segment being present.

    ``eval_ids`` (the held-out evaluate-split Sample ids) turns the OOF-only screen into a held-out-test
    screen: each segment's LR is fit on its non-eval genomes and ``eval_auroc`` is that model scored on its
    eval genomes. Default ``None`` → OOF only (eval fields empty).
    """
    segments = list(segment_matrices)
    if impute_absent_zero:
        if all_ids is None:
            raise ValueError("impute_absent_zero requires all_ids (the full read-genome id list).")
        y_all = np.array([label_map[s] for s in all_ids], dtype=int)
        dim = next(iter(segment_matrices.values()))[1].shape[1]
        results = Parallel(n_jobs=n_jobs)(
            delayed(fit_one_segment_imputed)(
                segment_matrices[g][0], segment_matrices[g][1], all_ids, y_all, dim,
                n_folds=n_folds, seed=seed, eval_ids=eval_ids)
            for g in segments
        )
    else:
        ys = {g: np.array([label_map[s] for s in segment_matrices[g][0]], dtype=int) for g in segments}
        results = Parallel(n_jobs=n_jobs)(
            delayed(fit_one_segment)(segment_matrices[g][0], segment_matrices[g][1], ys[g],
                                     n_folds=n_folds, seed=seed, eval_ids=eval_ids)
            for g in segments
        )
    fitted = {g: r for g, r in zip(segments, results, strict=True) if r is not None}
    return fitted

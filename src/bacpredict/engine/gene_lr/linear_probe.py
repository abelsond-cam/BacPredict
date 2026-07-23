"""The pinned locus-restricted linear-probe head: fit one ``LogisticRegression`` on TRAIN, score EVALUATE.

The single fit-and-score primitive every categorical / numeric feature step of the read-out ladder
shares, so steps differ only in their features, never in the estimator. The probe is fixed (user-confirmed)
at ``C=1.0`` L2 lbfgs, no class weight; the ``validate`` split only picks the Youden operating point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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
        Raw per-sample features indexed by Sample. Categorical (one-hot) for ``kind == "categorical"``
        (Step 1), numeric otherwise.
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

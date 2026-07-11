"""K-fold × m-seed harness for the linear read-out probes — confirm small AUROC deltas.

At the top of the localization ladder the inter-method AUROC deltas are small (concat 0.975 vs
ESM-rpoB 0.971 vs one-hot 0.960). A single canonical split can not tell whether those orderings are
real. This harness reruns any set of **pre-aligned** feature frames through the repo's canonical
k-fold (:func:`bacpredict.engine.finetune.split_utils.generate_kfold_splits` — a fixed evaluate holdout pinned by
``evaluate_seed``, validation folds rotated by ``seed``) for several seeds, scores every ``(fold,
seed)`` on the one fixed evaluate holdout with :func:`pangena_predict.snp_vs_esm_prediction.fit_score_step`,
and reports **mean ± sd** per frame plus **paired** per-run AUROC deltas between frames (the honest
test of "does A beat B" — same train rows, same evaluate rows, so the delta is paired).

Generic by design: the concat probe, the top-k-gene probe and the per-gene LR panel all feed it the
same way — a dict of named :class:`FeatureSpec` (frame, kind, standardise) over a common sample
universe. The universe defaults to the intersection of every frame's samples ∩ ``label_map`` so the
evaluate subset is identical across frames and the AUROCs are directly comparable.

Note: the k-fold evaluate holdout is the harness's *own* fixed 20 % (pinned by ``evaluate_seed``), not
the canonical deployed-model holdout — so its mean AUROC need not equal the single-split number. Its
job is the **significance of the deltas**, not reproducing the headline.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bacpredict.engine.finetune.split_utils import generate_kfold_splits
from pangena_predict.snp_vs_esm_prediction import fit_score_step

logger = logging.getLogger(__name__)

# Metrics aggregated per frame. AUROC/AUPRC are threshold-free (from ``metrics``); sens/spec/bal-acc
# are read at the Youden operating point selected on each fold's validate split (from ``operating_point``).
_THRESHOLD_FREE = ("auroc", "auprc")
_OPERATING_POINT = ("sensitivity", "specificity", "balanced_accuracy")
AGG_METRICS = (*_THRESHOLD_FREE, *_OPERATING_POINT)


@dataclass(frozen=True)
class FeatureSpec:
    """One named feature frame and how :func:`fit_score_step` should consume it.

    Parameters
    ----------
    frame
        Per-sample features indexed by ``Sample``. Must cover the harness universe.
    kind
        ``"numeric"`` (default) or ``"categorical"`` (one-hot, e.g. the RRDR genotype).
    standardise
        Fit a ``StandardScaler`` on each fold's train split (numeric frames only).
    """

    frame: pd.DataFrame
    kind: str = "numeric"
    standardise: bool = True


def _common_universe(feature_specs: dict[str, FeatureSpec], label_map: dict[str, int]) -> list[str]:
    """Sorted intersection of every frame's samples with the labelled set — the comparable universe."""
    sets = [set(spec.frame.index) for spec in feature_specs.values()]
    common = set.intersection(*sets) & set(label_map) if sets else set()
    return sorted(common)


def _run_metrics(res: dict) -> dict | None:
    """Pull the reported metrics out of one :func:`fit_score_step` result (or ``None`` on error)."""
    if "metrics" not in res:
        return None
    metrics = res["metrics"]
    op = res.get("operating_point") or {}
    out = {m: metrics.get(m) for m in _THRESHOLD_FREE}
    out.update({m: op.get(m) for m in _OPERATING_POINT})
    out["n_evaluate"] = res.get("n_evaluate")
    out["n_train"] = res.get("n_train")
    return out


def _aggregate(values: list[float | None]) -> dict | None:
    """Mean / sd (sample, ddof=1) / min / max / n over the non-null values of one metric."""
    vals = np.array([v for v in values if v is not None], dtype=float)
    if vals.size == 0:
        return None
    return {
        "mean": float(vals.mean()),
        "sd": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
        "min": float(vals.min()),
        "max": float(vals.max()),
        "n": int(vals.size),
    }


def _paired_deltas(per_run: dict[str, list[dict]], metric: str = "auroc") -> dict:
    """Per-run paired ``metric`` deltas between every unordered pair of frames.

    Each run is keyed by ``(seed, fold)`` so the two frames are compared on the *same* train/evaluate
    rows — a paired test. Reports the mean ± sd delta, the run count, and the fraction of runs the
    first frame wins (a sign-test summary of "does A reliably beat B").
    """
    keyed = {
        name: {(r["seed"], r["fold"]): r.get(metric) for r in runs}
        for name, runs in per_run.items()
    }
    names = list(per_run)
    out: dict = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            common = sorted(set(keyed[a]) & set(keyed[b]))
            deltas = np.array(
                [keyed[a][k] - keyed[b][k] for k in common if keyed[a][k] is not None and keyed[b][k] is not None],
                dtype=float,
            )
            if deltas.size == 0:
                continue
            out[f"{a}__minus__{b}"] = {
                "metric": metric,
                "mean_delta": float(deltas.mean()),
                "sd_delta": float(deltas.std(ddof=1)) if deltas.size > 1 else 0.0,
                "n_runs": int(deltas.size),
                "n_first_wins": int((deltas > 0).sum()),
                "win_fraction": float((deltas > 0).mean()),
            }
    return out


def run_kfold_probe(
    feature_specs: dict[str, FeatureSpec],
    label_map: dict[str, int],
    *,
    universe_ids: list[str] | None = None,
    n_folds: int = 5,
    seeds: Sequence[int] = (1, 2, 3),
    evaluate_seed: int = 1,
    evaluate_fraction: float = 0.20,
) -> dict:
    """Score every feature frame over ``n_folds × len(seeds)`` runs and aggregate.

    The evaluate holdout is fixed (pinned by ``evaluate_seed``) and identical across seeds and frames;
    only the train/validate folds rotate (by ``seed`` × fold). Every ``(seed, fold)`` scores every
    frame on that one holdout, so the per-frame distributions are paired run-for-run.

    Parameters
    ----------
    feature_specs
        Named :class:`FeatureSpec` frames to compare on identical folds.
    label_map
        ``Sample → 0/1`` labels.
    universe_ids
        Samples the folds are built over (default: intersection of every frame ∩ ``label_map``).
    n_folds, seeds, evaluate_seed, evaluate_fraction
        Passed to :func:`generate_kfold_splits` (called once per seed).

    Returns
    -------
    dict
        ``config`` (sizes + the fixed evaluate ids), ``frames`` (``per_run`` + ``aggregate`` mean±sd
        per metric), and ``paired_auroc_deltas`` (every unordered frame pair).
    """
    if not feature_specs:
        raise ValueError("feature_specs is empty — nothing to score.")
    if universe_ids is None:
        universe_ids = _common_universe(feature_specs, label_map)
    if len(universe_ids) < n_folds + 1:
        raise ValueError(
            f"universe too small ({len(universe_ids)}) for {n_folds}-fold CV — need ≥ {n_folds + 1} samples."
        )
    universe_df = pd.DataFrame({"Sample": universe_ids})

    # Build (seed, fold) → (train, validate); evaluate is fixed across every seed (assert it).
    runs: list[tuple[int, int, list[str], list[str]]] = []
    evaluate_ids: list[str] | None = None
    for seed in seeds:
        ev_set, folds = generate_kfold_splits(
            universe_df, n_folds=n_folds, seed=seed,
            evaluate_fraction=evaluate_fraction, evaluate_seed=evaluate_seed,
        )
        ev_sorted = sorted(ev_set)
        if evaluate_ids is None:
            evaluate_ids = ev_sorted
        elif evaluate_ids != ev_sorted:
            raise RuntimeError("evaluate holdout drifted across seeds — evaluate_seed not honoured by the harness.")
        for fold_idx, (tr_set, va_set) in enumerate(folds):
            runs.append((seed, fold_idx, sorted(tr_set), sorted(va_set)))
    assert evaluate_ids is not None  # the seeds loop runs ≥ once, so this is always set

    logger.info(
        "k-fold probe: %d frames × %d folds × %d seeds = %d runs/frame; universe=%d, evaluate=%d",
        len(feature_specs), n_folds, len(seeds), len(runs), len(universe_ids), len(evaluate_ids),
    )

    per_run: dict[str, list[dict]] = {name: [] for name in feature_specs}
    for seed, fold_idx, train_ids, validate_ids in runs:
        for name, spec in feature_specs.items():
            res = fit_score_step(
                spec.frame, kind=spec.kind, standardise=spec.standardise, label_map=label_map,
                train_ids=train_ids, validate_ids=validate_ids, evaluate_ids=evaluate_ids,
            )
            row = _run_metrics(res)
            if row is None:
                logger.warning("frame %s seed=%d fold=%d errored: %s", name, seed, fold_idx, res.get("error"))
                continue
            per_run[name].append({"seed": seed, "fold": fold_idx, **row})

    frames: dict[str, dict] = {}
    for name, runs_for_frame in per_run.items():
        aggregate = {m: _aggregate([r.get(m) for r in runs_for_frame]) for m in AGG_METRICS}
        frames[name] = {"per_run": runs_for_frame, "aggregate": aggregate}
        auroc = aggregate.get("auroc")
        if auroc is not None:
            logger.info("frame %-28s AUROC %.4f ± %.4f (n=%d)", name, auroc["mean"], auroc["sd"], auroc["n"])

    return {
        "config": {
            "n_folds": n_folds,
            "seeds": list(seeds),
            "evaluate_seed": evaluate_seed,
            "evaluate_fraction": evaluate_fraction,
            "n_universe": len(universe_ids),
            "n_evaluate": len(evaluate_ids),
            "n_runs_per_frame": len(runs),
            "evaluate_ids": evaluate_ids,
        },
        "frames": frames,
        "paired_auroc_deltas": _paired_deltas(per_run, metric="auroc"),
    }


def summarise_kfold(result: dict) -> str:
    """A compact text block — per-frame AUROC mean±sd and the paired deltas — for logging/print."""
    lines = ["k-fold × m-seed summary (AUROC mean ± sd over fold×seed, fixed evaluate holdout):"]
    rows = sorted(
        ((name, f["aggregate"]["auroc"]) for name, f in result["frames"].items() if f["aggregate"]["auroc"]),
        key=lambda kv: kv[1]["mean"],
        reverse=True,
    )
    for name, agg in rows:
        lines.append(f"  {name:<32} {agg['mean']:.4f} ± {agg['sd']:.4f}  [{agg['min']:.4f}, {agg['max']:.4f}]")
    if result.get("paired_auroc_deltas"):
        lines.append("paired AUROC deltas (Δ mean ± sd, win-fraction over runs):")
        for pair, d in result["paired_auroc_deltas"].items():
            lines.append(
                f"  {pair:<48} Δ={d['mean_delta']:+.4f} ± {d['sd_delta']:.4f}  win={d['win_fraction']:.2f} (n={d['n_runs']})"
            )
    return "\n".join(lines)

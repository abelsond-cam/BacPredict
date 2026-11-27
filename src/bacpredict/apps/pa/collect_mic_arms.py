r"""Collect the MIC arms into one table: a row per arm x drug x holdout x subset.

Each arm is a directory of ``<drug>/results.json`` + ``eval_scores*.npz`` written by
``finetune_amr --split-table`` (the incumbent binary arm) or by its MIC path (``--mic-objective``). Every drug
asked for gets a row from every arm: an arm with no ``results.json`` for a drug is a ``missing`` row, never an
absent one, so a half-finished array cannot read as complete.

**Two subsets per holdout, and the reason for the second.** The arms do not score the same genomes. Arm 0
scores every cell of the split table's holdout; the MIC-cell arms score only those with a measured MIC. The
sets are *nested*, not disjoint, but a difference between two arms computed on two different sets is not a
difference between the arms. So every metric is reported twice:

* ``own`` — each arm on the holdout it actually scored, which is the number that arm would report alone;
* ``common`` — every arm on the **intersection** of all the arms' holdouts for that drug, which is the only
  subset on which the comparison is paired. ``n_samples`` is carried on both, because a comparison whose
  common subset has shrunk is a comparison the reader must be able to see has shrunk.

**What is reported, beyond AUROC.** AUROC is rank-only, and on known-truth simulation a fit with half the
true MIC scale scores within 0.03 of a correct one (``tests/engine/finetune/test_mic_estimator_recovery.py``).
So the table also carries the R/S call (balanced accuracy, sensitivity, specificity), calibration (Brier and
ECE — for the MIC arms both at the validate-re-fitted ``sigma`` and at the training one, so the size of the
temperature-scaling correction is visible), and for the MIC arms the censored NLL, interval coverage, and
absolute and essential (+-1 dilution) agreement.

Calibration is recomputed here from each arm's ``eval_scores.npz`` rather than read from ``results.json``, so
that the incumbent binary arm — whose results predate any of this — is scored by exactly the same code as the
MIC arms. ``results.json`` is read for provenance and for the MIC block.

Usage::

    python -m bacpredict.apps.pa.collect_mic_arms --arm 0_bce_all_cells=$P/models \
        --arm censored_interval=$Q/models/censored_interval --drugs amikacin ... --out $Q/models/mic_arm_metrics.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bacpredict.engine.finetune.censored_mic_loss import (
    calibration_scores,
    censored_log_likelihood,
    fit_logit_temperature,
    mic_agreement,
    prob_resistant,
    prob_resistant_tempered,
)
from bacpredict.engine.finetune.metrics import compute_full_metrics

logger = logging.getLogger(__name__)

METRICS = ("auroc", "auprc", "sensitivity", "specificity", "balanced_accuracy", "f1", "prevalence")
MIC_KEYS = ("sigma", "sigma_train", "censored_nll", "censored_nll_at_sigma_train", "interval_coverage",
            "absolute_agreement", "essential_agreement", "bias_log2", "rmse_log2", "pearson_r",
            "slope_log2", "n_exact")
CALIBRATION_KEYS = ("brier", "ece", "brier_raw", "ece_raw", "temperature")


def _scores(run: Path, scores_file: str) -> dict | None:
    """One holdout's per-sample scores, or ``None`` when the arm has not written them."""
    return _load_npz(run / scores_file)


def _load_npz(path: Path) -> dict | None:
    """An npz as a plain dict of arrays, or ``None`` when it is not there."""
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    out["sample_ids"] = np.asarray(out["sample_ids"], dtype=str)
    return out


def _row(arm: str, drug: str, holdout: str, subset: str, z: dict, mic_block: dict | None, base: dict,
         temperature: float = float("nan")) -> dict:
    """Metrics for one arm on one set of samples, recomputed here so every arm is scored by one code path.

    **Calibration is reported twice for every arm, raw and tempered, and the two mean the same thing in each.**
    ``*_raw`` is the model's own probability, straight from its head; ``brier``/``ece`` are after the
    one-parameter scale fitted on **validate** — ``sigma`` for a MIC arm, a logit temperature for a binary one
    (:func:`~bacpredict.engine.finetune.censored_mic_loss.fit_logit_temperature`). Reporting a tempered MIC arm
    against an untempered binary one would credit the estimator with the scaling. Neither scale can move
    ``auroc``, ``auprc`` or the R/S call, so those columns are the same number either way.
    """
    y_true, y_prob = z["y_true"].astype(int), z["y_prob"].astype(float)
    full = compute_full_metrics(y_true, y_prob)
    row = {**base, "arm": arm, "drug": drug, "holdout": holdout, "subset": subset,
           "n_samples": int(y_true.size), "n_resistant": int(y_true.sum()),
           **{m: full[m] for m in METRICS}}
    if "mu" not in z:
        raw = calibration_scores(y_true, y_prob)                      # sigmoid(z), the head's own probability
        if np.isfinite(temperature) and "score" in z and y_true.size:
            tempered = calibration_scores(y_true, prob_resistant_tempered(z["score"], temperature))
        else:
            tempered = {"brier": np.nan, "ece": np.nan}               # no validate scores: cannot be tempered
        return {**row, **dict.fromkeys(MIC_KEYS), "brier": tempered["brier"], "ece": tempered["ece"],
                "brier_raw": raw["brier"], "ece_raw": raw["ece"], "temperature": float(temperature)}
    mu, lo, hi = z["mu"].astype(float), z["lo"].astype(float), z["hi"].astype(float)
    sigma = float(z["sigma"])
    sigma_train = float((mic_block or {}).get("sigma_train", np.nan))
    censor = np.asarray((mic_block or {}).get("censor_classes", [""] * mu.size), dtype=object)
    agree = mic_agreement(mu, lo, hi, censor)
    tempered = calibration_scores(y_true, y_prob)                     # y_prob is already Phi(mu/sigma_validate)
    raw = (calibration_scores(y_true, prob_resistant(mu, sigma_train))
           if np.isfinite(sigma_train) and mu.size else {"brier": np.nan, "ece": np.nan})
    row |= {
        "brier": tempered["brier"], "ece": tempered["ece"],
        "brier_raw": raw["brier"], "ece_raw": raw["ece"], "temperature": float("nan"),
        "sigma": sigma, "sigma_train": sigma_train,
        "censored_nll": -float(np.mean(censored_log_likelihood(mu, sigma, lo, hi))) if mu.size else np.nan,
        "censored_nll_at_sigma_train": (
            -float(np.mean(censored_log_likelihood(mu, sigma_train, lo, hi)))
            if mu.size and np.isfinite(sigma_train) else np.nan),
        "interval_coverage": agree["interval_coverage"], "absolute_agreement": agree["absolute_agreement"],
        "essential_agreement": agree["essential_agreement"], "bias_log2": agree["bias_log2"],
        "rmse_log2": agree["rmse_log2"], "pearson_r": agree["pearson_r"],
        "slope_log2": agree["slope_log2"], "n_exact": agree["n_exact"],
    }
    return row


def _censor_classes(intervals_dir: Path | None, drug: str, sample_ids: np.ndarray) -> list[str]:
    """Each scored genome's censoring class, read from the drug's interval table.

    The npz carries ``lo``/``hi`` but not the class, and ``exact`` cannot be recovered from the interval alone
    (a one-dilution-wide interval could also be a ``> X, < U`` pair). Without the table the class is left
    blank, which makes the agreement rates NaN rather than wrong.
    """
    table = None if intervals_dir is None else intervals_dir / f"{drug}_mic_intervals.csv"
    if table is None or not table.exists():
        return [""] * len(sample_ids)
    t = pd.read_csv(table, usecols=["Sample", "censor"])
    m = dict(zip(t["Sample"].astype(str), t["censor"].astype(str), strict=True))
    return [m.get(s, "") for s in sample_ids]


def collect(arms: dict[str, Path], drugs: list[str], holdouts: tuple[str, ...],
            intervals_dir: Path | None = None,
            validate_scores: dict[str, Path] | None = None) -> pd.DataFrame:
    """One row per arm x drug x holdout x subset, with the paired ``common`` subset computed per drug."""
    rows: list[dict] = []
    for drug in drugs:
        loaded: dict[str, dict] = {}
        for arm, root in arms.items():
            run = root / drug
            results = run / "results.json"
            if not results.exists():
                rows.append({"arm": arm, "drug": drug, "holdout": "", "subset": "", "status": "missing",
                             "results_json": str(results)})
                continue
            r = json.loads(results.read_text())
            base = {"status": "ok", "split_source": r["split"].get("source"),
                    "split_table_md5": r["split"].get("table_md5"), "git_sha": r["model"].get("git_sha"),
                    "objective": (r.get("mic") or {}).get("objective", "bce"),
                    "cells": (r.get("mic") or {}).get("cells", "all"), "results_json": str(results)}
            # a binary arm's temperature, fitted on ITS OWN validate scores (the same split a MIC arm
            # re-fits sigma on). Absent for a run written before validate scores were saved — the incumbent
            # arm 0 — whose tempered calibration is then NaN rather than silently its raw one.
            v = _scores(run, "eval_scores_validate.npz")
            if v is None and (validate_scores or {}).get(arm):
                # an arm trained before validate scores were saved, re-scored afterwards by
                # engine/finetune/score_split into a directory of its own
                v = _load_npz(Path(validate_scores[arm]) / f"{drug}_validate_scores.npz")
            temperature = float("nan")
            if v is not None and "score" in v and "mu" not in v:
                fit = fit_logit_temperature(v["score"], v["y_true"])
                temperature = fit["temperature"]
                base = {**base, "temperature_fit_n": fit["n"], "temperature_at_bound": fit["at_bound"]}
            loaded[arm] = {"run": run, "results": r, "base": base, "temperature": temperature}

        for holdout in holdouts:
            present = {}
            for arm, info in loaded.items():
                block = (info["results"].get("holdouts") or {}).get(holdout)
                if not block:
                    continue
                z = _scores(info["run"], block["scores"])
                if z is None or z["y_true"].size == 0:
                    continue
                mic_block = dict(block.get("mic") or {})
                if "mu" in z:
                    mic_block["censor_classes"] = _censor_classes(intervals_dir, drug, z["sample_ids"])
                present[arm] = (z, mic_block, info["base"], info["temperature"])
            if not present:
                continue
            common = set.intersection(*(set(z["sample_ids"]) for z, _, _, _ in present.values()))
            for arm, (z, mic_block, base, temperature) in present.items():
                rows.append(_row(arm, drug, holdout, "own", z, mic_block, base, temperature))
                keep = np.isin(z["sample_ids"], list(common))
                sub = {k: (v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == z["y_true"].shape else v)
                       for k, v in z.items()}
                if mic_block.get("censor_classes") is not None:
                    mic_block = {**mic_block,
                                 "censor_classes": list(np.asarray(mic_block["censor_classes"])[keep])}
                rows.append(_row(arm, drug, holdout, "common", sub, mic_block, base, temperature))
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    """CLI: every arm's holdout metrics, on its own holdout and on the arms' common subset."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", action="append", required=True, metavar="NAME=DIR",
                   help="an arm's name and its models directory (repeatable; order is the report order)")
    p.add_argument("--drugs", nargs="+", required=True)
    p.add_argument("--holdouts", nargs="+", default=["holdout", "holdout_st"])
    p.add_argument("--intervals-dir", help="directory of <drug>_mic_intervals.csv, for the censoring classes "
                                           "the agreement rates need (without it they are NaN, not wrong)")
    p.add_argument("--validate-scores", action="append", default=[], metavar="NAME=DIR",
                   help="for an arm whose run dir has no eval_scores_validate.npz (one trained before they "
                        "were saved): a directory of <drug>_validate_scores.npz from engine/finetune/"
                        "score_split. Without it that arm's tempered calibration is NaN, never its raw one")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    arms: dict[str, Path] = {}
    for spec in args.arm:
        if "=" not in spec:
            raise SystemExit(f"--arm wants NAME=DIR, got {spec!r}")
        name, directory = spec.split("=", 1)
        arms[name] = Path(directory)
    extra_validate: dict[str, Path] = {}
    for spec in args.validate_scores:
        if "=" not in spec:
            raise SystemExit(f"--validate-scores wants NAME=DIR, got {spec!r}")
        name, directory = spec.split("=", 1)
        extra_validate[name] = Path(directory)
    df = collect(arms, args.drugs, tuple(args.holdouts),
                 Path(args.intervals_dir) if args.intervals_dir else None, extra_validate)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    logger.info("wrote %s (%d rows)", args.out, len(df))

    show = df[(df["holdout"] == "holdout") & (df["subset"] == "common")]
    cols = ["drug", "arm", "n_samples", "auroc", "auprc", "balanced_accuracy", "brier", "ece", "ece_raw",
            "temperature", "sigma", "censored_nll", "pearson_r", "slope_log2", "rmse_log2",
            "essential_agreement", "absolute_agreement"]
    if not show.empty:
        print("\n=== holdout, on the subset every arm scored (the paired comparison) ===")
        with pd.option_context("display.width", 300, "display.max_columns", 40):
            print(show[cols].round(4).to_string(index=False))
    missing = df[df["status"] == "missing"]
    if not missing.empty:
        print(f"\n{len(missing)} arm x drug with no results.json yet:")
        print(missing[["arm", "drug"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

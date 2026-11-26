"""Known-truth recovery: does each objective get the latent MIC scale back, and what happens when it does not?

A simulation is the only place this question has an answer, because only here is the latent log2 MIC known.
Truth is a linear model with ``sigma = 1.5`` dilutions, observed through two label sources with different
tested dilution ranges (the narrow one censors most of its cells) and a quarter of cells reported as a bare
R/S call. Reported MICs are generated as ``(sign, value)`` pairs and pushed through the **shipped** interval
builder, so what is tested is the production table and the production loss, not a copy of them.

The finding these tests lock down is the one that makes the comparison worth running: **AUROC cannot tell the
arms apart, and everything that depends on the MIC scale can.** Point-imputed least squares attenuates the
latent scale by roughly a half and still scores an AUROC within noise of the censored fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from bacpredict.engine.ast_labels.mic_intervals import intervals_from_reported_mic
from bacpredict.engine.finetune.censored_mic_loss import (
    calibration_scores,
    fit_sigma,
    interval_censored_gaussian_nll,
    prob_resistant,
)

R_GT, LOG_R, TRUE_SIGMA = 8.0, 3.0, 1.5
PANELS = {0: (1, 4), 1: (-3, 8)}   # source -> (lowest, highest) tested log2 dilution
BARE_FRACTION = 0.25


def _simulate(n: int, p_signal: int, p_noise: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p_signal + p_noise))
    beta = np.zeros(p_signal + p_noise)
    beta[:p_signal] = rng.normal(0, 1.2, p_signal)
    mu_true = X @ beta
    mu_true -= np.median(mu_true)
    y = mu_true + rng.normal(0, TRUE_SIGMA, n)
    k = np.ceil(y)

    source = rng.integers(0, 2, n)
    sign, value = np.full(n, "=", dtype=object), np.empty(n)
    for s, (lo_cap, hi_cap) in PANELS.items():
        m = source == s
        value[m] = 2.0 ** np.clip(k[m], lo_cap, hi_cap) * R_GT
        sign[m & (k > hi_cap)] = ">"
        sign[m & (k <= lo_cap)] = "<="
    lo, hi, point = intervals_from_reported_mic(pd.Series(sign), pd.Series(value), pd.Series(np.full(n, np.nan)))
    lo, hi, point = lo - LOG_R, hi - LOG_R, point - LOG_R

    label = (y > 0).astype(int)
    bare = rng.random(n) < BARE_FRACTION
    lo = np.where(bare, np.where(label == 1, 0.0, -np.inf), lo)
    hi = np.where(bare, np.where(label == 1, np.inf, 0.0), hi)
    point = np.where(bare, np.nan, point)
    return {"X": X, "mu_true": mu_true, "label": label, "lo": lo, "hi": hi, "point": point, "bare": bare}


def _fit(X, objective, *, label=None, lo=None, hi=None, point=None, steps=700, seed=0):
    torch.manual_seed(seed)
    net = torch.nn.Linear(X.shape[1], 1)
    log_sigma = torch.nn.Parameter(torch.zeros(()))
    groups = [{"params": list(net.parameters()), "lr": 0.08}]
    if objective == "censored":
        groups.append({"params": [log_sigma], "lr": 0.08})
    opt = torch.optim.Adam(groups)
    Xt = torch.tensor(X, dtype=torch.float32)
    for _ in range(steps):
        opt.zero_grad()
        mu = net(Xt).reshape(-1)
        if objective == "bce":
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                mu, torch.tensor(label, dtype=torch.float32))
        elif objective == "mse":
            loss = torch.nn.functional.mse_loss(mu, torch.tensor(point, dtype=torch.float32))
        else:
            loss = interval_censored_gaussian_nll(mu, log_sigma, torch.tensor(lo), torch.tensor(hi))
        loss.backward()
        opt.step()
    return net, float(log_sigma.detach().exp())


def _mu(net, X):
    with torch.no_grad():
        return net(torch.tensor(X, dtype=torch.float32)).reshape(-1).numpy()


@pytest.fixture(scope="module")
def arms():
    """Fit all four arms on one simulated cohort and score them on its holdout."""
    n = 1500
    d = _simulate(n, 6, 40, seed=1)
    perm = np.random.default_rng(99).permutation(n)
    tr, va, ho = perm[: int(0.6 * n)], perm[int(0.6 * n): int(0.75 * n)], perm[int(0.75 * n):]
    mic = ~d["bare"]
    tr_mic, va_mic = tr[mic[tr]], va[mic[va]]

    out = {}
    for name, idx, objective in (("bce_all", tr, "bce"), ("bce_mic", tr_mic, "bce")):
        net, _ = _fit(d["X"][idx], objective, label=d["label"][idx])
        mu = _mu(net, d["X"][ho])
        prob = 1 / (1 + np.exp(-mu))
        out[name] = {"mu": mu, "prob": prob, "sigma": np.nan}
    net, _ = _fit(d["X"][tr_mic], "mse", point=d["point"][tr_mic])
    s = fit_sigma(_mu(net, d["X"][va_mic]), d["lo"][va_mic], d["hi"][va_mic])["sigma"]
    out["mse"] = {"mu": _mu(net, d["X"][ho]), "prob": prob_resistant(_mu(net, d["X"][ho]), s), "sigma": s}
    net, s_train = _fit(d["X"][tr], "censored", lo=d["lo"][tr], hi=d["hi"][tr])
    s = fit_sigma(_mu(net, d["X"][va]), d["lo"][va], d["hi"][va])["sigma"]
    out["censored"] = {"mu": _mu(net, d["X"][ho]), "prob": prob_resistant(_mu(net, d["X"][ho]), s),
                       "sigma": s, "sigma_train": s_train}
    for v in out.values():
        v["auroc"] = roc_auc_score(d["label"][ho], v["prob"])
        v["bal_acc"] = balanced_accuracy_score(d["label"][ho], (v["prob"] >= 0.5).astype(int))
        v["ece"] = calibration_scores(d["label"][ho], v["prob"])["ece"]
        v["slope"] = float(np.polyfit(d["mu_true"][ho], v["mu"], 1)[0])
    return out


def test_the_censored_fit_recovers_the_latent_mic_scale(arms):
    """Regressing the fitted ``mu`` on the true latent mean gives a slope of 1: no attenuation, no inflation."""
    assert arms["censored"]["slope"] == pytest.approx(1.0, abs=0.15)


def test_point_imputation_attenuates_that_scale_by_about_half(arms):
    """Replacing ``> X`` by ``2X`` and ``<= X`` by ``X`` pulls both tails in, and least squares shrinks to fit."""
    assert arms["mse"]["slope"] < 0.8
    assert arms["mse"]["slope"] < arms["censored"]["slope"]


def test_sigma_is_recovered_and_the_validate_refit_is_closer_than_the_trained_one(arms):
    """The learned sigma is fitted alongside weights that have already seen the data; validate corrects it."""
    assert arms["censored"]["sigma"] == pytest.approx(TRUE_SIGMA, rel=0.25)
    assert abs(arms["censored"]["sigma"] - TRUE_SIGMA) <= abs(arms["censored"]["sigma_train"] - TRUE_SIGMA) + 0.1


def test_auroc_cannot_tell_a_badly_attenuated_fit_from_a_good_one(arms):
    """The reason this comparison must report more than AUROC: ranking survives a monotone distortion.

    The least-squares arm has half the latent scale and is grossly over-dispersed, and its AUROC is within a
    couple of points of every other arm. A study that reported AUROC alone would call these arms equivalent.
    """
    assert abs(arms["mse"]["auroc"] - arms["censored"]["auroc"]) < 0.03


def test_what_auroc_misses_shows_up_in_the_call_and_the_calibration(arms):
    """Balanced accuracy at ``mu > 0`` and ECE both depend on the scale, and both collapse for the point fit."""
    assert arms["mse"]["bal_acc"] < arms["censored"]["bal_acc"] - 0.05
    assert arms["mse"]["ece"] > arms["censored"]["ece"] + 0.05


def test_the_censored_arm_is_at_least_as_well_calibrated_as_the_binary_one(arms):
    """It carries strictly more information per genome and is scored on the same labels, so it should be."""
    assert arms["censored"]["ece"] <= arms["bce_all"]["ece"] + 0.01

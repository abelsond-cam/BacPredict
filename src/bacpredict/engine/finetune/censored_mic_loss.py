r"""The interval-censored Gaussian (Tobit) likelihood on log2 MIC, and the MIC read-outs that go with it.

**The model.** For one drug, a genome's latent log2 MIC in dilutions from the EUCAST R breakpoint is
``y ~ Normal(mu, sigma^2)``, where ``mu`` is the network's scalar output for that genome and ``sigma`` is one
scalar for the drug — the residual spread of log2 MIC the genome does not explain, in doubling dilutions. The
assay never reveals ``y``; it reveals an interval ``(lo, hi]`` containing it
(:mod:`bacpredict.engine.ast_labels.mic_intervals`). The likelihood of that observation is therefore

.. math::  P(lo < y \le hi) = \Phi\!\left(\frac{hi-\mu}{\sigma}\right) - \Phi\!\left(\frac{lo-\mu}{\sigma}\right)

and the loss is its negative log. Either bound may be infinite: a left-censored ``<= X`` keeps only the
``\Phi(\cdot)`` term, a right-censored ``> X`` only ``1 - \Phi(\cdot)``, and a bare R/S call is the fully
open interval on one side of the breakpoint — which is exactly binary cross-entropy for that genome, so BCE is
the special case of this loss in which every observation is censored at ``0``.

**Resistance falls out of it.** The call is ``y > 0``, so ``P(R) = \Phi(\mu/\sigma)``, which is monotone in
``mu``. ``sigma`` therefore moves **no** ranking metric — AUROC, AUPRC and the R/S call at ``mu > 0`` are
identical for any ``sigma > 0`` — and moves only how confident the probability is. That is what makes the
decision of record safe: ``sigma`` is re-fitted on **validate** at the selected checkpoint
(:func:`fit_sigma`), which is one-parameter temperature scaling, and it cannot flatter the discrimination
numbers because it cannot touch them.

**Numerics are the whole difficulty.** In float32 ``1 - \Phi(z)`` is exactly 0 from ``z = 5.5``, so the naive
``log(\Phi(b) - \Phi(a))`` returns ``-inf`` for any confidently-predicted tail observation — on a 297-case
probe it produced **129 non-finite losses of 297**, which would poison the gradient. :func:`log_interval_prob`
mirrors the upper tail into the lower one and uses ``log_ndtr`` + ``log1mexp`` throughout: 0 non-finite on the
same probe, max relative error 5.5e-4 against a 50-digit mpmath reference. Infinite bounds are substituted
*before* the division by ``sigma``, so no infinity ever reaches an arithmetic op whose gradient would be NaN.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr as _log_ndtr_np
from scipy.special import ndtr as _ndtr_np

#: Standardised bound substituted for an infinite one. ``log_ndtr(-1e4) ~ -5e7``, finite in float32, and
#: ``log1mexp`` of a gap that large is 0 to machine precision — so the open end contributes nothing, which is
#: what an infinite bound means.
BIG = 1.0e4

#: Hard bounds on ``sigma`` (doubling dilutions), for numerical safety only — not a prior. A fitted value at a
#: bound is reported as such rather than silently used.
SIGMA_MIN, SIGMA_MAX = 0.02, 50.0

#: ISO 20776-2 essential agreement is +-1 doubling dilution: the assay's own reproducibility, so a residual SD
#: below it is measuring noise the assay cannot resolve. Reported beside the fitted value, never imposed on it.
ASSAY_REPRODUCIBILITY_DILUTIONS = 1.0


def _log1mexp(x: torch.Tensor) -> torch.Tensor:
    """``log(1 - exp(-x))`` for ``x >= 0``, by the branch that keeps full precision on each side of ``log 2``."""
    x = x.clamp_min(1e-30)
    return torch.where(x < math.log(2.0), torch.log(-torch.expm1(-x)), torch.log1p(-torch.exp(-x)))


def log_interval_prob(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``log(Phi(b) - Phi(a))`` for standardised bounds ``a < b``, accurate in both tails.

    Mirrors ``a > 0`` into ``(-b, -a)`` so the pair is always evaluated in the lower tail, where ``log_ndtr`` is
    exact, then subtracts with :func:`_log1mexp` rather than in probability space.
    """
    flip = a > 0
    a2 = torch.where(flip, -b, a)
    b2 = torch.where(flip, -a, b)
    la, lb = torch.special.log_ndtr(a2), torch.special.log_ndtr(b2)
    return lb + _log1mexp(lb - la)


def _standardise(
    mu: torch.Tensor, sigma: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``((lo-mu)/sigma, (hi-mu)/sigma)`` with infinite bounds replaced by ``-+BIG`` before the division.

    ``torch.where`` evaluates both branches and its backward pass propagates a NaN from the branch it does not
    select, so the infinities are removed by ``nan_to_num`` *first*; the ``where`` then picks the constant, whose
    gradient is correctly zero (an open end says nothing about ``mu``).
    """
    big = torch.full_like(mu, BIG)
    a = torch.where(torch.isneginf(lo), -big, (torch.nan_to_num(lo, neginf=0.0) - mu) / sigma)
    b = torch.where(torch.isposinf(hi), big, (torch.nan_to_num(hi, posinf=0.0) - mu) / sigma)
    return a.clamp(-BIG, BIG), b.clamp(-BIG, BIG)


def interval_censored_gaussian_nll(
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Negative log-likelihood of ``lo < y <= hi`` under ``y ~ N(mu, exp(log_sigma)^2)``.

    Everything is computed in float32 whatever the autocast dtype: ``log_ndtr`` is not an autocast-aware op and
    bf16's 8-bit mantissa cannot carry a tail log-probability.

    Parameters
    ----------
    mu : torch.Tensor
        The network's output, shape ``(N,)`` — the predicted mean log2 MIC in dilutions from the R breakpoint.
    log_sigma : torch.Tensor
        Scalar ``log sigma`` for the drug (one learned parameter).
    lo, hi : torch.Tensor
        The observed interval, shape ``(N,)``; ``-inf`` / ``+inf`` allowed.
    reduction : str
        ``"mean"`` (default, matching the BCE arm), ``"sum"`` or ``"none"``.
    """
    mu = mu.reshape(-1).float()
    lo, hi = lo.reshape(-1).float(), hi.reshape(-1).float()
    sigma = log_sigma.float().clamp(math.log(SIGMA_MIN), math.log(SIGMA_MAX)).exp()
    nll = -log_interval_prob(*_standardise(mu, sigma, lo, hi))
    if reduction == "none":
        return nll
    return nll.sum() if reduction == "sum" else nll.mean()


############################################################## numpy read-outs ##############################################################


def _log_interval_prob_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """:func:`log_interval_prob` in numpy — the same mirror-and-``log1mexp`` scheme."""
    flip = a > 0
    a2, b2 = np.where(flip, -b, a), np.where(flip, -a, b)
    la, lb = _log_ndtr_np(a2), _log_ndtr_np(b2)
    gap = np.maximum(lb - la, 1e-300)
    return lb + np.where(gap < math.log(2.0), np.log(-np.expm1(-gap)), np.log1p(-np.exp(-gap)))


def censored_log_likelihood(mu: np.ndarray, sigma: float, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Per-observation ``log P(lo < y <= hi)`` under ``y ~ N(mu, sigma^2)``, in numpy."""
    mu, lo, hi = (np.asarray(x, dtype=float).ravel() for x in (mu, lo, hi))
    sigma = float(np.clip(sigma, SIGMA_MIN, SIGMA_MAX))
    a = np.clip(np.where(np.isneginf(lo), -BIG, (np.nan_to_num(lo, neginf=0.0) - mu) / sigma), -BIG, BIG)
    b = np.clip(np.where(np.isposinf(hi), BIG, (np.nan_to_num(hi, posinf=0.0) - mu) / sigma), -BIG, BIG)
    return _log_interval_prob_np(a, b)


def fit_sigma(mu: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict:
    """Maximum-likelihood ``sigma`` with ``mu`` held fixed — the validate re-fit (one-parameter temperature scaling).

    ``mu`` comes from the selected checkpoint's predictions on a split the training loss never saw, so this is the
    held-out scale of the model's own residuals. It is a 1-D problem (Brent on ``log sigma``) because every ``mu``
    is fixed, so there is nothing here that can overfit beyond one number.

    Returns
    -------
    dict
        ``sigma``, ``mean_nll`` at it, ``n``, ``at_bound`` and ``sigma_assay_floor``. ``at_bound`` means a
        bound of :data:`SIGMA_MIN`/:data:`SIGMA_MAX` fits as well as the optimum, which happens when every
        interval lies on the correct side of its ``mu`` — the likelihood is then flat and ``sigma`` is **not
        identified**, so the returned number is a bound, not an estimate. ``sigma_assay_floor`` is the same
        value raised to :data:`ASSAY_REPRODUCIBILITY_DILUTIONS`, reported for comparison and not used.
    """
    mu, lo, hi = (np.asarray(x, dtype=float).ravel() for x in (mu, lo, hi))
    ok = np.isfinite(mu) & ~(np.isnan(lo) | np.isnan(hi))
    mu, lo, hi = mu[ok], lo[ok], hi[ok]
    if mu.size == 0:
        return {"sigma": float("nan"), "mean_nll": float("nan"), "n": 0, "at_bound": False,
                "sigma_assay_floor": float("nan")}

    def neg_ll(log_sigma: float) -> float:
        return -float(censored_log_likelihood(mu, math.exp(log_sigma), lo, hi).sum())

    lo_b, hi_b = math.log(SIGMA_MIN), math.log(SIGMA_MAX)
    res = minimize_scalar(neg_ll, bounds=(lo_b, hi_b), method="bounded", options={"xatol": 1e-6})
    # Brent is told a bracket but searches inside it, and the likelihood of perfectly separated open intervals
    # is FLAT as sigma -> 0 (every Phi is 1 to machine precision), so the interior optimum it returns can be an
    # arbitrary point of that plateau. Minimise over the closed interval instead, and say when a bound won:
    # sigma is then not identified by these data and the number must not be read as an estimate.
    candidates = [(res.fun, float(res.x)), (neg_ll(lo_b), lo_b), (neg_ll(hi_b), hi_b)]
    best_nll, best_log_sigma = min(candidates, key=lambda c: c[0])
    at_bound = bool(min(neg_ll(lo_b), neg_ll(hi_b)) <= best_nll + 1e-6 * max(mu.size, 1))
    sigma = float(math.exp(best_log_sigma))
    return {"sigma": sigma, "mean_nll": float(best_nll / mu.size), "n": int(mu.size), "at_bound": at_bound,
            "sigma_assay_floor": float(max(sigma, ASSAY_REPRODUCIBILITY_DILUTIONS))}


def prob_resistant(mu: np.ndarray, sigma: float) -> np.ndarray:
    """``P(R) = Phi(mu / sigma)``: the resistance probability implied by the censored model.

    Monotone in ``mu``, and ``>= 0.5`` exactly when ``mu >= 0`` — so changing ``sigma`` changes the calibration
    of this number and nothing about the ranking or the R/S call it implies.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = float(np.clip(sigma, SIGMA_MIN, SIGMA_MAX))
    return _ndtr_np(mu / sigma)


def predicted_ladder_mic(mu: np.ndarray) -> np.ndarray:
    """The doubling-ladder MIC the model predicts, in dilutions from R: ``ceil(mu)``.

    A reported MIC of ``k`` dilutions means the latent value lies in ``(k-1, k]``, so the ladder point a latent
    ``mu`` would be reported as is the one whose interval contains it. For a Gaussian centred at ``mu`` that is
    also the highest-probability ladder point, so the plug-in and the MAP prediction coincide.
    """
    return np.ceil(np.asarray(mu, dtype=float).ravel())


def mic_agreement(mu: np.ndarray, lo: np.ndarray, hi: np.ndarray, censor: np.ndarray) -> dict:
    """MIC read-outs: interval coverage over every cell, and agreement over the uncensored ones.

    ``interval_coverage`` is the one figure every cell can contribute to — the share whose observed interval
    contains ``mu`` — because a censored observation has no MIC to agree with. The two agreement rates are
    therefore computed on ``censor == "exact"`` cells only, whose reported MIC is a known ladder point ``k``
    (their interval is ``(k-1, k]``, so ``k`` is the upper bound):

    * ``absolute_agreement`` — the predicted ladder point equals the reported one;
    * ``essential_agreement`` — it is within one doubling dilution (ISO 20776-2). Reported because it is the
      field's convention and the assay's own reproducibility ceiling, **not** as the headline.

    ``bias_log2`` and ``rmse_log2`` compare ``mu`` with the midpoint of the observed interval, ``k - 0.5``,
    which is the unbiased point summary of "somewhere in ``(k-1, k]``".
    """
    mu, lo, hi = (np.asarray(x, dtype=float).ravel() for x in (mu, lo, hi))
    censor = np.asarray(censor, dtype=object).ravel()
    nan = float("nan")
    out: dict[str, float] = {"n": int(mu.size)}
    if mu.size == 0:
        return {**out, "interval_coverage": nan, "n_exact": 0, "absolute_agreement": nan,
                "essential_agreement": nan, "bias_log2": nan, "rmse_log2": nan}

    out["interval_coverage"] = float(np.mean((mu > lo) & (mu <= hi)))
    exact = censor == "exact"
    out["n_exact"] = int(exact.sum())
    if not exact.any():
        return {**out, "absolute_agreement": nan, "essential_agreement": nan, "bias_log2": nan, "rmse_log2": nan}
    k = hi[exact]
    k_hat = predicted_ladder_mic(mu[exact])
    resid = mu[exact] - (k - 0.5)
    out["absolute_agreement"] = float(np.mean(k_hat == k))
    out["essential_agreement"] = float(np.mean(np.abs(k_hat - k) <= 1.0))
    out["bias_log2"] = float(np.mean(resid))
    out["rmse_log2"] = float(np.sqrt(np.mean(resid ** 2)))
    return out


def calibration_scores(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """Brier score and expected calibration error (equal-width bins) — the two numbers ``sigma`` moves.

    ECE is ``sum_b (n_b / n) |accuracy_b - confidence_b|`` over ``n_bins`` equal-width probability bins, the
    standard definition used to report temperature scaling.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if y_true.size == 0:
        return {"brier": float("nan"), "ece": float("nan"), "n_bins": int(n_bins)}
    brier = float(np.mean((y_prob - y_true) ** 2))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += (m.sum() / y_true.size) * abs(y_true[m].mean() - y_prob[m].mean())
    return {"brier": brier, "ece": float(ece), "n_bins": int(n_bins)}

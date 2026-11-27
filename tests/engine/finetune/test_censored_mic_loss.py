"""The interval-censored Gaussian likelihood: its tails, its special cases, and the validate sigma re-fit."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from bacpredict.engine.finetune.censored_mic_loss import (
    SIGMA_MAX,
    SIGMA_MIN,
    calibration_scores,
    censored_log_likelihood,
    fit_sigma,
    interval_censored_gaussian_nll,
    log_interval_prob,
    mic_agreement,
    predicted_ladder_mic,
    prob_resistant,
)

INF = float("inf")


# ------------------------------------------------------------------ numerics


def test_tails_stay_finite_where_the_naive_difference_does_not():
    """``log(Phi(b) - Phi(a))`` by subtraction is -inf past z ~ 5.5 in float32; this formulation is not."""
    grid = torch.tensor([-40.0, -10.0, -6.0, -1.0, 0.0, 1.0, 6.0, 10.0, 40.0])
    a, b = torch.meshgrid(grid, grid, indexing="ij")
    keep = b > a
    got = log_interval_prob(a[keep], b[keep])
    naive = (torch.special.ndtr(b[keep]) - torch.special.ndtr(a[keep])).log()
    assert torch.isfinite(got).all()
    assert not torch.isfinite(naive).all()   # the failure this exists to avoid


def test_it_agrees_with_the_naive_form_where_the_naive_form_works():
    """In the body of the distribution the two must be the same number."""
    a = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5])
    b = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0])
    naive = (torch.special.ndtr(b) - torch.special.ndtr(a)).log()
    assert torch.allclose(log_interval_prob(a, b), naive, atol=1e-5)


def test_gradients_are_finite_including_at_infinite_bounds():
    """A ``torch.where`` propagates a NaN from the branch it does not take: the infinities go first."""
    mu = torch.tensor([-30.0, -2.0, 0.0, 2.0, 30.0], requires_grad=True)
    log_sigma = torch.tensor(math.log(0.8), requires_grad=True)
    lo = torch.tensor([-INF, -1.0, 0.0, 2.0, 3.0])
    hi = torch.tensor([0.0, 0.0, INF, 3.0, INF])
    interval_censored_gaussian_nll(mu, log_sigma, lo, hi, reduction="sum").backward()
    assert torch.isfinite(mu.grad).all() and torch.isfinite(log_sigma.grad).all()


def test_bf16_inputs_are_computed_in_float32():
    """bf16's 8-bit mantissa cannot carry a tail log-probability, and ``log_ndtr`` is not autocast-aware."""
    mu = torch.tensor([3.0, -3.0], dtype=torch.bfloat16)
    out = interval_censored_gaussian_nll(mu, torch.tensor(0.0, dtype=torch.bfloat16),
                                         torch.tensor([-INF, 0.0]), torch.tensor([0.0, INF]))
    assert out.dtype == torch.float32 and torch.isfinite(out)


# ------------------------------------------------------------------ the special cases


def test_one_sided_censoring_collapses_to_the_one_sided_form():
    """``(-inf, 0]`` must be exactly ``-log Phi(-mu/sigma)`` and ``(0, inf)`` exactly ``-log Phi(mu/sigma)``."""
    mu = torch.tensor([-3.0, 0.0, 2.0, 7.0])
    sigma = 1.4
    zeros, inf = torch.zeros_like(mu), torch.full_like(mu, INF)
    right = interval_censored_gaussian_nll(mu, torch.tensor(math.log(sigma)), zeros, inf, reduction="none")
    left = interval_censored_gaussian_nll(mu, torch.tensor(math.log(sigma)), -inf, zeros, reduction="none")
    assert torch.allclose(right, -torch.special.log_ndtr(mu / sigma), atol=1e-6)
    assert torch.allclose(left, -torch.special.log_ndtr(-mu / sigma), atol=1e-6)


def test_binary_cross_entropy_is_the_all_censored_special_case():
    """Every observation censored at the breakpoint, sigma = 1: the loss *is* probit BCE, to the last bit.

    This is the structural claim the whole comparison rests on — the censored arm does not replace the binary
    arm's objective with a different one, it generalises it — so it is checked as an identity, not an
    approximation.
    """
    g = torch.Generator().manual_seed(0)
    mu = torch.randn(2000, generator=g) * 3
    y = (torch.rand(2000, generator=g) < torch.special.ndtr(mu)).float()
    lo = torch.where(y > 0, torch.zeros_like(mu), torch.full_like(mu, -INF))
    hi = torch.where(y > 0, torch.full_like(mu, INF), torch.zeros_like(mu))
    censored = interval_censored_gaussian_nll(mu, torch.tensor(0.0), lo, hi)
    probit_bce = -(y * torch.special.log_ndtr(mu) + (1 - y) * torch.special.log_ndtr(-mu)).mean()
    assert censored.item() == pytest.approx(probit_bce.item(), abs=1e-6)


# ------------------------------------------------------------------ sigma


def test_fit_sigma_recovers_a_known_sigma_from_censored_data():
    """With ``mu`` fixed, the 1-D MLE is the held-out scale of the residuals — and it is unbiased."""
    rng = np.random.default_rng(7)
    for true_sigma in (0.6, 1.5, 3.0):
        mu = rng.normal(0, 2.0, 20000)
        k = np.ceil(mu + rng.normal(0, true_sigma, 20000))      # the ladder point the assay reports
        lo, hi = k - 1.0, k.copy()
        lo[k > 4], hi[k > 4] = 4.0, INF                          # censored above +4 dilutions
        lo[k <= -4], hi[k <= -4] = -INF, -4.0                    # and below -4
        assert fit_sigma(mu, lo, hi)["sigma"] == pytest.approx(true_sigma, rel=0.05)


def test_fit_sigma_flags_an_unidentified_sigma_rather_than_passing_it_off_as_an_estimate():
    """Open intervals every one of which already contains its ``mu`` make the likelihood flat as sigma -> 0.

    There is then no maximum to find — any small sigma fits identically — so what matters is that the result
    says so. A number read out of a flat likelihood and reported as an estimate is the failure mode here.
    """
    mu = np.array([-5.0, -4.0, 4.0, 5.0])
    out = fit_sigma(mu, np.array([-INF, -INF, 0.0, 0.0]), np.array([0.0, 0.0, INF, INF]))
    assert out["at_bound"]
    assert out["mean_nll"] == pytest.approx(0.0, abs=1e-9)        # the plateau: a perfect fit at any sigma
    assert SIGMA_MIN <= out["sigma"] <= SIGMA_MAX


def test_sigma_moves_calibration_and_cannot_move_ranking():
    """The guarantee that makes the validate re-fit safe: ``Phi(mu/sigma)`` is monotone in ``mu``."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(3)
    mu = rng.normal(0, 2, 4000)
    y = (mu + rng.normal(0, 1.5, 4000) > 0).astype(int)
    aurocs = [roc_auc_score(y, prob_resistant(mu, s)) for s in (1.0, 1.5, 5.0)]
    calls = [((prob_resistant(mu, s) >= 0.5).astype(int) == (mu >= 0).astype(int)).all() for s in (0.3, 1.0, 5.0)]
    assert aurocs[0] == pytest.approx(aurocs[1]) == pytest.approx(aurocs[2])
    assert all(calls)
    eces = {s: calibration_scores(y, prob_resistant(mu, s))["ece"] for s in (0.3, 1.5, 5.0)}
    assert eces[1.5] < eces[0.3] and eces[1.5] < eces[5.0]   # and the MLE is near the truth


def test_censored_log_likelihood_matches_the_torch_loss():
    """The numpy read-out path and the torch training path must not be two different models."""
    mu = np.array([-2.0, 0.3, 4.0])
    lo = np.array([-INF, 0.0, 3.0])
    hi = np.array([-1.0, 1.0, INF])
    np_nll = -censored_log_likelihood(mu, 1.3, lo, hi)
    torch_nll = interval_censored_gaussian_nll(
        torch.tensor(mu), torch.tensor(math.log(1.3)), torch.tensor(lo), torch.tensor(hi), reduction="none")
    assert np.allclose(np_nll, torch_nll.numpy(), atol=1e-5)


# ------------------------------------------------------------------ the MIC read-outs


def test_the_predicted_ladder_point_is_the_one_whose_interval_contains_mu():
    """A reported MIC of ``k`` means the latent lies in ``(k-1, k]``, so the prediction is ``ceil(mu)``."""
    assert list(predicted_ladder_mic([-1.2, 0.3, 2.7, 3.0])) == [-1.0, 1.0, 3.0, 3.0]


def test_agreement_rates_use_only_the_uncensored_cells():
    """A censored observation has no MIC to agree with, so it contributes to coverage and nothing else."""
    mu = np.array([0.4, 3.4, 9.0])
    lo = np.array([0.0, 1.0, 3.0])
    hi = np.array([1.0, 2.0, INF])
    out = mic_agreement(mu, lo, hi, np.array(["exact", "exact", "right"]))
    assert out["n"] == 3 and out["n_exact"] == 2
    assert out["interval_coverage"] == pytest.approx(2 / 3)      # the right-censored one is covered
    assert out["absolute_agreement"] == pytest.approx(0.5)       # ceil(0.4)=1 hits, ceil(3.4)=4 misses 2
    assert out["essential_agreement"] == pytest.approx(0.5)      # |4-2| = 2 is outside +-1


def test_the_slope_catches_an_attenuated_fit_that_the_correlation_misses():
    """Why read-out (b) reports a slope and not just r: a shrunk prediction keeps r and loses the scale.

    This is arm A's measured failure mode -- it recovers 0.584 of the latent scale on known-truth simulation
    while its AUROC stays within 0.005 of the censored fit's. A correlation alone would call it fine.
    """
    rng = np.random.default_rng(0)
    k = rng.integers(-4, 5, 400).astype(float)          # reported ladder points
    lo, hi = k - 1.0, k
    mid = k - 0.5
    censor = np.array(["exact"] * k.size, dtype=object)

    honest = mic_agreement(mid.copy(), lo, hi, censor)
    assert honest["pearson_r"] == pytest.approx(1.0)
    assert honest["slope_log2"] == pytest.approx(1.0)

    shrunk = mic_agreement(0.5 * mid, lo, hi, censor)   # half the scale, same ordering
    assert shrunk["pearson_r"] == pytest.approx(1.0)    # r is blind to it
    assert shrunk["slope_log2"] == pytest.approx(0.5)   # the slope is not

    noisy = mic_agreement(mid + rng.normal(0, 0.4, k.size), lo, hi, censor)
    assert 0.9 < noisy["slope_log2"] < 1.1 and noisy["pearson_r"] < 1.0


def test_mic_scale_statistics_are_nan_without_spread_to_regress_on():
    """One exact cell, or a cohort whose exact MICs are all the same point, has no slope -- not a slope of 0."""
    one = mic_agreement(np.array([0.5, 9.0]), np.array([0.0, 3.0]), np.array([1.0, INF]),
                        np.array(["exact", "right"], dtype=object))
    assert one["n_exact"] == 1 and math.isnan(one["slope_log2"]) and math.isnan(one["pearson_r"])
    flat = mic_agreement(np.array([0.4, 0.6]), np.array([0.0, 0.0]), np.array([1.0, 1.0]),
                         np.array(["exact", "exact"], dtype=object))
    assert flat["n_exact"] == 2 and math.isnan(flat["slope_log2"])


def test_a_holdout_with_no_uncensored_cell_reports_nan_not_zero():
    """An agreement rate over nothing is unknown, which is not the same number as nought."""
    out = mic_agreement(np.array([1.0]), np.array([0.0]), np.array([INF]), np.array(["right"]))
    assert out["n_exact"] == 0 and math.isnan(out["absolute_agreement"])


# ------------------------------------------------------------------ the binary arms' temperature


def test_the_temperature_fit_recovers_an_injected_over_confidence():
    """A binary arm has no sigma, so this is what makes its calibration comparable with a MIC arm's."""
    from bacpredict.engine.finetune.censored_mic_loss import fit_logit_temperature, prob_resistant_tempered
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1.5, 20000)
    y = (rng.random(20000) < 1 / (1 + np.exp(-z))).astype(int)   # calibrated at T = 1 by construction
    for injected in (0.4, 1.0, 2.5):
        fit = fit_logit_temperature(z * injected, y)
        assert fit["temperature"] == pytest.approx(injected, rel=0.05)
        before = calibration_scores(y, 1 / (1 + np.exp(-z * injected)))["ece"]
        after = calibration_scores(y, prob_resistant_tempered(z * injected, fit["temperature"]))["ece"]
        assert after <= before + 1e-9


def test_the_temperature_cannot_move_ranking_or_the_call():
    """Same guarantee as sigma's: sigmoid(z/T) is monotone in z and crosses 0.5 where z does."""
    from sklearn.metrics import roc_auc_score

    from bacpredict.engine.finetune.censored_mic_loss import prob_resistant_tempered
    rng = np.random.default_rng(1)
    z = rng.normal(0, 2, 3000)
    y = (rng.random(3000) < 1 / (1 + np.exp(-z))).astype(int)
    aurocs = [roc_auc_score(y, prob_resistant_tempered(z, t)) for t in (0.5, 1.0, 3.0)]
    assert aurocs[0] == pytest.approx(aurocs[1]) == pytest.approx(aurocs[2])
    for t in (0.5, 1.0, 3.0):
        assert np.array_equal((prob_resistant_tempered(z, t) >= 0.5).astype(int), (z >= 0).astype(int))


def test_a_single_class_validate_split_has_no_temperature_to_fit():
    """Returning 1.0 there would assert a calibration nobody measured."""
    from bacpredict.engine.finetune.censored_mic_loss import fit_logit_temperature
    out = fit_logit_temperature(np.array([1.0, 2.0, 3.0]), np.array([1, 1, 1]))
    assert math.isnan(out["temperature"])

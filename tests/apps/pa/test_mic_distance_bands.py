"""Banding MICs by distance from the breakpoint that decided the label."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from bacpredict.apps.pa.mic_distance_bands import (
    NO_MIC,
    UNDETERMINED,
    aggregate_to_genome,
    band_of,
    band_of_interval,
    band_tests,
    eucast_zone_of,
    interval_of,
    summarise,
)

# the real PsA rows: 7 drugs have no S category (s_le 0.001), meropenem a genuine band, amikacin none
BREAKPOINTS = pd.DataFrame([
    {"taxon_rank": "genus", "taxon": "Pseudomonas", "drug": "cefepime", "s_le_mg_l": 0.001, "r_gt_mg_l": 8.0},
    {"taxon_rank": "genus", "taxon": "Pseudomonas", "drug": "meropenem", "s_le_mg_l": 2.0, "r_gt_mg_l": 8.0},
    {"taxon_rank": "genus", "taxon": "Pseudomonas", "drug": "amikacin", "s_le_mg_l": 16.0, "r_gt_mg_l": 16.0},
])


def _test(sample: str, drug: str, sign: str = "=", value: float = float("nan"), *,
          upper: float = float("nan"), basis: str = "mic", r_gt: float = 8.0) -> dict:
    return {"Sample": sample, "species": "Pseudomonas aeruginosa", "drug": drug, "mic_sign": sign,
            "mic_value": value, "mic_upper": upper, "basis": basis, "r_gt_mg_l": r_gt}


class TestBandOf:
    """The partition is asymmetric about zero because the rule is *resistant iff MIC > R*."""

    @pytest.mark.parametrize(("d", "expected"), [
        (0.0, "near_S"),      # MIC exactly at R: susceptible, and one dilution from flipping
        (1.0, "near_R"),      # one dilution above R
        (0.5, "near_R"),
        (-0.999, "near_S"),
        (-1.0, "mid_S"),      # the open end belongs on the susceptible side
        (1.001, "mid_R"),
        (3.0, "mid_R"),
        (-3.0, "mid_S"),
        (3.001, "far_R"),
        (-3.001, "far_S"),
        (float("inf"), "far_R"),
        (float("-inf"), "far_S"),
    ])
    def test_boundaries(self, d: float, expected: str) -> None:
        assert band_of(d)[1] == expected

    def test_unsigned_is_the_signed_prefix(self) -> None:
        for d in (-5.0, -2.0, 0.0, 0.5, 2.0, 5.0):
            band, signed = band_of(d)
            assert signed.startswith(band)


class TestInterval:
    """⛔ A censored MIC is an interval. Banding it as its own bound would assert a distance nobody measured."""

    def test_exact_is_a_point(self) -> None:
        assert interval_of("=", 16.0, float("nan"), 8.0) == (1.0, 1.0)

    @pytest.mark.parametrize("sign", ["<=", "<"])
    def test_upper_bound_only(self, sign: str) -> None:
        lo, hi = interval_of(sign, 2.0, float("nan"), 8.0)
        assert lo == float("-inf") and hi == pytest.approx(-2.0)

    @pytest.mark.parametrize("sign", [">", ">="])
    def test_lower_bound_only(self, sign: str) -> None:
        lo, hi = interval_of(sign, 32.0, float("nan"), 8.0)
        assert lo == pytest.approx(2.0) and hi == float("inf")

    def test_both_bounds(self) -> None:
        lo, hi = interval_of(">,<", 8.0, 32.0, 8.0)
        assert lo == pytest.approx(0.0) and hi == pytest.approx(2.0)

    def test_no_usable_mic(self) -> None:
        lo, hi = interval_of("=", float("nan"), float("nan"), 8.0)
        assert math.isnan(lo) and math.isnan(hi)

    def test_a_straddling_censored_mic_is_not_banded(self) -> None:
        """``<= 2`` against R = 8 is consistent with anything from far_S to mid_S, so it gets no band."""
        assert band_of_interval(*interval_of("<=", 2.0, float("nan"), 8.0)) == (UNDETERMINED, UNDETERMINED)

    def test_a_decisive_censored_mic_is_banded(self) -> None:
        """``<= 0.5`` against R = 8 is d <= -4: the whole interval is far_S."""
        assert band_of_interval(*interval_of("<=", 0.5, float("nan"), 8.0)) == ("far", "far_S")

    def test_off_grid_etest_value(self) -> None:
        """Etest reports 0.75, 1.5, 3, 6 — not on the doubling grid, so bands must be continuous intervals."""
        band, signed = band_of_interval(*interval_of("=", 0.75, float("nan"), 0.5))
        assert signed == "near_R"  # log2(1.5) = 0.585


class TestEucastZone:
    """The zone is descriptive only — for 7 of 10 PsA drugs it is a tautology for the negative class."""

    def test_no_s_category_drug_calls_the_wild_type_intermediate(self) -> None:
        assert eucast_zone_of(*interval_of("=", 0.5, float("nan"), 8.0), 0.001, 8.0) == "I"
        assert eucast_zone_of(*interval_of("=", 8.0, float("nan"), 8.0), 0.001, 8.0) == "I"
        assert eucast_zone_of(*interval_of("=", 16.0, float("nan"), 8.0), 0.001, 8.0) == "R"

    def test_meropenem_has_all_three(self) -> None:
        assert eucast_zone_of(*interval_of("=", 1.0, float("nan"), 8.0), 2.0, 8.0) == "S"
        assert eucast_zone_of(*interval_of("=", 4.0, float("nan"), 8.0), 2.0, 8.0) == "I"
        assert eucast_zone_of(*interval_of("=", 16.0, float("nan"), 8.0), 2.0, 8.0) == "R"

    def test_a_drug_with_s_le_equal_to_r_gt_never_returns_intermediate(self) -> None:
        for mic in (1.0, 8.0, 16.0, 32.0):
            assert eucast_zone_of(*interval_of("=", mic, float("nan"), 16.0), 16.0, 16.0) in {"S", "R"}


class TestAggregation:
    def test_no_mic_is_its_own_stratum_not_far(self) -> None:
        """⛔ Folding a source-call label into ``far`` would assert a distance nobody measured."""
        t = band_tests(pd.DataFrame([_test("g1", "cefepime", basis="source_call")]), BREAKPOINTS)
        out = aggregate_to_genome(t)
        assert out.loc[0, "band"] == NO_MIC
        assert math.isnan(out.loc[0, "d_min_abs"])

    def test_a_straddling_censored_only_genome_is_undetermined_not_no_mic(self) -> None:
        """It HAD an MIC; we just cannot place it. That is a different thing from having none."""
        t = band_tests(pd.DataFrame([_test("g1", "cefepime", "<=", 2.0)]), BREAKPOINTS)
        out = aggregate_to_genome(t)
        assert out.loc[0, "band"] == UNDETERMINED
        assert out.loc[0, "n_mic_tests"] == 1

    def test_min_abs_d_wins_and_disagreement_is_flagged(self) -> None:
        """If ANY test sat at the cut the label was contestable, so min |d| is the primary rule."""
        t = band_tests(pd.DataFrame([
            _test("g1", "cefepime", "=", 0.125),  # d = -6, far_S
            _test("g1", "cefepime", "=", 8.0),    # d =  0, near_S
        ]), BREAKPOINTS)
        out = aggregate_to_genome(t)
        assert out.loc[0, "band_signed"] == "near_S"
        assert out.loc[0, "d_min_abs"] == pytest.approx(0.0)
        assert out.loc[0, "d_median"] == pytest.approx(-3.0)
        assert bool(out.loc[0, "band_disagreement"]) is True

    def test_agreeing_tests_are_not_flagged(self) -> None:
        t = band_tests(pd.DataFrame([
            _test("g1", "cefepime", "=", 8.0),
            _test("g1", "cefepime", "=", 16.0),
        ]), BREAKPOINTS)
        out = aggregate_to_genome(t)
        assert bool(out.loc[0, "band_disagreement"]) is True  # near_S vs near_R differ in SIDE
        t2 = band_tests(pd.DataFrame([
            _test("g1", "cefepime", "=", 16.0),
            _test("g1", "cefepime", "=", 16.0),
        ]), BREAKPOINTS)
        assert bool(aggregate_to_genome(t2).loc[0, "band_disagreement"]) is False

    def test_legacy_definition_is_reproduced(self) -> None:
        """The superseded prose rule (an ``=`` MIC with R/2 <= MIC <= 2R), kept as a check against the record."""
        t = band_tests(pd.DataFrame([
            _test("g1", "cefepime", "=", 4.0),    # R/2, near_legacy
            _test("g2", "cefepime", "=", 2.0),    # below R/2
            _test("g3", "cefepime", ">", 4.0),    # censored: legacy needs an exact MIC
        ]), BREAKPOINTS)
        out = aggregate_to_genome(t).set_index("Sample")
        assert bool(out.loc["g1", "near_legacy"]) is True
        assert bool(out.loc["g2", "near_legacy"]) is False
        assert bool(out.loc["g3", "near_legacy"]) is False

    def test_summary_rows_account_for_every_genome(self) -> None:
        t = band_tests(pd.DataFrame([
            _test("g1", "cefepime", "=", 8.0),
            _test("g2", "cefepime", "=", 0.125),
            _test("g3", "cefepime", "<=", 2.0),
            _test("g4", "cefepime", basis="source_call"),
        ]), BREAKPOINTS)
        s = summarise(aggregate_to_genome(t)).iloc[0]
        assert s["n"] == 4
        assert s["n_near"] + s["n_mid"] + s["n_far"] + s["n_no_mic"] + s["n_undetermined"] == 4

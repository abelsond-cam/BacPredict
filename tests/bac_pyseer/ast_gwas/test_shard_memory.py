"""The memory estimate must reproduce its calibration, and must fail loudly where it has none.

An estimator that quietly returns a plausible number for a cohort size it has never seen is worse
than no estimator, because a job then passes its own check and OOMs four hours later. These tests pin
both halves: the fit where data exists, and the honest width of the bracket where it does not.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bac_pyseer.ast_gwas.shard_memory import (
    ANCHOR_CPU,
    ANCHOR_N,
    ANCHOR_SWEEP,
    MB_PER_CORE,
    Observation,
    anchor_peak_gb,
    check_allocation,
    cores_for_mem,
    estimate_shard_peak,
    load_observations,
    main,
    record_observation,
)


@pytest.mark.parametrize(("unitigs", "measured"), ANCHOR_SWEEP)
def test_reproduces_the_invasion_calibration(unitigs: int, measured: float) -> None:
    """Within 5% of all three measured points — including the middle one, which is not fitted.

    The fit uses only the 10k and 100k ends, so 50k is a genuine held-out check of the logarithmic
    form rather than a restatement of the inputs.
    """
    assert anchor_peak_gb(unitigs) == pytest.approx(measured, rel=0.05)


def test_the_bracket_collapses_to_a_point_at_the_anchor() -> None:
    """Where the measurement was taken there is nothing to bracket."""
    low, high = estimate_shard_peak(ANCHOR_N, ANCHOR_CPU, 100_000)
    assert low == pytest.approx(high)
    assert low == pytest.approx(26.0, rel=0.05)


def test_the_bracket_widens_with_distance_from_the_anchor() -> None:
    """Extrapolating 2x in cohort size must visibly widen the honest range, not narrow it."""
    near_lo, near_hi = estimate_shard_peak(int(ANCHOR_N * 1.1), ANCHOR_CPU, 100_000)
    far_lo, far_hi = estimate_shard_peak(int(ANCHOR_N * 2.1), ANCHOR_CPU, 100_000)
    assert (far_hi - far_lo) > (near_hi - near_lo)
    assert far_lo > near_lo and far_hi > near_hi


def test_tb_rifampin_clears_the_default_bound_but_not_its_safety_margin() -> None:
    """The finding that motivated the gate, stated precisely.

    128 GiB does clear the estimated upper bound -- but by 1.12x, where the invasion run had ~5x, and
    that bound is extrapolated from a single cohort. "Nobody can currently say" is the honest verdict,
    and it is the one the gate must return.
    """
    low, high = estimate_shard_peak(28_508, 8, 100_000)
    assert low < high < 128, f"expected the bracket below 128 GiB, got {low:.0f}-{high:.0f}"
    assert 128 / high < 1.3, "margin over the upper bound is thin -- that is the point"
    assert check_allocation(128, 28_508, 8, 100_000)[0] is False


def test_more_workers_can_only_raise_the_upper_bound() -> None:
    """cpu is unresolved -- shared rotation means it does not matter, private means it does."""
    lo4, hi4 = estimate_shard_peak(28_508, 4, 100_000)
    lo8, hi8 = estimate_shard_peak(28_508, 8, 100_000)
    assert lo4 == pytest.approx(lo8)
    assert hi8 > hi4


def test_a_bigger_shard_costs_more_but_sub_linearly() -> None:
    """10x the unitigs must not cost 10x the memory, or sharding would buy nothing."""
    small, big = anchor_peak_gb(10_000), anchor_peak_gb(100_000)
    assert big > small
    assert big < 10 * small


def test_observations_collapse_the_bracket() -> None:
    """Two cohorts of measurement replace the guess with a fitted exponent."""
    wide_lo, wide_hi = estimate_shard_peak(28_508, 8, 100_000)
    obs = [
        Observation(n=7_172, cpu=8, unitigs_per_shard=100_000, max_rss_gb=14.0),
        Observation(n=13_976, cpu=8, unitigs_per_shard=100_000, max_rss_gb=27.0),
    ]
    tight_lo, tight_hi = estimate_shard_peak(28_508, 8, 100_000, obs)
    assert (tight_hi - tight_lo) < (wide_hi - wide_lo)
    assert tight_lo > 0


def test_one_observation_is_not_enough_to_fit_a_slope() -> None:
    """A single point cannot determine an exponent; the bracket must stay honest."""
    one = [Observation(n=7_172, cpu=8, unitigs_per_shard=100_000, max_rss_gb=14.0)]
    assert estimate_shard_peak(28_508, 8, 100_000, one) == estimate_shard_peak(28_508, 8, 100_000)


def test_observations_all_at_one_cohort_size_cannot_fit_a_slope() -> None:
    """Repeating a measurement is not the same as varying n -- this is the anchor's whole problem."""
    same_n = [
        Observation(n=7_172, cpu=8, unitigs_per_shard=50_000, max_rss_gb=11.0),
        Observation(n=7_172, cpu=8, unitigs_per_shard=100_000, max_rss_gb=14.0),
    ]
    assert estimate_shard_peak(28_508, 8, 100_000, same_n) == estimate_shard_peak(28_508, 8, 100_000)


def test_check_allocation_refuses_an_under_provisioned_job() -> None:
    ok, message = check_allocation(48, 28_508, 8, 100_000)
    assert not ok
    assert "INSUFFICIENT" in message
    assert "UNMEASURED" in message


def test_check_allocation_accepts_a_generous_one() -> None:
    ok, message = check_allocation(256, 28_508, 8, 100_000)
    assert ok
    assert "[OK]" in message


def test_check_allocation_sizes_against_the_upper_bound() -> None:
    """Sizing to the optimistic end is how a job passes its own check and dies anyway."""
    low, high = estimate_shard_peak(28_508, 8, 100_000)
    assert check_allocation(low * 1.3, 28_508, 8, 100_000)[0] is False
    assert check_allocation(high * 1.3, 28_508, 8, 100_000)[0] is True


def test_cores_for_mem_exposes_the_csd3_inflation() -> None:
    """128G at cpus-per-task=8 does not get 8 cores -- it gets 20, and bills for 20.

    The G suffix is GiB. Reading it as decimal GB gives 19 and understates the reservation.
    """
    assert cores_for_mem(128) == 20
    assert cores_for_mem(ANCHOR_CPU * MB_PER_CORE / 1024) == ANCHOR_CPU
    assert cores_for_mem(128 * 1000 / 1024) == 19, "the decimal misreading, pinned so it stays wrong"


def test_the_message_names_the_allocated_cores_not_the_requested_ones() -> None:
    """The reservation understatement was what held every Kp array in AssocGrpCPUMinutesLimit."""
    assert "allocates 20 cores" in check_allocation(128, 28_508, 8, 100_000)[1]


def test_observations_round_trip_through_the_calibration_file(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    assert load_observations(path) == []
    record_observation(path, Observation(7_172, 8, 100_000, 14.0, source="job 1"))
    held = record_observation(path, Observation(13_976, 8, 100_000, 27.0, source="job 2"))
    assert len(held) == 2
    assert [o.n for o in load_observations(path)] == [7_172, 13_976]
    assert json.loads(path.read_text())[0]["source"] == "job 1"


def test_a_missing_calibration_file_is_empty_not_an_error(tmp_path: Path) -> None:
    assert load_observations(tmp_path / "nope.json") == []
    assert load_observations(None) == []


def test_cli_exits_nonzero_when_the_allocation_is_short(capsys) -> None:
    with pytest.raises(SystemExit) as e:
        main(["--n", "28508", "--cpu", "8", "--unitigs-per-shard", "100000", "--mem-gb", "48"])
    assert e.value.code == 1
    assert "INSUFFICIENT" in capsys.readouterr().err


def test_cli_exits_zero_when_it_fits(capsys) -> None:
    with pytest.raises(SystemExit) as e:
        main(["--n", "28508", "--cpu", "8", "--unitigs-per-shard", "100000", "--mem-gb", "256"])
    assert e.value.code == 0
    assert "[OK]" in capsys.readouterr().err


def test_cli_reports_the_bracket_when_no_allocation_is_given(capsys) -> None:
    """Two numbers on stdout, so a shell can read them without parsing prose."""
    main(["--n", "28508", "--cpu", "8", "--unitigs-per-shard", "100000"])
    low, high = (float(x) for x in capsys.readouterr().out.split())
    assert low < high


def test_cli_records_a_measurement(tmp_path: Path, capsys) -> None:
    path = tmp_path / "cal.json"
    main(["--n", "7172", "--cpu", "8", "--unitigs-per-shard", "100000",
          "--record-max-rss-gb", "14.0", "--calibration", str(path), "--source", "job 42"])
    assert "recorded 14.0 GB" in capsys.readouterr().out
    assert load_observations(path)[0].source == "job 42"


def test_recording_without_a_calibration_path_is_refused() -> None:
    with pytest.raises(SystemExit):
        main(["--n", "7172", "--unitigs-per-shard", "100000", "--record-max-rss-gb", "14.0"])


@pytest.mark.parametrize(("n", "cpu", "unitigs"), [(0, 8, 1000), (100, 0, 1000), (100, 8, 0)])
def test_nonsense_inputs_raise(n: int, cpu: int, unitigs: int) -> None:
    with pytest.raises(ValueError):
        estimate_shard_peak(n, cpu, unitigs)

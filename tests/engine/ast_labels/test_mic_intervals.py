"""The censored-MIC interval table: the sign table, the ladder snap, the collapse, and the split-table check."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bacpredict.engine.ast_labels.mic_intervals import (
    check_against_split_table,
    collapse_to_cells,
    snap_to_ladder,
    intervals_from_reported_mic,
    ast_tests_to_intervals,
)

INF = float("inf")


def _tests(rows: list[dict]) -> pd.DataFrame:
    """A minimal ``ast_tests_called`` frame: the columns ``ast_tests_to_intervals`` reads."""
    base = {"species": "paeruginosa", "drug": "meropenem", "mic_sign": "=", "mic_value": np.nan,
            "mic_upper": np.nan, "basis": "mic", "y_test": np.nan, "r_gt_mg_l": 8.0}
    return pd.DataFrame([{**base, **r} for r in rows])


# ------------------------------------------------------------------ the sign table


@pytest.mark.parametrize(
    ("sign", "value", "upper", "lo", "hi", "point"),
    [
        ("=", 4.0, np.nan, 1.0, 2.0, 2.0),          # (next doubling below X, X]; point X
        (">", 8.0, np.nan, 3.0, INF, 4.0),          # (X, inf); point one dilution up
        (">=", 16.0, np.nan, 3.0, INF, 4.0),        # (next doubling below X, inf); point X
        ("<=", 0.5, np.nan, -INF, -1.0, -1.0),      # (-inf, X]; point X
        ("<", 0.5, np.nan, -INF, -1.0, -2.0),       # (-inf, X]; point one dilution down
        (">,<", 2.0, 16.0, 1.0, 4.0, 2.5),          # (X, U]; point the geometric mean
    ],
)
def test_sign_table_is_the_agreed_one(sign, value, upper, lo, hi, point):
    """Each reported form maps to the interval and the imputed point agreed with David."""
    got_lo, got_hi, got_point = intervals_from_reported_mic(
        pd.Series([sign]), pd.Series([value]), pd.Series([upper]))
    assert (got_lo[0], got_hi[0]) == (lo, hi)
    assert got_point[0] == pytest.approx(point)


def test_exact_mic_is_an_interval_one_dilution_wide():
    """``= X`` is never a point: the assay saw growth at X/2 and none at X."""
    lo, hi, _ = intervals_from_reported_mic(pd.Series(["="] * 3), pd.Series([0.25, 4.0, 64.0]), pd.Series([np.nan] * 3))
    assert list(hi - lo) == [1.0, 1.0, 1.0]


# ------------------------------------------------------------------ the ladder snap


def test_snap_fixes_the_16_001_artefact():
    """``16.001`` is 16: unsnapped its log2 is 4.00009, which lands it above an R = 16 breakpoint."""
    l2, on = snap_to_ladder(np.array([16.001, 4.001, 16.0]))
    assert list(on) == [True, True, True]
    assert list(l2) == [4.0, 2.0, 4.0]


def test_off_ladder_values_keep_their_real_log2_and_take_floor_to_ceil():
    """0.19 is not a power of two and is not pretended to be: its interval is ``(floor, ceil]``."""
    l2, on = snap_to_ladder(np.array([0.19, 640.0]))
    assert not on.any()
    lo, hi, _ = intervals_from_reported_mic(pd.Series(["=", "="]), pd.Series([0.19, 640.0]), pd.Series([np.nan] * 2))
    assert lo[0] == np.floor(l2[0]) and hi[0] == np.ceil(l2[0])
    assert lo[1] == np.floor(l2[1]) and hi[1] == np.ceil(l2[1])


def test_half_step_values_do_not_snap():
    """The tolerance is far below half a dilution, so 0.75 (log2 = -0.415) stays where it is."""
    _, on = snap_to_ladder(np.array([0.75, 1.5, 3.0, 6.0]))
    assert not on.any()


# ------------------------------------------------------------------ shifting to dilutions from R


def test_bounds_are_dilutions_from_the_breakpoint():
    """R = 8 means ``= 8`` is ``(-1, 0]`` — at the breakpoint, and susceptible, because R is ``MIC > R``."""
    out = ast_tests_to_intervals(_tests([{"Sample": "S1", "mic_sign": "=", "mic_value": 8.0}]), pd.DataFrame())
    assert (out["lo"].iloc[0], out["hi"].iloc[0]) == (-1.0, 0.0)


def test_bare_calls_become_the_open_interval_on_their_side():
    """A source's own R/S call says only which side of the breakpoint the MIC is: that is its interval."""
    out = ast_tests_to_intervals(_tests([
        {"Sample": "R1", "basis": "source_call", "y_test": 1.0},
        {"Sample": "S1", "basis": "source_call", "y_test": 0.0},
    ]), pd.DataFrame()).set_index("Sample")
    assert (out.loc["R1", "lo"], out.loc["R1", "hi"]) == (0.0, INF)
    assert (out.loc["S1", "lo"], out.loc["S1", "hi"]) == (-INF, 0.0)
    assert out["point"].isna().all()   # a bare call carries no MIC to impute


def test_unusable_rows_are_dropped_not_guessed():
    """``mic_undecided`` / ``no_breakpoint`` / a bare call with no call carry no interval and are dropped."""
    out = ast_tests_to_intervals(_tests([
        {"Sample": "A", "basis": "mic_undecided", "mic_sign": ">", "mic_value": 2.0},
        {"Sample": "B", "basis": "no_breakpoint", "mic_sign": "=", "mic_value": 2.0},
        {"Sample": "C", "basis": "source_call", "y_test": np.nan},
        {"Sample": "D", "basis": "mic", "mic_sign": "=", "mic_value": 2.0},
    ]), pd.DataFrame())
    assert list(out["Sample"]) == ["D"]


# ------------------------------------------------------------------ collapsing a cell


def _per_test(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"Sample": "S", "species": "paeruginosa", "drug": "meropenem", "lo": lo, "hi": hi,
          "point": pt, "is_mic": mic, "is_exact": ex} for lo, hi, pt, mic, ex in rows])


def test_duplicate_ingest_collapses_to_the_one_measurement():
    """Two ingests of one assay intersect to that assay — which is what deduplicates them."""
    cells = collapse_to_cells(_per_test([(1.0, 2.0, 2.0, True, True), (1.0, 2.0, 2.0, True, True)]))
    assert (cells["lo"].iloc[0], cells["hi"].iloc[0]) == (1.0, 2.0)
    assert not cells["hull"].iloc[0] and cells["censor"].iloc[0] == "exact"


def test_an_exact_mic_and_a_censored_one_intersect_to_the_tighter_statement():
    """``= 4`` and ``> 2`` against R = 8: ``(1, 2] ^ (-2, inf) = (1, 2]``."""
    cells = collapse_to_cells(_per_test([(1.0, 2.0, 2.0, True, True), (-2.0, INF, -1.0, True, False)]))
    assert (cells["lo"].iloc[0], cells["hi"].iloc[0]) == (1.0, 2.0)
    assert cells["censor"].iloc[0] == "interval"   # not every test was exact


def test_contradicting_tests_take_the_hull_and_are_flagged():
    """``= 4`` and ``= 8`` cannot both hold; the hull keeps the cell usable and visible, never resolved."""
    cells = collapse_to_cells(_per_test([(1.0, 2.0, 2.0, True, True), (2.0, 3.0, 3.0, True, True)]))
    assert (cells["lo"].iloc[0], cells["hi"].iloc[0]) == (1.0, 3.0)
    assert bool(cells["hull"].iloc[0])


def test_the_interval_call_is_r_only_when_the_whole_interval_is_above_the_breakpoint():
    """R iff ``lo >= 0``, S iff ``hi <= 0``; a straddling interval is no call at all."""
    cells = collapse_to_cells(pd.DataFrame([
        {"Sample": "R", "species": "p", "drug": "d", "lo": 0.0, "hi": INF, "point": 1.0, "is_mic": True, "is_exact": False},
        {"Sample": "S", "species": "p", "drug": "d", "lo": -INF, "hi": 0.0, "point": 0.0, "is_mic": True, "is_exact": False},
        {"Sample": "U", "species": "p", "drug": "d", "lo": -2.0, "hi": INF, "point": -1.0, "is_mic": True, "is_exact": False},
    ]))
    assert dict(zip(cells["Sample"], cells["interval_label"], strict=True)) == {
        "R": 1.0, "S": 0.0, "U": pytest.approx(np.nan, nan_ok=True)}


def test_censor_classes():
    """Every cell is classified by which ends are open, and whether every MIC test was exact."""
    cells = collapse_to_cells(pd.DataFrame([
        {"Sample": "e", "species": "p", "drug": "d", "lo": 1.0, "hi": 2.0, "point": 2.0, "is_mic": True, "is_exact": True},
        {"Sample": "i", "species": "p", "drug": "d", "lo": 1.0, "hi": 4.0, "point": 2.5, "is_mic": True, "is_exact": False},
        {"Sample": "l", "species": "p", "drug": "d", "lo": -INF, "hi": -1.0, "point": -1.0, "is_mic": True, "is_exact": False},
        {"Sample": "r", "species": "p", "drug": "d", "lo": 3.0, "hi": INF, "point": 4.0, "is_mic": True, "is_exact": False},
        {"Sample": "b", "species": "p", "drug": "d", "lo": 0.0, "hi": INF, "point": np.nan, "is_mic": False, "is_exact": False},
    ]))
    assert dict(zip(cells["Sample"], cells["censor"], strict=True)) == {
        "e": "exact", "i": "interval", "l": "left", "r": "right", "b": "bare"}
    assert dict(zip(cells["Sample"], cells["basis"], strict=True))["b"] == "bare_call"


# ------------------------------------------------------------------ the split table stays the authority


def _split_table(tmp_path, rows: list[tuple[str, int]]):
    path = tmp_path / "meropenem_split.csv"
    pd.DataFrame([{"Sample": s, "ast_label": y, "split": "train"} for s, y in rows]).to_csv(path, index=False)
    return path


def test_cells_outside_the_split_table_are_not_added(tmp_path):
    """The cohort of a MIC arm is the split table's, so a cell the table does not hold is never introduced."""
    cells = pd.DataFrame([{"Sample": s, "drug": "meropenem", "interval_label": 1.0} for s in ("A", "B", "NEW")])
    kept, conflicts = check_against_split_table(cells, _split_table(tmp_path, [("A", 1), ("B", 1)]))
    assert sorted(kept["Sample"]) == ["A", "B"] and conflicts.empty


def test_a_cell_whose_interval_contradicts_its_label_is_separated_not_overridden(tmp_path):
    """Neither side wins: the cell leaves the MIC arms and is written out for inspection."""
    cells = pd.DataFrame([{"Sample": "A", "drug": "meropenem", "interval_label": 1.0},
                          {"Sample": "B", "drug": "meropenem", "interval_label": 0.0}])
    kept, conflicts = check_against_split_table(cells, _split_table(tmp_path, [("A", 1), ("B", 1)]))
    assert list(kept["Sample"]) == ["A"]
    assert list(conflicts["Sample"]) == ["B"]
    assert conflicts["ast_label"].iloc[0] == 1.0 and conflicts["interval_label"].iloc[0] == 0.0


def test_an_undecided_interval_keeps_the_tables_label(tmp_path):
    """A straddling interval contradicts nothing, so the cell stays with the label the table gave it."""
    cells = pd.DataFrame([{"Sample": "A", "drug": "meropenem", "interval_label": np.nan}])
    kept, conflicts = check_against_split_table(cells, _split_table(tmp_path, [("A", 1)]))
    assert list(kept["Sample"]) == ["A"] and kept["ast_label"].iloc[0] == 1.0 and conflicts.empty

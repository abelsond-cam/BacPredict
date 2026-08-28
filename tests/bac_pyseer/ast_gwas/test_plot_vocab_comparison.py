"""The comparison figures must render from the table and must not invent columns."""

from __future__ import annotations

import pandas as pd
import pytest

from bac_pyseer.ast_gwas.plot_vocab_comparison import delta_caterpillar, paired_scatter, run


def _table(n=5):
    return pd.DataFrame({
        "drug": [f"drug{i}" for i in range(n)],
        "full_cohort_auroc": [0.97, 0.91, 0.84, 0.78, 0.71][:n],
        "trainval_vocab_auroc": [0.96, 0.92, 0.80, 0.77, 0.60][:n],
        "delta": [0.01, -0.01, 0.04, 0.01, 0.11][:n],
        "ci_lo": [-0.004, -0.03, 0.012, -0.02, 0.061][:n],
        "ci_hi": [0.026, 0.011, 0.070, 0.041, 0.160][:n],
        "separates_from_zero": [False, False, True, False, True][:n],
    })


def test_both_figures_render(tmp_path):
    a = paired_scatter(_table(), tmp_path / "scatter.png")
    b = delta_caterpillar(_table(), tmp_path / "cat.png")
    assert a.stat().st_size > 5000
    assert b.stat().st_size > 5000


def test_run_writes_both_and_counts_separating_drugs(tmp_path, capsys):
    csv = tmp_path / "cmp.csv"
    _table().to_csv(csv, index=False)
    assert run(csv, tmp_path / "figs", organism="kp") == 0
    out = capsys.readouterr().out
    assert "2 with a CI excluding zero" in out
    assert (tmp_path / "figs" / "vocab_paired_scatter_kp.png").exists()
    assert (tmp_path / "figs" / "vocab_delta_caterpillar_kp.png").exists()


def test_a_table_from_the_wrong_source_is_rejected(tmp_path):
    """Silently plotting whatever columns happen to exist is how a figure comes to mislead."""
    csv = tmp_path / "wrong.csv"
    pd.DataFrame({"drug": ["a"], "auroc": [0.9]}).to_csv(csv, index=False)
    with pytest.raises(SystemExit) as e:
        run(csv, tmp_path / "figs")
    assert "compare_vocab_arms" in str(e.value)


def test_empty_table_is_rejected(tmp_path):
    csv = tmp_path / "empty.csv"
    _table(0).to_csv(csv, index=False)
    with pytest.raises(SystemExit):
        run(csv, tmp_path / "figs")


def test_both_figures_tolerate_a_missing_separation_flag(tmp_path):
    """paired_delta_ci omits the key when every resample was single-class."""
    t = _table().drop(columns=["separates_from_zero"])
    assert paired_scatter(t, tmp_path / "s.png").exists()
    assert delta_caterpillar(t, tmp_path / "c.png").exists()


def test_a_degenerate_ci_is_never_drawn_as_separating(tmp_path, capsys):
    """NaN is truthy: a naive bool cast would assert the figure's strongest claim from a CI that
    could not be computed."""
    t = _table()
    # object dtype first: assigning NaN into a bool column warns about an incompatible set.
    t["separates_from_zero"] = t["separates_from_zero"].astype(object)
    t.loc[0, "separates_from_zero"] = float("nan")
    t.loc[1, "separates_from_zero"] = float("nan")
    csv = tmp_path / "cmp.csv"
    t.to_csv(csv, index=False)
    assert run(csv, tmp_path / "figs") == 0
    # rows 2 and 4 are the genuinely separating ones; the two NaNs must not join them.
    assert "2 with a CI excluding zero" in capsys.readouterr().out

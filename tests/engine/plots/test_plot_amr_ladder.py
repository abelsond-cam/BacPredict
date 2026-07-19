"""Stage-A smoke for the AMR concat-ladder figure (ft_mean + gene / + noncoding / + both, + ceiling)."""
from __future__ import annotations

import pandas as pd

from bacpredict.engine.plots import plot_amr_ladder as P


def _table(**over) -> pd.DataFrame:
    base = pd.DataFrame([
        {"rung": 1, "config": "ft_mean", "block": "", "noncoding_kind": "", "n_features": 768,
         "auroc": 0.83, "auprc": 0.80, "ceiling_auroc": 0.95, "ceiling_auprc": 0.93, "lift_vs_ft": 0.0},
        {"rung": 2, "config": "ft_mean+baclm_gene", "block": "ethA", "noncoding_kind": "", "n_features": 1728,
         "auroc": 0.86, "auprc": 0.84, "ceiling_auroc": 0.95, "ceiling_auprc": 0.93, "lift_vs_ft": 0.03},
        {"rung": 3, "config": "ft_mean+baclm_noncoding", "block": "upstream:fabg1", "noncoding_kind": "upstream",
         "n_features": 1728, "auroc": 0.90, "auprc": 0.88, "ceiling_auroc": 0.95, "ceiling_auprc": 0.93,
         "lift_vs_ft": 0.07},
        {"rung": 4, "config": "ft_mean+baclm_gene+baclm_noncoding", "block": "ethA | upstream:fabg1",
         "noncoding_kind": "upstream", "n_features": 2688, "auroc": 0.92, "auprc": 0.90, "ceiling_auroc": 0.95,
         "ceiling_auprc": 0.93, "lift_vs_ft": 0.09},
    ])
    return base.assign(**over) if over else base


def test_rung_labels_name_the_added_block():
    df = _table()
    assert P._rung_label(df.iloc[0]) == "FT genome-mean"
    assert P._rung_label(df.iloc[1]) == "+ baclm gene\n(ethA)"
    assert P._rung_label(df.iloc[2]) == "+ baclm noncoding\n(upstream:fabg1)"
    assert P._rung_label(df.iloc[3]) == "+ gene + noncoding\n(ethA | upstream:fabg1)"
    # an unmatched block still labels the config
    empty = df.iloc[2].copy()
    empty["block"] = ""
    assert P._rung_label(empty) == "+ baclm noncoding\n(none)"


def test_plot_writes_png(tmp_path):
    out = tmp_path / "amr_concat_ladder.png"
    P.plot_amr_ladder(_table(), out, species="tb", drug="ethionamide")
    assert out.exists() and out.stat().st_size > 0


def test_run_reads_table_and_renders(tmp_path):
    csv = tmp_path / "ethionamide_amr_ladder_table.csv"
    _table().to_csv(csv, index=False)
    out = tmp_path / "fig.png"
    table = P.run(species="tb", drug="ethionamide", table_csv=csv, out_path=out)
    assert out.exists() and len(table) == 4


def test_plot_survives_missing_ceiling(tmp_path):
    """A drug with no committed catalogue CSV (ceiling NaN) still renders the configs."""
    df = _table(ceiling_auroc=float("nan"), ceiling_auprc=float("nan"))
    out = tmp_path / "no_ceiling.png"
    P.plot_amr_ladder(df, out, species="kp", drug="colistin")
    assert out.exists() and out.stat().st_size > 0

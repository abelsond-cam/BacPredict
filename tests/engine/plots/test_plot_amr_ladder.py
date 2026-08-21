"""Stage-A smoke for the AMR concat-ladder figure (ft_mean + gene / + noncoding / + both, + ceiling)."""
from __future__ import annotations

import json
from pathlib import Path

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
    assert P._rung_bar_label(df.iloc[0]) == "FT"
    assert P._rung_bar_label(df.iloc[1]) == "FT ⊕ gene\n(ethA)"
    # the non-coding rung is relabelled by the shared region_label helper: upstream:fabg1 → "inhA promoter"
    assert P._rung_bar_label(df.iloc[2]) == "FT ⊕ IGR\n(inhA promoter)"
    assert P._rung_bar_label(df.iloc[3]) == "FT ⊕ gene ⊕ IGR\n(ethA | inhA promoter)"
    # an empty block just labels the rung
    empty = df.iloc[2].copy()
    empty["block"] = ""
    assert P._rung_bar_label(empty) == "FT ⊕ IGR"


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


def test_catalogue_has_noncoding_gates_the_red_hatch(tmp_path):
    """The red-bar "includes IGR" hatch is conditional: a non-coding determinant → True, coding-only → False."""
    nc = tmp_path / "eth.csv"
    pd.DataFrame([
        {"gene_name": "__ALL_WHO_one_hot__", "region": "all", "mut_auroc": 0.90, "mut_auprc": 0.9,
         "is_noncoding": False, "is_rrna": False},
        {"gene_name": "inhA", "region": "non-coding", "mut_auroc": 0.83, "mut_auprc": 0.7,
         "is_noncoding": True, "is_rrna": False},
        {"gene_name": "ethA", "region": "coding", "mut_auroc": 0.60, "mut_auprc": 0.4,
         "is_noncoding": False, "is_rrna": False},
    ]).to_csv(nc, index=False)
    assert P._catalogue_refs(nc, "auroc")[3] is True  # ethionamide catalogue includes the inhA promoter

    coding = tmp_path / "rif.csv"
    pd.DataFrame([
        {"gene_name": "__ALL_WHO_one_hot__", "region": "all", "mut_auroc": 0.95, "mut_auprc": 0.9,
         "is_noncoding": False, "is_rrna": False},
        {"gene_name": "rpoB", "region": "coding", "mut_auroc": 0.94, "mut_auprc": 0.9,
         "is_noncoding": False, "is_rrna": False},
    ]).to_csv(coding, index=False)
    assert P._catalogue_refs(coding, "auroc")[3] is False  # rifampin catalogue is coding-only → no red hatch
    assert P._catalogue_refs(tmp_path / "missing.csv", "auroc")[3] is False


def test_plot_accepts_conditional_catalogue_hatch(tmp_path):
    """plot_amr_ladder renders with the hatch gate both on and off (draw-level smoke)."""
    for has_nc in (True, False):
        out = tmp_path / f"ladder_{has_nc}.png"
        P.plot_amr_ladder(_table(), out, species="tb", drug="ethionamide", strongest_single=0.83,
                          ceiling=0.9, catalogue_has_noncoding=has_nc)
        assert out.exists() and out.stat().st_size > 0


def _unitig_results(tmp_path, *, name="lr", auroc=0.94, n_evaluate=200, n_unitigs=1837,
                    model="unitig_lr") -> Path:
    """A minimal unitig_lr results.json — only the keys the figure actually reads."""
    out = tmp_path / name
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.2", "task": "kleb_ast", "drug": "gentamicin",
        "model": {"name_or_path": model, "checkpoint_dir": str(out)},
        "split": {"source": "split_table", "n_evaluate": n_evaluate},
        # metrics.threshold is 0.5 and operating_point is Youden-on-holdout; the figure reads NEITHER,
        # because AUROC/AUPRC are threshold-free. Both are present here so the test would catch it if
        # a future change started consulting them.
        "metrics": {"auroc": auroc, "auprc": 0.91, "threshold": 0.5},
        "operating_point": {"objective": "youden_j", "selected_on": "holdout", "threshold": 0.71},
        "extra": {"n_unitigs": n_unitigs, "n_train": 800},
    }
    (out / "results.json").write_text(json.dumps(payload))
    return out / "results.json"


def test_unitig_arm_reads_the_threshold_free_metric(tmp_path):
    arm = P.unitig_arm(_unitig_results(tmp_path), metric="auroc")
    assert arm == {"value": 0.94, "n_unitigs": 1837, "n_evaluate": 200}
    assert P.unitig_arm(_unitig_results(tmp_path, name="b"), metric="auprc")["value"] == 0.91


def test_absent_unitig_results_is_not_an_error(tmp_path):
    # The normal state while a fan-out lands: most drugs have no unitig result yet.
    assert P.unitig_arm(None) is None
    assert P.unitig_arm(tmp_path / "nope" / "results.json") is None


def test_a_different_model_is_refused(tmp_path):
    # A results.json from some other arm must never be drawn as the unitig bar.
    assert P.unitig_arm(_unitig_results(tmp_path, model="finetune_amr")) is None


def test_holdout_size_mismatch_drops_the_bar(tmp_path):
    # 200 vs 341 means the two arms scored DIFFERENT genomes — refuse rather than compare across cohorts.
    res = _unitig_results(tmp_path, n_evaluate=200)
    assert P.unitig_arm(res, n_holdout=200) is not None
    assert P.unitig_arm(res, n_holdout=341) is None


def test_render_with_the_unitig_bar(tmp_path):
    out = tmp_path / "with_unitig.png"
    P.plot_amr_ladder(_table(), out, species="kp", drug="gentamicin", strongest_single=0.88,
                      strongest_name="aac(6')-Ib", ceiling=0.95,
                      unitig={"value": 0.94, "n_unitigs": 1837, "n_evaluate": 200})
    assert out.exists() and out.stat().st_size > 0


def test_render_with_both_purple_bars(tmp_path):
    out = tmp_path / "with_dedup.png"
    P.plot_amr_ladder(_table(), out, species="kp", drug="gentamicin", ceiling=0.95,
                      unitig={"value": 0.94, "n_unitigs": 1837, "n_evaluate": 200},
                      unitig_dedup={"value": 0.92, "n_unitigs": 61, "n_evaluate": 200})
    assert out.exists() and out.stat().st_size > 0


def test_render_without_any_unitig_arm_is_unchanged(tmp_path):
    # No purple group at all: the figure must still render, since that is every drug pre-fan-out.
    out = tmp_path / "no_unitig.png"
    P.plot_amr_ladder(_table(), out, species="kp", drug="gentamicin", ceiling=0.95)
    assert out.exists() and out.stat().st_size > 0


def test_run_threads_the_unitig_results_through(tmp_path):
    csv = tmp_path / "t.csv"
    _table().to_csv(csv, index=False)
    out = tmp_path / "run.png"
    table = P.run(species="kp", drug="gentamicin", table_csv=csv, out_path=out,
                  unitig_results=_unitig_results(tmp_path))
    assert out.exists() and len(table) == 4  # the ladder table itself is untouched — still four rungs

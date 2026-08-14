"""Stage-A smoke for the two-panel (imputed / carrier-only) catalogue-vs-LR causal comparison plot."""
from __future__ import annotations

import pandas as pd

from bacpredict.engine.plots import plot_causal_comparison as C


def test_synonyms_bridge_inha_fabg1():
    assert "fabg1" in C._synonyms("inhA")  # the mabA-inhA operon promoter naming gap
    assert C._synonyms("rpoB") == {"rpob"}


def test_prev_alpha_floor_and_nan():
    """Penetrance → opacity: near-universal solid, rare translucent (floored so it stays visible), NaN opaque."""
    assert C._prev_alpha(1.0) == 1.0
    assert C._prev_alpha(0.0) == 0.2
    assert 0.2 < C._prev_alpha(0.5) < 1.0
    assert C._prev_alpha(float("nan")) == 1.0
    assert C._prev_alpha(None) == 1.0


def test_best_by_key_floors_and_aliases():
    """Entries are ``(select_auroc, display_auroc, source, prevalence, key)``.

    The first two are separate because selection happens on the train-OOF ``lr_auroc_`` while the
    figure displays the deployment-holdout ``eval_auroc_``. With no ``eval_auroc_`` column the
    display value falls back to the selection value, which is what these fixtures exercise.
    """
    coding = pd.DataFrame([{"gene_name": "rpoB", "lr_auroc_x": 0.96, "n_pos": 500, "prevalence": 0.98},
                           {"gene_name": "art", "lr_auroc_x": 1.0, "n_pos": 3, "prevalence": 0.003}])
    upstream = pd.DataFrame([{"upstream_gene": "upstream:fabg1", "lr_auroc_x": 0.80, "n_pos": 900,
                              "prevalence": 0.997}])
    best = C._best_by_key([(coding, "gene_name", "coding"), (upstream, "upstream_gene", "upstream"),
                           (None, "unit", "per_unit")], min_n_pos=20)
    assert "rpob" in best
    assert best["rpob"][0] == 0.96 and best["rpob"][1] == 0.96  # select, then display (fallback)
    assert best["rpob"][2] == "coding" and best["rpob"][3] == 0.98
    assert "art" not in best  # n_pos < 20 → the low-n artifact is floored out
    assert "fabg1" in best and best["fabg1"][2] == "upstream"  # bare-gene alias of the upstream key
    assert "upstream:fabg1" in best


def test_top_gene_prefers_available_auroc():
    """The imputed rung-2 pick = the coding ranking's top row (the ◆ routed-in marker)."""
    df = pd.DataFrame([{"gene_name": "recE", "lr_auroc_cipro": 0.94},
                       {"gene_name": "gyrA", "lr_auroc_cipro": 0.92}])
    assert C._top_gene(df) == "rece"
    assert C._top_gene(None) is None
    assert C._top_gene(pd.DataFrame({"gene_name": []})) is None


def test_catalogue_parses_auroc_and_ceiling(tmp_path):
    csv = tmp_path / "cat.csv"
    pd.DataFrame({"gene_name": ["__ALL_CARD__", "eis", "eis", "rrs"],
                  "mut_auroc": [0.90, 0.55, 0.62, 0.78],  # eis appears twice → the max is kept
                  "mut_auprc": [0.7, 0.3, 0.35, 0.5]}).to_csv(csv, index=False)
    dets, au, ceiling = C._catalogue(csv)
    assert dets == {"eis", "rrs"}  # the __ALL ceiling row is split out of the determinant set
    assert au["eis"] == 0.62 and au["rrs"] == 0.78
    assert ceiling == 0.90


def test_panel_data_splits_determinants_and_lr_only():
    """One panel: catalogue determinant bars (with prevalence) vs LR-only bars (with prevalence)."""
    coding = pd.DataFrame([{"gene_name": "rpoB", "lr_auroc_x": 0.96, "n_pos": 500, "prevalence": 0.98},
                           {"gene_name": "recE", "lr_auroc_x": 0.94, "n_pos": 300, "prevalence": 0.22}])
    cat_bars, lr_only = C._panel_data([(coding, "gene_name", "coding"), (None, "upstream_gene", "upstream"),
                                       (None, "unit", "per_unit")], {"rpob"}, {"rpob": 0.95}, {},
                                      top_n_lr=10, min_n_pos=20)
    # _CatBar = (name, auroc, cat_ref, cov, prev, source, raw_key); coding determinant → source "coding".
    assert cat_bars[0][0] == "rpob" and cat_bars[0][1] == 0.96 and cat_bars[0][4] == 0.98
    assert cat_bars[0][5] == "coding"
    # _LrBar = (name, auroc, prev, source, raw_key).
    rece = next((t for t in lr_only if t[0] == "rece"), None)
    assert rece[:3] == ("rece", 0.94, 0.22) and rece[3] == "coding" and rece[4] == "rece"


def test_panel_data_carries_promoter_source_for_synonym_determinant():
    """A determinant won by its promoter (inha via upstream:fabg1) carries source ``upstream`` + the full
    ``upstream:fabg1`` raw key, so the bar is hatched and relabelled "inhA promoter" not "inha"."""
    coding = pd.DataFrame([{"gene_name": "inhA", "lr_auroc_x": 0.565, "n_pos": 400, "prevalence": 0.99}])
    upstream = pd.DataFrame([{"upstream_gene": "upstream:fabg1", "lr_auroc_x": 0.80, "n_pos": 900,
                              "prevalence": 0.997}])
    cat_bars, _ = C._panel_data([(coding, "gene_name", "coding"), (upstream, "upstream_gene", "upstream"),
                                 (None, "unit", "per_unit")], {"inha"}, {"inha": 0.83}, {},
                                top_n_lr=10, min_n_pos=20)
    inha = next(t for t in cat_bars if t[0] == "inha")
    assert inha[1] == 0.80 and inha[5] == "upstream" and inha[6] == "upstream:fabg1"


def test_routed_marks_from_ladder_table(tmp_path):
    """The ◆/★ bar-names resolve from the ladder table: rung-2 gene → the coding bar, rung-3 non-coding
    block (upstream:fabg1) → the inha determinant bar that folded the promoter in."""
    ladder = tmp_path / "ethionamide_amr_ladder_table.csv"
    pd.DataFrame([{"rung": 1, "block": ""}, {"rung": 2, "block": "rpoB"},
                  {"rung": 3, "block": "upstream:fabg1"},
                  {"rung": 4, "block": "rpoB | upstream:fabg1"}]).to_csv(ladder, index=False)
    cat = [("inha", 0.80, 0.83, float("nan"), 0.997, "upstream", "upstream:fabg1"),
           ("rpob", 0.77, 0.77, float("nan"), 0.98, "coding", "rpob")]
    coding_mark, noncoding_mark = C._routed_marks(ladder, cat, [])
    assert coding_mark == "rpob" and noncoding_mark == "inha"


def test_run_writes_two_panel_comparison(tmp_path):
    imp = tmp_path / "imp.csv"
    pd.DataFrame([{"gene_name": "rpoB", "lr_auroc_rifampin": 0.96, "n_pos": 500, "prevalence": 0.98},
                  {"gene_name": "katG", "lr_auroc_rifampin": 0.86, "n_pos": 400, "prevalence": 0.9}]).to_csv(
        imp, index=False)
    car = tmp_path / "car.csv"
    pd.DataFrame([{"gene_name": "rpoB", "lr_auroc_rifampin": 0.95, "n_pos": 500, "prevalence": 0.98},
                  {"gene_name": "recE", "lr_auroc_rifampin": 0.93, "n_pos": 120, "prevalence": 0.2}]).to_csv(
        car, index=False)
    up = tmp_path / "u.csv"
    pd.DataFrame([{"upstream_gene": "upstream:fabg1", "lr_auroc_rifampin": 0.80, "n_pos": 900,
                   "prevalence": 0.997}]).to_csv(up, index=False)
    cat = tmp_path / "c.csv"
    pd.DataFrame({"gene_name": ["__ALL_WHO_one_hot__", "rpoB", "inhA"],
                  "mut_auroc": [0.95, 0.96, 0.80], "mut_auprc": [0.9, 0.9, 0.7]}).to_csv(cat, index=False)
    out = C.run(species="tb", drug="rifampin", imputed_coding_csv=imp, carrier_coding_csv=car,
                upstream_csv=up, unit_csv=None, catalogue_csv=cat, out_dir=tmp_path)
    assert out.exists() and out.name == "causal_comparison.png"
    assert out.parts[-2] == "rifampicin"  # display_name(rifampin)


def test_run_survives_empty(tmp_path):
    cat = tmp_path / "c.csv"
    pd.DataFrame({"gene_name": ["rpoB"], "mut_auroc": [0.9], "mut_auprc": [0.8]}).to_csv(cat, index=False)
    out = C.run(species="tb", drug="rifampin", imputed_coding_csv=None, carrier_coding_csv=None,
                upstream_csv=None, unit_csv=None, catalogue_csv=cat, out_dir=tmp_path)
    assert not out.exists()  # no rankings on either panel → nothing to draw, no crash

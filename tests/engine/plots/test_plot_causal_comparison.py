"""Stage-A smoke for the catalogue-vs-LR causal comparison plot."""
from __future__ import annotations

import pandas as pd

from bacpredict.engine.plots import plot_causal_comparison as C


def test_synonyms_bridge_inha_fabg1():
    assert "fabg1" in C._synonyms("inhA")  # the mabA-inhA operon promoter naming gap
    assert C._synonyms("rpoB") == {"rpob"}


def test_best_by_key_floors_and_aliases():
    coding = pd.DataFrame([{"gene_name": "rpoB", "lr_auroc_x": 0.96, "n_pos": 500, "prevalence": 0.98},
                           {"gene_name": "art", "lr_auroc_x": 1.0, "n_pos": 3, "prevalence": 0.003}])
    upstream = pd.DataFrame([{"upstream_gene": "upstream:fabg1", "lr_auroc_x": 0.80, "n_pos": 900,
                              "prevalence": 0.997}])
    best = C._best_by_key([(coding, "gene_name", "coding"), (upstream, "upstream_gene", "upstream"),
                           (None, "unit", "per_unit")], min_n_pos=20)
    assert "rpob" in best and best["rpob"][0] == 0.96
    assert "art" not in best  # n_pos < 20 → the low-n artifact is floored out
    assert "fabg1" in best and best["fabg1"][1] == "upstream"  # bare-gene alias of the upstream key
    assert "upstream:fabg1" in best


def test_determinant_status_vs_own_auroc():
    """recovered/missed is judged against the determinant's OWN catalogue one-hot, not a global cutoff."""
    S = C._determinant_status
    assert S(0.593, 0.620, 0.05) == "recovered"  # kanamycin eis: LR within margin of its weak catalogue one-hot
    assert S(0.795, 0.778, 0.05) == "recovered"  # kanamycin rrs: LR beats the catalogue
    assert S(0.55, 0.90, 0.05) == "missed"       # a genuine gap the LR did not capture
    assert S(None, 0.62, 0.05) == "absent"
    assert S(float("nan"), 0.62, 0.05) == "absent"
    assert S(0.58, float("nan"), 0.05) == "recovered"  # unknown catalogue AUROC → chance + margin (0.55)
    assert S(0.52, float("nan"), 0.05) == "missed"


def test_catalogue_parses_auroc_and_ceiling(tmp_path):
    csv = tmp_path / "cat.csv"
    pd.DataFrame({"gene_name": ["__ALL_CARD__", "eis", "eis", "rrs"],
                  "mut_auroc": [0.90, 0.55, 0.62, 0.78],  # eis appears twice → the max is kept
                  "mut_auprc": [0.7, 0.3, 0.35, 0.5]}).to_csv(csv, index=False)
    dets, au, ceiling = C._catalogue(csv)
    assert dets == {"eis", "rrs"}  # the __ALL ceiling row is split out of the determinant set
    assert au["eis"] == 0.62 and au["rrs"] == 0.78
    assert ceiling == 0.90


def test_run_writes_comparison(tmp_path):
    coding = tmp_path / "g.csv"
    pd.DataFrame([{"gene_name": "rpoB", "lr_auroc_rifampin": 0.96, "n_pos": 500, "prevalence": 0.98},
                  {"gene_name": "katG", "lr_auroc_rifampin": 0.86, "n_pos": 400, "prevalence": 0.9}]).to_csv(
        coding, index=False)
    up = tmp_path / "u.csv"
    pd.DataFrame([{"upstream_gene": "upstream:fabg1", "lr_auroc_rifampin": 0.80, "n_pos": 900,
                   "prevalence": 0.997}]).to_csv(up, index=False)
    cat = tmp_path / "c.csv"
    pd.DataFrame({"gene_name": ["__ALL_WHO_one_hot__", "rpoB", "inhA"],
                  "mut_auroc": [0.95, 0.96, 0.80], "mut_auprc": [0.9, 0.9, 0.7]}).to_csv(cat, index=False)
    out = C.run(species="tb", drug="rifampin", coding_csv=coding, upstream_csv=up, unit_csv=None,
                catalogue_csv=cat, out_dir=tmp_path)
    assert out.exists() and out.name == "causal_comparison.png"
    assert out.parts[-2] == "rifampicin"  # display_name(rifampin)


def test_run_survives_empty(tmp_path):
    cat = tmp_path / "c.csv"
    pd.DataFrame({"gene_name": ["rpoB"], "mut_auroc": [0.9], "mut_auprc": [0.8]}).to_csv(cat, index=False)
    out = C.run(species="tb", drug="rifampin", coding_csv=None, upstream_csv=None, unit_csv=None,
                catalogue_csv=cat, out_dir=tmp_path)
    assert not out.exists()  # no rankings → nothing to draw, no crash

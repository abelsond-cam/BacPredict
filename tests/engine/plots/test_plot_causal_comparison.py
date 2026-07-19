"""Stage-A smoke for the catalogue-vs-LR causal comparison plot."""
from __future__ import annotations

import numpy as np
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


def test_run_writes_comparison(tmp_path):
    coding = tmp_path / "g.csv"
    pd.DataFrame([{"gene_name": "rpoB", "lr_auroc_rifampin": 0.96, "n_pos": 500, "prevalence": 0.98},
                  {"gene_name": "katG", "lr_auroc_rifampin": 0.86, "n_pos": 400, "prevalence": 0.9}]).to_csv(
        coding, index=False)
    up = tmp_path / "u.csv"
    pd.DataFrame([{"upstream_gene": "upstream:fabg1", "lr_auroc_rifampin": 0.80, "n_pos": 900,
                   "prevalence": 0.997}]).to_csv(up, index=False)
    cat = tmp_path / "c.csv"
    pd.DataFrame({"gene_name": ["__ALL_WHO_one_hot__", "rpoB", "inhA"]}).to_csv(cat, index=False)
    out = C.run(species="tb", drug="rifampin", coding_csv=coding, upstream_csv=up, unit_csv=None,
                catalogue_csv=cat, out_dir=tmp_path)
    assert out.exists() and out.name == "causal_comparison.png"
    assert out.parts[-2] == "rifampicin"  # display_name(rifampin)


def test_run_survives_empty(tmp_path):
    cat = tmp_path / "c.csv"
    pd.DataFrame({"gene_name": ["rpoB"]}).to_csv(cat, index=False)
    out = C.run(species="tb", drug="rifampin", coding_csv=None, upstream_csv=None, unit_csv=None,
                catalogue_csv=cat, out_dir=tmp_path)
    assert not out.exists()  # no rankings → nothing to draw, no crash
    _ = np  # keep the import used

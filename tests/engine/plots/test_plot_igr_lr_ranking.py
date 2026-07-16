"""Stage-A smoke for the per-region LR ranking plots (no cluster, tiny synthetic ranking)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bacpredict.engine.plots import plot_igr_lr_ranking as P


def _ranking(n=40, drug="ciprofloxacin"):
    rng = np.random.default_rng(0)
    genes = [f"g{i}" for i in range(n + 1)]
    rows = []
    for i in range(n):
        rows.append({
            "igr_pair": f"{genes[i]}→{genes[i + 1]}",
            "left_gene": genes[i], "right_gene": genes[i + 1],
            "prevalence": float(rng.uniform(0.05, 1.0)),
            f"lr_auroc_{drug}": float(rng.uniform(0.5, 0.99)),
            "n_train": 2000, "n_pos": 1000, "kept_filtered": False,
        })
    # make gyrA→parC the clear top hit (a causal pair)
    rows.append({"igr_pair": "gyrA→parC", "left_gene": "gyrA", "right_gene": "parC",
                 "prevalence": 0.95, f"lr_auroc_{drug}": 0.985, "n_train": 2000, "n_pos": 1000,
                 "kept_filtered": True})
    return pd.DataFrame(rows)


def _presence(rank, drug="ciprofloxacin"):
    return pd.DataFrame({"igr_pair": rank["igr_pair"],
                         f"presence_lr_auroc_{drug}": np.full(len(rank), 0.53)})


def test_run_writes_both_figures(tmp_path):
    rank = _ranking()
    csv = tmp_path / "per_igr_lr_ciprofloxacin.csv"
    rank.to_csv(csv, index=False)
    pcsv = tmp_path / "per_igr_presence_lr_ciprofloxacin.csv"
    _presence(rank).to_csv(pcsv, index=False)

    base = P.run(species="kp", drug="ciprofloxacin", method="per_igr", csv=csv, presence_csv=pcsv,
                 out_dir=tmp_path / "viz", causal_genes=["gyrA", "parC"], top_n=10)
    assert (base / "top10.png").exists()
    assert (base / "density.png").exists()
    # dir layout: <out>/<species>/<display_drug>/<method>/
    assert base.parts[-3:] == ("kp", "ciprofloxacin", "per_igr")


def test_causal_and_label_helpers():
    causal = P.load_causal(["GyrA", "ParC"], None)
    assert causal == {"gyra", "parc"}
    row = pd.Series({"igr_pair": "gyrA→parC", "left_gene": "gyrA", "right_gene": "parC"})
    assert P._is_causal(row, causal)
    assert not P._is_causal(pd.Series({"igr_pair": "x→y", "left_gene": "x", "right_gene": "y"}), causal)
    # named-feature suffixing
    feat = pd.Series({"feature_type": "rrna", "feature_name": "rrs"})
    assert P._region_label(feat) == "rrs (rRNA)"
    assert P._region_label(pd.Series({"igr_pair": "a→b"})) == "a→b"


def test_runs_without_presence(tmp_path):
    rank = _ranking()
    csv = tmp_path / "r.csv"
    rank.to_csv(csv, index=False)
    base = P.run(species="tb", drug="rifampin", method="whole_igr", csv=csv, presence_csv=None,
                 out_dir=tmp_path / "viz", causal_genes=["rpoB"])
    assert (base / "top10.png").exists() and (base / "density.png").exists()
    assert base.parts[-2] == "rifampicin"  # display_name maps rifampin -> rifampicin

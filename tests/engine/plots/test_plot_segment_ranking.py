"""Stage-A smoke for the per-gene LR ranking screen + the non-imputed >N-eval-carrier gate (Part 4)."""
from __future__ import annotations

import pandas as pd

from bacpredict.engine.plots import plot_segment_ranking as P


def _ranking(tmp_path, n_eval):
    csv = tmp_path / "per_gene_lr_rifampin.csv"
    pd.DataFrame([
        {"gene_name": "rpoB", "lr_auroc_rifampin": 0.96, "n_eval": n_eval[0]},
        {"gene_name": "katG", "lr_auroc_rifampin": 0.80, "n_eval": n_eval[1]},
        {"gene_name": "rare", "lr_auroc_rifampin": 0.99, "n_eval": n_eval[2]},  # would top the plot but rare
    ]).to_csv(csv, index=False)
    return csv


def test_ungated_renders(tmp_path):
    out = tmp_path / "screen.png"
    P.plot_ranking(_ranking(tmp_path, [500, 400, 5]), out, drug="rifampin")
    assert out.exists()


def test_gate_keeps_well_powered_genes(tmp_path):
    # `rare` (n_eval=5, AUROC 0.99) is dropped by the >100 gate; rpoB + katG survive → figure written.
    out = tmp_path / "gated.png"
    P.plot_ranking(_ranking(tmp_path, [500, 400, 5]), out, drug="rifampin", min_n_eval=100)
    assert out.exists()


def test_gate_empty_skips_without_crash(tmp_path):
    # every gene below the gate → no figure, no exception (the "no well-powered gene" case).
    out = tmp_path / "empty.png"
    P.plot_ranking(_ranking(tmp_path, [5, 3, 8]), out, drug="rifampin", min_n_eval=100)
    assert not out.exists()

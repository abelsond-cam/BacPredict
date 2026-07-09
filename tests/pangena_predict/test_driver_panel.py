"""Unit smoke for the driver panel: CSV parse, coding classification, Bacformer join, chart render."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pangena_predict.driver_panel import (
    _bacformer_frame_for_gene,
    _is_coding,
    _resolve_ast_column,
    parse_driver_csv,
    plot_drug_panel,
)

CSV = (
    "gene_name,region,site,mut_auroc,mut_auroc_sd,mut_auprc,mut_auprc_sd,n_variants,"
    "n_genomes_with_variant,embeddable,is_rrna,is_noncoding\n"
    "__ALL_WHO_one_hot__,all,__ALL_WHO_one_hot__,0.87,0.002,0.72,0.001,361,2845,False,False,False\n"
    "inhA,non-coding,inhA (promoter),0.83,0.0003,0.63,0.003,7,1893,False,False,True\n"
    "ethA,coding,ethA,0.61,0.003,0.31,0.004,347,1397,True,False,False\n"
    "rrs,non-coding,rrs,0.70,0.001,0.40,0.002,5,900,False,True,True\n"
)


def test_parse_driver_csv_splits_ceiling(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text(CSV)
    drivers, ceiling = parse_driver_csv(p)
    assert ceiling == {"auroc": 0.87, "auprc": 0.72}
    assert len(drivers) == 3
    assert set(drivers["gene_name"]) == {"inhA", "ethA", "rrs"}


def test_is_coding_only_embeddable_coding(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text(CSV)
    drivers, _ = parse_driver_csv(p)
    by_gene = {r["gene_name"]: _is_coding(r) for _, r in drivers.iterrows()}
    assert by_gene == {"ethA": True, "inhA": False, "rrs": False}  # promoter + rRNA are not codable


def test_bacformer_frame_lookup():
    npz = {"ethA__ids": np.array(["s1", "s2"]), "ethA__tok": np.zeros((2, 4), dtype=np.float32)}
    frame = _bacformer_frame_for_gene("ethA", npz)
    assert frame is not None and list(frame.index) == ["s1", "s2"] and frame.shape == (2, 4)
    assert _bacformer_frame_for_gene("katG", npz) is None  # gene not in the sweep


def test_resolve_ast_column_maps_rifampicin():
    cols = {"rifampin", "isoniazid", "ethionamide"}
    assert _resolve_ast_column("rifampicin", cols) == "rifampin"
    assert _resolve_ast_column("isoniazid", cols) == "isoniazid"
    assert _resolve_ast_column("levofloxacin", cols) is None  # absent -> skip


def test_plot_drug_panel_renders(tmp_path):
    table = pd.DataFrame([
        {"gene": "ethA", "site": "ethA", "region": "coding", "onehot_auroc": 0.61, "onehot_auprc": 0.31,
         "baclm_auroc": 0.60, "baclm_auprc": 0.30, "esm_auroc": 0.62, "esm_auprc": 0.32,
         "bacformer_auroc": None, "bacformer_auprc": None},
        {"gene": "inhA", "site": "inhA (promoter)", "region": "non-coding", "onehot_auroc": 0.83,
         "onehot_auprc": 0.63, "baclm_auroc": None, "baclm_auprc": None, "esm_auroc": None,
         "esm_auprc": None, "bacformer_auroc": None, "bacformer_auprc": None},
    ])
    out = tmp_path / "panel.png"
    plot_drug_panel({"drug": "ethionamide", "ceiling": {"auroc": 0.87, "auprc": 0.72}, "table": table}, out)
    assert out.exists() and out.stat().st_size > 0


def test_per_sample_genes_inverts_presence():
    import pandas as pd

    from pangena_predict.bacformer_gene_panel_vectors import _per_sample_genes

    presence = {
        "ethA": pd.DataFrame({"gene_flat_index": [10], "n_proteins": [4000]}, index=pd.Index(["s1"], name="Sample")),
        "inhA": pd.DataFrame({"gene_flat_index": [20, 21], "n_proteins": [4000, 4100]},
                             index=pd.Index(["s1", "s2"], name="Sample")),
    }
    per = _per_sample_genes(presence)
    assert per["s1"] == {"ethA": (10, 4000), "inhA": (20, 4000)}
    assert per["s2"] == {"inhA": (21, 4100)}

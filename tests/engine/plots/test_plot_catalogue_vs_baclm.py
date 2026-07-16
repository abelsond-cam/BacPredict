"""Stage-A smoke for the catalogue-vs-baclm join plot (no cluster, tiny synthetic tables).

Exercises the mechanism classification, the coding/upstream matching, and — critically — the
``inhA (promoter) → upstream:fabg1`` anchor bridge that recovers the determinant the old flank-pair
IGR screen dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bacpredict.engine.plots import plot_catalogue_vs_baclm as P


def _tb_catalogue() -> pd.DataFrame:
    """Minimal TB-Profiler catalogue: an all-drivers ceiling, a coding gene, a promoter, an rRNA."""
    return pd.DataFrame([
        {"gene_name": "__ALL_WHO_one_hot__", "region": "all", "site": "__ALL_WHO_one_hot__",
         "mut_auroc": 0.87, "mut_auprc": 0.72, "embeddable": False, "is_rrna": False, "is_noncoding": False},
        {"gene_name": "inhA", "region": "non-coding", "site": "inhA (promoter)",
         "mut_auroc": 0.826, "mut_auprc": 0.63, "embeddable": False, "is_rrna": False, "is_noncoding": True},
        {"gene_name": "ethA", "region": "coding", "site": "ethA",
         "mut_auroc": 0.61, "mut_auprc": 0.31, "embeddable": True, "is_rrna": False, "is_noncoding": False},
        {"gene_name": "rrs", "region": "non-coding", "site": "rrs",
         "mut_auroc": 0.55, "mut_auprc": 0.25, "embeddable": False, "is_rrna": True, "is_noncoding": True},
    ])


def test_build_table_matches_coding_promoter_and_rrna():
    cat = _tb_catalogue()
    is_all = cat["gene_name"].str.startswith("__ALL")
    drivers = cat[~is_all].reset_index(drop=True)

    coding_map = {"etha": 0.58}                 # ethA gene body ranked
    upstream_map = {"fabg1": 0.80}              # promoter embedded + ranked as upstream:fabg1

    table = P.build_table(drivers, kind="tbprofiler", species="tb",
                          coding_map=coding_map, upstream_map=upstream_map)

    by_gene = table.set_index("gene")
    # coding ethA → per_gene body
    assert by_gene.loc["ethA", "mechanism"] == "coding"
    assert by_gene.loc["ethA", "baclm_auroc"] == 0.58
    assert by_gene.loc["ethA", "matched_via"] == "per_gene:etha"
    # promoter inhA → the fabG1 upstream anchor (THE capture-bug fix)
    assert by_gene.loc["inhA", "mechanism"] == "promoter"
    assert by_gene.loc["inhA", "matched_via"] == "upstream:fabg1"
    assert by_gene.loc["inhA", "baclm_auroc"] == 0.80
    # rRNA → blank (per_unit re-embed pending)
    assert by_gene.loc["rrs", "mechanism"] == "rRNA"
    assert not bool(by_gene.loc["rrs", "baclm_matched"])
    assert np.isnan(by_gene.loc["rrs", "baclm_auroc"])


def test_kp_base_symbol_strips_mut_wt_qualifier():
    drivers = pd.DataFrame([
        {"gene_name": "GyrA (mut)", "site": "GyrA (mut)", "category": "chromosomal_mutation",
         "mut_auroc": 0.88, "mut_auprc": 0.92, "is_causal": True, "is_rrna": False, "is_noncoding": True},
        {"gene_name": "GyrA (WT)", "site": "GyrA (WT)", "category": "chromosomal_coding",
         "mut_auroc": 0.85, "mut_auprc": 0.88, "is_causal": False, "is_rrna": False, "is_noncoding": False},
    ])
    table = P.build_table(drivers, kind="card", species="kp",
                          coding_map={"gyra": 0.83}, upstream_map={})
    # both the mut and WT determinant collapse to the gyrA gene body
    assert set(table["matched_via"]) == {"per_gene:gyra"}
    assert (table["baclm_auroc"] == 0.83).all()
    assert list(table["mechanism"]) == ["chromosomal_mut", "coding"]


def test_run_writes_png_and_csv(tmp_path):
    cat_csv = tmp_path / "tbprofiler_gene_lr_ethionamide.csv"
    _tb_catalogue().to_csv(cat_csv, index=False)
    per_gene = tmp_path / "per_gene_lr_ethionamide.csv"
    pd.DataFrame([{"gene_name": "ethA", "annotation": "", "prevalence": 0.9,
                   "lr_auroc_ethionamide": 0.58, "n_train": 2000, "n_pos": 900,
                   "kept_filtered": False}]).to_csv(per_gene, index=False)
    upstream = tmp_path / "per_upstream_lr_ethionamide.csv"
    pd.DataFrame([{"upstream_gene": "upstream:fabg1", "gene": "fabg1", "prevalence": 0.95,
                   "lr_auroc_ethionamide": 0.80, "n_train": 2000, "n_pos": 900,
                   "kept_filtered": True}]).to_csv(upstream, index=False)

    out_png = tmp_path / "tb_profiler_vs_bac_lm.png"
    table = P.run(species="tb", drug="ethionamide", catalogue_kind="tbprofiler", catalogue_csv=cat_csv,
                  per_gene_csv=per_gene, upstream_csv=upstream, out_path=out_png)
    assert out_png.exists() and out_png.stat().st_size > 0
    assert out_png.with_suffix(".csv").exists()
    # the promoter row is matched via the fabG1 anchor
    inha = table.set_index("gene").loc["inhA"]
    assert inha["baclm_auroc"] == 0.80 and inha["matched_via"] == "upstream:fabg1"


def test_run_survives_missing_rankings(tmp_path):
    """A drug whose baclm rankings are absent still renders (all baclm bars blank)."""
    cat_csv = tmp_path / "tbprofiler_gene_lr_kanamycin.csv"
    _tb_catalogue().to_csv(cat_csv, index=False)
    out_png = tmp_path / "tb_profiler_vs_bac_lm.png"
    table = P.run(species="tb", drug="kanamycin", catalogue_kind="tbprofiler", catalogue_csv=cat_csv,
                  per_gene_csv=None, upstream_csv=None, out_path=out_png)
    assert out_png.exists()
    assert not table["baclm_matched"].any()

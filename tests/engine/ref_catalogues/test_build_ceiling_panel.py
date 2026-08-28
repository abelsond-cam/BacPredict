"""The ceiling panel must be reproducible from its sources, and must refuse a mislabelled estimator."""

from __future__ import annotations

import pandas as pd
import pytest

from bacpredict.engine.ref_catalogues import build_ceiling_panel as B


def _card_csv(path, drug, *, auroc=0.98, sd=0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"gene_name": "blaKPC", "site": "blaKPC", "category": "gene", "mut_auroc": 0.91,
         "mut_auroc_sd": sd, "mut_auprc": 0.88, "mut_auprc_sd": 0.0, "n_determinants": 12,
         "n_genomes_with_determinant": 300, "embeddable": True, "is_causal": True,
         "is_rrna": False, "is_noncoding": False},
        {"gene_name": "__ALL_CARD__", "site": "__ALL_CARD__", "category": "all", "mut_auroc": auroc,
         "mut_auroc_sd": sd, "mut_auprc": 0.99, "mut_auprc_sd": 0.0, "n_determinants": 168,
         "n_genomes_with_determinant": 2121, "embeddable": False, "is_causal": False,
         "is_rrna": False, "is_noncoding": False},
    ]).to_csv(path, index=False)


def _who_csv(path, drug, *, sd=0.00026):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"gene_name": "__ALL_WHO_one_hot__", "region": "all", "site": "__ALL_WHO_one_hot__",
         "mut_auroc": 0.9666, "mut_auroc_sd": sd, "mut_auprc": 0.9392, "mut_auprc_sd": 0.0004,
         "n_variants": 149, "n_genomes_with_variant": 11293, "embeddable": False,
         "is_rrna": False, "is_noncoding": False},
    ]).to_csv(path, index=False)


def test_card_panel_round_trips_the_source_values(tmp_path):
    root = tmp_path / "card_ceiling"
    _card_csv(root / "ertapenem" / "card_determinant_lr_ertapenem_allele.csv", "ertapenem", auroc=0.9828)
    _card_csv(root / "colistin" / "card_determinant_lr_colistin_allele.csv", "colistin", auroc=0.6563)

    panel = B.build_panel(
        B.discover_card(root, "allele"), schema_key="card", grain="allele",
        estimator="deployment_holdout", status="current",
    )
    assert list(panel.columns) == list(B.PANEL_COLUMNS)
    assert panel.loc[panel.drug == "ertapenem", "ceiling_auroc"].iloc[0] == pytest.approx(0.9828)
    assert panel.loc[panel.drug == "colistin", "ceiling_auroc"].iloc[0] == pytest.approx(0.6563)
    # Provenance travels per row — that is the whole point of the panel over a bare two-column CSV.
    assert set(panel.ceiling_catalogue) == {"CARD"}
    assert set(panel.ceiling_estimator) == {"deployment_holdout"}
    assert set(panel.ceiling_status) == {"current"}
    assert panel.loc[panel.drug == "ertapenem", "n_determinants"].iloc[0] == 168


def test_declaring_deployment_holdout_over_kfold_data_is_refused(tmp_path):
    """A non-zero spread contradicts "fit once", so the mislabel must fail loudly.

    This is the check that stops the TB ceiling being relabelled `current` to make a comparison look
    like-for-like when it is not — the single most consequential way this panel could go wrong.
    """
    root = tmp_path / "card_ceiling"
    _card_csv(root / "rifampin" / "card_determinant_lr_rifampin_allele.csv", "rifampin", sd=0.0025)
    with pytest.raises(ValueError, match="cannot have a spread"):
        B.build_panel(
            B.discover_card(root, "allele"), schema_key="card", grain="allele",
            estimator="deployment_holdout", status="current",
        )


def test_kfold_probe_accepts_a_spread_and_is_marked_provisional(tmp_path):
    root = tmp_path / "tbprofiler_gene_lr"
    _who_csv(root / "tbprofiler_gene_lr_rifampin.csv", "rifampin")

    panel = B.build_panel(
        B.discover_who(root), schema_key="who", grain="one_hot",
        estimator="kfold_probe", status="provisional",
    )
    assert panel.ceiling_auroc_sd.iloc[0] > 0
    assert panel.ceiling_estimator.iloc[0] == "kfold_probe"
    assert panel.n_determinants.iloc[0] == 149  # read from n_variants, the TB schema's name


def test_a_drug_with_no_all_row_is_skipped_not_zeroed(tmp_path, caplog):
    """TB is genuinely missing rifabutin. An absent drug must be absent, never a zero row."""
    root = tmp_path / "card_ceiling"
    _card_csv(root / "ertapenem" / "card_determinant_lr_ertapenem_allele.csv", "ertapenem")
    empty = root / "rifabutin" / "card_determinant_lr_rifabutin_allele.csv"
    empty.parent.mkdir(parents=True)
    pd.DataFrame([{"gene_name": "someGene", "site": "s", "category": "gene", "mut_auroc": 0.5,
                   "mut_auroc_sd": 0.0, "mut_auprc": 0.5, "mut_auprc_sd": 0.0,
                   "n_determinants": 1, "n_genomes_with_determinant": 2, "embeddable": True,
                   "is_causal": False, "is_rrna": False, "is_noncoding": False}]).to_csv(empty, index=False)

    panel = B.build_panel(
        B.discover_card(root, "allele"), schema_key="card", grain="allele",
        estimator="deployment_holdout", status="current",
    )
    assert list(panel.drug) == ["ertapenem"]


def test_the_who_manifest_is_not_mistaken_for_a_drug(tmp_path):
    root = tmp_path / "tbprofiler_gene_lr"
    _who_csv(root / "tbprofiler_gene_lr_rifampin.csv", "rifampin")
    (root / "tbprofiler_gene_lr_manifest.json").write_text("{}")
    assert sorted(B.discover_who(root)) == ["rifampin"]


def test_an_unknown_estimator_is_refused(tmp_path):
    root = tmp_path / "card_ceiling"
    _card_csv(root / "ertapenem" / "card_determinant_lr_ertapenem_allele.csv", "ertapenem")
    with pytest.raises(ValueError, match="estimator must be one of"):
        B.build_panel(
            B.discover_card(root, "allele"), schema_key="card", grain="allele",
            estimator="whatever_makes_it_pass", status="current",
        )


def test_who_discovery_prefers_the_nested_layout(tmp_path):
    """The rebuilt TB ceiling mirrors CARD: <dir>/<drug>/tbprofiler_gene_lr_<drug>.csv."""
    root = tmp_path / "who_ceiling"
    for drug in ("rifampin", "rifabutin"):
        _who_csv(root / drug / f"tbprofiler_gene_lr_{drug}.csv", drug)
    assert sorted(B.discover_who(root)) == ["rifabutin", "rifampin"]


def test_who_discovery_still_reads_the_flat_june_layout(tmp_path):
    """The retired probe wrote flat, its files are still on disk, and archived plot code reads them."""
    root = tmp_path / "tbprofiler_gene_lr"
    _who_csv(root / "tbprofiler_gene_lr_rifampin.csv", "rifampin")
    assert sorted(B.discover_who(root)) == ["rifampin"]


def test_a_nested_ceiling_still_cannot_be_mislabelled_as_the_deployment_holdout(tmp_path):
    """Finding a file is not endorsing it — the estimator check is on the data, not the layout.

    Otherwise moving the June CSVs into the new tree would launder them into looking current, which
    is the exact edit PROVENANCE.md warns against.
    """
    root = tmp_path / "who_ceiling"
    _who_csv(root / "rifampin" / "tbprofiler_gene_lr_rifampin.csv", "rifampin")  # non-zero sd
    with pytest.raises(ValueError, match="cannot have a spread"):
        B.build_panel(
            B.discover_who(root), schema_key="who", grain="one_hot",
            estimator="deployment_holdout", status="current",
        )


def test_a_stray_subdirectory_does_not_become_a_drug(tmp_path):
    """Nested discovery keys on <drug>/tbprofiler_gene_lr_<drug>.csv, so a logs/ dir is not a drug."""
    root = tmp_path / "who_ceiling"
    _who_csv(root / "rifampin" / "tbprofiler_gene_lr_rifampin.csv", "rifampin")
    (root / "logs").mkdir()
    (root / "logs" / "tbprofiler_gene_lr_isoniazid.csv").write_text("junk")
    assert sorted(B.discover_who(root)) == ["rifampin"]

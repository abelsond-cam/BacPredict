"""CABBAGE v2 → portal schema, and the v2 table labelled through the unchanged portal path."""

from __future__ import annotations

import pandas as pd
import pytest

from bacpredict.engine.ast_labels.build_binary_ast_labels import build_binary_ast_labels
from bacpredict.engine.ast_labels.cabbage_v2_phenotype import V2_COLUMNS, cabbage_v2_as_portal


def _row(**kw: str) -> dict[str, str]:
    return kw


def _v2(rows: list[dict]) -> pd.DataFrame:
    base = dict.fromkeys(V2_COLUMNS, "")
    return pd.DataFrame([{**base, **r} for r in rows], dtype=str)


@pytest.fixture
def v2():
    return _v2([
        # no sign: read as "=" (David, 2026-11-21) — 16 > R 8 → R even though the paper said S
        _row(BioSample_ID="S1", genus="Pseudomonas", species="aeruginosa", Antibiotic_name="meropenem",
             measurement_value="16", measurement_unit="mg/l", phenotype="susceptible", database="puffin"),
        _row(BioSample_ID="S1", genus="Pseudomonas", species="aeruginosa", Antibiotic_name="Piperacillin/Tazobactam",
             measurement_sign="<=", measurement_value="4/4", measurement_unit="mg/l", phenotype="susceptible",
             database="NCBI|puffin", Assembly_ID="GCA_1", SRA_run="SRR1", Pubmed_ID="123"),
        # bare call, no value: the sign stays empty and the source's call is the label
        _row(BioSample_ID="S2", genus="Pseudomonas", species="aeruginosa", Antibiotic_name="meropenem",
             phenotype="resistant", database="puffin"),
        # disk zone with no sign: filled, but mm is not an MIC so the source's call stands
        _row(BioSample_ID="S3", genus="Pseudomonas", species="aeruginosa", Antibiotic_name="meropenem",
             measurement_value="12", measurement_unit="mm", laboratory_typing_method="disk diffusion",
             phenotype="resistant", database="NCBI"),
        # no epithet: the genus alone, which still resolves to the Enterobacterales breakpoints
        _row(BioSample_ID="S4", genus="Escherichia", species="", Antibiotic_name="ciprofloxacin",
             measurement_value="1", measurement_unit="mg/l", database="puffin"),
        _row(BioSample_ID="", genus="Pseudomonas", species="aeruginosa", Antibiotic_name="meropenem",
             measurement_value="1", measurement_unit="mg/l"),
    ])


def test_mapping(v2):
    out, counts = cabbage_v2_as_portal(v2)
    assert counts == {"n_rows_in": 6, "n_rows_out": 5, "n_dropped_no_biosample": 1, "n_sign_filled_as_equals": 3,
                      "n_species_genus_only": 1}
    out = out.set_index(["BioSample_ID", "antibiotic_name"])
    assert out.loc[("S1", "meropenem"), "measurement_sign"] == "="
    assert out.loc[("S1", "piperacillin-tazobactam"), "database"] == "NCBI;puffin"
    assert out.loc[("S1", "piperacillin-tazobactam"), "SRA_accession"] == "SRR1"
    assert out.loc[("S1", "piperacillin-tazobactam"), "AMR_associated_publications"] == "123"
    assert pd.isna(out.loc[("S2", "meropenem"), "measurement_sign"])
    assert out.loc[("S4", "ciprofloxacin"), "species"] == "Escherichia"


def test_v2_labels_through_the_portal_path(v2, tmp_path):
    out, _ = cabbage_v2_as_portal(v2)
    out.to_parquet(tmp_path / "phenotype.parquet", index=False)
    res = build_binary_ast_labels(
        phenotype=tmp_path / "phenotype.parquet", species=["Pseudomonas aeruginosa"],
        drugs=["meropenem", "piperacillin-tazobactam"], portal_source="cabbage_v2", output_dir=tmp_path / "pa")
    b = res["binary"]
    assert b.loc["S1", "meropenem"] == 1.0 and b.loc["S1", "piperacillin-tazobactam"] == 0.0
    assert b.loc["S2", "meropenem"] == 1.0 and b.loc["S3", "meropenem"] == 1.0
    assert res["cohort"].loc["S1", "genome_route"] == "assembly"
    assert set(res["called"]["source"]) == {"cabbage_v2"}

    ent = build_binary_ast_labels(phenotype=tmp_path / "phenotype.parquet", genera=["Escherichia"],
                                  drugs=["ciprofloxacin"], portal_source="cabbage_v2", output_dir=tmp_path / "ent")
    assert ent["binary"].loc["S4", "ciprofloxacin"] == 1.0   # Enterobacterales ciprofloxacin R > 0.5

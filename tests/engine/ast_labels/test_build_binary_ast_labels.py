"""The per-test labelling rule, per-genome aggregation, and the end-to-end build on a toy portal release."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from bacpredict.engine.ast_labels.ast_tests import TEST_COLUMNS, aggregate_labels, call_agreement, call_tests
from bacpredict.engine.ast_labels.breakpoints import load_breakpoints
from bacpredict.engine.ast_labels.build_binary_ast_labels import build_binary_ast_labels

PA = "Pseudomonas aeruginosa"


def _test(sample, drug, sign=None, value=np.nan, units="mg/L", call=None, source="portal", species=PA,
          upper=np.nan, reads=None, st=None, asm=None, method=None):
    row = dict.fromkeys(TEST_COLUMNS, pd.NA)
    row.update(Sample=sample, species=species, drug=drug, source=source, mic_sign=sign, mic_value=value,
               mic_upper=upper, mic_units=units if not np.isnan(value) else None, source_call=call,
               reads_accession=reads, st_reported=st, assembly_phenotype=asm, method=method)
    return row


@pytest.fixture(scope="module")
def bp():
    return load_breakpoints()


def test_mic_beats_the_source_call_and_disk_keeps_it(bp):
    """A CLSI 'S' with MIC 16 meropenem becomes R; a disk test keeps its own call; I gives no call."""
    t = pd.DataFrame([
        _test("a", "meropenem", "=", 16, call="susceptible"),
        _test("b", "meropenem", "=", 22, units="mm", call="resistant"),   # disk zone — not an MIC
        _test("c", "meropenem", call="intermediate"),
        _test("d", "meropenem", "<=", 16, call="susceptible"),          # straddles R > 8 — not rescued
        _test("e", "gentamicin", "=", 4, call="resistant"),              # no breakpoint row
    ])
    c = call_tests(t, bp).set_index("Sample")
    assert c.loc["a", "y_test"] == 1.0 and c.loc["a", "basis"] == "mic"
    assert c.loc["b", "y_test"] == 1.0 and c.loc["b", "basis"] == "source_call"
    assert np.isnan(c.loc["c", "y_test"]) and c.loc["c", "basis"] == "no_call"
    assert np.isnan(c.loc["d", "y_test"]) and c.loc["d", "basis"] == "mic_undecided"
    assert np.isnan(c.loc["e", "y_test"]) and c.loc["e", "basis"] == "no_breakpoint"


def test_unitless_measurement_is_flagged_and_only_called_as_mic_on_request(bp):
    """No unit (CDC AR Bank style): by default the source's call stands; opt-in calls it as mg/L; disk excluded."""
    t = pd.DataFrame([
        _test("a", "meropenem", "=", 8, units=None, call="resistant"),           # CLSI R; EUCAST S (R > 8)
        _test("b", "meropenem", "=", 22, units=None, call="resistant", method="disk diffusion"),
    ])
    off = call_tests(t, bp).set_index("Sample")
    assert off.loc["a", "mic_units_missing"] and not off.loc["b", "mic_units_missing"]
    assert off.loc["a", "y_test"] == 1.0 and off.loc["a", "basis"] == "source_call"
    on = call_tests(t, bp, unitless_mic_as_mg_l=True).set_index("Sample")
    assert on.loc["a", "y_test"] == 0.0 and on.loc["a", "basis"] == "mic"
    assert on.loc["b", "y_test"] == 1.0 and on.loc["b", "basis"] == "source_call"
    assert aggregate_labels(call_tests(t, bp)).set_index("Sample").loc["a", "n_mic_units_missing"] == 1


def test_call_agreement_is_empty_not_a_crash_when_nothing_carries_both_calls(bp):
    t = pd.DataFrame([_test("a", "meropenem", "=", 4, species="Acinetobacter baumannii", call="susceptible")])
    out = call_agreement(call_tests(t, bp))
    assert out.empty and list(out.columns) == ["source", "drug", "n_both", "agree", "src_S_to_R", "src_R_to_S"]


def test_aggregation_averages_and_flags_conflict(bp):
    t = pd.DataFrame([
        _test("a", "meropenem", "=", 16), _test("a", "meropenem", "=", 32, source="paper"),  # R + R
        _test("b", "meropenem", "=", 16), _test("b", "meropenem", "=", 2, source="paper"),   # R + S
        _test("c", "meropenem", "=", 2), _test("c", "meropenem", call="susceptible"),        # mixed basis
        _test("d", "meropenem", "<=", 16),                                                     # undecided only
    ])
    a = aggregate_labels(call_tests(t, bp)).set_index("Sample")
    assert a.loc["a", "y"] == 1.0 and not a.loc["a", "conflict"] and a.loc["a", "sources"] == "paper;portal"
    assert a.loc["b", "y"] == 0.5 and a.loc["b", "conflict"]
    assert a.loc["c", "label_from"] == "mixed" and a.loc["c", "y"] == 0.0
    assert np.isnan(a.loc["d", "y"]) and a.loc["d", "n_called"] == 0 and a.loc["d", "n_mic_undecided"] == 1


def _toy_release(tmp_path):
    pheno = pd.DataFrame({
        "BioSample_ID": ["S1", "S1", "S2", "S3", "S4", "S5"],
        "species": [PA, PA, PA, PA, "Escherichia coli", PA],
        "antibiotic_name": ["meropenem", "ciprofloxacin", "meropenem", "meropenem", "meropenem", "colistin"],
        "measurement": ["16", "0.25", None, "4", "16", "4"],
        "measurement_sign": ["==", "<=", None, "=", "=", "="],
        "measurement_units": ["mg/L", "mg/L", None, "mg/L", "mg/L", "mg/L"],
        "laboratory_typing_method": ["broth dilution"] * 2 + [None] + ["agar dilution"] * 3,
        "platform": [None] * 6, "ast_standard": ["CLSI"] * 6,
        "resistance_phenotype": ["susceptible", "susceptible", "resistant", "susceptible", "resistant", "resistant"],
        "database": ["NCBI_antibiogram"] * 6,
        "assembly_ID": ["GCA_1", "GCA_1", None, None, "GCA_4", None],
        "SRA_accession": ["SRS1", "SRS1", "SRR2", None, None, None],
        "AMR_associated_publications": [None] * 6,
    })
    pheno.to_parquet(tmp_path / "phenotype.parquet")
    pd.DataFrame({"BioSample_ID": ["S3", "S3", "S1"], "assembly_ID": ["ERZ3", "ERZ3", "GCA_1"],
                  "species": [PA, PA, "Pseudomonas paraeruginosa"]}).to_parquet(tmp_path / "genotype.parquet")
    extra = pd.DataFrame([_test("S1", "meropenem", "=", 32, source="paper", reads="SRR1", st="235"),
                          _test("S9", "meropenem", ">", 8, source="paper", reads="SRR9", st="175")])
    extra.to_csv(tmp_path / "extra.csv", index=False)
    return tmp_path


def test_end_to_end_build(tmp_path):
    root = _toy_release(tmp_path)
    out = build_binary_ast_labels(
        phenotype=root / "phenotype.parquet", genotype=root / "genotype.parquet",
        extra_tests=[root / "extra.csv"], species=[PA], drugs=["meropenem", "ciprofloxacin"],
        output_dir=root / "out")
    b = pd.read_csv(root / "out" / "binary_ast.csv")
    assert b.columns[0] == "phenotype-BioSample_ID" and list(b.columns[1:]) == ["meropenem", "ciprofloxacin"]
    b = b.set_index("phenotype-BioSample_ID")
    assert set(b.index) == {"S1", "S2", "S3", "S9"}            # E. coli S4 and colistin-only S5 excluded
    assert b.loc["S1", "meropenem"] == 1.0 and b.loc["S1", "ciprofloxacin"] == 0.0
    assert b.loc["S2", "meropenem"] == 1.0                      # bare call kept (no MIC)
    assert b.loc["S3", "meropenem"] == 0.0                      # MIC 4 ≤ 8, CLSI call agrees
    assert b.loc["S9", "meropenem"] == 1.0
    cohort = out["cohort"]
    assert cohort.loc["S1", "genome_route"] == "assembly" and cohort.loc["S1", "st_reported"] == "235"
    assert cohort.loc["S1", "reads_accession"] == "SRR1"        # the paper's run beats the portal's sample acc
    assert cohort.loc["S1", "species_conflict"] and not cohort.loc["S3", "species_conflict"]
    assert cohort.loc["S2", "genome_route"] == "reads" and cohort.loc["S3", "assembly_genotype"] == "ERZ3"
    assert cohort.loc["S9", "sources"] == "paper" and cohort.loc["S1", "sources"] == "ebi_amr_portal;paper"
    counts = out["counts"].set_index("drug")
    assert counts.loc["meropenem", "n_labelled"] == 4 and counts.loc["meropenem", "n_only_paper"] == 1
    manifest = json.loads((root / "out" / "manifest.json").read_text())
    assert manifest["n_genomes_in_binary_ast"] == 4 and "breakpoints" in manifest


def test_build_writes_outputs_for_a_taxon_without_breakpoints(tmp_path):
    """Every MIC is no_breakpoint: the build must still write its counts (it used to crash in call_agreement)."""
    root = _toy_release(tmp_path)
    out = build_binary_ast_labels(phenotype=root / "phenotype.parquet", species=[PA], drugs=["colistin"],
                                  output_dir=root / "o")
    assert (root / "o" / "label_counts.csv").exists() and out["agreement"].empty
    assert out["counts"].set_index("drug").loc["colistin", "n_no_call_no_breakpoint"] == 1


def test_build_refuses_an_all_species_run(tmp_path):
    root = _toy_release(tmp_path)
    with pytest.raises(ValueError, match="species"):
        build_binary_ast_labels(phenotype=root / "phenotype.parquet", drugs=["meropenem"], output_dir=root / "o")

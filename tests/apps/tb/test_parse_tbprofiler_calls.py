"""Parsing the TB-Profiler call set is only legitimate if its provenance is recorded.

A ``results.json`` is a deterministic function of one assembly and one catalogue version, which is the
argument for re-parsing an existing call set rather than re-running TB-Profiler over 36k assemblies. The
argument collapses if the catalogue version is not actually checked — so these tests pin the three ways
the call set can be silently wrong: a mixed catalogue, a file that would not parse, and a cohort sample
that has no call at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bacpredict.apps.tb.parse_tbprofiler_calls import MANIFEST_NAME, UNCOVERED_NAME, load_cohort, run

DRUGS = ["rifampin", "isoniazid", "rifabutin"]


def _results_json(sample: str, *, drugs=(), commit="7fe4364e", sub_lineage="lineage4.3.4.2") -> dict:
    """One TB-Profiler results JSON, shaped like the real 6.7.0 / who_v2+ output."""
    return {
        "id": sample,
        "drtype": "MDR-TB" if drugs else "Sensitive",
        "main_lineage": sub_lineage.split(".")[0] if sub_lineage else None,
        "sub_lineage": sub_lineage,
        "schema_version": "1.0.0",
        "pipeline": {
            "software_version": "6.7.0",
            "db_version": {"name": "who_v2+", "commit": commit, "db-schema-version": "2.1.0"},
        },
        "dr_variants": [
            {"gene_name": "rpoB", "change": "p.Ser450Leu", "type": "missense_variant",
             "drugs": [{"drug": d} for d in drugs]}
        ] if drugs else [],
    }


def _cohort_dir(tmp_path: Path, samples: dict[str, dict]) -> Path:
    results = tmp_path / "results"
    results.mkdir(parents=True, exist_ok=True)
    for sample, payload in samples.items():
        (results / f"{sample}.results.json").write_text(json.dumps(payload))
    return results


def test_the_manifest_records_the_catalogue_that_produced_the_calls(tmp_path: Path) -> None:
    """Without this the 'deterministic function of assembly + catalogue' argument is unverifiable."""
    results = _cohort_dir(tmp_path, {
        "SAMEA1": _results_json("SAMEA1", drugs=["rifampicin"]),
        "SAMEA2": _results_json("SAMEA2"),
    })
    manifest = run(results, tmp_path / "out", DRUGS)
    assert manifest["catalogue_commits"] == ["7fe4364e"]
    (prov,) = manifest["provenance"]
    assert prov["db_name"] == "who_v2+"
    assert prov["software_version"] == "6.7.0"
    assert prov["db_schema_version"] == "2.1.0"
    assert prov["n_genomes"] == 2
    assert manifest["source_results_dir"] == str(results)
    on_disk = json.loads((tmp_path / "out" / MANIFEST_NAME).read_text())
    assert on_disk == manifest


def test_a_mixed_catalogue_is_refused(tmp_path: Path) -> None:
    """Two catalogue commits mean two variant vocabularies, and nothing downstream could tell."""
    results = _cohort_dir(tmp_path, {
        "SAMEA1": _results_json("SAMEA1", drugs=["rifampicin"], commit="7fe4364e"),
        "SAMEA2": _results_json("SAMEA2", drugs=["rifampicin"], commit="deadbeef"),
    })
    with pytest.raises(SystemExit) as e:
        run(results, tmp_path / "out", DRUGS)
    assert "catalogue commits" in str(e.value)


def test_a_mixed_catalogue_can_be_allowed_deliberately(tmp_path: Path) -> None:
    results = _cohort_dir(tmp_path, {
        "SAMEA1": _results_json("SAMEA1", commit="7fe4364e"),
        "SAMEA2": _results_json("SAMEA2", commit="deadbeef"),
    })
    manifest = run(results, tmp_path / "out", DRUGS, allow_mixed_catalogue=True)
    assert manifest["catalogue_commits"] == ["7fe4364e", "deadbeef"]
    assert len(manifest["provenance"]) == 2


def test_an_unparseable_file_is_fatal_and_named_not_logged_away(tmp_path: Path) -> None:
    """It was previously a logger.warning and a `continue` — a truncated call set with no trace."""
    results = _cohort_dir(tmp_path, {"SAMEA1": _results_json("SAMEA1")})
    (results / "SAMEA2.results.json").write_text('{"id": "SAMEA2", "dr_var')  # truncated
    with pytest.raises(SystemExit) as e:
        run(results, tmp_path / "out", DRUGS)
    assert "could not be parsed" in str(e.value)
    listed = (tmp_path / "out" / "tbprofiler_unparseable_files.txt").read_text()
    assert "SAMEA2.results.json" in listed


def test_unparseable_files_can_be_tolerated_explicitly(tmp_path: Path) -> None:
    results = _cohort_dir(tmp_path, {"SAMEA1": _results_json("SAMEA1")})
    (results / "SAMEA2.results.json").write_text("not json at all")
    manifest = run(results, tmp_path / "out", DRUGS, max_unparseable=1)
    assert manifest["n_unparseable"] == 1
    assert manifest["n_genomes"] == 1
    assert manifest["n_result_files"] == 2, "the file count must still report what was on disk"


def test_cohort_samples_with_no_call_are_named(tmp_path: Path) -> None:
    """The 69 uncovered TB samples must be excluded *and* enumerated, not silently absent."""
    results = _cohort_dir(tmp_path, {
        "SAMEA1": _results_json("SAMEA1"),
        "SAMEA2": _results_json("SAMEA2"),
    })
    cohort = tmp_path / "binary_ast_with_split.csv"
    pd.DataFrame({"Sample": ["SAMEA1", "SAMEA2", "SAMEA3"], "rifampin": [1, 0, 1]}).to_csv(cohort, index=False)
    manifest = run(results, tmp_path / "out", DRUGS, cohort_csv=cohort)
    cov = manifest["coverage"]
    assert cov["n_cohort"] == 3
    assert cov["n_cohort_with_calls"] == 2
    assert cov["n_cohort_uncovered"] == 1
    assert (tmp_path / "out" / UNCOVERED_NAME).read_text() == "SAMEA3\n"


def test_calls_outside_the_cohort_are_counted_not_conflated(tmp_path: Path) -> None:
    """TB-Profiler ran over a superset; that is fine, but it is a different number from coverage."""
    results = _cohort_dir(tmp_path, {
        "SAMEA1": _results_json("SAMEA1"),
        "SAMEA9": _results_json("SAMEA9"),
    })
    cohort = tmp_path / "c.csv"
    pd.DataFrame({"Sample": ["SAMEA1"]}).to_csv(cohort, index=False)
    manifest = run(results, tmp_path / "out", DRUGS, cohort_csv=cohort)
    assert manifest["coverage"]["n_cohort_uncovered"] == 0
    assert manifest["coverage"]["n_called_outside_cohort"] == 1


def test_a_drug_the_catalogue_never_calls_is_flagged(tmp_path: Path) -> None:
    """rifabutin is the real case: its resistance is recorded under rifampicin, so it gets zero calls
    and cannot support a WHO ceiling. That must be visible, not inferred from a zero in a table."""
    results = _cohort_dir(tmp_path, {"SAMEA1": _results_json("SAMEA1", drugs=["rifampicin", "isoniazid"])})
    manifest = run(results, tmp_path / "out", DRUGS)
    assert manifest["per_drug_resistant"] == {"rifampin": 1, "isoniazid": 1, "rifabutin": 0}
    assert manifest["drugs_with_no_calls"] == ["rifabutin"]


def test_rifampicin_normalises_to_our_drug_name(tmp_path: Path) -> None:
    """`rifampicin` vs `rifampin` silently dropping the headline drug is a documented trap."""
    results = _cohort_dir(tmp_path, {"SAMEA1": _results_json("SAMEA1", drugs=["rifampicin"])})
    run(results, tmp_path / "out", DRUGS)
    variants = pd.read_parquet(tmp_path / "out" / "tbprofiler_variants.parquet")
    assert variants["drugs"].tolist() == ["rifampin"]


def test_the_downstream_schemas_are_unchanged(tmp_path: Path) -> None:
    """tbprofiler_gene_lr reads the parquet and tb_lineage_from_tbprofiler reads the csv; adding a
    manifest must not have moved either."""
    results = _cohort_dir(tmp_path, {"SAMEA1": _results_json("SAMEA1", drugs=["rifampicin"])})
    run(results, tmp_path / "out", DRUGS)
    variants = pd.read_parquet(tmp_path / "out" / "tbprofiler_variants.parquet")
    assert set(variants.columns) == {"Sample", "variant_id", "gene_name", "change", "type", "drugs"}
    assert variants["variant_id"].tolist() == ["rpoB@p.Ser450Leu"]
    lineage = pd.read_csv(tmp_path / "out" / "tbprofiler_lineage.csv")
    assert {"Sample", "main_lineage", "sub_lineage"} <= set(lineage.columns)
    native = pd.read_parquet(tmp_path / "out" / "tbprofiler_native_calls.parquet")
    assert set(DRUGS) <= set(native.columns) and "drtype" in native.columns


def test_uncalled_lineages_are_counted_separately_from_genomes(tmp_path: Path) -> None:
    results = _cohort_dir(tmp_path, {
        "SAMEA1": _results_json("SAMEA1", sub_lineage="lineage4.3.4.2"),
        "SAMEA2": _results_json("SAMEA2", sub_lineage=None),
    })
    manifest = run(results, tmp_path / "out", DRUGS)
    assert manifest["n_genomes"] == 2
    assert manifest["n_with_sub_lineage"] == 1


def test_load_cohort_accepts_a_csv_or_a_reflist(tmp_path: Path) -> None:
    csv = tmp_path / "a.csv"
    pd.DataFrame({"Sample": ["S1", "S2"], "rifampin": [1, 0]}).to_csv(csv, index=False)
    reflist = tmp_path / "b.tsv"
    reflist.write_text("S1\t/data/S1.fa.gz\nS2\t/data/S2.fa.gz\n")
    assert load_cohort(csv) == load_cohort(reflist) == ["S1", "S2"]


def test_an_empty_results_dir_is_an_error(tmp_path: Path) -> None:
    empty = tmp_path / "results"
    empty.mkdir()
    with pytest.raises(RuntimeError):
        run(empty, tmp_path / "out", DRUGS)

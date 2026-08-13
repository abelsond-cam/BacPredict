"""Tests for taking lineage clusters from curated Kleborate Sublineage labels."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bac_pyseer.ast_gwas.lineage_from_distances import OTHER
from bac_pyseer.ast_gwas.sublineage_from_metadata import assign_clusters, load_sublineages
from bac_pyseer.ast_gwas.sublineage_from_metadata import run as sublineage_run


def _write_metadata(path: Path, rows: dict[str, str | None]) -> None:
    """metadata_v2 is a TSV keyed on Sample with a Sublineage column."""
    pd.DataFrame({"Sample": list(rows), "Sublineage": list(rows.values()), "ST": ["x"] * len(rows)}).to_csv(
        path, sep="\t", index=False
    )


def _write_reflist(path: Path, samples: list[str]) -> None:
    path.write_text("".join(f"{s}\t/data/{s}.fa.gz\n" for s in samples))


def test_load_sublineages_drops_placeholder_labels(tmp_path: Path) -> None:
    """Blank/NA/unknown are absences, not lineages — they must not become their own cluster."""
    meta = tmp_path / "metadata_v2.tsv"
    _write_metadata(meta, {"A": "SL258", "B": "unknown", "C": None, "D": "  SL307  "})
    got = load_sublineages(meta)
    assert got == {"A": "SL258", "D": "SL307"}  # whitespace stripped


def test_rare_sublineages_and_unlabelled_collapse_to_other(tmp_path: Path) -> None:
    """A permutation null cannot shuffle inside a cluster of one, so small ones join 'other'."""
    samples = [f"s{i}" for i in range(12)]
    sublineage_of = dict.fromkeys(samples[:10], "SL258")
    sublineage_of["s10"] = "SL999"  # rare
    # s11 has no label at all
    clusters = assign_clusters(samples, sublineage_of, min_size=5)
    assert clusters["s0"] == "SL258"
    assert clusters["s10"] == OTHER
    assert clusters["s11"] == OTHER


def test_sizes_are_counted_over_the_cohort_not_the_whole_sheet(tmp_path: Path) -> None:
    """A species-wide-common sublineage that is rare in THIS cohort still cannot support a null."""
    sublineage_of = {"a": "SL258", "b": "SL307", "c": "SL307", "d": "SL307"}
    clusters = assign_clusters(["a", "b", "c", "d"], sublineage_of, min_size=3)
    assert clusters["a"] == OTHER  # SL258 is huge species-wide, but n=1 here
    assert clusters["b"] == "SL307"


def test_run_writes_headerless_two_column_file_for_pyseer(tmp_path: Path) -> None:
    """Output must be drop-in interchangeable with lineage_from_distances' file."""
    samples = [f"s{i}" for i in range(10)]
    meta = tmp_path / "metadata_v2.tsv"
    _write_metadata(meta, dict.fromkeys(samples, "SL258"))
    reflist = tmp_path / "assembly_refs.txt"
    _write_reflist(reflist, samples)

    out = tmp_path / "lineage_clusters.tsv"
    manifest = sublineage_run(reflist=reflist, metadata_tsv=meta, out_tsv=out, min_size=5)

    rows = [line.split("\t") for line in out.read_text().splitlines()]
    assert len(rows) == 10
    assert all(len(r) == 2 for r in rows)
    assert manifest["n_clusters"] == 1
    assert manifest["coverage"] == 1.0
    assert json.loads(out.with_suffix(".manifest.json").read_text())["min_size"] == 5


def test_every_cohort_sample_gets_a_row_even_when_unlabelled(tmp_path: Path) -> None:
    """pyseer needs a cluster for every phenotyped sample; a missing label must not drop the row."""
    meta = tmp_path / "metadata_v2.tsv"
    _write_metadata(meta, {"a": "SL258", "b": "SL258"})
    reflist = tmp_path / "refs.txt"
    _write_reflist(reflist, ["a", "b", "c"])

    out = tmp_path / "clusters.tsv"
    manifest = sublineage_run(reflist=reflist, metadata_tsv=meta, out_tsv=out, min_size=1, min_coverage=0.5)
    assert dict(line.split("\t") for line in out.read_text().splitlines())["c"] == OTHER
    assert manifest["n_samples"] == 3
    assert manifest["n_with_label"] == 2


def test_a_failed_join_is_an_error_not_an_all_other_file(tmp_path: Path) -> None:
    """Wrong id column or wrong sheet must fail loudly rather than emit a uniform 'other' file."""
    meta = tmp_path / "metadata_v2.tsv"
    _write_metadata(meta, {"x": "SL258", "y": "SL307"})
    reflist = tmp_path / "refs.txt"
    _write_reflist(reflist, ["a", "b", "c"])
    with pytest.raises(SystemExit, match="min-coverage"):
        sublineage_run(reflist=reflist, metadata_tsv=meta, out_tsv=tmp_path / "c.tsv")

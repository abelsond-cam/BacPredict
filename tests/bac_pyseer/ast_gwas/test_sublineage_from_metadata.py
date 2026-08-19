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
    got, recovered = load_sublineages(meta)
    assert got == {"A": "SL258", "D": "SL307"}  # whitespace stripped
    assert recovered == {}  # nothing needed a fallback id here


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
    assert manifest["label_coverage"] == 1.0
    assert manifest["named_cluster_coverage"] == 1.0
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


def _write_metadata_with_alt_id(path: Path, rows: list[tuple[str | None, str, str | None]]) -> None:
    """metadata_v2 as it really is: some rows keyed by BioSample, some by GCA accession."""
    pd.DataFrame({
        "Sample": [r[0] for r in rows],
        "Sublineage": [r[1] for r in rows],
        "sample_accession": [r[2] for r in rows],
    }).to_csv(path, sep="\t", index=False)


def test_long_read_genomes_are_recovered_through_a_fallback_id(tmp_path: Path) -> None:
    """The BioSample key misses long-read genomes, which were deposited under a GCA accession.

    Those rows exist and carry a Sublineage; keying only on ``Sample`` drops them into ``other``.
    They are also the best-assembled genomes in the cohort, so losing them biases the clusters
    toward draft assemblies.
    """
    meta = tmp_path / "metadata_v2.tsv"
    _write_metadata_with_alt_id(meta, [
        ("SAMN0001", "SL258", "SAMN0001"),        # short read: keyed by BioSample
        ("GCA_00001.1", "SL307", "SAMN0002"),     # long read: Sample is a GCA accession
    ])
    got, recovered = load_sublineages(meta)
    assert got["SAMN0001"] == "SL258"
    assert got["SAMN0002"] == "SL307", "the GCA-keyed row must be reachable by its BioSample"
    assert recovered == {"sample_accession": {"SAMN0002"}}


def test_the_primary_key_wins_over_a_fallback(tmp_path: Path) -> None:
    """A fallback must never override a row that the primary key already resolved."""
    meta = tmp_path / "metadata_v2.tsv"
    _write_metadata_with_alt_id(meta, [
        ("SAMN0001", "SL258", "SAMN0001"),
        ("GCA_00009.1", "SL999", "SAMN0001"),  # same BioSample, different label, via fallback
    ])
    got, recovered = load_sublineages(meta)
    assert got["SAMN0001"] == "SL258"
    assert recovered == {}


def test_an_ambiguous_fallback_is_dropped_not_guessed(tmp_path: Path) -> None:
    """Two sublineages for one id is worse than none — it corrupts the null silently."""
    meta = tmp_path / "metadata_v2.tsv"
    _write_metadata_with_alt_id(meta, [
        ("GCA_00001.1", "SL258", "SAMN0002"),
        ("GCA_00002.1", "SL307", "SAMN0002"),  # same BioSample, conflicting labels
    ])
    got, recovered = load_sublineages(meta)
    assert "SAMN0002" not in got
    assert recovered == {}


def test_recovered_genomes_are_counted_in_the_manifest(tmp_path: Path) -> None:
    """The recovery must be visible as a number, or a change in keying returns silently."""
    meta = tmp_path / "metadata_v2.tsv"
    _write_metadata_with_alt_id(meta, [
        ("SAMN0001", "SL258", "SAMN0001"), ("SAMN0002", "SL258", "SAMN0002"),
        ("GCA_00001.1", "SL258", "SAMN0003"),
    ])
    reflist = tmp_path / "refs.txt"
    _write_reflist(reflist, ["SAMN0001", "SAMN0002", "SAMN0003"])
    manifest = sublineage_run(
        reflist=reflist, metadata_tsv=meta, out_tsv=tmp_path / "c.tsv", min_size=3
    )
    assert manifest["n_recovered_via_fallback_id"] == 1
    assert manifest["recovered_by_column"] == {"sample_accession": 1}
    # Without the recovery SL258 would be n=2 and collapse to `other` at min_size=3.
    assert manifest["n_clusters"] == 1
    assert manifest["n_in_other"] == 0


def _write_metadata_st(path: Path, rows: list[tuple[str, str | None, str | None]]) -> None:
    """metadata_v2 carries ST and Sublineage as SEPARATE columns, from different sources."""
    pd.DataFrame({
        "Sample": [r[0] for r in rows],
        "Sublineage": [r[1] for r in rows],
        "ST": [r[2] for r in rows],
    }).to_csv(path, sep="\t", index=False)


def test_st_clusters_are_labelled_st_and_never_renamed_to_sublineage(tmp_path: Path) -> None:
    """ST is NOT a sublineage and must never be presented as one.

    Sublineage comes from Pasteur BIGSdb LIN-typing — a specific algorithm. Kleborate has no
    LIN-coding module at all, so an ST cannot be converted into an SL by any rule, majority vote
    included. ST clustering is a stand-in while LIN-typing is pending, and the output has to say so
    or a downstream reader will quote ST clusters as sublineage clusters.
    """
    meta = tmp_path / "metadata_v2.tsv"
    _write_metadata_st(meta, [(f"s{i}", None, "ST258") for i in range(6)])
    reflist = tmp_path / "refs.txt"
    _write_reflist(reflist, [f"s{i}" for i in range(6)])

    manifest = sublineage_run(
        reflist=reflist, metadata_tsv=meta, out_tsv=tmp_path / "c.tsv",
        min_size=3, cluster_source="st",
    )
    assert manifest["cluster_source"] == "st"
    assert "NOT sublineage" in manifest["cluster_type"]
    assert manifest["label_column"] == "ST"

    labels = {line.split("\t")[1] for line in (tmp_path / "c.tsv").read_text().splitlines()}
    assert labels == {"ST258"}, "the ST string must survive verbatim"
    assert not any(lbl.upper().startswith("SL") for lbl in labels), (
        "an ST was emitted under a sublineage-looking label — ST and SL are different types"
    )


def test_sublineage_is_the_default_and_says_so(tmp_path: Path) -> None:
    """The method of record must stay the default; the stand-in has to be asked for explicitly."""
    meta = tmp_path / "metadata_v2.tsv"
    _write_metadata_st(meta, [(f"s{i}", "SL258", "ST258") for i in range(6)])
    reflist = tmp_path / "refs.txt"
    _write_reflist(reflist, [f"s{i}" for i in range(6)])

    manifest = sublineage_run(
        reflist=reflist, metadata_tsv=meta, out_tsv=tmp_path / "c.tsv", min_size=3
    )
    assert manifest["cluster_source"] == "sublineage"
    assert "BIGSdb" in manifest["cluster_type"]
    labels = {line.split("\t")[1] for line in (tmp_path / "c.tsv").read_text().splitlines()}
    assert labels == {"SL258"}


def test_choosing_st_does_not_silently_read_the_sublineage_column(tmp_path: Path) -> None:
    """Asking for ST must give ST, even when a Sublineage column is sitting right there."""
    meta = tmp_path / "metadata_v2.tsv"
    _write_metadata_st(meta, [(f"s{i}", "SL999", "ST258") for i in range(6)])
    reflist = tmp_path / "refs.txt"
    _write_reflist(reflist, [f"s{i}" for i in range(6)])

    sublineage_run(reflist=reflist, metadata_tsv=meta, out_tsv=tmp_path / "c.tsv",
                   min_size=3, cluster_source="st")
    labels = {line.split("\t")[1] for line in (tmp_path / "c.tsv").read_text().splitlines()}
    assert labels == {"ST258"} and "SL999" not in labels


def test_an_unknown_cluster_source_is_refused(tmp_path: Path) -> None:
    meta = tmp_path / "metadata_v2.tsv"
    _write_metadata_st(meta, [("s0", "SL258", "ST258")])
    reflist = tmp_path / "refs.txt"
    _write_reflist(reflist, ["s0"])
    with pytest.raises(SystemExit, match="cluster-source"):
        sublineage_run(reflist=reflist, metadata_tsv=meta, out_tsv=tmp_path / "c.tsv",
                       cluster_source="lincode_guess")


def _write_extra(path: Path, rows: list[dict[str, str]]) -> None:
    """Write an --extra-sublineage-tsv table in the shape genome_prep.lin_typing emits."""
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def test_extra_table_labels_genomes_metadata_has_no_row_for(tmp_path: Path) -> None:
    """The whole point: a genome absent from metadata_v2 gets its sublineage from the extra table."""
    samples = [f"m{i}" for i in range(3)] + [f"x{i}" for i in range(3)]
    meta = tmp_path / "metadata_v2.tsv"
    _write_metadata(meta, {f"m{i}": "SL258" for i in range(3)})
    reflist = tmp_path / "refs.txt"
    _write_reflist(reflist, samples)
    extra = tmp_path / "extra.tsv"
    _write_extra(extra, [{"Sample": f"x{i}", "Sublineage": "SL258", "passes_gate": "True"} for i in range(3)])

    out = tmp_path / "clusters.tsv"
    manifest = sublineage_run(
        reflist=reflist, metadata_tsv=meta, out_tsv=out, min_size=6,
        extra_sublineage_tsvs=[extra], min_coverage=0.0,
    )

    assert manifest["n_labels_added_from_extra"] == 3
    assert manifest["n_in_other"] == 0
    clusters = dict(line.split("\t") for line in out.read_text().splitlines())
    assert set(clusters.values()) == {"SL258"}


def test_extra_rows_failing_their_quality_gate_are_not_used(tmp_path: Path) -> None:
    """A nearest-profile call on a badly matched genome is a guess; it must not enter a cluster."""
    meta = tmp_path / "metadata_v2.tsv"
    _write_metadata(meta, {"a": "SL258", "b": "SL258"})
    reflist = tmp_path / "refs.txt"
    _write_reflist(reflist, ["a", "b", "c"])
    extra = tmp_path / "extra.tsv"
    _write_extra(extra, [{"Sample": "c", "Sublineage": "SL258", "passes_gate": "False"}])

    out = tmp_path / "clusters.tsv"
    manifest = sublineage_run(
        reflist=reflist, metadata_tsv=meta, out_tsv=out, min_size=2,
        extra_sublineage_tsvs=[extra], min_coverage=0.0,
    )

    assert manifest["n_labels_added_from_extra"] == 0
    assert manifest["n_extra_skipped_by_gate"] == 1
    assert dict(line.split("\t") for line in out.read_text().splitlines())["c"] == OTHER


def test_a_disagreement_between_sources_stops_the_run(tmp_path: Path) -> None:
    """Both sources claim the same algorithm, so a conflict means one is stale — never pick a winner."""
    meta = tmp_path / "metadata_v2.tsv"
    _write_metadata(meta, {"a": "SL258", "b": "SL307"})
    reflist = tmp_path / "refs.txt"
    _write_reflist(reflist, ["a", "b"])
    extra = tmp_path / "extra.tsv"
    _write_extra(extra, [{"Sample": "a", "Sublineage": "SL11", "passes_gate": "True"}])

    with pytest.raises(SystemExit, match="different"):
        sublineage_run(
            reflist=reflist, metadata_tsv=meta, out_tsv=tmp_path / "c.tsv", min_size=1,
            extra_sublineage_tsvs=[extra], min_coverage=0.0,
        )


def test_extra_sublineages_cannot_be_mixed_into_an_st_clustered_run(tmp_path: Path) -> None:
    """Blending LIN sublineages into ST clusters is the exact conflation --cluster-source prevents."""
    meta = tmp_path / "metadata_v2.tsv"
    pd.DataFrame({"Sample": ["a", "b"], "Sublineage": ["SL258", "SL258"], "ST": ["11", "11"]}).to_csv(
        meta, sep="\t", index=False
    )
    reflist = tmp_path / "refs.txt"
    _write_reflist(reflist, ["a", "b"])
    extra = tmp_path / "extra.tsv"
    _write_extra(extra, [{"Sample": "c", "Sublineage": "SL258", "passes_gate": "True"}])

    with pytest.raises(SystemExit, match="different label types"):
        sublineage_run(
            reflist=reflist, metadata_tsv=meta, out_tsv=tmp_path / "c.tsv", min_size=1,
            cluster_source="st", extra_sublineage_tsvs=[extra], min_coverage=0.0,
        )


def test_provenance_sidecar_names_where_each_label_came_from(tmp_path: Path) -> None:
    """Per-genome provenance, so a later reader can tell a v1-sheet label from a fresh LIN-typed one."""
    meta = tmp_path / "metadata_v2.tsv"
    _write_metadata(meta, {"a": "SL258", "b": "SL258"})
    reflist = tmp_path / "refs.txt"
    _write_reflist(reflist, ["a", "b", "c", "d"])
    extra = tmp_path / "mist_lin.tsv"
    _write_extra(extra, [{"Sample": "c", "Sublineage": "SL258", "passes_gate": "True"}])

    out = tmp_path / "clusters.tsv"
    sublineage_run(
        reflist=reflist, metadata_tsv=meta, out_tsv=out, min_size=1,
        extra_sublineage_tsvs=[extra], min_coverage=0.0,
    )

    rows = [line.split("\t") for line in out.with_suffix(".provenance.tsv").read_text().splitlines()[1:]]
    source = {r[0]: r[3] for r in rows}
    assert source == {"a": "metadata_v2", "b": "metadata_v2", "c": "mist_lin.tsv", "d": "none"}

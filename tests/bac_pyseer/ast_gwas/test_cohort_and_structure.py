"""Tests for the once-per-organism stages: assembly resolution and lineage clustering."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bac_pyseer.ast_gwas.lineage_from_distances import OTHER, cluster_distances
from bac_pyseer.ast_gwas.lineage_from_distances import run as cluster_run
from bac_pyseer.ast_gwas.resolve_ast_assemblies import (
    assemblies_dir,
    cohort_samples,
    load_file_list,
    resolve,
    resolve_via_file_list,
)
from bac_pyseer.ast_gwas.resolve_ast_assemblies import run as resolve_run


# --------------------------------------------------------------------------------------- #
# assembly resolution
# --------------------------------------------------------------------------------------- #
def test_assemblies_dir_uses_the_raw_naming_asymmetry(tmp_path: Path) -> None:
    """raw/kleb_ast vs raw/tb — the raw dirs disagree even though the processed ones do not."""
    assert assemblies_dir("kp", tmp_path) == tmp_path / "raw/kleb_ast/assemblies"
    assert assemblies_dir("tb", tmp_path) == tmp_path / "raw/tb/assemblies"


def test_resolve_finds_assemblies_and_reports_misses(tmp_path: Path) -> None:
    """Present assemblies resolve by suffix; absent ones are returned, not silently dropped."""
    asm = tmp_path / "assemblies"
    asm.mkdir()
    (asm / "SAMN1.fa.gz").write_text("")
    (asm / "SAMN2.fna").write_text("")
    resolved, missing = resolve(["SAMN1", "SAMN2", "SAMN3"], asm)
    assert [s for s, _ in resolved] == ["SAMN1", "SAMN2"]
    assert resolved[0][1].name == "SAMN1.fa.gz"
    assert missing == ["SAMN3"]


def test_resolve_run_writes_reflist_and_missing_list(tmp_path: Path) -> None:
    """The reflist is the Sample<TAB>path contract GGCAT -d and mash both consume."""
    asm = tmp_path / "assemblies"
    asm.mkdir()
    for s in ("SAMN1", "SAMN2"):
        (asm / f"{s}.fa.gz").write_text("")
    sheet = tmp_path / "binary_ast_with_split.csv"
    pd.DataFrame({"Sample": ["SAMN1", "SAMN2", "SAMN9"], "ertapenem": [1, 0, 1]}).to_csv(sheet, index=False)

    out = tmp_path / "assembly_refs.txt"
    manifest = resolve_run(organism_key="kp", out_tsv=out, ast_sheet=sheet, asm_dir=asm)

    lines = [line.split("\t") for line in out.read_text().splitlines()]
    assert [s for s, _ in lines] == ["SAMN1", "SAMN2"]
    assert manifest["n_resolved"] == 2
    assert manifest["n_missing"] == 1
    assert (tmp_path / "assembly_refs.missing.txt").read_text().split() == ["SAMN9"]


# --------------------------------------------------------------------------------------- #
# file-list resolution — CSD3's Kp layout, where there is no flat BioSample-keyed directory
# --------------------------------------------------------------------------------------- #
def _write_file_list(path: Path, rows: list[tuple[str, Path]], *, header: bool = True) -> None:
    """Write the Sample<TAB>path TSV CSD3 ships as raw/assemblies_file_list.tsv."""
    lines = ["Sample\tpath"] if header else []
    lines.extend(f"{s}\t{p}" for s, p in rows)
    path.write_text("\n".join(lines) + "\n")


def test_load_file_list_skips_the_header(tmp_path: Path) -> None:
    """CSD3's file list carries a `Sample<TAB>path` header; it must not become a sample."""
    listing = tmp_path / "assemblies_file_list.tsv"
    _write_file_list(listing, [("SAMEA1", tmp_path / "a.fa.gz"), ("SAMEA2", tmp_path / "b.fa.gz")])
    mapping = load_file_list(listing)
    assert set(mapping) == {"SAMEA1", "SAMEA2"}
    assert mapping["SAMEA1"] == tmp_path / "a.fa.gz"


def test_load_file_list_without_a_header_keeps_every_row(tmp_path: Path) -> None:
    """A headerless list is the format this module itself emits — round-trips cleanly."""
    listing = tmp_path / "refs.txt"
    _write_file_list(listing, [("SAMEA1", tmp_path / "a.fa.gz")], header=False)
    assert set(load_file_list(listing)) == {"SAMEA1"}


def test_load_file_list_rejects_an_empty_list(tmp_path: Path) -> None:
    """An empty mapping means a wrong path, not a cohort with no genomes."""
    listing = tmp_path / "empty.tsv"
    listing.write_text("Sample\tpath\n")
    with pytest.raises(SystemExit, match="no Sample"):
        load_file_list(listing)


def test_resolve_via_file_list_reports_absent_and_broken_entries(tmp_path: Path) -> None:
    """Absent from the map, or mapped to a file that is not there, both count as missing."""
    real = tmp_path / "SAMEA1.fa.gz"
    real.write_text("")
    mapping = {"SAMEA1": real, "SAMEA2": tmp_path / "gone.fa.gz"}
    resolved, missing = resolve_via_file_list(["SAMEA1", "SAMEA2", "SAMEA3"], mapping)
    assert [s for s, _ in resolved] == ["SAMEA1"]
    assert missing == ["SAMEA2", "SAMEA3"]


def test_resolve_via_file_list_can_skip_the_stat(tmp_path: Path) -> None:
    """--no-check-exists trusts the list, for when the store is on a slow shared mount."""
    mapping = {"SAMEA1": tmp_path / "gone.fa.gz"}
    resolved, missing = resolve_via_file_list(["SAMEA1"], mapping, check_exists=False)
    assert [s for s, _ in resolved] == ["SAMEA1"]
    assert missing == []


def test_run_via_file_list_records_the_strategy(tmp_path: Path) -> None:
    """The manifest must say how paths were resolved — the two clusters differ, and it matters."""
    asm = tmp_path / "store"
    asm.mkdir()
    (asm / "SAMEA1.fa.gz").write_text("")
    listing = tmp_path / "assemblies_file_list.tsv"
    _write_file_list(listing, [("SAMEA1", asm / "SAMEA1.fa.gz"), ("SAMEA9", asm / "SAMEA9.fa.gz")])
    sheet = tmp_path / "sheet.csv"
    pd.DataFrame({"phenotype-BioSample_ID": ["SAMEA1", "SAMEA2"]}).to_csv(sheet, index=False)

    out = tmp_path / "assembly_refs.txt"
    manifest = resolve_run(organism_key="kp", out_tsv=out, ast_sheet=sheet, file_list=listing)

    assert manifest["resolution"] == "file_list"
    assert manifest["n_resolved"] == 1
    assert manifest["n_missing"] == 1
    assert out.read_text().splitlines() == [f"SAMEA1\t{asm / 'SAMEA1.fa.gz'}"]


def test_file_list_takes_precedence_over_a_directory(tmp_path: Path) -> None:
    """Both supplied is not an error: the explicit list wins, since it is the narrower statement."""
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    (decoy / "SAMEA1.fa.gz").write_text("")
    real = tmp_path / "real"
    real.mkdir()
    (real / "SAMEA1.fa.gz").write_text("")
    listing = tmp_path / "list.tsv"
    _write_file_list(listing, [("SAMEA1", real / "SAMEA1.fa.gz")])
    sheet = tmp_path / "sheet.csv"
    pd.DataFrame({"Sample": ["SAMEA1"]}).to_csv(sheet, index=False)

    resolve_run(organism_key="kp", out_tsv=tmp_path / "refs.txt", ast_sheet=sheet,
                asm_dir=decoy, file_list=listing)
    assert str(real) in (tmp_path / "refs.txt").read_text()


def test_cohort_samples_accepts_either_id_column(tmp_path: Path) -> None:
    """The AST sheet keys on Sample or phenotype-BioSample_ID depending on how it was built."""
    sheet = tmp_path / "sheet.csv"
    pd.DataFrame({"phenotype-BioSample_ID": ["SAMN1", "SAMN1", "SAMN2"]}).to_csv(sheet, index=False)
    assert cohort_samples("kp", ast_sheet=sheet) == ["SAMN1", "SAMN2"]  # deduplicated


def test_cohort_samples_from_split_table_spans_all_splits(tmp_path: Path) -> None:
    """Restricting to a drug must still cover train+validate+holdout, not just the fitted rows."""
    split = tmp_path / "ertapenem_split.csv"
    pd.DataFrame({
        "Sample": ["A", "B", "C"], "ast_label": [1, 0, 1],
        "split": ["train", "validate", "holdout"],
    }).to_csv(split, index=False)
    assert sorted(cohort_samples("kp", split_table=split)) == ["A", "B", "C"]


# --------------------------------------------------------------------------------------- #
# lineage clustering
# --------------------------------------------------------------------------------------- #
def _block_diagonal(sizes: list[int], within: float = 0.001, between: float = 0.2) -> tuple[list[str], np.ndarray]:
    """A distance matrix with clearly separated blocks — the structure clustering must recover."""
    n = sum(sizes)
    d = np.full((n, n), between)
    names, start = [], 0
    for b, size in enumerate(sizes):
        d[start:start + size, start:start + size] = within
        names.extend(f"blk{b}_{i}" for i in range(size))
        start += size
    np.fill_diagonal(d, 0.0)
    return names, d


def test_cluster_recovers_block_structure() -> None:
    """Three well-separated blocks become three clusters, each internally consistent."""
    names, d = _block_diagonal([120, 110, 100])
    clusters = cluster_distances(names, d, threshold=0.02, min_size=50)
    assert len({c for c in clusters.values() if c != OTHER}) == 3
    for prefix in ("blk0", "blk1", "blk2"):
        labels = {c for s, c in clusters.items() if s.startswith(prefix)}
        assert len(labels) == 1  # a block never splits across clusters


def test_small_clusters_collapse_to_other() -> None:
    """Below min_size a cluster is not its own lineage — it joins 'other', as in production."""
    names, d = _block_diagonal([150, 5])
    clusters = cluster_distances(names, d, threshold=0.02, min_size=100)
    assert {c for s, c in clusters.items() if s.startswith("blk1")} == {OTHER}
    assert {c for s, c in clusters.items() if s.startswith("blk0")} == {"sl0001"}


def test_clusters_are_named_by_descending_size() -> None:
    """sl0001 is the largest cluster — stable, readable ordering in the lineage report."""
    names, d = _block_diagonal([200, 120])
    clusters = cluster_distances(names, d, threshold=0.02, min_size=50)
    assert clusters["blk0_0"] == "sl0001"
    assert clusters["blk1_0"] == "sl0002"


def _write_triangle(path: Path, names: list[str], d: np.ndarray) -> None:
    """Write a mash-triangle lower-triangular PHYLIP file."""
    lines = [f"\t{len(names)}"]
    for i, name in enumerate(names):
        lines.append("\t".join([name, *(f"{d[i, j]:.6f}" for j in range(i))]))
    path.write_text("\n".join(lines) + "\n")


def test_run_writes_pyseer_cluster_file(tmp_path: Path) -> None:
    """Output is the headerless Sample<TAB>cluster file --lineage-clusters expects."""
    names, d = _block_diagonal([120, 110])
    triangle = tmp_path / "mash_triangle.txt"
    _write_triangle(triangle, names, d)

    out = tmp_path / "lineage_clusters.tsv"
    manifest = cluster_run(triangle=triangle, out_tsv=out, threshold=0.02, min_size=50)

    rows = [line.split("\t") for line in out.read_text().splitlines()]
    assert len(rows) == 230
    assert all(len(r) == 2 for r in rows)
    assert manifest["n_clusters"] == 2
    assert manifest["n_in_other"] == 0
    assert json.loads((tmp_path / "lineage_clusters.manifest.json").read_text())["min_size"] == 50


def test_run_can_restrict_to_a_subset(tmp_path: Path) -> None:
    """--keep narrows a cohort-wide triangle to one drug's genomes without re-sketching."""
    names, d = _block_diagonal([120, 110])
    triangle = tmp_path / "mash_triangle.txt"
    _write_triangle(triangle, names, d)
    keep = tmp_path / "keep.txt"
    keep.write_text("\n".join(n for n in names if n.startswith("blk0")))

    manifest = cluster_run(
        triangle=triangle, out_tsv=tmp_path / "clusters.tsv", threshold=0.02, min_size=50, keep=keep
    )
    assert manifest["n_samples"] == 120


def test_run_rejects_a_disjoint_keep_list(tmp_path: Path) -> None:
    """A --keep list sharing no ids with the triangle is a mistake, not an empty result."""
    names, d = _block_diagonal([10, 10])
    triangle = tmp_path / "mash_triangle.txt"
    _write_triangle(triangle, names, d)
    keep = tmp_path / "keep.txt"
    keep.write_text("NOT_A_SAMPLE\n")
    with pytest.raises(SystemExit, match="none of the"):
        cluster_run(triangle=triangle, out_tsv=tmp_path / "c.tsv", keep=keep)

"""Unit tests for the generic gene presence table (``bacpredict.engine.gene_lr.locate_gene``).

Build small nested ``*_protein_sequences.parquet`` files (lists-of-lists per contig, as
``flatten_proteins`` expects) and check ``build_gene_presence_table``: single-copy genomes are kept
with the right flat index / protein count / annotation; absent and multi-copy genomes are skipped.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from bacpredict.engine.gene_lr.locate_gene import build_gene_presence_table

SUFFIX = "_protein_sequences.parquet"


def _write_genome(parquet_dir: Path, sid: str, contigs: list[list[tuple[str, str]]]) -> None:
    """Write one genome's nested parquet from ``[[(gene_name, protein_name), ...] per contig]``."""
    n = len(contigs)
    df = pd.DataFrame([{
        "contig_idx": list(range(n)),
        "gene_name": [[g for g, _ in c] for c in contigs],
        "protein_name": [[p for _, p in c] for c in contigs],
        "protein_sequence": [[f"SEQ_{g}" for g, _ in c] for c in contigs],
        "start": [[1 for _ in c] for c in contigs],
        "end": [[99 for _ in c] for c in contigs],
        "protein_id": [[f"id_{g}" for g, _ in c] for c in contigs],
    }])
    df.to_parquet(parquet_dir / f"{sid}{SUFFIX}")


def test_single_copy_kept_absent_and_multicopy_skipped(tmp_path: Path) -> None:
    """Only the single-copy genome survives, with the correct flat index, count, and annotation."""
    # g1: rpoB single-copy at flat index 1 (after gyrA), 3 proteins total across 2 contigs.
    _write_genome(tmp_path, "g1", [[("gyrA", "gyrase"), ("rpoB", "RNA polymerase")], [("katG", "catalase")]])
    _write_genome(tmp_path, "g2", [[("gyrA", "gyrase"), ("katG", "catalase")]])              # rpoB absent
    _write_genome(tmp_path, "g3", [[("rpoB", "RNA polymerase")], [("rpoB", "RNA polymerase")]])  # multi-copy

    qc_log = tmp_path / "qc.log"
    table = build_gene_presence_table(["g1", "g2", "g3"], tmp_path, "rpoB", qc_log_path=qc_log)

    assert list(table.index) == ["g1"]
    row = table.loc["g1"]
    assert row["protein_index"] == 1
    assert row["n_proteins"] == 3
    assert row["gene_name"] == "rpoB"
    assert row["annotation"] == "RNA polymerase"
    assert "single_copy=1" in qc_log.read_text()


def test_case_insensitive_and_aliases(tmp_path: Path) -> None:
    """Matching is case-insensitive and honours alias symbols."""
    _write_genome(tmp_path, "g1", [[("RPOB", "RNA polymerase")]])      # different case
    _write_genome(tmp_path, "g2", [[("Rv0667", "RNA polymerase")]])    # locus-tag alias
    table = build_gene_presence_table(["g1", "g2"], tmp_path, "rpoB", aliases=("Rv0667",))
    assert set(table.index) == {"g1", "g2"}
    assert (table["protein_index"] == 0).all()


def test_missing_parquet_is_skipped(tmp_path: Path) -> None:
    """A sample with no parquet file is counted as missing, not an error, and dropped."""
    _write_genome(tmp_path, "g1", [[("rpoB", "RNA polymerase")]])
    table = build_gene_presence_table(["g1", "absent_sample"], tmp_path, "rpoB")
    assert list(table.index) == ["g1"]


def test_no_hits_returns_empty_framed_table(tmp_path: Path) -> None:
    """When the gene is nowhere, an empty (correctly-columned) table comes back, not a crash."""
    _write_genome(tmp_path, "g1", [[("gyrA", "gyrase")]])
    table = build_gene_presence_table(["g1"], tmp_path, "rpoB")
    assert table.empty
    assert list(table.columns) == ["protein_index", "n_proteins", "gene_name", "annotation"]

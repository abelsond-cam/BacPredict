"""Unit smoke for the CARD gene locator: sidecar → presence table (same schema as the Bakta locator)."""

from __future__ import annotations

import pandas as pd

from bacpredict.engine.gene_lr.card_gene_locator import build_card_presence, sidecar_dir_available


def _write_sidecar(path, rows):
    pd.DataFrame(rows).to_parquet(path, engine="pyarrow")


def _write_protein_parquet(path, n_proteins):
    # locate_gene.flatten_proteins expects nested per-contig list columns; one contig, n proteins.
    pd.DataFrame([{
        "gene_name": [[f"g{i}" for i in range(n_proteins)]],
        "protein_sequence": [[f"M{i}" for i in range(n_proteins)]],
        "start": [[1] * n_proteins], "end": [[9] * n_proteins],
        "protein_id": [[f"p{i}" for i in range(n_proteins)]], "contig_idx": [[0]],
    }][0]).to_parquet(path, engine="pyarrow")


def test_build_card_presence_single_copy(tmp_path):
    amr = tmp_path / "amr"
    amr.mkdir()
    pq = tmp_path / "pq"
    pq.mkdir()
    # s1: single-copy AAC(6') at flat 3; a Bakta-missed (-1) ArmA row that must be ignored.
    _write_sidecar(amr / "s1_amr.parquet", [
        {"flat_index": 3, "amr_gene_family": "AAC(6')", "amr_allele": "AAC(6')-Ib"},
        {"flat_index": -1, "amr_gene_family": "ArmA", "amr_allele": "ArmA"},
    ])
    # s2: two AAC(6') copies (flat 3 and 7) -> multi-copy -> dropped.
    _write_sidecar(amr / "s2_amr.parquet", [
        {"flat_index": 3, "amr_gene_family": "AAC(6')", "amr_allele": "AAC(6')-Ib"},
        {"flat_index": 7, "amr_gene_family": "AAC(6')", "amr_allele": "AAC(6')-Il"},
    ])
    for s in ("s1", "s2"):
        _write_protein_parquet(pq / f"{s}_protein_sequences.parquet", n_proteins=10)

    tables = build_card_presence(["s1", "s2", "s3"], amr, pq, [("AAC(6')", ())])
    t = tables["AAC(6')"]
    assert list(t.index) == ["s1"]                       # s2 multi-copy dropped, s3 no sidecar
    assert int(t.loc["s1", "gene_flat_index"]) == 3
    assert int(t.loc["s1", "n_proteins"]) == 10
    assert t.loc["s1", "gene_name"] == "AAC(6')"


def test_sidecar_dir_available(tmp_path):
    amr = tmp_path / "amr"
    amr.mkdir()
    assert not sidecar_dir_available(amr, ["s1"])
    _write_sidecar(amr / "s1_amr.parquet", [{"flat_index": 1, "amr_gene_family": "X", "amr_allele": "x"}])
    assert sidecar_dir_available(amr, ["s1"])
    assert not sidecar_dir_available(tmp_path / "nope", ["s1"])

"""Golden test for the shared organism config — pins the exact store layout the refactor merged.

``engine.config.store_paths`` replaced two copy-pasted ``default_paths(species)`` functions
(``coding_amr_lr`` + ``igr_amr_lr``). These assertions freeze the paths those functions produced so
the merge cannot silently change where the engine looks for an organism's data.
"""
import pytest

from bacpredict.engine.config import KP, TB, StorePaths, organism, store_paths


def test_store_paths_tb_matches_legacy_layout(monkeypatch):
    monkeypatch.setenv("SCRATCHDIR", "/scratch/x")
    p = store_paths("tb")
    root = "/scratch/x/processed/train_tb_ast"
    assert str(p.ast_sheet) == f"{root}/binary_ast_with_split.csv"
    assert str(p.esm_dir) == f"{root}/esm"
    assert str(p.baclm_dir) == f"{root}/baclm"
    assert str(p.parquet_dir) == f"{root}/protein_sequences"
    assert str(p.input_csv) == f"{root}/embedding_input.csv"
    assert p.esm_suffix == "_esm_embeddings.pt"
    assert p.baclm_suffix == "_baclm_embeddings.pt"
    assert p.parquet_suffix == "_protein_sequences.parquet"


def test_store_paths_kp_uses_kleb_task(monkeypatch):
    monkeypatch.setenv("SCRATCHDIR", "/scratch/x")
    assert str(store_paths("kp").ast_sheet) == "/scratch/x/processed/train_kleb_ast/binary_ast_with_split.csv"


def test_store_paths_mutable_for_cli_override(monkeypatch):
    # The CLI overrides individual fields in place (paths.esm_dir = args.esm_dir), so StorePaths
    # must not be frozen.
    monkeypatch.setenv("SCRATCHDIR", "/scratch/x")
    p = store_paths("tb")
    p.esm_dir = "override"  # must not raise
    assert p.esm_dir == "override"
    assert isinstance(p, StorePaths)


def test_organism_registry_and_unknown_key():
    assert organism("tb") is TB
    assert organism("kp") is KP
    with pytest.raises(ValueError, match="Unknown organism"):
        organism("ecoli")

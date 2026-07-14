"""Golden test for the shared organism config — pins the exact store layout the refactor merged.

``engine.config.store_paths`` replaced two copy-pasted ``default_paths(species)`` functions
(``coding_amr_lr`` + ``igr_amr_lr``). These assertions freeze the paths those functions produced so
the merge cannot silently change where the engine looks for an organism's data.
"""
import pytest

from bacpredict.engine.config import KP, TB, StorePaths, organism, resolve_data_root, store_paths


@pytest.fixture(autouse=True)
def _clear_data_root_env(monkeypatch):
    """Isolate every test from a dev shell that happens to export $BACPREDICT_DATA_ROOT/$SCRATCHDIR."""
    monkeypatch.delenv("BACPREDICT_DATA_ROOT", raising=False)
    monkeypatch.delenv("SCRATCHDIR", raising=False)


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


def test_resolve_data_root_priority(monkeypatch):
    # explicit arg wins over everything
    monkeypatch.setenv("BACPREDICT_DATA_ROOT", "/env/root")
    monkeypatch.setenv("SCRATCHDIR", "/scratch/x")
    assert str(resolve_data_root("/explicit")) == "/explicit"
    # BACPREDICT_DATA_ROOT wins over $SCRATCHDIR
    assert str(resolve_data_root()) == "/env/root"
    # $SCRATCHDIR used when it's the only one set (the Isambard default)
    monkeypatch.delenv("BACPREDICT_DATA_ROOT")
    assert str(resolve_data_root()) == "/scratch/x"


def test_resolve_data_root_unresolvable_raises(monkeypatch):
    # neither env var set, and force the CSD3 autodetect path to look absent
    monkeypatch.setattr("bacpredict.engine.config._CSD3_DATA_ROOT", __import__("pathlib").Path("/no/such/csd3"))
    with pytest.raises(RuntimeError, match="BACPREDICT_DATA_ROOT"):
        resolve_data_root()


def test_data_root_honours_explicit_override():
    # the CLI --data-root seam: OrganismConfig.data_root(root=...) bypasses env entirely
    assert str(TB.data_root("/custom")) == "/custom/processed/train_tb_ast"

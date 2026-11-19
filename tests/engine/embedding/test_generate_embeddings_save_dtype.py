"""The stored dtype of ESM-C embeddings: bf16 for the embeddings only, ids and masks untouched."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

import bacpredict.engine.embedding.generate_embeddings as ge


@pytest.mark.parametrize("save_dtype", [torch.float32, torch.bfloat16])
def test_save_dtype_casts_embeddings_but_never_contig_ids(tmp_path, monkeypatch, save_dtype):
    rng = np.random.default_rng(0)
    n_contigs = 300  # contig ids above 256 are exactly what bf16 would round
    fake = [[rng.standard_normal(960).astype(np.float32)] for _ in range(n_contigs)]
    monkeypatch.setattr(ge, "compute_genome_protein_embeddings", lambda **kw: fake)
    parquet = tmp_path / "S1_protein_sequences.parquet"
    pd.DataFrame({"sample_id": ["S1"], "protein_sequence": [[["M"]] * n_contigs]}).to_parquet(parquet)

    sample, ok, err = ge.process_genome_from_protein_sequences(
        parquet, esm_model=None, esm_tokenizer=None, bacformer_model=None, device="cpu", esm_dir=tmp_path,
        bacformer_dir=None, max_proteins=9000, save_dtype=save_dtype)
    assert ok, err
    saved = torch.load(tmp_path / "S1_esm_embeddings.pt", weights_only=False)
    assert saved["protein_embeddings"].dtype == save_dtype
    assert saved["protein_embeddings"].shape == (1, n_contigs, 960)
    assert saved["contig_ids"].dtype == torch.float32 and saved["attention_mask"].dtype == torch.float32
    assert saved["contig_ids"][0].tolist() == [float(i) for i in range(n_contigs)]
    exact = torch.tensor(np.stack([c[0] for c in fake]))
    assert torch.allclose(saved["protein_embeddings"][0].float(), exact, rtol=1e-2, atol=1e-2)


def test_a_save_cut_short_leaves_no_file_skip_existing_would_trust(tmp_path, monkeypatch):
    """--skip-existing trusts any file at the final name, so an interrupted save must leave none there."""
    real_save = torch.save

    def dies_mid_write(obj, f):
        real_save(obj, f)
        with open(f, "r+b") as fh:
            fh.truncate(100)
        raise OSError("Disk quota exceeded")

    monkeypatch.setattr(ge.torch, "save", dies_mid_write)
    out = tmp_path / "S1_esm_embeddings.pt"
    with pytest.raises(OSError):
        ge._atomic_torch_save({"x": torch.zeros(3)}, out)
    assert list(tmp_path.iterdir()) == []

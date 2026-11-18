"""Unit tests for the captum gene-ablation stage.

The load-time guards are the point of most of these: an interleaved store or a truncated genome
would not crash, it would silently attribute the wrong gene, which is the failure mode this whole
analysis cannot survive. The toy-model test pins the two conventions the numbers depend on — that a
protein's whole embedding is ablated as one group, and that the stored score is the logit shift.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("torch")
import numpy as np
import pandas as pd
import torch

from bacpredict.engine.explain.gene_ablation_attribution import (
    LogitForward,
    delta_probability,
    load_genome,
    shard,
    verify_protein_count,
)


class _ToyOutput:
    """Minimal stand-in for the model output object, carrying only ``logits``."""

    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _ToyModel(torch.nn.Module):
    """A model whose logit is the sum of each protein's first embedding dimension.

    Ablating protein ``t`` therefore shifts the logit by exactly ``protein_embeddings[0, t, 0]``,
    which makes the correct attribution known in closed form.
    """

    def forward(self, protein_embeddings, attention_mask=None, contig_ids=None, return_dict=True):
        """Return the sum of each protein's first embedding dimension as the genome logit."""
        return _ToyOutput(protein_embeddings[:, :, 0].sum(dim=1, keepdim=True))


def _store(tmp_path: Path, name: str, n_proteins: int = 6, width: int = 4, **extra) -> Path:
    """Write a plain per-protein embedding store."""
    store = {
        "protein_embeddings": torch.arange(n_proteins * width, dtype=torch.float32).reshape(1, n_proteins, width),
        "attention_mask": torch.ones(1, n_proteins),
        "contig_ids": torch.zeros(1, n_proteins, dtype=torch.long),
    }
    store.update(extra)
    path = tmp_path / name
    torch.save(store, path)
    return path


def test_ablation_recovers_the_known_logit_shift(tmp_path: Path) -> None:
    """On a model with a closed-form answer, every protein's attribution must match it exactly."""
    pytest.importorskip("captum")
    from bacpredict.engine.explain.gene_ablation_attribution import attribute_genome

    inputs = {
        "protein_embeddings": torch.arange(24, dtype=torch.float32).reshape(1, 6, 4),
        "attention_mask": torch.ones(1, 6),
        "contig_ids": torch.zeros(1, 6, dtype=torch.long),
    }
    wrapped = LogitForward(_ToyModel())
    attributions, base_logit = attribute_genome(wrapped, inputs, check_group_constant=True)

    expected = inputs["protein_embeddings"][0, :, 0].numpy().astype(float)
    assert base_logit == pytest.approx(float(expected.sum()))
    np.testing.assert_allclose(attributions, expected, rtol=1e-6, atol=1e-6)


def test_batched_perturbations_give_the_same_answer(tmp_path: Path) -> None:
    """perturbations_per_eval is a throughput knob and must not change a single number."""
    pytest.importorskip("captum")
    from bacpredict.engine.explain.gene_ablation_attribution import attribute_genome

    inputs = {
        "protein_embeddings": torch.randn(1, 7, 5),
        "attention_mask": torch.ones(1, 7),
        "contig_ids": torch.zeros(1, 7, dtype=torch.long),
    }
    wrapped = LogitForward(_ToyModel())
    one, base_one = attribute_genome(wrapped, inputs, perturbations_per_eval=1)
    many, base_many = attribute_genome(wrapped, inputs, perturbations_per_eval=4)

    assert base_one == pytest.approx(base_many)
    np.testing.assert_allclose(one, many, rtol=1e-6, atol=1e-6)


def test_interleaved_store_is_rejected(tmp_path: Path) -> None:
    """A bundle-layout store would shift every flat index, so it must fail loudly."""
    path = _store(tmp_path, "bundle.pt", special_tokens_mask=torch.full((1, 6), 4))
    with pytest.raises(ValueError, match="interleaved"):
        load_genome(path, "cpu", torch.float32)


def test_padded_store_is_rejected(tmp_path: Path) -> None:
    """A padded mask means some rows are not proteins; attributing them is meaningless."""
    path = _store(tmp_path, "padded.pt")
    store = torch.load(path, map_location="cpu", weights_only=False)
    store["attention_mask"][0, -2:] = 0
    torch.save(store, path)
    with pytest.raises(ValueError, match="padded attention_mask"):
        load_genome(path, "cpu", torch.float32)


def test_plain_store_loads_with_the_expected_shapes(tmp_path: Path) -> None:
    """The happy path returns exactly the three forward kwargs the model takes."""
    inputs = load_genome(_store(tmp_path, "plain.pt"), "cpu", torch.float32)
    assert set(inputs) == {"protein_embeddings", "attention_mask", "contig_ids"}
    assert inputs["protein_embeddings"].shape == (1, 6, 4)


def test_protein_count_mismatch_is_fatal(tmp_path: Path) -> None:
    """A genome whose embedding and parquet disagree has untrustworthy flat indices."""
    parquet_dir = tmp_path / "protein_sequences"
    parquet_dir.mkdir()
    pd.DataFrame([{
        "sample_id": "S1", "contig_idx": [0], "gene_name": [["a", "b"]], "protein_name": [[None, None]],
        "protein_sequence": [["M", "K"]], "start": [[1, 10]], "end": [[9, 20]], "protein_id": [[None, None]],
    }]).to_parquet(parquet_dir / "S1_protein_sequences.parquet", index=False)

    assert verify_protein_count("S1", 2, parquet_dir, "_protein_sequences.parquet") == 2
    with pytest.raises(ValueError, match="flat index is not trustworthy"):
        verify_protein_count("S1", 3, parquet_dir, "_protein_sequences.parquet")


def test_missing_parquet_warns_rather_than_failing(tmp_path: Path) -> None:
    """A genome with no parquet still attributes; the guard simply cannot run."""
    assert verify_protein_count("nope", 10, tmp_path, "_protein_sequences.parquet") is None


def test_delta_probability_matches_the_direct_sigmoid_difference() -> None:
    """The stored logit shift must reproduce the probability-scale score exactly."""
    base = 3.2
    attribution = np.array([0.0, 0.5, -1.5, 8.0])
    direct = 1 / (1 + np.exp(-base)) - 1 / (1 + np.exp(-(base - attribution)))
    np.testing.assert_allclose(delta_probability(base, attribution), direct, rtol=1e-12, atol=1e-12)


def test_delta_probability_stays_in_range_when_saturated() -> None:
    """Even at a logit the sigmoid cannot resolve, the converted score stays a valid probability shift."""
    out = delta_probability(40.0, np.array([0.001, 5.0, -5.0]))
    assert np.all(out >= -1.0) and np.all(out <= 1.0)


def test_shards_partition_the_cohort_exactly() -> None:
    """Every genome lands in exactly one shard — no gaps, no double work."""
    frame = pd.DataFrame({"Sample": [f"S{i:03d}" for i in range(37)]})
    pieces = [shard(frame, i, 5) for i in range(5)]
    seen = pd.concat(pieces)["Sample"].tolist()
    assert sorted(seen) == sorted(frame["Sample"].tolist())
    assert len(seen) == len(set(seen))


def test_shard_rejects_an_out_of_range_index() -> None:
    """An array sized wrongly against --n-shards must fail rather than silently do nothing."""
    frame = pd.DataFrame({"Sample": ["a", "b"]})
    with pytest.raises(ValueError, match="out of range"):
        shard(frame, 5, 5)

"""Unit tests for the gated-attention MIL genome-classification head.

Pure-torch (no model download): the pool is exercised directly, and the
classifier wrapper is tested against a tiny stub backbone so freeze/grad
behaviour can be checked without pulling Bacformer weights. Skipped where torch
is unavailable (e.g. the local MacBook env without the GPU stack).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from tl.train.attention_pool import (  # noqa: E402 - after importorskip
    BacformerAttnPoolForGenomeClassification,
    GatedAttentionMILPool,
)


@pytest.fixture(autouse=True)
def _seed() -> None:
    """Deterministic init/inputs for every test."""
    torch.manual_seed(0)


# ---------------------------------------------------------------------------
# GatedAttentionMILPool
# ---------------------------------------------------------------------------


def test_pool_shapes_and_weights_sum_to_one() -> None:
    """Pooled is (B, H), weights are (B, N) and sum to 1 over valid instances."""
    pool = GatedAttentionMILPool(hidden=16, attn_dim=8).eval()
    h = torch.randn(3, 7, 16)
    mask = torch.ones(3, 7)
    mask[0, 5:] = 0  # two padded instances in row 0
    pooled, weights = pool(h, mask)

    assert pooled.shape == (3, 16)
    assert weights.shape == (3, 7)
    valid_sums = (weights * mask).sum(dim=1)
    assert torch.allclose(valid_sums, torch.ones(3), atol=1e-5)
    # Padded instances carry (essentially) zero weight.
    assert torch.allclose(weights[0, 5:], torch.zeros(2), atol=1e-6)


def test_pool_padding_invariance() -> None:
    """Appending masked instances does not change the pooled vector."""
    pool = GatedAttentionMILPool(hidden=12, attn_dim=8).eval()
    h = torch.randn(2, 4, 12)
    with torch.no_grad():
        pooled_a, _ = pool(h, torch.ones(2, 4))
        h_padded = torch.cat([h, torch.randn(2, 3, 12)], dim=1)
        mask = torch.cat([torch.ones(2, 4), torch.zeros(2, 3)], dim=1)
        pooled_b, _ = pool(h_padded, mask)
    assert torch.allclose(pooled_a, pooled_b, atol=1e-5)


def test_pool_permutation_invariance() -> None:
    """The pooled vector is invariant to the order of instances."""
    pool = GatedAttentionMILPool(hidden=12, attn_dim=8).eval()
    h = torch.randn(2, 6, 12)
    perm = torch.randperm(6)
    with torch.no_grad():
        pooled, _ = pool(h, torch.ones(2, 6))
        pooled_perm, _ = pool(h[:, perm, :], torch.ones(2, 6))
    assert torch.allclose(pooled, pooled_perm, atol=1e-5)


def test_pool_uniform_equals_masked_mean() -> None:
    """With zero scoring weights the attention is uniform ⇒ pooled == masked mean."""
    pool = GatedAttentionMILPool(hidden=10, attn_dim=8).eval()
    nn.init.zeros_(pool.w.weight)  # constant logits ⇒ uniform softmax over valid
    h = torch.randn(2, 5, 10)
    mask = torch.ones(2, 5)
    mask[0, 3:] = 0
    with torch.no_grad():
        pooled, _ = pool(h, mask)
    masked_mean = (h * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
    assert torch.allclose(pooled, masked_mean, atol=1e-5)


def test_pool_gradients_flow() -> None:
    """All pool parameters receive gradients."""
    pool = GatedAttentionMILPool(hidden=8, attn_dim=4)
    h = torch.randn(2, 5, 8)
    pooled, _ = pool(h, torch.ones(2, 5))
    pooled.sum().backward()
    for name, p in pool.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all()


# ---------------------------------------------------------------------------
# BacformerAttnPoolForGenomeClassification (stub backbone)
# ---------------------------------------------------------------------------


class _StubBackbone(nn.Module):
    """Minimal stand-in for ``.bacformer``: returns (last_hidden_state,)."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, protein_embeddings: torch.Tensor, **kwargs) -> tuple[torch.Tensor]:
        return (self.proj(protein_embeddings),)


def _batch(hidden: int = 16, b: int = 3, n: int = 6):
    pe = torch.randn(b, n, hidden)
    am = torch.ones(b, n)
    am[0, 4:] = 0  # padding in row 0
    cid = torch.zeros(b, n, dtype=torch.long)
    labels = torch.tensor([0.0, 1.0, 1.0])
    return pe, am, cid, labels


def test_wrapper_forward_output() -> None:
    """Forward returns (B, num_labels) logits, a finite scalar loss, and stashed weights."""
    hidden = 16
    model = BacformerAttnPoolForGenomeClassification(_StubBackbone(hidden), hidden, attn_dim=8).eval()
    pe, am, cid, labels = _batch(hidden)
    out = model(protein_embeddings=pe, attention_mask=am, contig_ids=cid, labels=labels)

    assert out.logits.shape == (3, 1)
    assert out.loss is not None and out.loss.ndim == 0 and torch.isfinite(out.loss)
    assert model.last_attention_weights.shape == (3, 6)
    valid_sums = (model.last_attention_weights * am).sum(dim=1)
    assert torch.allclose(valid_sums, torch.ones(3), atol=1e-5)
    # last_attention_weights is detached (interpretability snapshot only)
    assert not model.last_attention_weights.requires_grad


def test_wrapper_no_labels_returns_no_loss() -> None:
    """Without labels the forward yields logits but no loss."""
    hidden = 16
    model = BacformerAttnPoolForGenomeClassification(_StubBackbone(hidden), hidden).eval()
    pe, am, cid, _ = _batch(hidden)
    out = model(protein_embeddings=pe, attention_mask=am, contig_ids=cid)
    assert out.loss is None
    assert out.logits.shape == (3, 1)


def test_wrapper_frozen_backbone_grad_isolation() -> None:
    """Frozen backbone gets no gradient; the pool + head still train."""
    hidden = 16
    model = BacformerAttnPoolForGenomeClassification(_StubBackbone(hidden), hidden, freeze_backbone=True)
    pe, am, cid, labels = _batch(hidden)
    model(protein_embeddings=pe, attention_mask=am, contig_ids=cid, labels=labels).loss.backward()

    assert model.out_proj.weight.grad is not None
    assert model.pool.V.weight.grad is not None
    assert model.pool.w.weight.grad is not None
    assert all(p.grad is None for p in model.bacformer.parameters())


def test_frozen_backbone_pinned_to_eval_through_train() -> None:
    """A frozen backbone stays in eval() even after model.train() (deterministic features)."""
    model = BacformerAttnPoolForGenomeClassification(_StubBackbone(16), 16, freeze_backbone=True)
    model.train()  # Trainer flips the whole module to train mode at training start
    assert not model.bacformer.training, "frozen backbone must remain in eval (dropout off)"
    assert model.pool.training, "pool/head must follow train mode"
    # Unfreezing lets the backbone follow the module's train/eval state again.
    model.set_backbone_frozen(False)
    model.train()
    assert model.bacformer.training
    model.eval()
    assert not model.bacformer.training


def test_wrapper_unfrozen_backbone_receives_grad() -> None:
    """After unfreezing, the backbone receives gradients end-to-end."""
    hidden = 16
    model = BacformerAttnPoolForGenomeClassification(_StubBackbone(hidden), hidden, freeze_backbone=True)
    model.set_backbone_frozen(False)
    model.zero_grad(set_to_none=True)
    pe, am, cid, labels = _batch(hidden)
    model(protein_embeddings=pe, attention_mask=am, contig_ids=cid, labels=labels).loss.backward()

    assert model.bacformer.proj.weight.grad is not None
    assert model.out_proj.weight.grad is not None

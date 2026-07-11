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

from bacpredict.engine.finetune.attention_pool import (  # noqa: E402 - after importorskip
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


def test_pool_value_states_default_equals_hidden() -> None:
    """Omitting ``value_states`` pools ``hidden_states`` (byte-identical to the old pool)."""
    pool = GatedAttentionMILPool(hidden=10, attn_dim=8).eval()
    h = torch.randn(2, 5, 10)
    mask = torch.ones(2, 5)
    with torch.no_grad():
        a, wa = pool(h, mask)
        b, wb = pool(h, mask, value_states=h)
    assert torch.allclose(a, b, atol=1e-6)
    assert torch.allclose(wa, wb, atol=1e-6)


def test_pool_split_score_value_widths() -> None:
    """Scoring from a wider input while pooling a narrower value gives ``(B, value_dim)``."""
    pool = GatedAttentionMILPool(hidden=10, attn_dim=8, score_hidden=13).eval()
    assert pool.V.in_features == 13 and pool.U.in_features == 13
    score = torch.randn(2, 5, 13)
    value = torch.randn(2, 5, 10)
    pooled, weights = pool(score, torch.ones(2, 5), value_states=value)
    assert pooled.shape == (2, 10)
    assert weights.shape == (2, 5)
    # Weights depend on the score input, the pooled value on the value input:
    # the pooled vector lives in value space (10-d), not score space (13-d).
    assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-5)


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


# ---------------------------------------------------------------------------
# Surprisal-panel modes (none / att_head / e2e)
# ---------------------------------------------------------------------------

PANEL_DIM = 9


def _panel(b: int = 3, n: int = 6, dim: int = PANEL_DIM):
    return torch.randn(b, n, dim)


def test_panel_mode_none_sizes_unchanged() -> None:
    """``panel_mode='none'`` keeps the panel-free module widths (regression guard)."""
    hidden = 16
    model = BacformerAttnPoolForGenomeClassification(_StubBackbone(hidden), hidden, panel_mode="none")
    assert model.panel_dim == 0
    assert model.pool.V.in_features == hidden  # scoring width
    assert model.norm.normalized_shape == (hidden,)  # value/head width
    assert model.out_proj.in_features == hidden


def test_panel_mode_none_logits_match_default() -> None:
    """``panel_mode='none'`` is byte-identical to the default (no-panel) construction."""
    hidden = 16
    torch.manual_seed(123)
    plain = BacformerAttnPoolForGenomeClassification(_StubBackbone(hidden), hidden).eval()
    torch.manual_seed(123)
    none_mode = BacformerAttnPoolForGenomeClassification(_StubBackbone(hidden), hidden, panel_mode="none").eval()
    pe, am, cid, _ = _batch(hidden)
    with torch.no_grad():
        a = plain(protein_embeddings=pe, attention_mask=am, contig_ids=cid).logits
        b = none_mode(protein_embeddings=pe, attention_mask=am, contig_ids=cid).logits
    assert torch.allclose(a, b, atol=0.0)


def test_panel_mode_att_head_sizing_and_forward() -> None:
    """att_head: score width = hidden+panel, pooled value + head stay hidden-wide."""
    hidden = 16
    model = BacformerAttnPoolForGenomeClassification(
        _StubBackbone(hidden), hidden, panel_mode="att_head", panel_dim=PANEL_DIM
    ).eval()
    assert model.pool.V.in_features == hidden + PANEL_DIM  # gate sees the panel
    assert model.norm.normalized_shape == (hidden,)  # value stays the pure backbone token
    assert model.out_proj.in_features == hidden
    pe, am, cid, labels = _batch(hidden)
    out = model(protein_embeddings=pe, attention_mask=am, contig_ids=cid, labels=labels, panel=_panel())
    assert out.logits.shape == (3, 1)
    assert torch.isfinite(out.loss)


def test_panel_mode_e2e_sizing_and_forward() -> None:
    """e2e: score, pooled value and head are all hidden+panel wide."""
    hidden = 16
    model = BacformerAttnPoolForGenomeClassification(
        _StubBackbone(hidden), hidden, panel_mode="e2e", panel_dim=PANEL_DIM
    ).eval()
    assert model.pool.V.in_features == hidden + PANEL_DIM
    assert model.norm.normalized_shape == (hidden + PANEL_DIM,)
    assert model.out_proj.in_features == hidden + PANEL_DIM
    pe, am, cid, labels = _batch(hidden)
    out = model(protein_embeddings=pe, attention_mask=am, contig_ids=cid, labels=labels, panel=_panel())
    assert out.logits.shape == (3, 1)
    assert torch.isfinite(out.loss)


def test_invalid_panel_mode_raises() -> None:
    """An unknown panel mode fails loudly at construction."""
    with pytest.raises(ValueError, match="panel_mode"):
        BacformerAttnPoolForGenomeClassification(_StubBackbone(16), 16, panel_mode="bogus")


@pytest.mark.parametrize("mode", ["att_head", "e2e"])
def test_panel_required_when_mode_on(mode: str) -> None:
    """Panel modes raise if no panel tensor is supplied."""
    model = BacformerAttnPoolForGenomeClassification(_StubBackbone(16), 16, panel_mode=mode, panel_dim=PANEL_DIM).eval()
    pe, am, cid, _ = _batch(16)
    with pytest.raises(ValueError, match="panel"):
        model(protein_embeddings=pe, attention_mask=am, contig_ids=cid)


@pytest.mark.parametrize("mode", ["att_head", "e2e"])
def test_panel_padding_invariance(mode: str) -> None:
    """Appending masked instances (and their panel rows) leaves the logits unchanged."""
    hidden = 16
    model = BacformerAttnPoolForGenomeClassification(_StubBackbone(hidden), hidden, panel_mode=mode, panel_dim=PANEL_DIM).eval()
    pe = torch.randn(2, 4, hidden)
    panel = torch.randn(2, 4, PANEL_DIM)
    cid = torch.zeros(2, 4, dtype=torch.long)
    with torch.no_grad():
        a = model(protein_embeddings=pe, attention_mask=torch.ones(2, 4), contig_ids=cid, panel=panel).logits
        pe_pad = torch.cat([pe, torch.randn(2, 3, hidden)], dim=1)
        panel_pad = torch.cat([panel, torch.randn(2, 3, PANEL_DIM)], dim=1)
        mask = torch.cat([torch.ones(2, 4), torch.zeros(2, 3)], dim=1)
        cid_pad = torch.zeros(2, 7, dtype=torch.long)
        b = model(protein_embeddings=pe_pad, attention_mask=mask, contig_ids=cid_pad, panel=panel_pad).logits
    assert torch.allclose(a, b, atol=1e-5)


def test_panel_reaches_gate() -> None:
    """Changing only the panel (att_head) shifts the attention distribution — panel → gate.

    In att_head the pooled *value* is the pure backbone token, so when only the panel changes,
    any change in the attention weights is attributable solely to the panel reaching the gate.
    """
    hidden = 16
    model = BacformerAttnPoolForGenomeClassification(
        _StubBackbone(hidden), hidden, panel_mode="att_head", panel_dim=PANEL_DIM
    ).eval()
    pe, am, cid, _ = _batch(hidden)
    base_panel = torch.zeros(3, 6, PANEL_DIM)
    with torch.no_grad():
        model(protein_embeddings=pe, attention_mask=am, contig_ids=cid, panel=base_panel)
        w_base = model.last_attention_weights.clone()
        pert = base_panel.clone()
        pert[1, 2, 0] = 50.0  # a single "this protein is wildly anomalous" panel spike
        model(protein_embeddings=pe, attention_mask=am, contig_ids=cid, panel=pert)
        w_pert = model.last_attention_weights.clone()
    # The row whose panel was perturbed must change; an untouched row must not.
    assert (w_pert[1] - w_base[1]).abs().max() > 1e-3
    assert torch.allclose(w_pert[0], w_base[0], atol=1e-6)


def test_panel_config_stamped() -> None:
    """Panel sizing is stamped on the config so a reload rebuilds the right shapes."""

    class _Cfg:
        hidden_size = 16

    cfg = _Cfg()
    model = BacformerAttnPoolForGenomeClassification(
        _StubBackbone(16), 16, panel_mode="att_head", panel_dim=PANEL_DIM, attn_dim=8, config=cfg
    )
    assert cfg.panel_mode == "att_head"
    assert cfg.panel_dim == PANEL_DIM
    assert cfg.attn_dim == 8
    assert model is not None

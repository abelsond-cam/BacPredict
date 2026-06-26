"""Gated-attention MIL pooling head for the Bacformer genome classifier.

The deployed Bacformer genome head **mean-pools** the per-protein tokens
(``einsum("ijk,ij->ik", features, mask) / mask.sum``), which dilutes a single
causal protein — e.g. an *rpoB* RRDR point mutation — into the ~4,000 others in
the genome. The Task-7 diagnostic localised the TB-AST defect to exactly this
step: the frozen *rpoB*-token AUROC (0.953) collapses to 0.788 once mean-pooled,
and fine-tuning the mean-pool head recovers only to 0.905
(``src/snp_embeddings/docs/PROGRESS_REPORT.md``).

This module swaps that mean for a learned **gated-attention multiple-instance
-learning (MIL) pool** (Ilse, Tomczak & Welling, 2018), so the genome
representation can concentrate on the few signal-bearing proteins. It reuses the
pretrained Bacformer backbone unchanged — only the pool + classification head are
new — and returns a :class:`~transformers.modeling_outputs.SequenceClassifierOutput`
so it drops into a Hugging Face ``Trainer`` exactly like the stock model.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput


class GatedAttentionMILPool(nn.Module):
    r"""Gated-attention MIL pool over a variable-length set of instances.

    Implements the gated attention of Ilse, Tomczak & Welling (2018):

    .. math::
        a_n = \operatorname{softmax}_n\!\big(w^\top (\tanh(V h_n) \odot
        \sigma(U h_n))\big), \qquad z = \sum_n a_n\, h_n

    The attention is **mask-aware**: padded instances are pushed to the smallest
    representable logit before the softmax, so they receive (effectively) zero
    weight and the pooled vector is invariant to padding.

    The **scoring** input (which drives the gate ``V`` / ``U``) is decoupled from
    the **value** that is pooled: ``forward`` accepts a separate ``value_states``
    so the attention can be computed from a panel-augmented token while the pooled
    vector stays the pure backbone token (the ``att_head`` panel mode). When
    ``value_states`` is omitted it defaults to ``hidden_states`` — identical to the
    original single-input pool.

    Parameters
    ----------
    hidden : int
        Value (pooled instance) embedding dimension — the width of the pooled
        output ``z``.
    attn_dim : int, default 128
        Width of the attention hidden layer (``V`` / ``U``).
    dropout : float, default 0.1
        Dropout applied to the gated attention activation.
    score_hidden : int or None, default None
        Width of the **scoring** input fed to ``V`` / ``U``. Defaults to
        ``hidden`` (scoring == value, the original behaviour); set larger to score
        from a panel-augmented token (e.g. ``hidden + panel_dim``).
    """

    def __init__(
        self, hidden: int, attn_dim: int = 128, dropout: float = 0.1, score_hidden: int | None = None
    ) -> None:
        super().__init__()
        score_hidden = hidden if score_hidden is None else score_hidden
        self.V = nn.Linear(score_hidden, attn_dim, bias=False)
        self.U = nn.Linear(score_hidden, attn_dim, bias=False)
        self.w = nn.Linear(attn_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        value_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool over the instance (protein) axis, scoring from ``hidden_states``.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Per-instance **scoring** embeddings (drive the gate), shape
            ``(batch, n, score_hidden)``.
        attention_mask : torch.Tensor or None
            ``1`` for valid instances, ``0`` for padding, shape ``(batch, n)``.
            ``None`` treats every instance as valid.
        value_states : torch.Tensor or None
            Per-instance **value** embeddings that are actually pooled, shape
            ``(batch, n, hidden)``. Defaults to ``hidden_states`` (scoring ==
            value).

        Returns
        -------
        pooled : torch.Tensor
            Attention-weighted sum of ``value_states``, shape ``(batch, hidden)``.
        weights : torch.Tensor
            Per-instance attention weights summing to 1 over valid instances,
            shape ``(batch, n)`` (kept for interpretability).
        """
        if value_states is None:
            value_states = hidden_states
        gated = torch.tanh(self.V(hidden_states)) * torch.sigmoid(self.U(hidden_states))
        gated = self.dropout(gated)
        logits = self.w(gated).squeeze(-1)  # (batch, n)

        if attention_mask is not None:
            pad = attention_mask <= 0
            logits = logits.masked_fill(pad, torch.finfo(logits.dtype).min)

        weights = torch.softmax(logits, dim=-1)
        pooled = torch.einsum("bn,bnh->bh", weights, value_states)
        return pooled, weights


class MultiheadAttentionPool(nn.Module):
    r"""Multi-head attention pool over a variable-length set of instances.

    A learnable **query** token attends over the protein tokens via
    :class:`torch.nn.MultiheadAttention` — the standard, well-tested multi-head
    attention (closer to how Bacformer itself attends, 15-head RoPE self-attention)
    than the single-query gated-attention MIL pool. Drop-in alternative: same
    ``forward(hidden_states, attention_mask, value_states)`` signature returning
    ``(pooled, weights)``.

    The pooled output is always ``hidden``-wide (``embed_dim``); ``nn.MultiheadAttention``'s
    value projection maps ``value_dim`` → ``hidden`` internally, so a panel-augmented value
    (``e2e``) is absorbed by that projection and the head stays ``hidden``-wide — which also
    keeps ``embed_dim`` divisible by ``num_heads`` (the panel-augmented ``969`` is not).

    Parameters
    ----------
    hidden : int
        Pooled-output (query / ``embed_dim``) width. Must be divisible by ``num_heads``.
    num_heads : int, default 8
        Number of attention heads.
    dropout : float, default 0.1
        Attention dropout.
    score_hidden : int or None, default None
        Width of the **scoring** (key) input. Defaults to ``hidden``; set larger to score
        from a panel-augmented token (``hidden + panel_dim``).
    value_dim : int or None, default None
        Width of the **value** input that is pooled. Defaults to ``hidden``; set to
        ``hidden + panel_dim`` to carry the panel into the value (``e2e``).
    """

    def __init__(
        self,
        hidden: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        score_hidden: int | None = None,
        value_dim: int | None = None,
    ) -> None:
        super().__init__()
        if hidden % num_heads != 0:
            raise ValueError(f"MHA embed_dim {hidden} not divisible by num_heads {num_heads}")
        score_hidden = hidden if score_hidden is None else score_hidden
        value_dim = hidden if value_dim is None else value_dim
        self.query = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.normal_(self.query, std=0.02)
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=num_heads, dropout=dropout,
            kdim=score_hidden, vdim=value_dim, batch_first=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        value_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool via a learnable query attending over the (key=``hidden_states``) tokens.

        ``value_states`` defaults to ``hidden_states``. Returns
        ``(pooled (batch, hidden), weights (batch, n))`` — weights averaged over heads,
        for parity with the gated-MIL pool's ``last_attention_weights``.
        """
        if value_states is None:
            value_states = hidden_states
        batch = hidden_states.shape[0]
        query = self.query.expand(batch, -1, -1).to(hidden_states.dtype)
        key_padding_mask = (attention_mask <= 0) if attention_mask is not None else None
        pooled, weights = self.mha(
            query, hidden_states, value_states,
            key_padding_mask=key_padding_mask, need_weights=True, average_attn_weights=True,
        )
        return pooled.squeeze(1), (weights.squeeze(1) if weights is not None else None)


class BacformerAttnPoolForGenomeClassification(nn.Module):
    """Bacformer genome classifier with a gated-attention MIL pool head.

    Reuses a pretrained ``BacformerLargeModel`` backbone (contextualised
    per-protein tokens) and replaces the stock mean-pool genome head with a
    :class:`GatedAttentionMILPool` followed by LayerNorm → Dropout → Linear.
    Mirrors the upstream ``BacformerLargeForGenomeClassification`` forward
    signature and loss, and returns a
    :class:`~transformers.modeling_outputs.SequenceClassifierOutput`, so it is a
    drop-in for a Hugging Face ``Trainer``.

    Parameters
    ----------
    backbone : nn.Module
        The pretrained ``.bacformer`` encoder. Its ``forward`` must accept
        ``protein_embeddings`` / ``attention_mask`` / ``contig_ids`` /
        ``return_dict`` and return a sequence output whose element ``[0]`` is
        ``last_hidden_state`` of shape ``(batch, n, hidden)``.
    hidden : int
        Backbone hidden size.
    num_labels : int, default 1
        Output dimension (``1`` ⇒ a single binary logit).
    attn_dim : int, default 128
        Attention hidden width.
    dropout : float, default 0.1
        Dropout for the attention pool and the classification head.
    freeze_backbone : bool, default False
        If ``True`` the backbone runs under ``torch.no_grad`` and its parameters
        are frozen — only the pool + head train.
    problem_type : str, default "binary_classification"
        Loss selector; ``"single_label_classification"`` (with ``num_labels>1``)
        uses cross-entropy, everything else binary-cross-entropy-with-logits.
    panel_mode : {"none", "att_head", "e2e"}, default "none"
        How the per-protein surprisal **panel** (``panel_dim`` extra features
        concatenated onto each backbone token, *after* the backbone) is wired in:

        ===========  ================  =============  =========================
        mode         V/U in (score)    einsum value   head in (norm / out_proj)
        ===========  ================  =============  =========================
        ``none``     ``hidden``        ``hidden``     ``hidden``
        ``att_head`` ``hidden+panel``  ``hidden``     ``hidden``
        ``e2e``      ``hidden+panel``  ``hidden+panel`` ``hidden+panel``
        ===========  ================  =============  =========================

        ``none`` is byte-identical to the panel-free model. ``att_head`` lets the
        panel steer the gate while the pooled value stays the pure backbone token.
        ``e2e`` carries the panel into the pooled value (and head) too. In both
        panel modes the panel is glued onto the backbone *output*, never its input
        — Bacformer's 960-d input and pretraining are untouched.
    panel_dim : int, default 9
        Number of per-protein panel features (ignored when ``panel_mode="none"``).
    pool_type : {"gated_mil", "mha"}, default "gated_mil"
        Pooling mechanism: ``gated_mil`` is the single-query gated-attention MIL pool
        (Ilse 2018); ``mha`` is a learnable-query :class:`torch.nn.MultiheadAttention` pool
        (multi-head, closer to Bacformer's own attention). Composes with every ``panel_mode``;
        for ``mha`` the head is always ``hidden``-wide (its value projection absorbs the panel).
    num_heads : int, default 8
        Number of heads when ``pool_type="mha"`` (must divide ``hidden``).
    """

    def __init__(
        self,
        backbone: nn.Module,
        hidden: int,
        *,
        num_labels: int = 1,
        attn_dim: int = 128,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        problem_type: str = "binary_classification",
        panel_mode: str = "none",
        panel_dim: int = 9,
        pool_type: str = "gated_mil",
        num_heads: int = 8,
        config=None,
    ) -> None:
        super().__init__()
        if panel_mode not in ("none", "att_head", "e2e"):
            raise ValueError(f"panel_mode must be 'none', 'att_head' or 'e2e', got {panel_mode!r}")
        if pool_type not in ("gated_mil", "mha"):
            raise ValueError(f"pool_type must be 'gated_mil' or 'mha', got {pool_type!r}")
        self.config = config  # backbone PretrainedConfig (HF Trainer reads model.config); None in unit tests
        self.bacformer = backbone
        self.panel_mode = panel_mode
        self.panel_dim = panel_dim if panel_mode != "none" else 0
        self.pool_type = pool_type
        self.num_heads = num_heads
        self.hidden = hidden
        # Size table (single source of truth): scoring width feeds the gate/keys; value width
        # is what gets pooled. gated_mil pools the value directly (head = value width); mha's
        # value projection maps the value to `hidden`, so its head is always `hidden`-wide.
        score_hidden = hidden + self.panel_dim if panel_mode in ("att_head", "e2e") else hidden
        value_hidden = hidden + self.panel_dim if panel_mode == "e2e" else hidden
        if pool_type == "gated_mil":
            self.pool = GatedAttentionMILPool(
                value_hidden, attn_dim=attn_dim, dropout=dropout, score_hidden=score_hidden
            )
            head_in = value_hidden
        else:  # mha
            self.pool = MultiheadAttentionPool(
                hidden, num_heads=num_heads, dropout=dropout, score_hidden=score_hidden, value_dim=value_hidden
            )
            head_in = hidden
        self.norm = nn.LayerNorm(head_in)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(head_in, num_labels)
        self.num_labels = num_labels
        self.problem_type = problem_type
        self.attn_dim = attn_dim
        self.last_attention_weights: torch.Tensor | None = None
        self.set_backbone_frozen(freeze_backbone)
        # Stamp panel/pool config so a reloaded checkpoint rebuilds the right sizes.
        if self.config is not None:
            self.config.panel_mode = panel_mode
            self.config.panel_dim = self.panel_dim
            self.config.attn_dim = attn_dim
            self.config.pool_type = pool_type
            self.config.num_heads = num_heads

    def set_backbone_frozen(self, frozen: bool = True) -> None:
        """Freeze or unfreeze the backbone (toggles ``requires_grad`` + the no-grad forward).

        A frozen backbone is also put in ``eval()`` so its dropout is off and the
        extracted features are deterministic — otherwise the pool/head chase a moving
        target each step and cannot fit (see :meth:`train`).
        """
        self.freeze_backbone = frozen
        for p in self.bacformer.parameters():
            p.requires_grad = not frozen
        if frozen:
            self.bacformer.eval()
        else:
            self.bacformer.train(self.training)

    def train(self, mode: bool = True):
        """Set training mode, but keep a frozen backbone in ``eval()`` (dropout off).

        ``Trainer.train()`` flips the whole module to train mode at the start of
        training, which would re-enable backbone dropout and make the frozen features
        stochastic; this override pins the frozen backbone back to eval each time.
        """
        super().train(mode)
        if self.freeze_backbone:
            self.bacformer.eval()
        return self

    def _encode(
        self,
        protein_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None,
        contig_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run the backbone and return its ``last_hidden_state`` ``(batch, n, hidden)``."""
        out = self.bacformer(
            protein_embeddings=protein_embeddings,
            attention_mask=attention_mask,
            contig_ids=contig_ids,
            return_dict=True,
        )
        return out[0]

    def forward(
        self,
        protein_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        contig_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool = True,
        special_tokens_mask: torch.Tensor | None = None,
        panel: torch.Tensor | None = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        """Backbone → gated-attention pool → norm/dropout/linear → logits (and loss if labelled).

        The signature mirrors the upstream classifier (``special_tokens_mask`` /
        extra kwargs accepted for collate compatibility); the pooled attention
        weights are stashed on ``self.last_attention_weights`` for inspection.

        When ``panel_mode`` ≠ ``"none"`` a per-protein ``panel`` tensor of shape
        ``(batch, n, panel_dim)`` is concatenated onto the backbone tokens to form
        the scoring input; ``att_head`` pools the pure backbone value, ``e2e`` pools
        the panel-augmented value.
        """
        if self.freeze_backbone:
            with torch.no_grad():
                last_hidden = self._encode(protein_embeddings, attention_mask, contig_ids)
        else:
            last_hidden = self._encode(protein_embeddings, attention_mask, contig_ids)

        if self.panel_mode == "none":
            pooled, weights = self.pool(last_hidden, attention_mask)
        else:
            if panel is None:
                raise ValueError(f"panel_mode={self.panel_mode!r} requires a `panel` tensor")
            panel = panel.to(dtype=last_hidden.dtype)
            score = torch.cat([last_hidden, panel], dim=-1)
            value = score if self.panel_mode == "e2e" else last_hidden
            pooled, weights = self.pool(score, attention_mask, value_states=value)
        self.last_attention_weights = weights.detach()

        x = self.norm(pooled)
        x = self.dropout(x)
        logits = self.out_proj(x)  # (batch, num_labels)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.problem_type == "single_label_classification" and self.num_labels > 1:
                loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1).long())
            else:
                loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1).type_as(logits))

        return SequenceClassifierOutput(loss=loss, logits=logits)

    @classmethod
    def from_pretrained_backbone(
        cls,
        model_id: str,
        *,
        num_labels: int = 1,
        freeze_backbone: bool = False,
        attn_dim: int = 128,
        dropout: float = 0.1,
        panel_mode: str = "none",
        panel_dim: int = 9,
        pool_type: str = "gated_mil",
        num_heads: int = 8,
        **load_kwargs,
    ) -> BacformerAttnPoolForGenomeClassification:
        """Load the upstream classifier, lift its pretrained ``.bacformer`` backbone, swap the head.

        Parameters
        ----------
        model_id : str
            Hugging Face model id or local path of the Bacformer
            complete-genomes model.
        num_labels, freeze_backbone, attn_dim, dropout, panel_mode, panel_dim, pool_type, num_heads
            Forwarded to :meth:`__init__`.
        **load_kwargs
            Passed through to ``from_pretrained`` (e.g. ``dtype="auto"`` so a
            CPU smoke stays fp32 while GPU runs in bf16).

        Returns
        -------
        BacformerAttnPoolForGenomeClassification
            Wrapper whose new pool + head match the backbone dtype.
        """
        full = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=num_labels,
            problem_type="binary_classification",
            trust_remote_code=True,
            **load_kwargs,
        )
        backbone = full.bacformer  # discard full.classifier (the stock mean-pool head)
        hidden = full.config.hidden_size
        model = cls(
            backbone,
            hidden,
            num_labels=num_labels,
            attn_dim=attn_dim,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            panel_mode=panel_mode,
            panel_dim=panel_dim,
            pool_type=pool_type,
            num_heads=num_heads,
            config=full.config,
        )
        # Match the freshly-built modules to the backbone dtype (dtype="auto" ⇒ bf16
        # on GPU), so the pooled bf16 hidden states feed dtype-compatible Linears.
        backbone_dtype = next(backbone.parameters()).dtype
        model.pool.to(dtype=backbone_dtype)
        model.norm.to(dtype=backbone_dtype)
        model.out_proj.to(dtype=backbone_dtype)
        return model

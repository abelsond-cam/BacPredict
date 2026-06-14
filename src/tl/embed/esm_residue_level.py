"""Residue-level ESM-C (ESM++) operations shared by the ``snp_embeddings`` task.

This is the shared home for *per-residue* ESM-C work — the opposite end from the
production embedding path (:mod:`tl.embed.generate_embeddings`), which only ever
keeps the mean-pooled protein vector.

Step 3a (masked-marginal LLR) needs the masked-LM head and masked-marginal
log-probabilities at chosen residue positions. Step 3b (the geometry probe) adds
:func:`residue_states` (per-residue hidden states across all layers),
:func:`production_mean_pool` (the residue→protein mean the production path bakes
in, so ``d_pool`` measures the *real* pool), and :func:`apply_point_mutation`
(in-silico single-residue WT→mutant edits).

Tokeniser facts (verified against the cached ``modeling_esm_plusplus.py``):

- ESM++ wraps each sequence as ``<cls> A <eos>``, so amino-acid position ``p``
  (0-based in the protein string) lives at **token index ``p + 1``**.
- ``ESMplusplusForMaskedLM`` returns an ``ESMplusplusOutput`` with
  ``logits`` of shape ``[batch, seq_len, vocab]``.
- The vocabulary is character-level; ``tokenizer.convert_tokens_to_ids("L")``
  maps a one-letter amino acid to its logit column, and ``<mask>`` is a real
  token (``tokenizer.mask_token_id``).
"""

from __future__ import annotations

import logging

import torch
from transformers import AutoModelForMaskedLM

logger = logging.getLogger(__name__)

ESMC_MODEL_PATH = "Synthyra/ESMplusplus_small"

# Pin the trust_remote_code revision. Upstream pushed a broken
# modeling_esm_plusplus.py to `main` (commit 286a9db, 2026-06-09) whose
# ESMplusplusForMaskedLM subclasses an *undefined* `FastPLMTestTimeTrainingMixin`,
# so it raises NameError at import — and HF auto-downloads whatever `main` points
# to. We pin the 960-d ESM-C revision that produced the production embedding store
# (and that Bacformer Large consumes): the May-2026 `0c0b9c57` upload. PROVEN by
# byte-match — pooling this revision's per-residue states reproduces a stored
# pooled rpoB vector at cosine 0.999 (residual = bf16 rounding). The earlier
# Feb-2026 `d0be1083` upload is a *different model* (cosine 0.81 vs the same stored
# vector; its config.json claims 960 but its forward is not the store's ESM-C), so
# it must NOT be used. Pinning also dodges the broken `main` (286a9db).
ESMC_REVISION = "0c0b9c57a7c3da867c8512176ecddb3922816f80"

# Token-index of amino-acid position p, given the leading <cls> token.
_CLS_OFFSET = 1


def load_esmc_mlm(device: str = "cpu", dtype: str = "auto"):
    """Load the ESM++ masked-LM head + tokeniser.

    Pinned to :data:`ESMC_REVISION` — the 960-d ESM-C upload behind the production
    embedding store (proven by byte-match) — so it never pulls the broken ``main``.
    Uses ``dtype="auto"`` (not a manual ``.to(bfloat16)`` cast) so the model keeps
    its checkpoint dtype and Stage A CPU smoke tests work — matching the repo's
    Bacformer-loading idiom.

    Parameters
    ----------
    device : str, default "cpu"
        Torch device string (``"cpu"`` or ``"cuda:0"``).
    dtype : str, default "auto"
        Passed straight to ``from_pretrained(dtype=...)``.

    Returns
    -------
    tuple
        ``(model, tokenizer)`` — the model in ``.eval()`` mode on ``device``, and
        its ``EsmSequenceTokenizer`` (``model.tokenizer``).
    """
    model = AutoModelForMaskedLM.from_pretrained(
        ESMC_MODEL_PATH,
        revision=ESMC_REVISION,
        trust_remote_code=True,
        dtype=dtype,
    )
    model = model.to(device).eval()
    tokenizer = model.tokenizer
    return model, tokenizer


@torch.no_grad()
def masked_marginals(
    model,
    tokenizer,
    seq: str,
    positions: list[int],
    *,
    device: str = "cpu",
    batch_size: int = 16,
    expected_residues: dict[int, str] | None = None,
) -> dict[int, torch.Tensor]:
    """Masked-marginal log-probabilities at chosen residue positions.

    For each 0-based amino-acid position ``p`` in ``positions``, replace its
    token with ``<mask>``, run a forward pass, and return the
    log-softmax over the vocabulary at that position — i.e. ``log P(aa | rest of
    sequence)`` for every amino acid. The classic single-substitution effect
    score is then ``log_probs[p][alt] - log_probs[p][wt]``.

    Parameters
    ----------
    model, tokenizer
        As returned by :func:`load_esmc_mlm`.
    seq : str
        The protein sequence (the exact string ESM-C embedded). **Not** truncated
        here — pass the full sequence so masked positions keep their real context.
    positions : list of int
        0-based amino-acid indices to score.
    device : str, default "cpu"
        Torch device.
    batch_size : int, default 16
        Number of masked variants to forward at once.
    expected_residues : dict[int, str], optional
        Optional ``{position: amino_acid}`` map; each is asserted against the
        residue actually present at that position (catches off-by-one / wrong
        sequence). Raises ``ValueError`` on mismatch.

    Returns
    -------
    dict[int, torch.Tensor]
        ``{position: log_prob_vector}`` where each vector is shape ``[vocab]`` on
        CPU (float32).
    """
    if expected_residues:
        for p, aa in expected_residues.items():
            if not 0 <= p < len(seq):
                raise ValueError(f"position {p} out of range for sequence of length {len(seq)}")
            if seq[p] != aa:
                raise ValueError(f"expected {aa!r} at position {p}, found {seq[p]!r}")

    enc = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
    base_ids = enc["input_ids"][0]
    base_attn = enc["attention_mask"][0]
    mask_id = tokenizer.mask_token_id

    results: dict[int, torch.Tensor] = {}
    for start in range(0, len(positions), batch_size):
        batch_pos = positions[start : start + batch_size]
        rows = base_ids.unsqueeze(0).repeat(len(batch_pos), 1).clone()
        for r, p in enumerate(batch_pos):
            rows[r, p + _CLS_OFFSET] = mask_id
        attn = base_attn.unsqueeze(0).repeat(len(batch_pos), 1)
        out = model(input_ids=rows.to(device), attention_mask=attn.to(device))
        log_probs = torch.log_softmax(out.logits.float(), dim=-1)
        for r, p in enumerate(batch_pos):
            results[p] = log_probs[r, p + _CLS_OFFSET].cpu()
    return results


def aa_log_prob(log_prob_vector: torch.Tensor, tokenizer, amino_acid: str) -> float:
    """Read one amino acid's log-probability out of a masked-marginal vector."""
    token_id = tokenizer.convert_tokens_to_ids(amino_acid)
    return float(log_prob_vector[token_id].item())


def substitution_llr(
    log_prob_vector: torch.Tensor,
    tokenizer,
    wt: str,
    observed: str,
) -> float:
    """Masked-marginal log-likelihood ratio ``log P(observed) - log P(wt)``.

    Negative for a deleterious / unexpected substitution (the observed residue is
    less probable than wild-type given the rest of the protein); zero when the
    sample is wild-type at this site.
    """
    return aa_log_prob(log_prob_vector, tokenizer, observed) - aa_log_prob(log_prob_vector, tokenizer, wt)


@torch.no_grad()
def unmasked_logprobs(model, tokenizer, seq: str, *, device: str = "cpu") -> torch.Tensor:
    """Per-residue ``log P(observed | full context)`` from one unmasked forward.

    The cheap "surprisal" / naturalness score: forward the protein as-is and read,
    at every residue, the log-softmax of the observed amino acid given the *whole*
    sequence (the model can see ``x_i`` itself, so this is attenuated relative to
    the masked marginal — but it costs **one** forward for the entire protein, so
    it scales genome-wide). Low = the model is surprisald the residue is there =
    anomalous. Reference-free: no wild-type or alignment needed.

    Parameters
    ----------
    model, tokenizer
        As returned by :func:`load_esmc_mlm`.
    seq : str
        The protein sequence (the exact string ESM-C embedded). Passed with
        ``truncation=False`` so every residue keeps its real context.
    device : str, default "cpu"
        Torch device.

    Returns
    -------
    torch.Tensor
        ``[L]`` float32 on CPU, ``log P(observed)`` at each residue, aligned 1:1
        with ``seq`` (token ``p + 1`` → amino acid ``p``).
    """
    enc = tokenizer(seq, return_tensors="pt", add_special_tokens=True, truncation=False)
    ids = enc["input_ids"][0]
    out = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))
    log_probs = torch.log_softmax(out.logits[0].float(), dim=-1).cpu()  # [seq_len, vocab]
    length = len(seq)
    res = slice(_CLS_OFFSET, _CLS_OFFSET + length)
    return log_probs[res].gather(1, ids[res].unsqueeze(1)).squeeze(1)


@torch.no_grad()
def masked_logprobs(
    model,
    tokenizer,
    seq: str,
    *,
    device: str = "cpu",
    positions: list[int] | None = None,
    batch_size: int = 16,
) -> torch.Tensor:
    r"""Per-position masked-marginal ``log P(observed | context\i)`` (the gold standard).

    For each position in ``positions``, mask the residue, forward, and read the
    log-probability of the residue actually present given the rest of the protein
    — the ablation-by-masking surprisal. One forward **per position** (``L`` per
    protein if ``positions`` is the whole sequence), so for a genome-wide pass
    prefer :func:`unmasked_logprobs`; ``positions`` lets a caller restrict the
    masked computation to a small window (e.g. a resistance hotspot ±W).

    Parameters
    ----------
    model, tokenizer
        As returned by :func:`load_esmc_mlm`.
    seq : str
        The protein sequence.
    device : str, default "cpu"
        Torch device.
    positions : list of int, optional
        0-based amino-acid indices to score. ``None`` scores every position.
    batch_size : int, default 16
        Masked variants forwarded at once (passed to :func:`masked_marginals`).

    Returns
    -------
    torch.Tensor
        ``[len(positions)]`` float32, ``log P(observed)`` at each scored position,
        in the order of ``positions``.
    """
    if positions is None:
        positions = list(range(len(seq)))
    vectors = masked_marginals(model, tokenizer, seq, positions, device=device, batch_size=batch_size)
    return torch.tensor([aa_log_prob(vectors[p], tokenizer, seq[p]) for p in positions])


@torch.no_grad()
def residue_states(
    model,
    tokenizer,
    seq: str,
    *,
    device: str = "cpu",
    all_layers: bool = True,
    return_cls: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Per-residue ESM-C hidden states, ``<cls>``/``<eos>`` stripped, no truncation.

    Runs one forward pass with ``output_hidden_states=True`` and returns the
    residue rows aligned 1:1 with ``seq`` (token ``p + 1`` → amino acid ``p``).
    The full sequence is passed (``truncation=False``) so long proteins such as
    rpoB (~1,178 aa > the production 1,024 cap) keep every residue — the geometry
    probe needs them all.

    Parameters
    ----------
    model, tokenizer
        As returned by :func:`load_esmc_mlm`.
    seq : str
        Protein sequence (the exact string ESM-C embedded).
    device : str, default "cpu"
        Torch device.
    all_layers : bool, default True
        If True, return every hidden-state layer (embeddings + each transformer
        block) stacked as ``[n_layers, L, dim]``. If False, only the final layer
        ``[L, dim]``.
    return_cls : bool, default False
        If True, also return the ``<cls>`` token state(s) (``[n_layers, dim]`` or
        ``[dim]``) — used for the geometry probe's ``d_cls``.

    Returns
    -------
    torch.Tensor or tuple
        Residue states ``[n_layers, L, dim]`` (``all_layers=True``) or ``[L, dim]``,
        float32 on CPU, where ``L == len(seq)``. With ``return_cls`` a
        ``(residues, cls)`` tuple.

    Raises
    ------
    RuntimeError
        If ESM++ returns no hidden states (the day-one check for Step 3b).
    """
    enc = tokenizer(seq, return_tensors="pt", add_special_tokens=True, truncation=False)
    out = model(
        input_ids=enc["input_ids"].to(device),
        attention_mask=enc["attention_mask"].to(device),
        output_hidden_states=True,
    )
    hidden = out.hidden_states
    if not hidden:
        raise RuntimeError(
            "ESM++ returned no hidden_states — Step 3b needs output_hidden_states; "
            "check the cached modeling_esm_plusplus.py forward signature."
        )
    length = len(seq)
    residue_slice = slice(_CLS_OFFSET, _CLS_OFFSET + length)

    def _strip(layer: torch.Tensor) -> torch.Tensor:
        return layer[0, residue_slice].float().cpu()

    def _cls(layer: torch.Tensor) -> torch.Tensor:
        return layer[0, 0].float().cpu()

    if all_layers:
        residues = torch.stack([_strip(layer) for layer in hidden], dim=0)
        cls = torch.stack([_cls(layer) for layer in hidden], dim=0)
    else:
        residues = _strip(hidden[-1])
        cls = _cls(hidden[-1])
    return (residues, cls) if return_cls else residues


def production_mean_pool(
    residue_matrix: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    max_residues: int | None = None,
) -> torch.Tensor:
    """Mask-normalised residue→protein mean — the pool the production path bakes in.

    The ESM-C per-protein vector the embedding store holds is the **straight mean**
    over a protein's residue hidden states. This reproduces it as an einsum so
    ``d_pool`` in the geometry probe measures the *real* pool:
    ``einsum("ld,l->d", H, m) / m.sum()``. With ``mask=None`` this is exactly
    ``residue_matrix.mean(0)``.

    Parameters
    ----------
    residue_matrix : torch.Tensor
        ``[L, dim]`` per-residue states for one protein.
    mask : torch.Tensor, optional
        ``[L]`` 1/0 real-residue mask. Defaults to all-ones.
    max_residues : int, optional
        Pool over only the first ``max_residues`` rows — set to the production
        ``max_prot_seq_len`` (1,024) to byte-match a stored pooled vector for a
        protein longer than the cap.

    Returns
    -------
    torch.Tensor
        ``[dim]`` pooled vector (float32).
    """
    h = residue_matrix.float()
    if max_residues is not None:
        h = h[:max_residues]
    length = h.shape[0]
    m = torch.ones(length, dtype=torch.float32) if mask is None else mask[:length].float()
    denom = m.sum()
    if denom <= 0:
        raise ValueError("production_mean_pool: empty mask (no residues to pool)")
    return torch.einsum("ld,l->d", h, m) / denom


def apply_point_mutation(seq: str, position: int, new_aa: str, *, expected_wt: str | None = None) -> str:
    """Return ``seq`` with the residue at 0-based ``position`` replaced by ``new_aa``.

    Parameters
    ----------
    seq : str
        Wild-type protein sequence.
    position : int
        0-based residue index to mutate.
    new_aa : str
        Replacement amino acid (single letter).
    expected_wt : str, optional
        If given, assert the residue currently at ``position`` matches (guards
        against a wrong reference / off-by-one).

    Returns
    -------
    str
        The mutated sequence (same length).
    """
    if not 0 <= position < len(seq):
        raise ValueError(f"position {position} out of range for sequence of length {len(seq)}")
    if len(new_aa) != 1:
        raise ValueError(f"new_aa must be a single residue, got {new_aa!r}")
    if expected_wt is not None and seq[position] != expected_wt:
        raise ValueError(f"expected wild-type {expected_wt!r} at position {position}, found {seq[position]!r}")
    return seq[:position] + new_aa + seq[position + 1 :]

"""Residue-level ESM-C (ESM++) operations shared by the ``snp_embeddings`` task.

This is the shared home for *per-residue* ESM-C work — the opposite end from the
production embedding path (:mod:`tl.embed.generate_embeddings`), which only ever
keeps the mean-pooled protein vector.

Increment 1 (Stage 1.1 ceiling ladder) needs the masked-LM head and
masked-marginal log-probabilities at chosen residue positions. Increment 2
(Stage 1.2 geometry probe) will extend this module with ``residue_states`` for
per-residue hidden states across all layers.

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

# Token-index of amino-acid position p, given the leading <cls> token.
_CLS_OFFSET = 1


def load_esmc_mlm(device: str = "cpu", dtype: str = "auto"):
    """Load the ESM++ masked-LM head + tokeniser.

    Uses ``dtype="auto"`` (not a manual ``.to(bfloat16)`` cast) so the model
    keeps its checkpoint dtype and Stage A CPU smoke tests work — matching the
    repo's Bacformer-loading idiom.

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

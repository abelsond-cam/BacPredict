"""Low-level protein/genome vector primitives shared by every Bacformer forward producer.

Three primitives, extracted from the (previously copy-forked) per-genome loops in the cache/vector
producers so the store-layout seam, the flat-order guard, and the mask-mean pool live in **one** place:

- :func:`real_protein_indices` — which rows of a stored ``.pt`` are real proteins (flat order); the seam
  between the two on-disk store layouts and the flat protein order everything keys on
  (:func:`bacpredict.engine.gene_lr.locate_gene.flatten_proteins`). Torch-only and tiny so ``locate_gene``
  (parquet, no torch) need not import the model stack to reason about flat indices.
- :func:`real_protein_rows` — the real-protein rows of one genome's ``last_hidden_state``, in flat order,
  with the day-one length guard (the Bacformer output must align 1:1 with the input rows, or every flat
  index is off by an injected CLS). Producers used to inline this block with a ``length_checked`` flag that
  guarded only the first genome; here the guard runs per genome (a shape compare, negligible next to the
  forward) so a late misalignment can't slip through.
- :func:`genome_mean_pool` — the mask-normalised genome mean over those rows: exactly the pool the
  genome-classification head averages, and gene/drug-agnostic.
"""

from __future__ import annotations

import numpy as np
import torch

# bacformer SPECIAL_TOKENS_DICT — real protein rows carry PROT_EMB in the stored tensor's
# special_tokens_mask.
PROT_EMB_TOKEN_ID = 4


def real_protein_indices(store: dict, n_rows: int) -> torch.Tensor:
    """Raw row indices of the real proteins (flat order) in a stored ``.pt``.

    Two store layouts exist:

    - **Bacformer-input bundle** (``protein_embeddings_to_inputs``): interleaves CLS/SEP/pad rows with
      real proteins, flagged by ``special_tokens_mask == 4``.
    - **Plain per-protein** (TB store): one row per protein already, with an ``attention_mask`` marking
      real vs padding.

    Returns the raw indices of the real-protein rows, in flat order, matching
    :func:`bacpredict.engine.gene_lr.locate_gene.flatten_proteins`. Working with indices (not a
    boolean-masked copy) lets the caller read a single row out of an mmap'd tensor instead of
    materialising the whole ``[T, dim]`` block.
    """
    if "special_tokens_mask" in store:
        mask = store["special_tokens_mask"][0] == PROT_EMB_TOKEN_ID
    elif "attention_mask" in store:
        mask = store["attention_mask"][0].bool()
    else:
        return torch.arange(n_rows)
    return torch.nonzero(mask, as_tuple=False).flatten()


def real_protein_rows(last_hidden_state: torch.Tensor, real_idx: torch.Tensor, *, input_len: int) -> torch.Tensor:
    """Real-protein rows of a Bacformer forward, flat order ``[n_real, dim]`` float.

    ``last_hidden_state`` is the model's output for one genome (a ``[T, dim]`` tensor, or ``[1, T, dim]``
    which is squeezed). ``real_idx`` comes from :func:`real_protein_indices` on the same store, and
    ``input_len`` is the store's input row count (``protein_embeddings.shape[1]``).

    A **day-one guard** asserts the output aligns 1:1 with the input rows before any row is trusted: an
    injected CLS (or any length change) would shift every flat index. Returns ``last_hidden_state`` indexed
    at ``real_idx`` and cast to float — the contextualised real-protein tokens the genome mean pools over
    and the per-segment token extraction indexes into.
    """
    lhs = last_hidden_state[0] if last_hidden_state.dim() == 3 else last_hidden_state
    if lhs.shape[0] != input_len:
        raise RuntimeError(
            f"Bacformer last_hidden_state length {lhs.shape[0]} != input length {input_len}: "
            f"the real-protein rows would be misaligned. Aborting."
        )
    return lhs[real_idx].float()


def genome_mean_pool(real_rows: torch.Tensor | np.ndarray) -> np.ndarray:
    """Mask-normalised genome mean over the flat real-protein rows → ``[dim]`` float32 numpy.

    The pool the genome-classification head averages over — gene/drug-agnostic. Accepts the torch rows
    from :func:`real_protein_rows` or a numpy copy of them (some producers materialise numpy to index
    per-segment tokens), reducing over the row axis either way.
    """
    if torch.is_tensor(real_rows):
        return real_rows.mean(dim=0).cpu().numpy()
    return real_rows.mean(axis=0)

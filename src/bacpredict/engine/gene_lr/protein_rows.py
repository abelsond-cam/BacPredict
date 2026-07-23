"""Recover the real-protein rows (flat order) of a stored ``.pt`` embedding bundle.

The seam between the two on-disk store layouts and the flat protein order everything else keys on
(:func:`bacpredict.engine.gene_lr.locate_gene.flatten_proteins`). Kept torch-only and tiny so
``locate_gene`` (parquet, no torch) need not import the model stack to reason about flat indices.
"""

from __future__ import annotations

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

"""Transition re-export — the split utilities now live in :mod:`bacpredict.engine.splits.generate_kfold_splits`.

Kept ONLY so out-of-scope / not-yet-migrated importers keep working during the ``segment_amr_lr`` refactor —
notably the **parked** ``src/kleb_iso_source/train_isolation_source.py`` (an uncommitted edit that must not be
touched) and the two read-out modules being deleted in the unification (``gene_lr/kfold_probe.py``,
``gene_lr/coding_amr_lr.py``). Remove this shim once those land / are deleted and every consumer imports from
``bacpredict.engine.splits``.
"""

from __future__ import annotations

from bacpredict.engine.splits.generate_kfold_splits import add_splits, generate_kfold_splits

__all__ = ["add_splits", "generate_kfold_splits"]

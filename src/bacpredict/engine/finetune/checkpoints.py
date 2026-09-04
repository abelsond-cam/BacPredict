"""Choose which checkpoint a resumed run should continue from.

⚠ **Shared engine module** — organism-agnostic by design; the isolation-source trainer is its first
caller and the AMR trainer wants the same behaviour.

``transformers.trainer_utils.get_last_checkpoint`` returns the highest-numbered ``checkpoint-N``
directory and asks nothing about whether it is usable. That is fine when a run ends cleanly and wrong
when the thing that ended it was a wall clock: a job killed mid-save leaves a directory holding a
model file and no ``trainer_state.json``, and every subsequent link of a chained run would pick that
same corpse, fail, and take the rest of the chain with it. Over an unattended multi-day sweep that
turns one unlucky second into a lost week.

:func:`pick_resume_checkpoint` walks the checkpoints newest-first and returns the first *complete*
one. Completeness is the presence of ``trainer_state.json``, which HF writes **last** in
``_save_checkpoint`` — after the model, the optimizer, the scheduler and the RNG state — so its
presence means everything before it landed.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")

#: Written last by ``Trainer._save_checkpoint``, so it is the marker that the whole save completed.
COMPLETION_MARKER = "trainer_state.json"


def list_checkpoints(output_dir: str | Path) -> list[Path]:
    """Every ``checkpoint-N`` directory under *output_dir*, newest (highest N) first."""
    root = Path(output_dir)
    if not root.is_dir():
        return []
    found = [(int(m.group(1)), p) for p in root.iterdir()
             if p.is_dir() and (m := CHECKPOINT_RE.match(p.name))]
    return [p for _, p in sorted(found, key=lambda t: t[0], reverse=True)]


def is_complete(checkpoint: str | Path) -> bool:
    """Whether a checkpoint directory was fully written."""
    return (Path(checkpoint) / COMPLETION_MARKER).is_file()


def pick_resume_checkpoint(output_dir: str | Path) -> Path | None:
    """The newest complete checkpoint in *output_dir*, or ``None`` to start fresh.

    Returning ``None`` for an empty or non-existent directory is the point of the ``auto`` mode: link
    1 of a chain has no checkpoint and every later link does, so both are the same submission.

    Any incomplete checkpoint newer than the chosen one is logged by name — silently stepping back
    would hide that a save was interrupted, and losing an eval interval of training is worth a line in
    the log rather than a mystery in the step count.
    """
    checkpoints = list_checkpoints(output_dir)
    skipped = []
    for ckpt in checkpoints:
        if is_complete(ckpt):
            if skipped:
                logger.warning(
                    "skipping %d incomplete checkpoint(s) newer than %s (no %s — a save was "
                    "interrupted, most likely by the wall clock): %s",
                    len(skipped), ckpt.name, COMPLETION_MARKER, ", ".join(p.name for p in skipped),
                )
            return ckpt
        skipped.append(ckpt)
    if skipped:
        logger.warning(
            "found %d checkpoint(s) but none is complete (%s missing from all of them) — starting "
            "fresh: %s", len(skipped), COMPLETION_MARKER, ", ".join(p.name for p in skipped),
        )
    return None

r"""Stage B (CLAUDE.md §0.2): did a 10-genome smoke run overfit its train = validate = holdout set?

Reads one ``finetune_amr --split-table --n-samples 10`` run directory:

- the whole run's training history from ``<run>/trainer_state.json``, which split-table mode writes after
  ``train()`` (``Trainer.save_state``). Never from ``checkpoint-*/trainer_state.json``: with ``save_total_limit=1``
  and ``load_best_model_at_end`` HF deletes every checkpoint but the best at the end of training, and the best is
  the first evaluation to reach the top AUROC, so its history stops there, often with the loss still near its start;
- the holdout AUROC from ``<run>/results.json`` (holdout = train on this path).

PASS needs all of: the run went its whole schedule (``global_step == max_steps``: early stopping on a saturated
train = validate AUROC ends it before the loss has fallen, so the smoke runs with patience off), the logged loss
reaches the run's last logging interval, the last logged train loss ``<= max_loss``, and holdout AUROC
``>= min_auroc``.

Usage::

    python -m bacpredict.engine.finetune.check_overfit <run dir> [--max-loss 0.1] [--min-auroc 0.99]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def stage_b(run_dir: str | Path, max_loss: float = 0.1, min_auroc: float = 0.99) -> tuple[bool, list[str]]:
    """Judge Stage B on one run directory; return ``(passed, report lines)``.

    Raises ``FileNotFoundError`` when the run has no top-level ``trainer_state.json`` or ``results.json`` (not
    finished, failed, or trained by a trainer that did not save its state).
    """
    run = Path(run_dir)
    state_file, results_file = run / "trainer_state.json", run / "results.json"
    for f in (state_file, results_file):
        if not f.exists():
            raise FileNotFoundError(f"no {f}: the smoke has not finished, failed, or predates the trainer saving its "
                                    "state (checkpoint-*/trainer_state.json stops at the best checkpoint: never read)")
    state = json.loads(state_file.read_text())
    global_step, max_steps = state["global_step"], state["max_steps"]
    logging_steps = state.get("logging_steps") or 1
    history = state["log_history"]
    loss = [(h["step"], h["loss"]) for h in history if "loss" in h]
    auroc = [h["eval_auroc"] for h in history if "eval_auroc" in h]
    holdout = json.loads(results_file.read_text())["holdouts"]["holdout"]["metrics"]["auroc"]

    lines, problems = [], []
    if global_step < max_steps:
        problems.append(f"the run stopped at step {global_step} of {max_steps} (early stopping?): Stage B judges the "
                        "loss at the end of the whole schedule")
    if not loss:
        problems.append("no train loss logged")
    else:
        last_step, last_loss = loss[-1]
        if last_step < global_step - logging_steps:
            problems.append(f"the logged loss ends at step {last_step}, the run at {global_step}")
        lines.append(f"train loss: first {loss[0][1]:.4f} (step {loss[0][0]}), min {min(v for _, v in loss):.4f}, "
                     f"last {last_loss:.4f} (step {last_step} of {global_step}; {len(loss)} logged)")
        if last_loss > max_loss:
            problems.append(f"last train loss {last_loss:.4f} > {max_loss}")
    if auroc:
        lines.append(f"eval AUROC (validate = train): last {auroc[-1]:.3f}, max {max(auroc):.3f}")
    lines.append(f"holdout (= train) AUROC {holdout:.3f}")
    if not holdout >= min_auroc:
        problems.append(f"holdout AUROC {holdout:.3f} < {min_auroc}")
    lines += [f"FAIL {p}" for p in problems]
    ok = not problems
    lines.append(f"Stage B ({'PASS' if ok else 'FAIL'}): whole schedule run, last train loss <= {max_loss}, "
                 f"train = holdout AUROC >= {min_auroc}")
    return ok, lines


def main(argv: list[str] | None = None) -> int:  # noqa: D103
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run_dir")
    p.add_argument("--max-loss", type=float, default=0.1)
    p.add_argument("--min-auroc", type=float, default=0.99)
    a = p.parse_args(argv)
    try:
        ok, lines = stage_b(a.run_dir, a.max_loss, a.min_auroc)
    except FileNotFoundError as e:
        print(e)
        return 1
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

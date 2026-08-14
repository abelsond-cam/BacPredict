# Memory snapshot — 2026-08-14

A verbatim copy of all 52 files in the agent memory directory
`~/.claude/projects/-Users-davidabelson-developer-BacPredict/memory/`, taken **before** the
consolidation that reduces them to ~18 durable memories.

## Why this exists

The memory directory is not under version control, so it has no history and no rollback. Several of
these files are the **sole record** of a decision — including two data-safety guards whose loss
causes destructive mistakes:

- the git-rm scope: untracked `*_amr_ladder_table.csv` regenerate freely, but curated
  `*_card_ladder_table_family.csv` / `rif_ladder_table.md` need David's confirmation before deletion;
- `min_size` stays 100 for the invasion lineage clusters — **do not regenerate min50**.

This snapshot makes the consolidation reversible. Decisions of record are harvested out of these
files into `PROJECT_STATE.md` §6 as part of the same work.

## What is in here — and what it is not

**These files are a historical record, not guidance.** Do not act on anything in this directory.
They are exactly what was being cleaned up, and they are known to contain:

- **stale numbers** — pre-July-re-run fine-tune AUROCs (e.g. colistin 0.8072, since 0.9094) and
  pre-leak-fix ladder numbers (Kp azithromycin 0.918, honestly 0.799);
- **dead live-run state** — hundreds of SLURM job IDs, `RUNNING`/`PENDING`/`NEXT:` markers, and
  cluster-posture claims that were superseded within days;
- **at least one actively harmful instruction** — `bacformer_loading_idiom.md` recommends
  `dtype="auto"`, which silently loads Bacformer-large in fp32 and is the documented root cause of a
  ~5 pp AUROC loss;
- **~18 direct contradictions between files**, which is what prompted the audit.

The current, verified statement of where the project stands is **`PROJECT_STATE.md`** at the repo
root. Where this snapshot and that file disagree, that file wins — without exception.

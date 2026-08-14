---
name: isambard-ft-fanout-run-mechanics
description: How to launch BacPredict AMR fine-tunes on Isambard — worktree/PYTHONPATH/QOS/launch-command mechanics that were non-obvious and cost a turn to work out
metadata:
  node_type: memory
  type: project
  originSessionId: 287d1555-8fdc-4241-9446-ecac54a396be
---

Operational how-to for the AMR FT fan-out on Isambard (worked out 2026-07-17). Related:
[[bacpredict-engine-consolidation]], [[bacpredict-dual-cluster-data-root]], [[real-numbers-causal-lr-plan]].

**Where my branch lives.** Isambard `$HOME/BacPredict` is checked out on branch **`dev`** (another agent) —
NOT `refactor/consolidate-engine`. My branch is a **detached worktree** at
`$SCRATCHDIR/worktrees/consolidate` (`$SCRATCHDIR`=`/scratch/u6fp/dca36.u6fp`). Never `git checkout` the
shared `$HOME` tree. Advance the worktree with:
`git -C $WT fetch -q origin refactor/consolidate-engine && git -C $WT checkout -qf <sha>`.

**Two worktrees (to honour the coordination caveat).** While the FT fan-out is PENDING, `consolidate`
stays **pinned** at the FT commit and a **second** worktree `$SCRATCHDIR/worktrees/concat` carries all
concurrent CPU/cache work (WS-A upstream/IGR LR edits, WS-B3 FT-mean cache). Create it detached:
`git -C $WT worktree add --detach $SCRATCHDIR/worktrees/concat <sha>`. Both are mine; adding a worktree is
additive and does not touch `$HOME`(dev) or the pinned tree.

**⚠ Tracking-ref foot-gun (cost a turn).** `git fetch origin <branch>` (branch by NAME) updates only
`FETCH_HEAD`, NOT `refs/remotes/origin/<branch>` — so `git worktree add … origin/refactor/consolidate-engine`
checked out a **stale** tip (316d13a, not the pushed HEAD) even though the objects were present. Fixes:
checkout the **explicit SHA** (`git -C $NEW checkout -qf <sha>`), or refresh the tracking ref with an
explicit refspec `git -C $WT fetch -q origin '+refs/heads/refactor/consolidate-engine:refs/remotes/origin/refactor/consolidate-engine'`.
The memory'd advance command is fine because it checks out an explicit `<sha>` (never the tracking ref).

**PYTHONPATH is mandatory.** `bacpredict` is **NOT pip-installed** in the gpu-venv
(`$SCRATCHDIR/envs/bacpredict-gpu-venv/bin/python`), and `$HOME/BacPredict/src` (on `dev`) has **no
`bacpredict` package**. So the FT scripts import nothing unless PYTHONPATH points at the worktree. Both
`train_on_slurm_amr.sh` (Kp) and `train_on_slurm_amr_tb.sh` (TB) now use
`export PYTHONPATH="${BACPREDICT_REPO:-$HOME/BacPredict}/src:…"` — launch with `BACPREDICT_REPO=$WT`.

**Launch command** (single canonical Stage-C run = fold0/seed1; both scripts DEFAULT to `--array=0-14`
kfold sweep — you MUST override `--array=0`). TB now honours `DRUG` too (`drug=${DRUG:-rifampin}`):
```
WT=/scratch/u6fp/dca36.u6fp/worktrees/consolidate
sbatch --array=0 --time=24:00:00 --job-name=<org>_<drug> \
  --export=ALL,BACPREDICT_REPO=$WT,DRUG=<drug> \
  $WT/src/bacpredict/apps/{kleb,tb}/scripts/train_on_slurm_amr{,_tb}.sh
```
Results land: Kp `$SCRATCHDIR/processed/train_kleb_ast/models/finetune/klebsiella_pneumoniae_<drug>_lr_0.00015_finetuned_fold00_seed1/`;
TB `…/train_tb_ast/checkpoints/mycobacterium_tuberculosis_<drug>_lr_0.00015_finetuned_fold00_seed1/`
(each writes `results.json` on completion).

**QOS wall caps + GPU mem — CORRECTED 2026-07-20 (cost a turn AGAIN; the old note below was wrong).** My
assoc has ONLY `qos=normal`; **`restricted48` is NOT grantable to me** → `Invalid qos specification`. The
effective wall is `workq_qos` **MaxWall=24h with `DenyOnLimit`**, and requesting `--time=24:00:00` (exactly
the cap) is **REJECTED** (`QOSMaxWallDurationPerJobLimit`) — use **`--time=23:00:00`**. A 1-GPU job = **one
socket = 72 cores + ~115 GB** (`workq` `DefMemPerGPU=115000`, per-socket allocation); the committed FT
scripts' **`--mem=250G` blows the per-socket size** → `Job violates ... user's size ... limits`. Override
**`--mem=110G`** (FT is batch-size-1 lazy-load, real peak ≪115G). **Canonical GPU submit:**
`sbatch --array=0 --qos=normal --time=23:00:00 --mem=110G --export=ALL,BACPREDICT_REPO=$WT,DRUG=<drug> <launcher>`.
Proven 2026-07-20: 20-drug fan-out `5733635`–`5733654` accepted with exactly this. Checkpoints save each eval
so a wall-hit is `--resume-from-checkpoint`-able. (TODO: fix the committed scripts' `--mem`/`--time` defaults.)

**⚠ Short CPU array jobs: request a TIGHT `--time`, not the "be generous" default (David flagged 2026-07-21).**
Wall time is metered on *use*, so an over-long request costs no compute — BUT it wrecks **backfill**: the
scheduler can slot a 1 h job into a gap ahead of a big job, not an 8 h one, so over-long CPU arrays sit
`PENDING (Priority)` behind the GPU job for ages. **Measured 2026-07-22 — the important fact: ALL these LR rankings finish in MINUTES**, not tens of minutes.
Per-gene (`5743034`/`5743036`) ran **~2–6 min/drug** (max ~10; one 6 s early-skip); whole-IGR (`5743579`) —
the *heaviest* of them — still only **~4–13 min/drug** (most ~8–11). 32c/128G/task, baclm store. So a
**`--time=00:15:00` (per-gene) / `00:30:00` (whole-IGR)** wall is already generous headroom — the old "~30 min"
guess overstated them badly; NEVER hours. (Overnight, when backfill timing stops mattering, a "to be sure" bump
to ~2.5 h is a fine exception, as David did — but that is the exception, not the size of the job.) Already-queued jobs: lower in place
with `scontrol update JobId=<id> TimeLimit=01:00:00` (works on a pending array's remaining tasks, keeps queue
position; reason flips `Priority`→`None`). The "be generous with `--time`, 24 h default" rule is for the long
GPU FT jobs (a wall-hit throws away GPU-hours) — it does NOT apply to short CPU rankings.

**⚠ Worktree-coordination caveat.** FT jobs `import bacpredict` from the worktree **at job start** (runtime),
not submit time. A RUNNING job holds its imports (safe), but a job still PENDING when you advance the worktree
picks up the NEW code on start. So while an FT fan-out is pending, do NOT advance `$SCRATCHDIR/worktrees/consolidate`
with Python changes — use a **separate worktree** for concurrent CPU work (WS-A) or wait until all FT jobs are
RUNNING. (Shell-script-only advances are safe: SLURM copies the sbatch script at submit.)

**Dry-check before launch** (login-safe, import+argparse only, no GPU):
`BACPREDICT_REPO=$WT PYTHONPATH=$WT/src $PY -m bacpredict.engine.finetune.finetune_amr --help`.
SSH: `ssh -o IdentitiesOnly=yes -o BatchMode=yes u6fp.aip2.isambard` (key-based, no MFA).

---
name: hpc-no-data-in-home-use-rds-scratch
description: "UoHPC/CSD3: /home is quota-limited & code-only — stage data/logs/scratch to ~/rds/hpc-work or project_k (agnostic $HOME rule now in global CLAUDE.md; Isambard uses $SCRATCHDIR/$PROJECTDIR)"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 43f9dbd0-7aa0-4579-ab80-2e7170ce8eb3
---

> **Agnostic rule ("$HOME = code only; stage data/logs/scratch elsewhere") now in
> `~/.claude/CLAUDE.md` → "Working on HPC clusters".** Paths below are **UoHPC (CSD3)**
> (`~/rds/hpc-work`, `project_k`) — see `~/.claude/cluster_uohpc.md`. On Isambard the equivalents
> are `$SCRATCHDIR` (5 TiB, 60-day) + `$PROJECTDIR` (`/projects/u6fp`) — see `~/.claude/cluster_isambard.md`.

On CSD3, **/home (`/home/dca36`, ~52 GB quota) is strictly for code** — never put data, SLURM
`--output`/`--error` logs, large staged files, or scratch there. (Caught with a 1 GB `mash_sketch.msh`
+ experiment intermediates in `~/lb_scratch`, which had pushed /home to 99.6% of quota; the user called
it "TERRIBLE practice".)

Use instead:
- **`~/rds/hpc-work/`** (personal_rds, 1 TB RDS) — scratch: staged helper scripts, SLURM logs,
  caches (`UV_CACHE_DIR`), experiment intermediates. A dedicated `~/rds/hpc-work/pyseer_scratch` now exists.
- **`project_k/david/processed/<task>/`** (20 TB RDS) — real working data + result artifacts.

**Why:** /home is small and shared with the git checkouts + uv/pixi envs; filling it to quota breaks
running jobs and the user's own login shell. The storage-tier table in global CLAUDE.md documents the
tiers but did not stop me defaulting a working dir to `~` — hence this sharper rule.

**How to apply:** in every HPC script, point SBATCH `--output`/`--error`, staging dirs, and all data
outputs at RDS — default scratch to `~/rds/hpc-work`, bulk data to `project_k`. Never default a working
dir to `~`/`$HOME`. When tidying, `mv` data to RDS then delete from home. Relates to
[[csd3-spack-module-breaks-uv]].

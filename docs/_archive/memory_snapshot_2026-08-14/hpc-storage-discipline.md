---
name: hpc-storage-discipline
description: "Hard rules for HPC storage — du ban, count×size, nothing in $HOME, logs/envs/caches off home (agnostic rules now in global CLAUDE.md; CSD3 numbers = UoHPC)"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b4b920d3-63cd-41a6-a408-e0811a78fceb
---

> **The cluster-agnostic rules below now live in `~/.claude/CLAUDE.md` → "Working on HPC clusters"
> → STORAGE DISCIPLINE, applying on every cluster.** The specific quotas/paths here (~52 GB `/home`,
> `project_k`, `home_envs`, backup script) are **UoHPC (CSD3)** — see `~/.claude/cluster_uohpc.md`.
> The Isambard equivalents (`$HOME` 100 GiB / `$SCRATCHDIR` 5 TiB / `$PROJECTDIR` 200 TiB) are in
> `~/.claude/cluster_isambard.md`.

David was angry (twice) that agents keep overflowing the tiny ~52 GB `/home` quota with logs,
caches, and whole conda envs, and bloating scratch/project_k.

**Why:** `/home` is ~52 GB and chronically over quota; recursive `du` on these trees (hundreds
of thousands of files) takes *days* and crashes — it is explicitly banned, even inside a SLURM job.

**How to apply:**
- **Never run recursive / across-directory `du`** on `/home`, RDS, or any large tree. Size a
  directory by **count × one representative file**: `ls DIR | grep '\.pt' | wc -l`, then
  `du -h ONEFILE` (or `find DIR -maxdepth 1 -name '*.pt' -printf '%s %f\n' -quit`), multiply.
  Single-file / handful-of-named-files `du -h` is the only allowed `du`.
- **`$HOME` = code + dotfiles only.** No data, no run output, no caches.
- **SLURM `--output`/`--error`/`--chdir` → `project_k/david/processed/<task>/logs/` or scratch** —
  never `$HOME`, never the git repo.
- **Conda/mamba envs live at `project_k/david/home_envs/{conda_envs,mamba/envs}`** (point via
  `.condarc` envs_dirs + `.bashrc` MAMBA_ROOT_PREFIX); caches on scratch (`HF_HOME`,
  `UV_CACHE_DIR`, `XDG_CACHE_HOME`, …). Never create an env under `$HOME`.
- **Deletions are dry-run-first**: list paths + count×one-file estimate, get approval, then delete.
  Check mtimes (modified today ⇒ probably active) and `squeue` — never delete another agent's live work.
- **`bacformer RDS` (`rds-9X9gY1OFt4M`, 13 TB at 98%) is a shared lab mount — hands off.**

Full rules now codified in global `~/.claude/CLAUDE.md` ("STORAGE DISCIPLINE"),
`~/.claude/hpc_storage_overview.md`, and both repo `CLAUDE.md` files. Tiers/paths in
[[hpc-no-data-in-home-use-rds-scratch]].

**State after 2026-06-26 cleanup (durable facts):**
- Conda/mamba envs were relocated out of `$HOME` to `/rds/project/.../david/home_envs/{conda_envs,mamba/envs}`
  via `conda create --clone`; `.condarc` envs_dirs/pkgs_dirs + `.bashrc` `MAMBA_ROOT_PREFIX` repoint there
  (home left only as fallback). Activate by name as normal. `$HOME` brought 54→~40 GB.
- `~/storage_mgt/backup_rds_to_rcs.sh` mirrors **`project_k/david`, `~/rds/hpc-work`, `~/workspace`** → RCS
  on login (no `$HOME` backup). RCS `/rcs3` ~2.4/50 TB — ample for archival; prefer **bulk tarballs** over
  many small files (e.g. envs archived as one `.tar.gz` under RCS `David/env_backups/`).
- project_k (`rds-4k08a2yyQLw`) is **shared**: `aaron/ adam/ camila/ david/ klebsiella/`. Stay in `david/`
  unless cross-user dedupe is explicitly approved with proof-of-location.
- ESM embedding store ≈ 1.88 TB (87,718 × ~21 MB) — keep; the dedupe target is *regenerated/duplicate* copies.

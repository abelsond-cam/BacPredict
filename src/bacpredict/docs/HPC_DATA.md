# HPC (CSD3/UoHPC) data map — BacPredict

Where BacPredict's data lives on **CSD3/UoHPC** (Cambridge), and how the data-root resolver maps onto
it. Sibling doc: [`ISAMBARD_DATA.md`](ISAMBARD_DATA.md) (live-verified). Cluster reference:
[`~/.claude/cluster_uohpc.md`](~/.claude/cluster_uohpc.md).

> ⚠️ **Seeded from `cluster_uohpc.md`, NOT live-verified — CSD3 is down (data-centre failure,
> 27 Jun 2026).** Roots and tiers below are from the cluster doc; **all per-store contents, file
> counts, and checkpoint/`final/` locations are `TODO (confirm)` on CSD3's return.** Once verified,
> promote this to the same live-checked status as `ISAMBARD_DATA.md`.

## Root — the single knob

```bash
export BACPREDICT_DATA_ROOT=~/rds/rds-floto-bacterial-4k08a2yyQLw/david   # = project_k/david
```

BacPredict's working data is the **`project_k/david` RDS allocation** (20 TB, the main working-data
home). The code resolves the root via `bacpredict.engine.config.resolve_data_root()`: `--data-root`
arg → `$BACPREDICT_DATA_ROOT` → `$SCRATCHDIR` → **this CSD3 path if it exists on disk** → error. On
CSD3 there is no `$SCRATCHDIR` in the Isambard sense, so **set `$BACPREDICT_DATA_ROOT` explicitly**
in `.bashrc` (the autodetect fallback works only because the path exists, but the export is clearer).

## Storage tiers

| Tier | Path | Size | Holds for BacPredict |
|---|---|---|---|
| Home | `~/workspace` = `/home/dca36/workspace/` | ~52 GB (all `/home`) | **code only** — the `BacPredict` checkout (chronically near-full; code + dotfiles only) |
| `project_k` (RDS) | `~/rds/rds-floto-bacterial-4k08a2yyQLw/david` | 20 TB (shared) | **all working data** — raw/ processed/ final/ (stay in `david/`) |
| `personal_rds` (RDS) | `~/rds/hpc-work/` | 1 TB | scratch — SLURM logs, caches, staged scripts, intermediates |
| `bacformer_rds` (RDS) | `~/rds/rds-flotolab-9X9gY1OFt4M/` | 13 TB (shared) | Bacformer mount — **hands off**, shared lab mount |
| `cold_storage` (RCS) | `~/rcs/rcs-vgm23-lcms/David/` | ~50 TB | backups + published outputs (archival; write tarballs, not many small files) |

Caches/env off `/home` via `.bashrc` (`UV_CACHE_DIR`, `HF_HOME`, `TORCH_HOME`, `TMPDIR=~/rds/hpc-work/tmp`, …).

## Layout under `project_k/david` (the root)

Same three-part shape as Isambard: `raw/<task>/`, `processed/<task>/`, `final/`. The per-organism
processed stores mirror `ISAMBARD_DATA.md` — `processed/train_tb_ast/` and `processed/train_kleb_ast/`
each with `esm/ baclm/ bacformer/ protein_sequences/ intergenic/` plus `binary_ast_with_split.csv`,
`embedding_input.csv`, etc. **TODO (confirm):** exact contents, file counts, and whether the
pre-outage CSD3 stores are the same cohort as the Isambard re-download.

## Compute (SBATCH)

- **CPU:** `--account=FLOTO-PROJECT-K-SL2-CPU --partition=icelake-himem` (icelake-himem is less
  oversubscribed; ~76 cores/node, ~6.7 GB/core). Avoid the small personal `FLOTO-SL2-CPU` (budget holds).
- **GPU:** `--account=FLOTO-SL2-GPU --partition=ampere` (~£0.55/GPU-h — keep saturated) or
  `FLOTO-SL3-GPU` (free, low priority). No project_k GPU account (project_k is CPU-only).
- Logs → `~/rds/hpc-work/logs/` (a `#SBATCH` line needs the literal path). Checkout at `~/workspace/BacPredict`.
- **Gotcha:** `module load` of a spack tool leaks python-3.9 onto `PYTHONPATH` and breaks `uv` — use a
  pixi env for tool binaries, or `unset PYTHONPATH PYTHONHOME` after `module load`.

**TODO (confirm on return):** aarch64-vs-x86 env (CSD3 is x86 `icelake`/`ampere` — a *different* uv
solve from the Isambard aarch64 env); the actual env path; per-store file counts; `final/` contents.

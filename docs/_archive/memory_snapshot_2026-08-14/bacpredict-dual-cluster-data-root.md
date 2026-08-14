---
name: bacpredict-dual-cluster-data-root
description: "BacPredict resolves the cluster data ROOT via resolve_data_root()/$BACPREDICT_DATA_ROOT; Isambard=$SCRATCHDIR, CSD3=rds-floto/david; input/output stays CLI-flexible"
metadata:
  node_type: memory
  type: reference
  originSessionId: 287d1555-8fdc-4241-9446-ecac54a396be
---

BacPredict runs on **two clusters** with different storage roots. **CSD3/UoHPC is fully operational
again as of 2026-07-29** (SSH, RDS, ampere GPU + CPU jobs all running; only cold-storage *tape*
files remain inaccessible) after its 27 Jun–late Jul 2026 outage. Either cluster may be in use —
confirm which each session. (BacPredict's recent work has been on Isambard; check with the user
before assuming a cluster.) Resolved by one env var so the same code runs on both (H1–H3, 2026-07-13,
on branch `refactor/consolidate-engine` — see [[bacpredict-engine-consolidation]]).

## The resolver — `bacpredict.engine.config`
- `resolve_data_root(explicit=None) -> Path` — the working-data **ROOT** (holds `raw/ processed/
  final/`). Priority: explicit `--data-root` arg → `$BACPREDICT_DATA_ROOT` → `$SCRATCHDIR` (Isambard)
  → the CSD3 `~/rds/rds-floto-bacterial-4k08a2yyQLw/david` path *if it exists* → `RuntimeError`.
  **Never** a silent relative fallback. Resolution is **lazy** — called only after `parse_args()`
  (inside `main`), never at import/`add_argument`, so `--help` works on a laptop with nothing set.
- Siblings: `raw_root()` = `<root>/raw`, `final_root()` = `<root>/final`,
  `visualisations_dir(org)` = `<bacpredict>/visualisations/<org>` (source-tree, not data-root).
- `OrganismConfig.data_root()` = `resolve_data_root()/processed/train_{tb,kleb}_ast`.

## Design principle (user's steer)
Only the ROOT is env-resolved. **Input/output filenames stay CLI-flexible** (`--input`/`--output`/
`--esm-dir`/…, default `None`, resolved after parse) — the resolver is deliberately NOT a
derive-every-path system, because filenames vary by model/task. Migrated modules: drop the hardcoded
CSD3 constant → path args `default=None` → `x = args.x or <root-relative default>` in `main`. Shared
constants that other modules import (e.g. `validate_amr_annotation.default_metadata()/default_sidecar_dir()`)
became **lazy functions**, not module-level Paths.

## Per-cluster values (also in `src/bacpredict/docs/{ISAMBARD_DATA,HPC_DATA}.md`)
- **Isambard (active):** `BACPREDICT_DATA_ROOT=$SCRATCHDIR` = `/scratch/u6fp/dca36.u6fp`. Single root;
  `raw/{tb,kleb_ast}` + `processed/train_{tb,kleb}_ast/{esm,baclm,bacformer,protein_sequences,
  intergenic}`. Model weights are the exception: `HF_HOME`/`TORCH_HOME` on `$PROJECTDIR/david/cache/`.
  `final/` NOT present yet (stage `metadata_v2` there before catalogue runs). Env: aarch64
  `$SCRATCHDIR/envs/bacpredict-gpu-venv` (py3.11). SBATCH `--partition=workq --account=brics.u6fp
  --qos=normal`, logs `/scratch/u6fp/dca36.u6fp/logs/`.
- **CSD3 (operational again 2026-07-29):** root `~/rds/rds-floto-bacterial-4k08a2yyQLw/david`; x86;
  `FLOTO-*` accounts, `icelake-himem`/`ampere`. RDS is live again (only tape/cold-storage still
  inaccessible). HPC_DATA.md was seeded from docs while CSD3 was down — verify paths against the live
  RDS now that it's reachable.

Shell scripts follow the Gen-A pattern: `: "${BACPREDICT_DATA_ROOT:="$SCRATCHDIR"}"; D=…;
PY="$SCRATCHDIR/envs/bacpredict-gpu-venv/bin/python"; PYTHONPATH="$HOME/BacPredict/src"`. `#SBATCH`
lines can't read env vars → active directives are Isambard, with a commented CSD3 block.

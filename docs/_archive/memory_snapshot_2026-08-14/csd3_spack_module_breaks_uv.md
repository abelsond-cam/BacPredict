---
name: csd3-spack-module-breaks-uv
description: "UoHPC/CSD3-specific: module-loading a spack tool (bcftools/samtools) breaks uv's python numpy import; use a pixi env instead"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 43f9dbd0-7aa0-4579-ab80-2e7170ce8eb3
---

> **UoHPC (CSD3) only** — this gotcha is now also codified in `~/.claude/cluster_uohpc.md`
> ("Gotcha — spack modules break `uv`"). Does not apply on Isambard (Lmod/pixi, aarch64).

On CSD3, `module load <spack tool>` (e.g. `bcftools/1.14/...`, `samtools/...`) injects the
spack stack's **python-3.9 site-packages** onto `PYTHONPATH`. A subsequent `uv run python`
(python 3.12) then imports that python-3.9 numpy and dies with
`ImportError: numpy.core._multiarray_umath` / "Importing the numpy C-extensions failed".

**Why:** the spack module env-vars persist into the shell; uv does not clear `PYTHONPATH`, so
the wrong numpy shadows the venv's.

**How to apply:** prefer a dedicated **pixi env** for non-Python tool binaries (bioconda
bcftools/samtools/etc.) and call the binary by absolute path
(`src/<pkg>/.pixi/envs/default/bin/<tool>`) — conda binaries find their libs via RPATH, so no
module/`LD_LIBRARY_PATH` is needed and `PYTHONPATH` stays clean. This mirrors BacHGT's
`bac_isescan`/`bac_ariba` pixi convention. If you must keep a module, `unset PYTHONPATH PYTHONHOME`
after `module load`, before any `uv run`. The bac_pyseer collation uses the pixi route.

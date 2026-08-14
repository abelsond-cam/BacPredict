---
name: isambard-cpu-jobs-need-gpu
description: "CORRECTED — Isambard CPU-only sbatch jobs DO schedule; PENDING(None) is transient, not a stall. Set --mem."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: f3c1d41f-8ba4-45e4-98f4-8a0db0e1b64a
---

**CORRECTION (2026-07-08, verified).** The earlier claim in this note — "CPU-only sbatch jobs stall
`PENDING(None)` indefinitely; add `--gres=gpu:1`" — was **WRONG**, and David pushed back on grabbing a
GPU for CPU work. Empirically retested: a no-`gres` CPU job (`--cpus-per-task=8`, with and without
`--mem`) reaches **`PENDING(Priority)`** within ~20–40 s and queues normally — exactly like a GPU job.
The `PENDING(None)` I'd seen was the **transient first-few-seconds state right after submit**, before
the scheduler evaluates the job; I read it too early and cancelled, then mis-attributed the "stall" to
the missing GPU. **Do NOT add a GPU handle to a CPU-only job**, and do not cancel on an early `(None)`.

**Isambard `workq` facts** (`scontrol show partition workq`, `sinfo`): single partition `workq`
(1320 nodes, 288 cores each, `gpu:4(S:0-3)`), `OverSubscribe=NO`, `JobDefaults=DefCpuPerGPU=72,
DefMemPerGPU=115000`. Memory + CPU defaults are **GPU-anchored** — so a GPU-less job inherits **no
memory default** and **MUST set `--mem` explicitly** (e.g. `--mem=64G`); that (not the missing GPU) is
the real gotcha. `DefaultTime=04:00:00`.

**How to run CPU-bound pangena_predict work (LR probes, parquet sweeps, audits):**
- **Preferred: a plain CPU sbatch — `--partition=workq --account=brics.u6fp --qos=normal
  --cpus-per-task=N --mem=64G --time=…`, NO `--gres`.** It schedules off the login node and survives
  the session. This is what `src/pangena_predict/scripts/coding_amr_ladder.sh` /
  `coding_amr_lr_panel.sh` now do.
- Login node only for genuinely short (<~5–15 min) single-process CPU orchestration; it dies on SSH
  disconnect (and `/compact` drops the connection), so never for a 30–90 min job.

See [[pangena-predict-stage2-state]]; storage/HPC discipline in the global CLAUDE.md.

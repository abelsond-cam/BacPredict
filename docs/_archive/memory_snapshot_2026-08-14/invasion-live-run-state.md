---
name: invasion-live-run-state
description: "LIVE CSD3 job IDs and the exact next action for each, for the blood-vs-faeces invasion work as of 2026-08-11 evening"
metadata: 
  node_type: memory
  type: project
  originSessionId: 65fba51f-bd25-41d2-813a-24d9abb4255d
  modified: 2026-08-12T02:14:05.192Z
---

★ POST-COMPACT ENTRY POINT for the invasion (blood-vs-faeces) work. Results so far are in
[[invasion-comparators-2026-08]]; precision + flash-attn context in
[[invasion-model-was-fp32-not-bf16]]. **Re-verify every job state with `sacct` before acting** —
these were live at 2026-08-11 ~21:40 BST and will have moved on.

Branch `refactor/consolidate-engine`, HEAD `fe52e8c`, all pushed. CSD3 checkout
`~/workspace/BacPredict` is on the same commit. 16 commits this session, explicit paths only —
the ~45 modified visualisation PNGs / ladder CSVs / PROGRESS_UNITIGS.md belong to **other agents**;
never stage them.

## Live jobs and what to do when each lands

| Job | What | Next action on COMPLETED |
|---|---|---|
| ~~`33494112`~~ **DONE** | `score_cohort_iso` — all 14,119 genomes, deployed fp32 ckpt | **Complete, results in [[invasion-comparators-2026-08]].** Six tables + plots written by `scripts/stratified_tables_iso_source.sh` (now an **sbatch** job — inline it costs ~12 min/table in `uv` startup). Re-run it against the bf16 cohort scores with `SCORES_NPZ=<npz> sbatch …`. |
| `33505898[0-63]→33505899` | Leakage-free unitig LMM on the train+validate cohort (`sampled_country_2_1_all_trainval`, n=10,887). **Third launch**; prep `33505897` **DONE 3h08m** (cache 905 MB = the (10887/13602)²=0.64 scaling off the full-cohort 1.4 GB; 64 chunks × 98,786 = 6,322,304 unitigs). | Two steps after combine, **no new code needed**: (1) `COHORT=sampled_country_2_1_all_trainval sbatch --export=ALL,PHASE=select … map_unitig_hits_genomad.sh` — the existing `select` phase extracts a hits_submatrix for the trainval hit set from the **full** 77 GB matrix, so it covers every sample (the holdout genomes need presence values to be scored). Only the SELECTION is holdout-free, never the rows. (2) `SELECTION_SCOPE=trainval_only HITS_SUBMATRIX=<that> sbatch … run_unitig_presence_model.sh`. That is the publication number. |
| `33476292/3/4` | bf16 Stage C re-runs (pooled / stratified / all_samples) → `models_bf16/` | **Always pass the models dir — the defaults point at the fp32 run.** Evaluate: `MODELS_DIR=models_bf16 bash …/evaluate_iso_source.sh <cohort>` (fixed in `ccd8b9a`; before that `OUT_DIR` was hardcoded to `models/` and would have overwritten the deployed 0.786 `eval_results.json` + `eval_scores.npz` that every comparator joins against). Score cohort: `CKPT_SUBDIR=models_bf16 sbatch …/score_cohort_iso_source.sh` (already safe). Then `SCORES_NPZ=<bf16 cohort_scores.npz> sbatch …/stratified_tables_iso_source.sh`, and re-run the unitig head-to-head with `BAC_SCORES`/`BAC_CKPT` pointed at `models_bf16`. Compare vs fp32 0.786 / 0.762 / 0.827. |

## Hard-won gotchas (all cost a failed job today)

- **`$TMPDIR`** — CSD3 job scripts must set it (`/home/dca36/rds/hpc-work/tmp`). Bash here-docs land
  there; the node-local default is small and shared, and a prep job died "No space left on device"
  while the project tier had 2.5 TB free.
- **`zcat | head -N` under `set -o pipefail`** — SIGPIPE makes it exit 13 and kills the job. Wrap in
  `( set +o pipefail; … )` and assert the output. Latent for the pipeline's whole life because every
  earlier run reused an existing `lmm_cache.npz` and skipped the block.
- **`--no-distances` is rejected outright with `--lmm`** ("Cannot use --no-distances with --lmm",
  pyseer `__main__.py`). `--similarity` alone satisfies pyseer's structure-argument check, and with
  neither `--lineage` nor `--distances` it never loads a distance matrix — so the primed cache is
  identical. Fixed in `fe56de4`. **Three latent bugs now, all in the same priming block**, which every
  earlier run skipped via an existing `lmm_cache.npz`: never trust a code path a rerun has never taken.
- **The combine step deletes the shared chunks on success**, so a new cohort re-splits the 77 GB
  matrix in prep (~hour of IO, needs ~80 GB on `/rds/user/dca36`, which had 287 GB free).
- **`ls glob | wc -l`** aborts under `set -e` on a no-match glob → use `find`.
- **`[ test ] && cmd` under `set -e` — the folk rule is WRONG, measured 2026-08-12.** A false test
  does **not** abort mid-script at top level: `set -e` exempts non-final commands of an AND-OR list.
  It **is** fatal in exactly two places — as a script's **last** command (the non-zero status becomes
  the exit status and **SLURM reports FAILED**) and as a **function's last** command (the function
  returns 1 and `set -e` kills the caller). Prefer `if` anyway: immune to both, and to someone later
  moving the line to the end.
- **Unitig shard dirs**: chunks are cohort-independent and shared per pair (`CHUNK_DIR`, ~77 GB);
  per-shard `.assoc`/`patterns` MUST be per-cohort (`SHARD_DIR`) or two cohorts silently merge.
- **CSD3 SSH rate-limits** on rapid successive connections (`Permission denied
  (keyboard-interactive,hostbased)`). Back off ~5 min; jobs are unaffected.
- `train_isolation_source_cohort.sh` writes to `models_bf16/` — the three old `stage_c*.sh` copies
  hardcoded `<cohort>/models` and would overwrite the deployed fp32 checkpoints.

## Open, not yet started

- **Score a user-supplied strain list** for predicted invasiveness (David to provide the list).
- **bac-LM → per-gene LR → concat ladder.** Gate PASSED. **Demoted to the LAST job** by David.
  Plan: port assemblies+GFFs to Isambard, embed under the GH200 flash-attn env writing the
  **`baclm_reembed`** schema, bring the store back. See [[invasion-model-was-fp32-not-bf16]] for why
  CSD3 cannot run it.

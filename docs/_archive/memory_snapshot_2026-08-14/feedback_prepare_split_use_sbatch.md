---
name: feedback-prepare-split-use-sbatch
description: "Run the AMR prepare/split-regeneration script as a CPU sbatch job (~1h), not on the HPC login node — it crawls stat-ing tens of thousands of embedding files"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 965bdc91-22aa-4d38-9677-c15707d973a6
---

When regenerating a split CSV with `prepare_esmc_embeddings_and_labels_to_finetune_amr.py` (TB or Kp), submit it as a **CPU SLURM job** (a ~1 hour wall clock is plenty), not on the HPC login node.

**Why:** the script's `validate_embeddings_and_prune` step does a per-sample existence check over the whole embedding store (~38k `.pt` files for TB). On a busy login node (load avg seen ~30) those stats over RDS crawl — a run that took seconds on a quiet node took 10+ minutes and still hadn't finished. Login nodes are also CPU-policed. User feedback after watching it crawl: "Might be best to run on sbatch next time for 1 hr clock."

**How to apply:** for any future split regeneration (or similar file-validation-heavy prep over the embedding store), wrap the `uv run python ... prepare_esmc_embeddings_and_labels_to_finetune_amr.py` call in a small CPU sbatch script (partition `icelake`/`icelake-himem`, account `FLOTO-SL2-CPU`, ~01:00:00 wall) instead of running it interactively over SSH. The output CSV (`binary_ast_with_split.csv`) is the same either way; only the runtime placement changes.

Related: the same login-node CPU policy is why TB Stage A smoke had to move to a short GPU job rather than CPU. See [[refreshed-bacformer-complete-genomes-model]].

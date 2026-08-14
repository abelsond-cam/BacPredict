---
name: amr-ladder-holdout-leakage-csv-vs-kfold
description: "AMR ladder/cache score k-fold-trained FT models on the CSV single-split holdout — 81% of those genomes were in k-fold TRAIN/VAL, so ft_mean AUROC (Kp azithro 0.918/0.936) is a LEAKAGE artifact; honest holdout = 0.799. Root cause: resolve_clean_splits hardcodes n_folds=None."
metadata:
  node_type: memory
  type: project
  originSessionId: aac091b8-20e4-4661-ab5d-762fe4b1c697
---

**FIX LANDED IN CODE 2026-07-23 — commit `9060617` (branch refactor/consolidate-engine), 345 tests pass.**
Decompose+fix (Parts A/B/C/E of plan inherited-doodling-peacock): deleted `snp_vs_esm_prediction.py` →
`finetune/holdout.py` (canonical resolver + `resolve_deployed_holdout` reads results.json split provenance;
`resolve_clean_splits(checkpoint_dir=…)`, `==csv` assert dropped) + `gene_lr/{linear_probe,protein_rows,
pooled_cds_vectors}.py`. Ladder now fit-on-FT-train / test-on-FT-k-fold-holdout (was OOF over leaked
universe), selects blocks on train-OOF `lr_auroc` (was test `eval_auroc`), and GUARDS cache holdout
coverage (refuses the leak signature). `cache_bacformer_gene_embeddings` forwards deployed train-sample +
full k-fold holdout, scope-tags `ft_genome_mean_<drug>_<scope>.npz`, stamps provenance. STILL TODO:
(1) GPU re-cache azithro+rifampin scope=trainholdout on Isambard → rebuild ladders → verify azithro ft_mean
≈0.80, rifampin ≈0.96 (Part I verification); (2) `git rm` the contaminated tracked azithro+rifampin CSVs/
figures + untracked visualisations/*_amr_ladder_table.csv, regenerate; (3) hygiene Parts F/G/H/I + rename
fit_one_gene→fit_one_segment + gene_lr→segment_lr (defer dir rename to Phase III); (4) fan out all drugs.
`bacformer_token_cache` (reliable-carrier path) NOT yet fixed — same leak class, tracked for fan-out.

CORRECTED 2026-07-22 (supersedes the earlier WRONG "within-holdout CV probe / no leakage /
eval-method" conclusion — that was hypothesis-stated-as-fact; David pushed for rigour and the
real cause is a split-provenance BUG). Settling why Kp azithromycin "reads 0.918 on the ladder
but 0.799 deployed".

**The ladder/cache holdout ≠ the deployed model's holdout. 81% leakage. Measured, not argued.**
- Deployed Kp/TB AST models are **k-fold**-trained (`train_on_slurm_amr*.sh` pass `--n-folds 5`;
  `--array=0` = fold0/seed1, evaluate_seed=1). azithro results.json: `source="kfold"`, n_eval=**384**.
- `resolve_clean_splits` (engine/gene_lr/snp_vs_esm_prediction.py:74) hardcodes
  `resolve_holdouts(..., n_folds=None, fold=0, seed=1, evaluate_seed=1)` and **asserts source=="csv"**
  → it uses the **CSV `train_val_eval` single-split** evaluate column (=**370** for azithro), NOT the
  k-fold holdout. Its docstring claim "the identical holdout the deployed model scored on" is FALSE
  for k-fold-trained models. Every concat/gene_lr/ladder/driver_panel module calls it → same bug.
- The two "evaluate" sets share only **69** genomes. Of the ladder's 370 cached genomes:
  **244 in deployed k-fold TRAIN, 57 in VAL, only 69 in the true k-fold EVALUATE** → 301/370 =
  **81.4% were seen in training/val**. (cached set == CSV-evaluate exactly; `generate_kfold_splits`
  eval=384.)
- Clinching split of the head's own linear (`classifier.out_proj`, read from the checkpoint
  safetensors) applied to the cached FT genome-mean, by k-fold membership:
  LEAKED (301) AUROC **0.969** · CLEAN true-holdout (69) AUROC **0.788** · pooled 370 = 0.936.
  Deployed reported 0.7993 on its 384. So clean 0.788 ≈ deployed 0.799; the 0.918/0.936 are
  the leaked-dominated average.

**Consequences.**
- The AMR-ladder `ft_mean` **absolute** AUROC (and every rung's absolute AUROC, and the head-direction
  0.936) are inflated by ~81% train/val leakage for EVERY k-fold-trained drug (Kp AND TB). Not a
  cluster/precision/GPU effect; not benign CV optimism. `lift_vs_ft` deltas are measured against a
  near-saturated contaminated baseline (headroom suppressed) — the ladder needs rebuilding, not just
  re-reading. Related deliverable now suspect: [[real-numbers-causal-lr-plan]], [[amr-ladder-raw-recovery-framing]].
- Deployed FT eval-holdout numbers (results.json, 0.799 Isa) ARE honest (true k-fold holdout).

**The fix (proposed, needs David's go-ahead — shared engine + GPU re-cache + rebuilds committed CSVs).**
Thread `n_folds/fold/seed/evaluate_seed` through `resolve_clean_splits` (drop the `=="csv"` assert;
`resolve_holdouts` already does k-fold at evaluate.py:104) so the cache+ladder reproduce the SAME
k-fold holdout (5/fold0/seed1/evalseed1) the deployed model used. Then re-run
`cache_bacformer_gene_embeddings` (GPU forward — only 69 of the true 384 are cached now; need ~315
more) and rebuild ladders. Expect ft_mean ≈ 0.799 for azithro. The 22-drug "no Cambridge-vs-Isambard
gap" claim stands (that was deployed-vs-deployed). Env to re-run cheap CPU checks: Isambard login,
`$SCRATCHDIR/envs/bacpredict-gpu-venv/bin/python` + `PYTHONPATH=$SCRATCHDIR/worktrees/consolidate/src`.
See also [[dont-conflate-penetrance-with-lineage]].

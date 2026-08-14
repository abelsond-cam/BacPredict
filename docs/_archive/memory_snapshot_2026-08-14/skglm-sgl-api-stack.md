---
name: skglm-sgl-api-stack
description: Verified skglm 0.5 sparse-group-lasso stack (datafit/penalty/solver/grp_converter) — the engine for gene_array_lasso; signatures move between versions so pin 0.5
metadata: 
  node_type: memory
  type: reference
  originSessionId: 3240ca4c-5459-4b1e-97f4-9f2775c0f04c
---

Sparse-group lasso on skglm **0.5** (verified by introspection 2026-06-26; signatures drift between versions —
pin `skglm==0.5` and re-verify on upgrade). Stack:

`GeneralizedLinearEstimator(datafit, penalty, solver)` where:
- **datafit** = `LogisticGroup(grp_ptr, grp_indices)` (binary AMR/phenotype; `QuadraticGroup` for regression).
  Has full **sparse** methods (`gradient_sparse`, `get_lipschitz_sparse`, …) ⇒ accepts scipy **CSR X** — absent
  zero-blocks never densify (the memory win; groupyr couldn't, see [[groupyr-sgl-quadratic-memory-failure]]).
- **penalty** = `WeightedL1GroupL2(alpha, weights_groups, weights_features, grp_ptr, grp_indices)`. **No `tau`**:
  the L1-vs-group-L2 split is the two weight vectors — `weights_features` (per-feature L1), `weights_groups`
  (per-group L2). "Conservative tau" = small `weights_features` relative to `weights_groups`. Uniform group
  weight = `sqrt(960)`. `.generalized_support` → selected support (the selected-gene readout).
- **solver** = `GroupBCD(max_iter,max_epochs,p0,tol,fit_intercept,warm_start,ws_strategy,verbose)` **or**
  `GroupProxNewton(p0,max_iter,max_pn_iter,tol,…)`. For the **logistic** datafit prox-Newton is usually the right
  pairing (handover named GroupBCD) — resolve in the Stage-1 smoke.
- **groups** via `skglm.utils.data.grp_converter(groups, n_features)` → `(grp_ptr, grp_indices)`; pass `960` for
  uniform blocks. `grp_ptr`/`grp_indices` go to **both** datafit and penalty.

skglm = sklearn-contrib, Bertrand et al. NeurIPS 2022 "Beyond L1" (arXiv:2204.07826); coordinate descent +
working sets + Anderson. It does **not** lazy-load. See [[gene-array-lasso-skglm-migration]].

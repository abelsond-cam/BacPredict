---
name: groupyr-sgl-quadratic-memory-failure
description: "groupyr's sparse-group-lasso prox builds a dense O(n_groups²) mask — unusable past ~3000 gene-groups; the reason gene_array_lasso migrated to skglm"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 3240ca4c-5459-4b1e-97f4-9f2775c0f04c
---

groupyr's `SparseGroupL1` prox (`groupyr/_prox.py`) allocates a **dense `(n_groups × n_features)` int64
membership mask**: `np.full((len(groups), n_features), 0)`. For our pangenome blocks (n_features =
n_groups × 960) that is **O(960 · n_groups²)** memory:

- @>5% (7,161 genes) → **367 GiB** → OOM-killed the colistin fit (peak RSS 179 GB before the 367 GB alloc).
- @>1% (~24,000 genes) → **~4.4 TB** → never.
- Practical ceiling ~**3,000 gene-groups** regardless of RAM, data sparsity, or float32 — it is the *mask*, not X.

It is an **implementation flaw of groupyr (proximal-gradient, touches every feature, dense overlap-group mask),
not a property of the method.** The sparse-group penalty is block-separable: for contiguous equal 960-blocks the
per-group norm is a reshape, O(n_features), no mask. Confirmed on a real run 2026-06-26 (job 31113483, colistin
@5%, 7161 groups). This drove the migration to [[skglm-sgl-api-stack]] / [[gene-array-lasso-skglm-migration]].

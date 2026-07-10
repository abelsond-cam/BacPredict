# Handover plan: sparse-group lasso on pangenome gene embeddings (skglm)

**For:** Claude Code (implementation) · **From:** scoping discussion **Status:** plan only — no implementation decisions locked beyond those marked AGREED.

---

## 0\. What this is: a migration from groupyr to skglm

We are MIGRATING the sparse-group lasso from **groupyr** to **skglm**. The plan below assumes skglm throughout; this section is the why, and it goes up front because it is the framing for everything after.

Why move off groupyr:

- **skglm is part of the scikit-learn universe** (scikit-learn-contrib), standard fit/predict API — the reassuring, maintained, ecosystem-standard home for this method. groupyr is a smaller standalone package.  
- **Most up-to-date results:** skglm is the NeurIPS 2022 paper (Bertrand et al., "Beyond L1", arXiv:2204.07826) — current state-of-the-art for sparse GLMs.  
- **Much faster at our scale:** groupyr is proximal-gradient and touches every feature every iteration (no screening), which is exactly why it struggles on 24,000 groups × 960 dims. skglm uses coordinate descent \+ working sets \+ Anderson acceleration, discarding inactive groups from the iteration. For pangenome data (small active fraction expected) this is the favourable regime.  
- Bonus skglm gains groupyr lacks: non-convex penalties (MCP, L2/3) with lower bias and better support recovery — useful for rare-gene recovery (see step G).

What the migration does NOT solve — state plainly:

- **The PRIMARY CONSTRAINT remains MEMORY: fitting X in RAM.** skglm's speed is about COMPUTE (iterating over fewer variables), not RESIDENCY. arXiv:2204.07826 IS the skglm paper, not a separate out-of-core algorithm — "trial skglm" and "trial 2204.07826" are the same task, there is no second system to migrate to for memory. Dense X \~ 1.1 TB; the memory engineering (§3) is ours to do.

Context / target problem:

- Design matrix is genes-as-groups × embedding-dims: \~24,000 gene groups × 960 ESM-C/Bacformer dims × thousands of genomes.

The skglm estimator stack we migrate TO:

- `GeneralizedLinearEstimator` \+ `QuadraticGroup`/`LogisticGroup` datafit \+ `WeightedL1GroupL2` penalty \+ `GroupBCD` solver \= sparse-group lasso (group selection AND within-group sparsity), the groupyr equivalent we are replacing.

## 1\. The absence-encoding decision (conceptual, must be settled before coding)

The hard problem: penalised regression needs a fully-populated design matrix; there is no NA the solver skips. Absent genes therefore occupy their 960 columns with *some* value that gets multiplied by coefficients.

- ESM embeddings are NEVER near zero (assumed certain; VERIFY empirically in step A as a one-liner — min L2 norm of any real gene embedding).  
- So a zero block is an out-of-distribution FABRICATED point, not "no signal."  
- BUT this is least-harmful under group-L2 specifically: an all-zero block has zero group norm, so the group penalty ignores absent genes for SELECTION. That is why this stack tolerates the placeholder far better than ridge/plain-lasso.

**AGREED encoding (core plan):** zero embedding block for absent genes (sparse, never stored). Nothing else. The regression reads presence/absence directly off the embeddings — a present gene has its real vector, an absent one is the zero block — so no separate presence channel is needed. Methods text states plainly: absent-gene blocks are structural zeros, not imputed values; a modelling choice, not a clean solution.

**Do NOT** mean/centroid-impute absent genes (densifies X, destroys the memory win, muddies absence-as-signal).

**DOWNSTREAM OPTION (NOT core) — explicit presence/absence matrix.** A meaningful presence signal is not one column: it is a full N × n\_groups INTEGER (0/1) matrix, one binary column per gene. Reasons it is excluded from the core plan:

- It mixes incommensurate scales into one lasso path — binary 0/1 columns against standardised continuous embedding coordinates — forcing awkward weighting and muddying the 960-dim penalty scaling.  
- It is very likely redundant: the embeddings already encode presence (real vector) vs absence (zero block); the regression should handle this unaided.  
- It only earns inclusion if embeddings are empirically shown to MISS absence signal — a downstream test, not a core assumption. Try it later as an ADDITION if results suggest absence is under-modelled; do not build it into Stage 1\.

## 2\. The rare-collective-AMR constraint (shapes grouping, not Stage 1\)

Prevalence filtering is memory-cheap but biologically WRONG for this project: the target genes are rare AMR variants, individually \<5% penetrance, collectively high penetrance, and multiple low-frequency homologue groups each cullable.

- Stage 1 MAY use a \>5% filter purely to get a small fast smoke-test set. This is a scaffolding hack, explicitly NOT the long-term design.  
- Long-term fix (Stage 3+): define groups by **embedding clustering**, not Panaroo orthology, so rare homologues merge into one functional group at high *collective* prevalence — surviving any filter, selectable as a unit, and sidestepping Panaroo paralog/merge artefacts. Apply any prevalence threshold AFTER clustering, on cluster-union frequency.  
- **Architecture requirement:** group definition must be a SWAPPABLE input (a group-membership spec), never hard-wired to Panaroo. Switching from orthogroups to embedding clusters should change only that input \+ the column layout, nothing in the estimator assembly.

## 3\. Memory levers (the real engineering, ranked)

1. **scipy CSR sparse X** — absent gene blocks never materialise. With \~5k present genes/genome this is the dominant win.  
2. **float32 storage** — halves value memory. MUST be validated (step C) to not move the solution vs float64. Distributed SNP signal sits orders of magnitude above float32 epsilon, so expected to pass trivially; once it does, the precision question is formally closed.  
3. **Warm-started decreasing alpha path** — start near alpha\_max (all-zero, tiny working set, fast/cheap), walk down with warm starts. The active set is what's densely manipulated, so a small working set is also a runtime-memory win, not just speed.  
4. **Grouping redefinition (§2)** — stops 24k rare genes each owning a group.  
5. **LAST RESORT if 1–4 miss the node ceiling:** true out-of-core — memory-mapped X or a chunked custom datafit. skglm does NOT provide this; it is a real build with a real cost. Decide explicitly before committing; do not assume it.

CSD3: target a high-memory node (e.g. icelake-himem). Step D produces the real RSS-vs-N number that sizes the node — do not guess the node from back-of-envelope.

## 4\. Hyperparameters — first-pass values to AGREE, then CV later

| Param | First-pass | Rationale / how tuned later |
| :---- | :---- | :---- |
| datafit | QuadraticGroup (reg) | LogisticGroup for binary AMR/invasion phenotype |
| penalty | WeightedL1GroupL2 | the SGL penalty |
| solver | GroupBCD | block coordinate descent for groups |
| group weights | sqrt(group\_size) | all blocks are 960 → uniform sqrt(960); affects L1-vs-groupL2 balance |
| tau (L1 vs grpL2) | CONSERVATIVE (low) | embeddings have correlated dims; aggressive within-group L1 over-prunes. CV later |
| alpha | path from \~alpha\_max | geomspace down, \~20 points, warm-started |
| tol | strict (e.g. 1e-8 sm) | must close, not spin |
| max\_iter | low ceiling | so a non-closing gap FAILS LOUDLY |

VERIFY against the INSTALLED skglm version: constructor signatures (esp. whether grp\_ptr/grp\_indices go to datafit, penalty, or both; whether the split param is `tau` or an alpha-ratio) have moved between versions. Pin the version. Treat the argument wiring as something Stage 1 VERIFIES, not assumes.

## 5\. Build order (the actual work)

**Step A — verify the premise.** Load a real embedding sample; confirm min embedding L2 norm ≫ 0 (justifies "zero \= OOD fabrication, not no-signal"). One number, goes in methods.

**Step B — group plumbing.** Implement group-membership as a swappable spec → grp\_indices/grp\_ptr via grp\_converter. Uniform embedding blocks of 960, one per gene group. HARD TEST: group g maps to exactly columns \[g\*960:(g+1)\*960\]; off-by-960 / transposed blocks are SILENT and produce plausible garbage. This is the highest-risk correctness point.

**Step C — Stage 1 smoke test (AGREED scope: 100 genomes, genes \>5%).** Assemble the full stack on the small set. Four gates, all must pass before any scaling:

1. duality gap CLOSES (reaches tol, not max\_iter).  
2. group indexing correct (step B test).  
3. float32-X coefficients \== float64-X coefficients (precision gate).  
4. sparse-path coefficients \== dense-path coefficients (the group datafit's sparse path is the least battle-tested corner — likeliest scaling surprise). Output: agreed first-pass hyperparameters \+ a clean reproducible fit.

**Step D — MEMORY CHARACTERISATION EXPERIMENT (primary deliverable of phase 1).** Reframed from "profile a point" to "map where the wall is." The output is a concrete finding: the memory cutpoints across genome count and gene penetrance. This is information we do not currently have and explicitly want to capture.

What actually drives memory — measure against THIS, not the knobs:

- The RAM-consuming quantity is **nnz** (stored nonzeros in CSR), NOT N or p directly. nnz is computable BEFORE fitting — compute and log it every run.  
- Lowering penetrance (10%→1%) adds many groups but they are RARE, so each adds FEW nonzeros: group COUNT explodes, nnz grows slowly. (Encouraging for the hypothesis that low penetrance may still fit.)  
- Raising N adds nnz \~linearly per retained gene AND shifts which genes pass a fixed p (1% of 2000 \= 20 carriers; 1% of 100 \= 1). So N and p INTERACT — the same threshold admits a different gene set at each N. Do not treat the axes as independent.  
- \=\> Plot peak RSS vs nnz. Expect a near-linear law (\~12 B/nonzero float32 \+ indices, plus working-set overhead). Then any (N, p) cell becomes a LOOKUP: compute its nnz, predict RSS, including cells too big to actually reach.

Sweep protocol (genome axis, coarse-to-fine, adaptive):

- Up to 2000 genomes available. Start N=100 at p\>10%.  
- If fast and well under memory: step N up in 100s.  
- If slow / near ceiling: switch to coarser N grid (250s or 500s).  
- Record at each N the peak RSS and wall-clock.

Penetrance axis (after the genome axis is mapped at 10%):

- Relax threshold: 10% → 5% → 4% → 3% → 2% → 1%.  
- Repeat the genome sweep at each. Expect the RSS-vs-nnz law to hold ACROSS thresholds (that's the validation the law is real, not a per-p artefact).

THIRD axis — alpha / working-set memory (this is the lever, measure it):

- Peak RSS at fit is X-memory PLUS the solver working set, which holds a densified slice of the ACTIVE groups. Higher alpha → fewer active groups → smaller working set → LOWER peak RSS. This is the mechanism behind the "prune-first-then-scale" plan below, so confirm it empirically: at a fixed (N, p), measure peak RSS across a few alpha values from high to low, and find the alpha at which the working set blows the budget.

Convergence is part of the datapoint — DO NOT contaminate the map:

- A run can "fit in memory" by silently hitting max\_iter and returning a non-optimal vector, UNDERSTATING the peak RSS a real converged fit would need.  
- Record three DISTINCT outcomes per cell: CONVERGED / OOM-KILLED / HIT-MAX-ITER. Only CONVERGED points define the true memory wall. A HIT-MAX-ITER point is NOT a valid "it fits" datapoint.

Concrete findings to produce:

- RSS-vs-nnz law (slope, intercept, R^2) — validated across p.  
- The (N, p) memory cutpoints: largest converged cell per threshold.  
- The alpha-vs-peak-RSS curve at a representative (N, p).  
- A predicted-RSS table for cells too large to run, from the law.

Run-isolation note: each cell must be a FRESH process (peak RSS is per-process and CSR/working-set memory is not reliably freed within a session). Use /usr/bin/time \-v or resource.getrusage(RUSAGE\_SELF).ru\_maxrss per child process; do not measure peak RSS from inside a long-lived loop.

**Step E — PRUNE-FIRST SCALING (principled strategy, informed by step D's map).** If the map shows we can fit a given cell only at high alpha but not at the relaxed alpha we ultimately want (e.g. 1000 genomes at 1% fits at high regularisation but OOMs as alpha relaxes), exploit the working-set mechanism:

1. Fit at HIGH alpha on the affordable cell — small working set, low peak RSS, prunes the obviously-irrelevant groups (their coefficients → exact zero).  
2. DROP the pruned groups from X entirely → smaller matrix, fewer nonzeros.  
3. Scale up genomes / relax penetrance / lower alpha on the REDUCED group set, warm-starting from the high-alpha solution.  
4. Iterate: each round the surviving group set is smaller, so each subsequent (larger-N, lower-alpha) fit is cheaper than it would have been cold. Caveats to record, not hide:  
- **Pruning is CONSTRAINED TO CORE GENES** (present in \~all genomes). This is the clean fix to the rare-AMR deletion risk: a core gene is present at full N too, so pruning it on a subset CANNOT silently delete a signal that only appears at larger N — the failure mode literally cannot occur for core genes. Accessory/rare genes are NEVER pruned at this stage; they are carried through (or re-admitted) regardless of their high-alpha coefficient on a subset.  
- This makes prune-first safe as a first pass: it shrinks X by removing uninformative CORE groups (of which there are many — core genes are mostly housekeeping and unlikely to drive the phenotype), without touching the rare accessory groups the project exists to find.  
- Rare-accessory re-introduction (admitting all rarer genes back when scaling N) is a SUBSEQUENT step, layered on top once core-only pruning has bought the headroom. Note for later; not the first pass.  
- Still greedy in the weak sense that a pruned core group might matter at full N, but bounded: confine pruning to groups zero across a RANGE of high alphas.  
- This is a first-pass tractability strategy, not a substitute for a single full-set fit if memory ever allows one. Report it as such.

**Step F — full scale-up.** Full genome set, warm-started alpha path, gap-closure checked at every alpha. Use §4 path settings. Reachable directly if step D shows headroom; via step E if it doesn't.

**Step G (parallel track, NOT a migration) — non-convex penalties.** Since 2204.07826/skglm also offers MCP / L2/3, trial these for the rare-gene recovery: lower bias → better support recovery than L1. This is a STATISTICAL upgrade for the same library, evaluated for whether it recovers rare AMR groups L1 misses. Not a memory fix.

## 6\. Explicit non-goals / cautions for the implementer

- skglm does not lazy-load. Do not architect around an imagined out-of-core mode.  
- Do not let \>5% prevalence filtering leak into the long-term design (§2).  
- Do not treat float32 as safe until step C says so.  
- Do not trust remembered skglm API signatures — check the installed version.  
- The embedding-clustering grouping (Stage 3\) has its own memory question: cluster on per-gene CENTROID embeddings (one 960-vec/gene), not per-sample-per-gene; decide handling of variants spanning two clusters. Note for later, do not build now.


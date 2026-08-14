---
name: csd3-ast-assembly-resolution
description: On CSD3 the Kp AST assemblies are NOT in a flat BioSample-keyed dir — resolve them via raw/assemblies_file_list.tsv; TB works as-is. Canonical cohorts live under bac_ast_prediction/.
metadata: 
  node_type: memory
  type: project
  originSessionId: a8e12097-62e8-4781-953f-3400a478a807
  modified: 2026-08-13T20:12:12.014Z
---

Verified live on CSD3, 2026-08-11, while gating step 4 of the `ast_gwas` plan
(`src/bac_pyseer/ast_gwas/docs/PLAN.md`). Data root `R=~/rds/rds-floto-bacterial-4k08a2yyQLw/david`.

**The canonical AST task root is `$R/bac_ast_prediction/`, not `$R/processed/`.**
`$R/bac_ast_prediction/raw` is a symlink to `$R/raw`. Cohort sizes settle the ambiguity:

| Sheet | Rows | Status |
|---|---|---|
| `$R/bac_ast_prediction/processed/train_kleb_ast/binary_ast_with_split.csv` | 7,088 | **canonical** (matches Isambard) |
| `$R/processed/train_kleb_ast/binary_ast_with_split.csv` | 6,838 | deprecated May cohort |
| `$R/bac_ast_prediction/processed/train_tb_ast/…` | 36,692 | **canonical** |

Split tables: 22 Kp + 10 TB `<drug>_split.csv` under `bac_ast_prediction/processed/train_{kleb,tb}_ast/splits/`.

**The assembly-resolution asymmetry — this is the non-obvious part.**
`resolve_ast_assemblies.py` assumes flat, BioSample-keyed dirs (`raw/kleb_ast/assemblies`,
`raw/tb/assemblies`). That is the **Isambard** layout. On CSD3:

- **TB works unchanged.** `$R/raw/tb/assemblies/SAMEA*.fa.gz`, flat, BioSample-keyed, ~1.4 MB each.
  36,692/36,692 resolve. The resolver's default path is already correct.
- **Kp does NOT.** `$R/raw/kleb_ast/assemblies` does not exist. `$R/raw/assemblies` is the *whole-Klebsiella*
  81,061-file store keyed by **GCA accession** (`GCA_900451215.1_44310_G01_genomic.fna.gz`), symlinked into
  `seb/assemblies/NCTC/`. A filename join against BioSample IDs resolves **zero**.

**The fix is already on disk:** `$R/raw/assemblies_file_list.tsv` is a 95,131-row `Sample<TAB>path`
TSV, BioSample-keyed, pointing at `seb/assemblies_2/klebsiella_pneumoniae__NN/<BioSample>.fa.gz`
(~1.5 MB each; sharded into batch dirs, which is *why* there is no flat dir). That is byte-identical
in format to the reflist `resolve_ast_assemblies` emits and `run_ggcat_unitigs.sh` consumes — so Kp
needs a **filter, not new resolution logic**. 7,080/7,088 resolve (8 missing, 0.1%).

GGCAT input volume (count × one-file): Kp ≈ 10.6 GB, TB ≈ 51 GB.

**Scale flag for TB:** `mash triangle` over 36,692 genomes is ~673 M pairs — a multi-GB text
triangle, and `parse_triangle` materialising 36,692² float64 is ~10.8 GB RAM. The plan's per-drug
kinship subsetting avoids this downstream, but `build_cohort_once.sh` still builds the cohort-wide
triangle once.

Toolchain gate is **moot on CSD3** — it is `linux-64`, exactly what `src/bac_pyseer/pixi.toml`
targets. All seven binaries resolve in the installed env: pyseer 1.4.1, ggcat 2.2.0, unitig-caller
1.3.0, mash 2.3, bcftools/samtools 1.23.1. Caveat: **`mash` is in `pixi.lock` but not declared in
`pixi.toml`**, so a clean re-solve would drop it — declare it before publishing.

## Live run state (2026-08-13)

**PR #1 MERGED** into `refactor/consolidate-engine` as `ffa7abc`. Both worktrees removed —
local `~/developer/BacPredict` and CSD3 `~/workspace/BacPredict` are again the single checkout on
each machine, both at `ffa7abc`, 228 tests green. Drive everything from the main checkout now; its
own `.venv` runs the `ast_gwas` CLI. (The temporary worktree env
`/home/dca36/rds/hpc-work/venvs/ast_gwas` is now orphaned and can be deleted.)

**Step 5 (Kp cohort build) COMPLETE** — jobs `33506578`/`33506579`, and far cheaper than budgeted:
ggcat **54 min / ~38 GB peak** (asked 36 h / 250 G), mash **3.7 min / ~13 GB** (asked 6 h / 210 G).
Use that to size TB rather than the plan's guesses.

| Output | Value |
|---|---|
| Unitigs → testable features | 5,829,181 → **3,760,582** (8,196,279 colour segments) |
| `unitigs.pyseer.gz` | **27 GB** (plan guessed 40–60 GB) |
| af filters applied | `--min-samples` 71 (1%), `--max-samples` 7009 (99%) |
| `mash_triangle.txt` | 240 MB |

**The step-5 `[look]` found the mash lineage clustering had COLLAPSED** — `n_clusters: 1`, 6,852 of
7,080 in `sl0001`. The 0.02 average-linkage threshold (≈98% ANI) is far too loose for *K. pneumoniae*,
whose whole species complex sits inside that radius. That is inert, not merely coarse: `--lineage`
carries no information and the step-10 within-lineage permutation null degenerates into a global one.

**RESOLVED (David's call, 2026-08-13): Kp now uses curated Kleborate `Sublineage` from `metadata_v2`**
— new module `ast_gwas/sublineage_from_metadata.py` (`f9b1d7f`), output byte-interchangeable with
`lineage_from_distances`. Live result: **10 clusters** at `min_size=100` (SL258 1,302 · SL307 766 ·
SL17 443 · SL15 440 · SL147 236 · SL37 188 · SL101 142 · SL45 134 · SL14 126 · SL405 113), label
coverage 6,458/7,080 (91.2%). The collapsed mash version is preserved beside it as
`mash_lineage_clusters.tsv` for the methods comparison — do not delete.

**`other` bucket — RESOLVED (David, 2026-08-13): exclude it from the null, and also look at
`min_size=50` at step 10 to see how far broader claims can be pushed.**
`permute_phenotype_within_lineage.py` gained `--exclude-cluster` (`f31da8f`, default unchanged so
invasion outputs stay reproducible); it *drops* those samples rather than leaving them unshuffled,
since leaving them in would carry real association into a table meant to contain none.
⚠ **The paired real run must be scored on the same subset** — otherwise λ_perm (55% subset) is
compared against λ_real (full cohort). The tool prints this at runtime.

**`min_size` stays at 100 — David dropped the 50 variant (2026-08-13): not worth the complexity.**
Measured before dropping it: `min_size=50` gave 17 clusters and moved `other` from 45% → 38%
(+493 permutable genomes, 3,890 → 4,383), all 7 new clusters in the 50–99 range. Too small a gain.
`lineage_clusters_min50.*` was deleted; **do not regenerate it.** The single cluster file of record
is `…/pyseer_ast/kp/structure/lineage_clusters.tsv` (10 clusters, min_size=100).

**TB still needs the mash route** — no TB lineage labels exist until TB-Profiler runs over ~39k
assemblies, so the two organisms derive clusters differently and methods must say so.

**Step 6 (Kp `ertapenem`, the positive control) — GWAS COMPLETE 2026-08-13, fast.** Chain
`33571499` prep (36 min) → `33571500` array[0-63] (~5 min each) → `33571501` combine (3.6 min),
all `icelake-himem`/`cpu1`. Phenotype n=**1,697** (train+validate only; holdout never enters).

| | |
|---|---|
| Variants tested | 3,371,827 (of 3,760,582 — pyseer's own MAF filter over 1,697 samples) |
| Unique patterns | 1,601,208 → Bonferroni **3.12e-08** |
| **λ (genomic inflation)** | **4.198** — inflated; the permutation null is what decides signal vs structure |
| Significant unitigs | **31,856** |

⚠ **BUG FOUND AND FIXED (`8604b9e`) — the `pheno_var` fix was bypassed by the inherited chain.** The
combine phase ran its *own* `pyseer_postprocess` without `--phenotype-tsv`, so ertapenem's summary
recorded `"pheno_var": 0.249, "pheno_var_source": "default"` — the exact defect the PR fixed.
Measured correct value **0.2248**, so the literal overstates the denominator ~11% and understates
`var_explained_pct` by the same. It cannot reorder hits *within* a drug (constant divisor on a
monotone key), which is why it hid — but the fan-out's cross-drug VE league table ranks on it.
Second defect in the same call: **`--neg-label` was hardcoded to `faeces`**, so every AMR figure
would have been captioned with an isolation-source class. `run_drug.sh` now passes
`POS_LABEL=resistant NEG_LABEL=susceptible PAIR_TITLE=…`, and the driver exports `NEG_LABEL`.
Ertapenem's `$DRUG_DIR/` summary was regenerated correctly by hand; **the `gwas/` copy still has the
0.249 default — use the `$DRUG_DIR/` one.** Later drugs get it right automatically.

**★ STEP-6 GATE PASSED (job `33575297`, 12m51s).** Splits train 1,357 / validate 340 / holdout 424.

| Model | Unitigs | Holdout AUROC | AUPRC | Sens | Spec | Bal acc |
|---|---|---|---|---|---|---|
| unitig-LR (all hits) | 31,856 | **0.9775** | 0.9853 | 0.960 | 0.938 | 0.949 |
| unitig-LR (LD-deduped) | 11,522 | **0.9777** | 0.9845 | 0.960 | 0.932 | 0.946 |
| Bacformer FT (reference) | — | 0.9870 | 0.9843 | | | |
| CARD ceiling (reference) | — | 0.977 | | | | |

Two clean control results worth carrying forward:
1. **LD dedup removes 64% of features (31,856 → 11,522) and changes AUROC by +0.0002.** The hit
   *count* is LD-inflated; the information is not. Confusion matrices differ by one genome.
2. **C barely matters here.** Validate preferred C=0.001 (0.9781) over the repo-pinned C=1.0
   (0.9666) — a 1.2 pp validate gap consistent with the sibling's overfitting finding — but on
   holdout the two are within noise (0.9775 vs pinned 0.9786). Ertapenem's signal is saturated, so
   this does *not* settle the C question; colistin is the drug that will.

**★ ERTAPENEM HEAD-TO-HEAD (identical 424-genome holdout, all shared):**

| Arm | AUROC | AUPRC | Bal acc @Youden |
|---|---|---|---|
| Bacformer FT | **0.9878** | 0.9937 | 0.9804 |
| unitig-LR | 0.9775 | 0.9853 | 0.9530 |

Paired bootstrap: **delta (unitig−FT) = −0.0103, 95% CI [−0.0187, −0.0031], separates from zero.**
So on ertapenem the fine-tune genuinely beats the unitig screen — **unlike the invasion phenotype,
where the same comparison was a tie** ([[invasion-comparators-2026-08]]). It is the saturated
positive control, so this is a real but ~1 pp edge near the ceiling; the informative drugs are the
ones with catalogue gaps (colistin next).

**Threshold convention (David, 2026-08-13): Youden on the HOLDOUT, one convention for both arms.**
Validate-selection was tried and ditched — it transfers badly at these split sizes (ertapenem: a
340-genome validate picked thr 0.797 → balacc 0.925, *worse* than a flat 0.5's 0.949). Landed in
`59dd3a9`: `collect_comparison.operating_point()` recomputes it from each arm's own
`eval_scores.npz` so the arms cannot drift apart; the 0.5 block is out of the tables.
⚠ These sens/spec/balacc are the **best achievable operating point, optimistically biased** — the
caveat rides in `operating_point.caveat` of every results.json. AUROC/AUPRC unaffected. C is still
swept on validate; no refit.

**FT per-sample scores are NOT saved by training** — `results.json` has aggregate metrics at 0.5
only. Recovering them needs `engine.finetune.evaluate` per checkpoint (`--n-folds 5 --fold 0 --seed 1
--evaluate-seed 1`), which writes `eval_scores.npz`. Ertapenem done as job `33579047` (**13m48s**,
ampere, `FLOTO-SL2-GPU`). **Bacformer does NOT need flash-attn** (absent on CSD3) — that wall is
baclm-only. This pass is also the *only* way to get the paired CI, so it is not optional if a
head-to-head claim is wanted.

⚠ Three slightly different FT ertapenem AUROCs are in circulation: **0.9878** (this re-score),
0.9882 (stored `results.json`), 0.9870 (quoted in PLAN.md/CLAUDE.md). Reconcile which is of record
before it goes in a paper table.

**Step 7 (Kp `colistin`) — GWAS COMPLETE 2026-08-13** (prep 36 min, array ~2–3 min/task, combine
2.6 min). n=1,128 (297 R / 831 S). **The combine-phase fix is confirmed working**:
`pheno_var 0.19397, pheno_var_source: computed:…`. **λ = 1.232** — far better controlled than
ertapenem's 4.198. 9,277 significant of 2,486,812 tested, 960,320 patterns.
**★ COLISTIN HEAD-TO-HEAD COMPLETE** (`33585112` 11m36s + `33585113` 7m41s). Identical 282-genome
holdout (75 R), all shared:

| Arm | AUROC | AUPRC | Bal acc | Sens | Spec |
|---|---|---|---|---|---|
| Bacformer FT | 0.9100 | **0.8333** | 0.8444 | 0.800 | 0.889 |
| unitig-LR | **0.9188** | 0.8077 | 0.8477 | **0.947** | 0.749 |
| unitig-LR dedup | 0.9186 | 0.8027 | 0.8483 | 0.933 | 0.763 |

**Paired delta = +0.0088, CI [−0.0171, +0.0347] — does NOT separate from zero: a statistical TIE**,
like the invasion comparator and unlike ertapenem. Two drugs, two different answers, so neither
generalises yet. Note the arms trade off differently at Youden (unitig much more sensitive, FT much
more specific) and FT wins AUPRC despite losing AUROC — worth a look when more drugs land.

## ★ FT-NUMBER AUDIT — DONE 2026-08-13, all fixed in `a1b595d`

**All four pilot FT numbers in the docs were stale** (they predate the July re-runs; David
confirmed "we have rerun them all"). Ground truth = **each deployed checkpoint's own
`results.json`** — 32 runs (22 Kp + 10 TB), 15–21 Jul 2026, all `kfold` fold 0 / seed 1 on
`bacformer-large-masked-complete-genomes`.

| Drug | Doc said | **Actual** | Ceiling | Effect |
|---|---|---|---|---|
| Kp ertapenem | 0.9870 | **0.9882 / 0.9937** | 0.9828 CARD | trivial |
| Kp colistin | 0.8072 | **0.9094 / 0.8330** | 0.6563 CARD | +0.10 — rationale inverted |
| TB rifampin | 0.9046 | **0.9642 / 0.9160** | 0.9666 WHO | +0.06 — rationale inverted |
| TB ethionamide | 0.7742 | **0.8097 / 0.5962** | 0.8706 WHO | +0.04 |

Two selection rationales did **not** survive: colistin is *not* the worst Kp drug (that is
**azithromycin 0.7993**, ceiling 0.5584), and rifampin does *not* underperform its catalogue — it
matches it. The drugs stay, on corrected reasoning: colistin has the **largest FT-over-catalogue gap
in Kp (+0.253)**, ethionamide is the one that genuinely underperforms (−0.061). Worst TB =
`moxifloxacin` 0.7945 / AUPRC 0.5002.

**Ceiling provenance:** Kp = `__ALL_CARD__` row of
`train_kleb_ast/card_ceiling/<drug>/card_determinant_lr_<drug>_{allele,family}.csv` (all 22 drugs).
**TB has NO `card_ceiling` dir** — its ceilings only exist in the 5-drug figure panel.

### ⛔ NEVER take a fine-tune number from `*_amr_summary_panel.csv` — 3 silent failure modes

1. The panels carry **`concat_auroc`/`concat_auprc`** = the **concat-ladder model**, not the plain
   FT. `collect_comparison` looked for `ft_auroc`, found none, and silently emitted a table with
   **no fine-tune arm and no delta**. Fixed: FT columns now come from its own `eval_scores.npz`.
2. The panels are **partial** — Kp 7 of 22 (**no ertapenem**), TB 5 of 10. Missing → NaN, reading as
   "no ceiling exists". Now warned per drug.
3. TB panel keys **`rifampicin` (UK)** vs AST column **`rifampin` (US)** → headline TB drug matched
   nothing. Fixed via `PANEL_DRUG_ALIASES`.

The test that should have caught (1) asserted an *invented* panel shape containing `ft_auroc`, so it
passed while the real panels never had that column — now tests the real shape.
Also note the kp panel is misfiled under `visualisations/tb/kp_amr_summary_panel.csv`.

## Standing conventions from here (David, 2026-08-13)

1. **Report AUROC, AUPRC and balanced accuracy per arm** — David: "balanced accuracy shows it best".
   Plus the paired delta + CI. Drop the raw-0.5 block from tables.
2. **The `evaluate.py` re-score is the FT number of record**, not the training `results.json`
   ("use this run, it is fine") — so ertapenem FT AUROC = **0.9878**, resolving the
   0.9878/0.9882/0.9870 three-way discrepancy.
3. **Do the FT re-score for every drug going forward** ("can do the same going forward"), i.e. run
   `ast_gwas/scripts/run_readout.sh` per drug — it submits the CPU unitig read-out *and* the GPU FT
   re-score together (`2798f5d`). ~14 min GPU per drug. `SKIP_FT=1` where no checkpoint exists.

`run_readout.sh` refuses to assume three things that each silently corrupt the comparison: per-drug
checkpoint step counts (750/2500/7750/31000 → glob), the organism path split (Kp
`models/finetune/klebsiella_pneumoniae_*` vs TB `checkpoints/mycobacterium_tuberculosis_*`), and the
k-fold params, which must match training or a *different* holdout is rebuilt and the arms are scored
on different genomes. **TB FT checkpoints exist** (ethionamide/isoniazid/rifampin seen) under
`train_tb_ast/checkpoints/`, not a `models/` dir.

Note `run_unitig_lmm_sharded.sh` hardcodes `ACCT=FLOTO-PROJECT-K-SL2-CPU`/`icelake-himem` and passes
no `--qos`, so it is already CSD3-correct and ignores the `ACCT` env var. Its array asks cpu=8/mem=128G;
with `MaxMemPerCPU=6760` SLURM buys the extra memory with extra cores, which is the proven
2026-06-24 iso-source calibration (50k unitigs/shard ≈ 21 GB peak) — ours is ~58.8k/shard over 64 shards.

Cosmetic gotcha seen in the ggcat log: `OUT_NAME=blood_faeces` (its default) is printed even though
`OUT_DIR` was correctly overridden, so outputs landed in the right place. Harmless here, but if
`OUT_DIR` were ever left unset the build would overwrite the iso-source cohort.

**Cluster invocation that works** (the plan's Isambard defaults must all be overridden):
`BACPREDICT_DATA_ROOT=$R/bac_ast_prediction`, `REPO=~/workspace/BacPredict-ast-gwas`,
`FILE_LIST=$R/raw/assemblies_file_list.tsv`, `ACCT=FLOTO-PROJECT-K-SL2-CPU`, `PART=icelake-himem`,
`QOS=` (empty). Three gotchas, all fixed or worked around:

1. **`QOS=` must omit `--qos`, not substitute a default.** `${QOS:-normal}` substituted on empty too;
   CSD3 rejects `--qos=normal` (the FLOTO associations allow only `cpu1,intr`). Fixed to `${QOS-normal}`
   in both drivers (`3694b7c`).
2. **`MaxMemPerCPU=6760` on icelake-himem** — request ≤ cores × 6760 MB or SLURM silently inflates the
   core count. 38 cores → ≤250G; 32 cores → ≤210G.
3. **Envs must stay off `/home`** (37.4 of 52.4 GB used; the main `.venv/lib` alone is 8.3 GB). The
   worktree's uv env went to `UV_PROJECT_ENVIRONMENT=/home/dca36/rds/hpc-work/venvs/ast_gwas`. For pixi,
   point `PIXI_MANIFEST` at the **sibling checkout's** manifest — pixi keeps envs detached on RDS
   (`david/nuna/envs/`), so this reuses the proven ggcat 2.2.0 env read-only instead of re-solving from
   my branch's lock, which still lacks ggcat.

Related: [[bacpredict-dual-cluster-data-root]] · [[bacpredict-clean-architecture-plan]]

---
name: real-numbers-causal-lr-plan
description: LIVE plan (2026-07-17) — AMR concat-LADDER (FT-mean→+baclm gene→+baclm UPSTREAM region vs ceiling) + the IGR-iteration matrix that must be sorted first; does the non-coding rung lift headroom drugs (eth/strep/kan/isoniazid + Kp amik/azithro/colistin/tetra/aztreonam/cefoxitin)?
metadata: 
  node_type: memory
  type: project
  originSessionId: 287d1555-8fdc-4241-9446-ecac54a396be
---

**Deliverable = per-drug AMR score LADDER:** RED catalogue one-hot ceiling vs BLUE additive ladder FT
genome-mean → +best baclm gene → +best baclm **non-coding region**; does the non-coding rung lift the
**headroom drugs** toward the ceiling? Full plan: `~/.claude/plans/the-cambridge-hpc-and-dreamy-thacker.md`.
History: [[bacpredict-isambard-validation-program]], [[pangena-predict-stage2-state]], [[baclm-build-defects]].

> ⚠ **2026-07-22 — ALL committed ladder CSVs/figures are CONTAMINATED by holdout leakage.** The FT-mean
> cache + ladder score k-fold-trained models on the CSV single-split holdout, 81% of which is k-fold
> TRAIN/VAL → absolute AUROCs inflated (Kp azithro ft_mean 0.918 vs honest ~0.799), `lift_vs_ft` measured
> against a near-saturated baseline. Must fix the split (thread k-fold through `resolve_clean_splits`),
> re-cache FT-mean on the true k-fold holdout (GPU), and REBUILD every ladder before any of these numbers
> are trusted. Full detail + fix: [[amr-ladder-holdout-leakage-csv-vs-kfold]].

## CHECKPOINT 2026-07-22/23 — Phase A DISPLAY fixes + A3 (convergent merge + gate-free ladder) DONE; TB ladder REBUILD running

Plan `the-cambridge-hpc-and-dreamy-thacker.md` Phase A. Branch refactor/consolidate-engine, **HEAD `92d8a22`, pushed.**
⚠ The holdout-leakage warning at the top is UNCHANGED by this work — A3 fixes *which* region is routed + the
display; absolute AUROCs stay contaminated until [[amr-ladder-holdout-leakage-csv-vs-kfold]] lands. Orthogonal.

**Plots DONE + committed** (4 rounds, David live-review): `301548f` shared `engine/plots/labels.py::region_label`
(upstream:fabg1→"inhA promoter", rrna:rrs→"rrs rRNA") used by BOTH ladder+causal; causal ◆(coding)/★(non-coding)
markers sourced from the **ladder table** (`--ladder-table`, `_routed_marks`) not the tallest bar; terse
subheadings. `1d82c3b` unified `////`=includes-IGR hatch, dropped not-ranked hatch. `bdfb9ad` causal colours =
catalogue **purple** `#4a1486` / non-catalogue **deep blue** `#08306b` (penetrance rides on alpha now; light-blue
washed out), black ◆, grey opacity colourbar, wording. `92d8a22` ladder red-bar IGR hatch **CONDITIONAL** (David
confirmed) via `_catalogue_has_noncoding()` reading TB-Profiler is_noncoding/is_rrna/region flags — ethionamide/
kanamycin red bars hatched, rifampin/cipro (coding-only) solid; FT⊕IGR rungs always hatched.

**A3 DONE + committed `90e5cd3`:** (1) `_flank_pair` sorts the pair → canonical/orientation-invariant → merges
the convergent split (kanamycin `mura→ogt` = rrn/rrs operon **VERIFIED one key @prev0.965**, was ogt→mura+mura→ogt
~50/50; it's the top per_igr region imputed AUROC 0.790). (2) `build_amr_ladder` DROPPED the ≥0.9 core gate → the
non-coding rung selects **top imputed AUROC, no gate**, across upstream ∪ per_unit ∪ **per_igr** (new 3rd source
via `load_baclm_igr_block`); `_select_noncoding` replaces `_select_core_noncoding`; `run()` gains `igr_ranking_csv`,
drops `core_min_*`. (3) Launchers gain `FEATURE=imputed_full` (band 0.01–1.0 incl. core) — the accessory
`imputed` mode capped at 0.99, EXCLUDING the core regions the ladder routes (inhA promoter 0.997, rrs). 68 engine
tests green, ruff-clean on my files (repo has 176 pre-existing E402 etc — not mine).

**A3 cluster (Isambard):** concat worktree `$SCRATCHDIR/worktrees/concat` advanced to `90e5cd3` (no FT pending —
squeue was empty; safe). TB `imputed_full` rankings DONE (`5751487` upstream + `5751488` per_igr, 10/10 each, on
`baclm_reembed`, MAX_TRAIN=2000, ~2–7 min); `per_unit_lr_ranking_imputed` REUSED (already full-band). Ladder default
dirs now: `upstream_lr_ranking_imputed_full` / `per_unit_lr_ranking_imputed` / `per_igr_lr_ranking_imputed_full`.
**TB ladder REBUILD `5758597` RUNNING** (--array=0-9 --mem=250G --time=1:30 --export REPO=$WT; PENDING(Priority)
on the fairshare floor). Monitor bg-task `bauga0jnt`.

**NEXT (post-compact):** (A-verify) pull the 10 rebuilt TB `<drug>_amr_ladder_table.csv` → `render_amr_figs.sh tb
ethionamide ethionamide` + `... kanamycin kanamycin` → confirm kanamycin routes the merged `mura→ogt` (or rrs) as
its ★ non-coding rung + the single merged bar; **RE-ASSESS "linear ceiling = ft_mean"** — pre-A3 ethionamide
already showed FT 0.863→FT⊕IGR(inhA promoter) **0.920** (+0.057), so the promoter/convergent rung MAY genuinely
lift (do NOT record the stale "all lifts≈0" — re-check the rebuilt tables). Then **Kp**: same `FEATURE=imputed_full`
re-runs `--array=0-21` + `SPECIES=kp` ladder rebuild; then Phase B (rif/iso tables already exist; summary panels;
Part 6 whole_igr/fragment distribution plots; Kp cipro FT cache). GBDT still deferred (own plan/subpackage).

## CHECKPOINT 2026-07-21b — LADDER-CORRECTNESS reframe + two-panel causal + Parts 1–4 DONE (supersedes the "Bug a IN PROGRESS" state below)

**David reframed bug (a): the Kp ladder is MATHEMATICALLY WRONG, not suboptimal.** Gene rung was SELECTED by carrier-only AUROC but the block is zero-imputed into a **linear** head → a low-penetrance gene (iME4) becomes a mostly-zero block a linear model reads as "no contribution" → concat can DEGRADE vs FT-mean (the tetracycline regression). Only the gene rung; the non-coding rung is already prevalence-gated ≥0.9 (safe). Plan rewritten; GBDT split out (below).

**Parts 1–4 DONE + committed + pushed (branch refactor/consolidate-engine):**
- **`d571092` (Part 1+2):** `default_gene_ranking` HARD-FAILS (FileNotFoundError) if the imputed ranking is absent — NO carrier-only fallback; explicit `--gene-ranking-csv` validated by `_assert_imputed_ranking` vs a new `impute_mode` column (`carrier_only|imputed_zero|presence`) stamped by all 4 ranking builders. Engine `build_per_gene_lr_ranking.sh` gains `FEATURE=imputed` (→ `per_gene_lr_ranking_imputed_baclm/`, reproducible TB+Kp; replaces the ad-hoc kleb launcher). 44 tests green.
- **`0491495` (Part 3):** `plot_causal_comparison.py` → TWO stacked panels: TOP=presence-imputed (acquired determinants recover to their red tick; imputed rung-2 gene marked ◆), BOTTOM=carrier-only (recomputed top-10 = the GBDT candidate pool). **Penetrance → bar opacity** (floored; ScalarMappable colourbar). `run()` now takes `imputed_coding_csv`+`carrier_coding_csv` (non-coding shared). Visually verified.
- **`dc5237b` (Part 4):** `min_n_eval` gate (>100 eval carriers) on BOTH per-gene screen plotters (TB + Kp) for the non-imputed report; carrier-vs-imputed density already served by `plot_igr_lr_ranking` (method=per_gene, --imputed-csv).

**Cluster jobs — all PENDING, from concat worktree `$SCRATCHDIR/worktrees/concat`@`d571092` (DO NOT advance it until they finish):** imputed per-gene Kp `5743034`(0-21) + TB `5743036`(0-9) [I cancelled the stale-$HOME-launcher `5741470` + relaunched via the engine FEATURE=imputed launcher = correct PYTHONPATH + impute_mode column]; Kp whole-IGR `5743579`(0-21). fp32 rifampin `5734578` still RUNNING. Launch idiom: `--time=08:00:00` (QOS rejects exactly 24h), `REPO=$WT` on `--export`.

**GBDT accessory concat = DEFERRED to its OWN plan** (David: "careful separate planning" — own subpackage NOT engine/concat, own data-output folder). v0 draft parked `scratchpad/gbdt_draft/gbdt_accessory_amr.py` (sklearn HGB, zero-imputed no-presence v1, top-K from carrier-only rankings; v2 = presence-indicator gated).

**Fragment channel finding (Part 6 D):** `fragment_*` = fragmented IGRs split by adjacent RNA/truncation → NOT cleanly CDS-flanked → the flank-pair namer drops them → NOT a relabel; needs its OWN keying scheme + reader. No `per_igr_fragment` ranking dir exists on cluster (TB plots = uncommitted variant). RECOMMEND deferring fragment; whole-IGR parity delivered via `5743579`.

**Next (queue-gated):** Part 5 = rebuild 12 ladders on imputed (`build_amr_ladder.sh` now hard-fails without imputed) → verify tetracycline rung-2 ≠ iME4 + rung-2 ≥ FT-mean; re-render 12 ladders + 2 two-panel causals + 2 summary panels; hold the ladder-table CSVs until then. Part 6 C = cause-histogram re-render. Part 4 real render needs imputed + the eval carrier rankings.

## CHECKPOINT 2026-07-21 — plot redesign done + tetracycline two-bug fix (plan: the-cambridge-hpc-and-dreamy-thacker.md, rewritten)

**Plot redesign DONE + committed `aede420`** (branch refactor/consolidate-engine): 12 per-drug ladders + 12 causal-comparisons + 2 summary panels, per David's art direction. Ladder = two RED catalogue bars (strongest-single hatched + all-determinant ceiling) then BLUE FT (mid) + concat heads (royal, one colour) with `⊕` labels ("circled plus") + capped block names; title just "Kp <drug> prediction". Causal = VERTICAL bars, dark-blue catalogue / light-blue LR-only / red catalogue refs, retitled "Logistic Regression on bacLM Genomic Regions (defined by Bakta)". Summary = 2 series only (catalogue ceiling RED vs FT⊕gene⊕IGR concat=rung4 dark-blue). Plot modules: `engine/plots/plot_amr_ladder.py`, `plot_causal_comparison.py`, `apps/kleb/plot_amr_summary_panel.py`. Local render helper: scratchpad `render_amr_figs.sh` (findcsv searches wsd_rankings+cluster_pull).

**Tetracycline exposed TWO real bugs (both generalise):**
- **Bug b (causal join) = FIXED + committed `aede420`.** CARD names determinants `TetA/AAC(6')` (no parens); Bakta/LR names them `tet(a)/aac(6')-Ib` → string-match failed → real determinants read "not ranked". Fix = **data-driven CARD→Bakta map** (David's method, NOT string parsing): every `{Sample}_amr.parquet` row pairs `amr_gene_family`+`bakta_gene_name` (minimap overlap); combined store `processed/train_kleb_ast/amr_annotation/amr_calls_all.parquet` → `groupby(amr_gene_family).bakta_gene_name` majority+set. Builder `apps/kleb/card_bakta_map.py`, committed map `apps/kleb/refs/card_bakta_gene_map.csv` (94 families). Engine plot loads it generically via `--card-bakta-map` (Kp; TB passes none). Validated: tetracycline tet(a)/tet(d) + amikacin 6/12 acquired now dark-blue; TB byte-unchanged. Low-coverage flagged `‡`.
- **Bug a (concat picks the wrong gene) = IN PROGRESS.** Gene rung `_best_from_ranking` selects by CARRIER-ONLY AUROC but the block is presence-IMPUTED in usage ("selection≠usage"). Zero-imputed ranking `per_gene_lr_ranking_imputed_baclm/` is populated only for Kp cipro → tetracycline picked `iME4` (18% penetrance, NOT a catalogue determinant — do NOT call it lineage, see [[dont-conflate-penetrance-with-lineage]]) over tet(D). Fix = generate imputed rankings for all AST drugs → rebuild ladders. **Kp imputed job 5741470 RUNNING** (baclm, array 0-21, from concat worktree via a PYTHONPATH-sed'd copy of `build_per_gene_lr_ranking_imputed.sh` in `$SCRATCHDIR/jobs/`, EMBEDDING_STORE=baclm). Next: rebuild the 12 ladders (verify tetracycline rung-2 → tet(D)/tetR(D), not iME4) + re-render causals with imputed coding (presence-aware determinant AUROCs).

**tetR(D) finding (surfaced to David):** ground-truth map = `TetD→tet(d)`, `TetR→tetR(B)`; the strong `tetr(d)` region (0.89, top LR-only) is the tet(D) operon **repressor** — a distinct Bakta gene the minimap maps NO CARD determinant onto → legitimately LR-only (a presence proxy), not a mislabel. OPEN: operon-group it with TetD (domain choice) or accept ground truth.

**Still PENDING (David wants it done properly, "All AST drugs"):** B3 = imputed AND non-imputed (present-embeddings-only, gated >100 EVAL carriers = `n_eval>100` on carrier CSV) per-gene screens in SEPARATE plots + carrier-vs-imputed KDE (`plot_igr_lr_ranking.py --method per_gene --csv <carrier> --imputed-csv <imputed>`; the two screens via `plot_kleb_per_gene_ranking.py` pointed at each CSV). C = cause "histograms" (=ranked bars, `plot_kleborate_cause_histogram.py`; NO binned histogram exists). D = Kp whole-IGR (`build_upstream_region_lr_ranking.sh` CONVERGENT=1 BACLM_DIR=…/baclm_reembed → `whole_igr_lr_ranking/`) + FRAGMENT which has **NO committed reader** (`fragment_*` channel unread; resolve TB convention first). TB imputed rankings also still needed (no imputed dir on TB at all).

**Cluster mechanics (Isambard):** concat worktree `78afe4b` for CPU work; `$HOME/BacPredict` is stale branch `dev` (lacks imputed launcher + consolidated module paths) AND the FT/fp32 GPU jobs run from it → DON'T touch it; the per-gene launcher hardcodes `$HOME/BacPredict/src` PYTHONPATH so sed it to the concat-worktree src. fp32 rifampin `5734578` still RUNNING (~10h/24h); 16/20 FT done, 4 TB FT running. Kp `baclm_reembed` store EXISTS (3-channel).

## CHECKPOINT 2026-07-20b — rifampin HPC→Isambard investigation + fp32 experiment (plan: the-cambridge-hpc-and-dreamy-thacker.md)

**0.9046(CSD3)→0.9642(Isambard) rifampin — MEASURED bit-for-bit (concurrent SSH both clusters), not guessed:**
base model **BIT-IDENTICAL** (HF rev `ab3a91a2` on both, same safetensors bytes; Bacformer fork sha unchanged;
the HF refresh landed 2026-05-25 BEFORE the CSD3 run) → model is NOT the cause. The two runs scored
**DIFFERENT genomes** (CSD3 single-split drug-agnostic 20% n=7075 @0.9046 vs Isambard kfold rifampin-only
evaluate_seed=1 n=7127 @0.9642) → the jump is confounded by the eval set. **fp32→bf16 master weights (commit
b047ed8)** = the one real code lever; minor: early-stop thresh 0→0.005, A100→GH200/flash-attn. Hyperparams /
split-logic / AUROC metric / lib-versions all UNCHANGED (uv.lock identical). Provenance gap (results.json had
no dtype/versions/seed; git_sha null on Isambard) = why it was hard.

**Part 1+2 DONE — commit `0167e1c` pushed (branch refactor/consolidate-engine):** finetune_amr
`--precision {bf16,fp32}` (fp32 skips the line-302 `.to(bfloat16)` cast = native fp32 weights + bf16 AMP =
the CSD3/pre-b047ed8 condition; default bf16 = byte-identical). results.json **schema 1.2** (+`model.revision`
+`run`{precision,seed,lr,patience,eval_steps,warmup,max_steps} +`versions`{torch,transformers}); `_git_sha()`
resolves HEAD in the repo dir not cwd (was null on Isambard); TB launcher threads PRECISION → `_fp32` ckpt
dir; both FT launchers' Isambard SBATCH defaults fixed (`--mem 250G→110G` per-socket, `--time 24h→23h`
workq_qos DenyOnLimit — see [[isambard-ft-fanout-run-mechanics]]). 18 finetune tests pass, ruff-clean.

**fp32 rifampin RUNNING — job `5734578`** (worktree `$SCRATCHDIR/worktrees/fp32probe`@0167e1c, `--array=0` ⇒
SAME kfold fold0/seed1/evaluate_seed1 ⇒ SAME 7127 eval genomes as bf16 0.9642; PRECISION=fp32, 23h/110G).
Output → `…_rifampin_lr_0.00015_finetuned_fp32_fold00_seed1/results.json`. ~24h (may wall at 23h → resume).
**DECISION RULE:** fp32-eval ≈0.96 ⇒ bf16 NOT the cause (jump = eval-set + minor); ≈0.90–0.92 ⇒ bf16 genuinely
helps → lesson "train AMR heads in bf16."

**Part 3 (plots/results) IN FLIGHT:** 20 bf16 FT `5733635-54` (~3–4h, David) + IGR rankings `5733364/66/68/70`
+ Wave-2 ladders (isoniazid `5732777` + 5 Kp `5732779`). 6/12 per-drug ladder figs rendered. **TOMORROW AM:**
verify fp32 vs bf16; render remaining 12+20 ladders; summary-panel new-schema adapter
(`apps/kleb/plot_amr_summary_panel.py`: read rung/config/block/ceiling_auroc → ceiling / FT-mean(rung1) /
+both(rung4)); reconcile framing (deployed-head vs within-eval CV probe vs catalogue + eval-set caveat);
commit 32 ladder CSVs to visualisations/. **3 worktrees:** consolidate@bc627ee (20 bf16 FT, PINNED — don't
advance) · fp32probe@0167e1c (fp32 rifampin) · concat@78afe4b (rankings/plots).

## CHECKPOINT 2026-07-20 — plot corrections done+pushed; convergent fan-out verifying; THEN concat runs + ladders

Reviewing WS-D diagnostics surfaced 2 plot defects, both FIXED + committed + pushed (branch
refactor/consolidate-engine; concat worktree `$SCRATCHDIR/worktrees/concat` advanced 5263b71→**78afe4b**):
- **Part A `8a917cf` — causal_comparison scores each determinant vs its OWN catalogue mut_auroc, not a global
  top-10 cutoff.** `_catalogue()` reuses `driver_panel.parse_driver_csv` → {gene→mut_auroc} + __ALL__ ceiling;
  `_determinant_status(lr,cat,margin=0.05)`: recovered (lr≥cat−margin) / under-recovered / absent. Draws a
  per-determinant catalogue tick + ceiling line; collapses LR-only bare/prefixed dup rows. Re-rendered all 6
  diagnostics: kanamycin eis 0.593 vs own 0.620 = RECOVERED (was falsely "missed"); eth inhA(→fabg1)/etha/ethr
  recovered vs weak ticks; azithro's 7 CARD determinants correctly ABSENT (hollow) not missed. 6 tests green.
- **Part B `78afe4b` — build_upstream_region_lr_store `--include-convergent` flank-pair fallback.** The upstream
  screen keys only a gene's 5′ end, so convergent-flanked regions (rrn/rrs = murA(+)→ogt(−), both 3′-abut) get
  NO key → absent from per_igr_whole. Fallback emits `between:<left>→<right>` (reuse `_flank_pair`) for unclaimed
  regions; DEFAULT OFF (existing rankings byte-identical); driver `CONVERGENT=1` → own dir
  `whole_igr_lr_ranking<SUFFIX>/` so the ladder's `upstream_lr_ranking` input is UNTOUCHED. 8 store tests green.
  FINDING (Q(ii)): rrs 0.795 isolated (per_unit `rrna:rrs`) vs 0.723 in the whole 5555bp region (flank-pair
  murA→ogt) = ~0.07 dilution + STRUCTURAL absence from upstream screen (convergent); per_unit is the screen of
  record for rRNA. Q(i): per_igr_fragment = NAMED non-CDS bodies only (rrna/trna/crispr/**regulatory**/ncrna/
  oric), excludes unnamed spacers → ~551 units vs 4580 upstream anchors.
- **IN FLIGHT:** convergent fan-out **job 5727932** (TB rif/strep/eth/kan, `CONVERGENT=1 SUFFIX=_reembed
  BACLM_DIR=…/train_tb_ast/baclm_reembed`) → `whole_igr_lr_ranking_reembed/<drug>/`. Watcher **ba0bwe36w** polls
  drain + greps `between:` keys. PENDING post-drain: pull → re-render per_igr_whole → verify `between:mura→ogt`
  appears (kanamycin). Kp convergent fan-out NOT yet run. per_igr_whole render = plot_igr_lr_ranking `--method
  per_igr_whole` fed the whole_igr_lr_ranking_reembed CSV.

**GO-FORWARD (David 2026-07-20): finish these plots/corrections → then START THE CONCAT RUNS + FINISH THE
LADDERS.** = WS-C: the diagnostic ladders **5721477-9 have DRAINED** (inspect tables first — verify the
non-coding rung loads carriers post-5263b71 fix), THEN full TB+Kp ladder panels (`--array=0-9` /
`SPECIES=kp --array=0-21`) + plot_amr_ladder figs + refresh `visualisations/{tb,kp}/amr_summary_panel`. All
per_igr_whole/causal/ladder PNGs are gitignored (on disk only, regenerate from CSVs).

## TWO CORRECTIONS (David, high-effort re-plan 2026-07-17) — I had drifted:
1. **Non-coding rung MUST use `upstream:<gene>` (single flank), NOT flank-pair `left→right`.** My cipro proof
   wrongly used flank-pair (pptA→lamB = the OLD store we moved away from). Upstream anchoring is what recovered
   ethionamide inhA-promoter = `upstream:fabg1` 0.80@prev0.997. `build_amr_ladder` default `--igr-kind` → `upstream`.
2. **Selection≠usage + IGR variants half-built.** Concat feeds a ZERO-IMPUTED block → must SELECT by zero-imputed
   whole-cohort AUROC, which DOESN'T EXIST for IGRs. Carrier-only rankings pick low-prev artifacts (cipro
   upstream:ytfp 0.95@prev0.20, flank pptA→lamB@0.10). Need the zero-imputed ranking + its DISTRIBUTION to decide
   **accessory-vs-core** for the concat. (INVENTED lineage + mechanism steps stay DELETED.)

## IGR ITERATION MATRIX (be precise — several variants):
- **AXIS A parcel:** (i) regulators-only-initial=`intergenic_*` (baclm/, stale, gaps between ALL features);
  (ii) whole-IGR=`noncoding_*` (baclm_reembed/, CDS→CDS run); (iii) fragmented=`fragment_*`+`feature_*`
  (baclm_reembed/, split at feature bounds; named bodies rrna/trna/crispr/regulatory/oriC) — **EMBEDDED but NOT
  RANKED** (no LR reads feature_*).
- **AXIS B key×absence:** flank-pair `build_per_igr_lr_store` {carrier-only✓ / --impute-absent-zero✓ (⚠ overwrites
  carrier file, name branches only on presence) / --feature presence✓}; **upstream** `build_upstream_region_lr_store`
  {carrier-only ONLY, hardcoded — NO impute, NO presence}; per-unit body-key `build_per_unit_lr_store` = **DOESN'T
  EXIST**. All read whole-IGR channel today.

## WORKSTREAMS (WS-A CPU + WS-B GPU in PARALLEL [David chose parallel]; WS-C after). Diagnostics-first [David].
Diagnostic drugs: ethionamide (upstream win) · streptomycin/kanamycin/azithromycin (rRNA→feature channel) ·
ciprofloxacin/rifampin (coding controls).
- **WS-A (sort IGR, CPU):** A1 ✅DONE (416f32f): `impute_absent_zero`+`feature=presence`+`max_prevalence`
  (a (min,max] band) on build_upstream_region_lr_store, mirroring build_per_igr_lr_store; imputed/presence
  route to distinct --out-dir (upstream_lr_ranking_imputed/, upstream_presence_lr_ranking/) so carrier file
  never overwritten; presence writes per_upstream_presence_lr_<drug>.csv/presence_lr_auroc_. Stage-A test
  green (both strands of the 5′ index + band + 3 run modes). A2 ✅DONE (e743c1b): added `FEATURE=imputed` to
  build_per_igr_lr_ranking.sh (→ per_igr_lr_ranking_imputed/, `--impute-absent-zero` band 0.01–0.99, own dir
  so carrier file safe) + BACLM_DIR/MAX_TRAIN overrides; also added FEATURE={imputed,presence} to
  build_upstream_region_lr_ranking.sh so A1's module modes are runnable. Both shared engine launchers,
  additive (bash -n + dispatch verified). ⚠ upstream imputed/presence need REPO=concat worktree (416f32f+).
  Module already had impute/presence for the IGR side (pre-session); THESE ARE JUST LAUNCHER MODES — the
  actual diagnostic runs (choosing baclm vs baclm_reembed = AXIS A parcel) are WS-A5, NOT launched yet. A3 DONE (16e48ab module + 5c1d523 launcher): build_per_unit_lr_store.py keys re-embed feature_* by
  <type>:<name> (rrna:rrs / rrna:rrl / regulatory_region / crispr), MEAN-POOLS multi-copy bodies to one
  row/genome (the relaxed gate - NOT single-copy: rRNA is multi-copy), carrier/imputed/presence reuse
  fit_per_gene, NO GFF (units self-identify), fragments OUT (anonymous spacers, no type/name). 11-case
  Stage-A test green. Launcher build_per_unit_lr_ranking.sh: BACLM_DIR defaults to baclm_reembed; bands
  rRNA-aware (embedding+imputed KEEP prev 1.0 since rrs is ubiquitous+point-mut; only presence caps 0.99).
  A4 DONE (a6caaac): plot_igr_lr_ranking overlays carrier/zero-imputed/presence KDEs on density.png + a 3rd
  top10 bar; _key_col generalises the join to igr_pair|upstream_gene|unit|gene_name (upstream + per-unit now
  plot too); run() + both app drivers gain imputed_csv / --imputed-root (additive, existing paths unchanged).
  A5 RUNS LAUNCHED 2026-07-18 (jobs 5702465-78, 14 array-jobs = {upstream,per_igr,per_unit}×{carrier,imputed,
  presence} on baclm_reembed for diag drugs TB rif/strep/eth/kan (array 0,6,7,9) + Kp cipro/azithro (array 5,20);
  REPO=concat, MAX_TRAIN=2000). Existing on-disk carrier/presence rankings are PARCEL-i (stale baclm/ default —
  both synteny launchers default BACLM_DIR to the stale store), so re-ran ALL modes on baclm_reembed for a
  parcel-consistent 3-series. All write FRESH namespaced dirs (upstream uses SUFFIX=_reembed → upstream_lr_ranking
  _reembed / _imputed_reembed / upstream_presence_lr_ranking_reembed; per_igr imputed → per_igr_lr_ranking_imputed;
  per_unit → per_unit_lr_ranking{,_imputed,_presence}) so nothing collides with stale files. Pending behind the GPU
  jam. Then pull CSVs + render 3-series plots (plot_tb/kp_igr_rankings --rank-root/--imputed-root/--presence-root,
  --method upstream|per_unit) → **DECIDE (parcel,key,absence,prev-band)**; accessory-vs-core = does zero-imputed
  accessory AUROC stay above its presence baseline? (user-facing modeling decision — surface before baking in.)
  A5 RESULTS 2026-07-18 (all 14 drained; analysed via scratchpad/wsa5_analyze.py on-cluster): **DETERMINANTS
  ARE CORE, not accessory.** eth inhA-promoter = upstream:fabg1 0.797@prev0.997 (carrier, TOP of ranking) —
  SURVIVES the re-embed (was 0.80 stale); but EXCLUDED from the imputed ranking by the 0.99 band cap, so best
  imputed = 0.63 noise. rRNA drugs need the PER_UNIT key: strep rrna:rrs 0.65@0.997, kan rrna:rrs 0.80@0.999
  (+ upstream:eis 0.61@0.986 promoter); upstream:rpsl = 0.49 (coding, not the determinant). Kp azithro: maca
  0.70@0.993 + mph(a) 0.70@0.30 (upstream), per_unit rrs/rrl ~0.71-0.74. presence one-hot uniformly weak
  (~0.5-0.6). **LINEAGE CONFOUND severe:** coding controls rif(rpoB)+cipro(gyrA/parC) show ZERO determinant
  signal yet core/accessory regions still hit 0.70 (rif frmr/oriC) / 0.90 (cipro imputed cadb, carrier ytfp
  0.95@0.20) — and upstream:mlad is TOP for BOTH strep(0.82) AND rif(0.71) = same region diff drugs = proven
  lineage marker. So a high core-region AUROC ≠ mechanism. **DECISION (CONFIRMED David 2026-07-19):** non-coding
  rung = CORE region (prev≥~0.9), carrier/zero-imputed embedding (≈equal for core), NOT presence; SELECT best
  across upstream ∪ per_unit (promoters need upstream, rRNA needs per_unit); parcel = baclm_reembed (per_unit
  rRNA bodies exist ONLY there). REQUIRED FIX: the imputed/selection prevalence band must INCLUDE core up to 1.0
  (current (0.01,0.99] excludes EVERY real determinant) + an n_pos floor (~20-50) to kill low-n crispr artifacts
  (crispr 1.0@prev0.003 n=6). **DO NOT net out lineage** (David 2026-07-19: NOT relevant — the exercise is raw AUROC recovery with simple
  measures; rif/cipro stay in as controls that show the null/baseline lift, but we report RAW recovered AUROC, no
  subtraction/adjustment). See [[amr-ladder-raw-recovery-framing]].
- **WS-B (FT, GPU background):** B1 ✅DONE: Kp patience 30→15 — early-stop (no-improvement window) is the
  control, max-steps stays high/non-binding; NO hard epoch cap (my `--max-epochs 15` was WRONG, reverted per
  David). TB unchanged (eval1000/patience45≈15ep window). +worktree PYTHONPATH fix +TB `DRUG` override +--time
  24h (bc627ee/ba0a125). B2 ✅LAUNCHED 2026-07-17 all 10 PENDING (Kp amikacin 5693962/azithro 5693965/colistin
  5693966/tetra 5693967/aztreonam 5693968/cefoxitin 5693969; TB eth 5693970/iso 5693971/kan 5693972/strep
  5693973); monitor armed. **RUN MECHANICS → [[isambard-ft-fanout-run-mechanics]]** (worktree/PYTHONPATH/QOS).
  B3 ✅TB rifampin cache LAUNCHED 2026-07-17 (job 5695361, PENDING behind the FT fan-out) via NEW reusable
  launcher `apps/tb/scripts/cache_ft_bacformer_gene_embeddings_tb.sh` (fe035ad; DRUG-overridable single-job,
  BACPREDICT_REPO PYTHONPATH fix). Invokes `cache_bacformer_gene_embeddings --mode finetuned --eval-only`
  (organism-agnostic, CALLS-FREE); ckpt=checkpoints/…_rifampin_…_finetuned_fold00_seed1 (parent dir;
  resolve_checkpoint_dir finds checkpoint-31000), rank=pangena_predict/per_gene_lr_ranking_baclm/rifampin/
  per_gene_lr_rifampin.csv (rpoB 0.9625; ranking only picks top-N gene side-output — genome-mean is over ALL
  proteins so any valid ranking works), out=pangena_predict/ft_bacformer_cache/rifampin → ft_genome_mean_
  rifampin.npz. Fire one per headroom TB drug as its FT lands (DRUG=<drug>). Kp uses the existing
  apps/kleb cache_ft_bacformer_gene_embeddings.sh (⚠ still hardcodes $HOME/BacPredict/src — needs the same
  BACPREDICT_REPO fix — FIXED 2026-07-17 (f28d832 PYTHONPATH + 9734e8d ranking->per_gene_lr_ranking_baclm;
  old per_gene_lr_ranking_imputed path stale/absent). Launch per drug: sbatch --array=<idx>
  --export=ALL,BACPREDICT_REPO=$WT (idx=Kp DRUGS pos: amikacin13 cefoxitin16 tetracycline17 aztreonam18
  azithro20 colistin21). 4 Kp headroom cached 2026-07-17: amikacin(5696806) azithro(5696716) colistin(5696730)
  tetracycline(5696729), guarded on results.json (clean early-stop). Genome-mean ~27MB; top-N tokens ~1GB/drug.
  Cipro FT-mean already cached (n=865 eval-holdout).
- **WS-C (ladder — START NOW, David 2026-07-19):** re-point build_amr_ladder to the CORE non-coding rung —
  select the best region with prevalence ≥ ~0.9 (core) AND an n_pos floor (~20-50, kills crispr 1.0@0.003 low-n
  artifacts) across upstream ∪ per_unit on baclm_reembed (promoters via upstream:<gene>, rRNA via per_unit
  rrna:rrs/rrl); carrier≈imputed embedding for core (NOT presence). Ladder configs on the FT genome-mean rung:
  **+best coding gene, +best IGR (core non-coding), +both** vs the RED catalogue one-hot ceiling. Run diagnostics
  first (rif/cipro controls + eth/strep/kan/azithro) → then full TB+Kp panels + refresh
  visualisations/{tb,kp}/amr_summary_panel.{csv,png}. **NO lineage subtraction** — raw recovery only (see A5 note
  + [[amr-ladder-raw-recovery-framing]]).
- **WS-D (visualisation — David 2026-07-19):** (a) 3-series density + top-10 (carrier/imputed/presence) for the 6
  diagnostic drugs via plot_igr_lr_ranking; (b) RESTRUCTURE visualisations/{species}/{drug}/per_igr →
  **per_igr_whole** (whole CDS→CDS IGR = noncoding_* channel; upstream/flank key) + **per_igr_fragment** (the
  named feature bodies = feature_* channel: regulatory_region + rrna/ribosomal = the per_unit key); (c) HISTOGRAM
  plots of the LR-AUROC distributions for whole vs fragment; (d) CAUSAL COMPARISON per drug: catalogue-causal vs
  LR-causal (which regions/genes the WHO/CARD catalogue flags vs which the LR ranks top); (e) plots of the CODING
  proteins too (per-gene coding LR = per_gene_lr_ranking_baclm) alongside the non-coding.

## CHECKPOINT 2026-07-19 (pre-compact) — WS-C ladder RUNNING + WS-D plots DONE
**Repo HEAD 5263b71** (branch refactor/consolidate-engine). Session commits: fe5cc4b (ladder core rung + 4
configs + per_unit loader + launcher + tests), 664edc0 (4 plotter bug fixes), 921d64c (causal-comparison
module), 4e8004b (ladder-plot test→4 configs + launcher FT_CACHE override), 5263b71 (ladder eval_auroc fix).

**WS-D DONE.** For all 6 diagnostic drugs, rendered per_igr_whole (upstream key) + per_igr_fragment (per_unit
named bodies rrna/regulatory/crispr) + per_gene (coding) 3-series (presence/carrier/zero-imputed) top10+density,
PLUS causal_comparison.png, into src/bacpredict/visualisations/{tb,kp}/<drug>/. ⚠ PNGs are GITIGNORED (*.png;
on disk only, NOT committed) — regenerate via the app drivers. NEW engine/plots/plot_causal_comparison.py:
catalogue determinants (recovered/missed/absent) vs LR-only top hits on one AUROC axis; curated synonym map
bridges inhA↔upstream:fabg1 (mabA-inhA operon); alias-aware (no double-count). Fixed 4 bugs in
plot_igr_lr_ranking: _region_label handles upstream_gene/gene/gene_name/unit + _cap truncation; _is_causal
matches gene/gene_name; plot_top10+density n_pos floor (default 20) kills low-n crispr artifacts. 24 plot+concat
tests green. Density KDE = the "histogram" (David confirmed). Ranking CSVs pulled to scratchpad/wsd_rankings.
RENDER: plot_{tb,kp}_igr_rankings --rank-root <reembed/per_unit/per_gene dir> --imputed-root --presence-root
--method per_igr_whole|per_igr_fragment|per_gene --prefix per_upstream_lr|per_unit_lr|per_gene_lr; plus
plot_causal_comparison --coding-csv --upstream-csv --unit-csv --catalogue-csv (TB tbprofiler_gene_lr_<drug>.csv,
Kp card_determinant_lr_<drug>_family.csv).

**WS-C ladder RUNNING.** build_amr_ladder re-pointed: 4 configs ft_mean / +best coding gene (baclm/ store) /
+best CORE non-coding (prev≥0.9 + n_pos≥50, best across upstream ∪ per_unit on baclm_reembed) / +both, each
re-scored by zero-imputed OOF LR over the FT eval-holdout vs the catalogue ceiling. Launcher build_amr_ladder.sh
(CPU array, FT_CACHE overridable). Canary (Kp azithro 5718697) validated end-to-end BUT exposed the eval_auroc
bug (all-NaN eval_auroc_<drug> col made selection return None → non-coding rung silently fell to ft_mean; fixed
5263b71). Azithro first table: ft_mean 0.918 (>> catalogue 0.555 — FT crushes the CARD ceiling for azithro),
+gene(rfbC) 0.934. 6 diagnostics RE-LAUNCHED post-fix: 5721477 (TB rif/strep/eth/kan array 0,6,7,9), 5721478
(Kp azithro), 5721479 (Kp cipro — FT_CACHE=$SCRATCHDIR/processed/train_kleb_ast/ft_amr_cache/ciprofloxacin, its
CP-0 cache is NOT at the standard ft_bacformer_cache path). Watcher bne82ybcu cats the 6 tables on drain. NEXT:
inspect diagnostic ladders → full TB+Kp panels (--array=0-9 / SPECIES=kp --array=0-21) + plot_amr_ladder figs +
refresh visualisations/{tb,kp}/amr_summary_panel. concat worktree @ 5263b71; all FT-mean caches done (11/11).

## STATE (2026-07-17): ladder code done+proven (316d13a). FT models: rif(TB)+cipro(Kp) DONE + 10 headroom
DONE (early-stop CLEAN, 9/10): FT genome-means cached for ALL 10 non-isoniazid drugs — amik/azithro/colistin/
tetra/aztreonam/cefoxitin (Kp) + rif/eth/kan/strep (TB); the 5 new caches (jobs 5702115-19) COMPLETED 2026-07-18
(~3-5 min each, exit 0, all npz present). Only isoniazid's cache remains, blocked on its FT (still RUNNING ~16.7h;
new monitor b90eupom8 fires when it leaves the queue → launch its cache then via the TB cache launcher fe035ad).
isoniazid is the ONLY FT still training (best@ckpt-24000, ~15h, healthy - inside patience window, not stuck).
TB re-run eval AUROCs MEET/BEAT the CSD3 anchors: eth 0.810 (>0.774), kan 0.843 (>0.833), strep 0.873
(>0.834); Kp aztreonam 0.863, cefoxitin 0.921. **CSD3 back online** (ssh key-based, no MFA) — TB Stage-C checkpoints exist there
(`…/processed/train_tb_ast/checkpoints`, complete-genomes base, May29) but DECIDED **redo-on-Isambard, NOT
import** (no Kp checkpoints there → Kp runs here anyway; provenance uniformity; results.json `git_sha` is CODE
commit not weight-revision). CSD3 anchors (FT head, cross-check targets for the re-runs): ethionamide 0.774,
kanamycin 0.833, streptomycin 0.834. baclm_reembed 4-channel COMPLETE. Held-out scoring = fit_one_gene over FT
eval-holdout (OOF). Repo HEAD 5c1d523 (WS-A1-A4 + per-unit launcher ALL done; TB cache launcher fe035ad; Kp cache f28d832/9734e8d). **TWO worktrees now:**
FT worktree `$SCRATCHDIR/worktrees/consolidate` pinned @ bc627ee (isoniazid FT 5693971 still RUNNING ~15.5h;
5 caches 5702115-19 now COMPLETED — all 10 non-isoniazid npz present); `$SCRATCHDIR/worktrees/concat` advanced to 5c1d523 (cache
module unchanged 9734e8d→5c1d523 so the 5 pending caches are unaffected) for
all WS-A/WS-B3 CPU+cache work (keeps FT worktree untouched — the coordination caveat). Advance-worktree
foot-gun fixed: see [[isambard-ft-fanout-run-mechanics]]. aarch64 serial-sweep+process-fit.

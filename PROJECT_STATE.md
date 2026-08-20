# BacPredict — project state

> **Last verified: 2026-08-14 @ `23bf069`** (branch `refactor/consolidate-engine`).
> **Verification scope:** repo tree read directly; all 32 fine-tune numbers read from each
> checkpoint's own `results.json` on CSD3; both catalogue ceilings re-extracted from their source
> CSVs; split-table ↔ deployed-holdout equivalence checked for all 32 drugs (0 mismatches).
> Not re-verified this pass: the invasion GWAS numbers (taken from the sibling agent's committed
> write-ups) and the concat-ladder rungs (pending regeneration).
> Added 2026-08-14 (§3.4 model comparison): both invasion score archives re-scored from their npz on
> load and matched to the AUROC of record (0.785816 / 0.765471) before any threshold was derived.
> Added 2026-08-18 (§3.2 `all_samples_2`): metrics read from that run's own `results.json`; its
> holdout verified **identical genome-for-genome** to the pooled cohort's (2,822/2,822, zero leakage
> into train/validate) from `split_manifest.json` and both split CSVs, not inferred from matching n.
> Lab-collection split composition (154/22/31/44) counted from `lab_collection_invasion_predictions.csv`.
> Added 2026-08-19 (§3.4 Q1): unitig+sublineage numbers read from that run's own
> `unitig_model_results.json`; the plain-vs-+SL score agreement and per-sublineage AUROCs
> recomputed from both `unitig_cohort_scores.npz` archives, which were checked to cover the same
> genomes, labels and splits before being compared.
> Added 2026-08-20 (§3.2 sublineage): 432 genomes LIN-typed on CSD3 with MiST 1.3.0; every number here read from that run's own manifests, not from a summary. Validated twice — 7,193 archived calls vs metadata_v2 (0 conflicts) and 48 already-labelled genomes re-called from scratch (48/48 agree). The proposed clusters are NOT yet live on Isambard.

## 0. How to use this file

This is the **single authority on current state** for this repository.

| Question | Answer lives |
|---|---|
| What is the current status / next step of anything? | **Here**, §3, in that layer's section |
| What number do I quote? | **Here**, §3 "Numbers of record" — and only if it names its artifact |
| How does a thing work, and why was it built that way? | The sub-project's `CLAUDE.md` or `docs/` write-up |
| What are the global conventions? | Root [`CLAUDE.md`](CLAUDE.md) |
| What is the argument behind a result? | The `PROGRESS*.md` / write-up docs — those keep their science |

Three rules make this work:

1. **Status, numbers and next-steps live only here.** Sub-project docs hold mechanics and argument.
   When a doc and this file disagree about state, **this file wins**.
2. **A number is of record only if §3 names the artifact it was read from.** A number without a path
   is a quotation, not a fact. This file was written because quotations had been circulating for
   three weeks after the runs behind them were redone.
3. **Update this file in the same commit as the work that changed a fact**, and move the stamp above.

Where a memory and this file disagree, **this file wins** — memories are not versioned and cannot be
reviewed.

---

## 1. What this project is

Fine-tune [Bacformer](https://github.com/macwiatrak/Bacformer) genome embeddings to predict bacterial
phenotypes, and work out **which genes carry the signal**.

Two organisms, two phenotype families:

- **AMR / AST** — 22 *Klebsiella pneumoniae* drugs, 10 *M. tuberculosis* drugs. The programme
  hypothesis is that Bacformer reads **HGT / acquired** resistance well and **chromosomal point
  mutation** less well. Kp is the HGT-rich regime; TB is almost purely chromosomal.
- **Invasion / isolation source** — blood vs faeces in *Klebsiella*, as a virulence proxy.

Deliverables, in the order they matter:

1. **An honest headline per drug**, against a catalogue ceiling that says what a known-determinant
   model can do, and against a mechanism-agnostic unitig GWAS baseline that says what *any* sequence
   feature can do.
2. **A descriptive gene-level account** — the concat ladder — of which genes and weights the model
   uses, compared against CARD / WHO. This is **descriptive, not causal**: see §6.
3. **Downstream application** — resistance trends over time, and ranking a lab collection for
   *Galleria* testing.

---

## 2. Repo map

`src/` holds one wrapper package plus six standalone packages.

| Path | What it is |
|---|---|
| [`src/bacpredict/engine/`](src/bacpredict/engine/) | **The organism-agnostic pipeline.** Every AMR stage lives here |
| [`src/bacpredict/apps/tb/`](src/bacpredict/apps/tb/) | TB adapter — WHO / TB-Profiler catalogue, tbprofiler pixi env |
| [`src/bacpredict/apps/kleb/`](src/bacpredict/apps/kleb/) | Kp adapter — CARD + Kleborate, AMR sidecar, metadata curation |
| [`src/bacpredict/visualisations/`](src/bacpredict/visualisations/) | Publication mirror — see its [`PROVENANCE.md`](src/bacpredict/visualisations/PROVENANCE.md) |
| [`src/bacpredict/_archive/`](src/bacpredict/_archive/) | Concluded TB SNP diagnostic. Excluded from wheel/ruff/pytest |
| [`src/bac_pyseer/`](src/bac_pyseer/) | Pyseer GWAS — `kleb_iso_source` (invasion) and `ast_gwas` (AMR) |
| [`src/kleb_iso_source/`](src/kleb_iso_source/) | Invasion fine-tuning (distinct from the GWAS subfolder of the same name) |
| [`src/amr_over_time/`](src/amr_over_time/) | Predict AST across ~80k genomes → resistance trends |
| [`src/gene_array_lasso/`](src/gene_array_lasso/) | Sparse-group-lasso over gene-presence arrays |
| [`src/genome_prep/`](src/genome_prep/) | Assembly/GFF preparation shared by the GWAS packages |
| [`src/dp_short_read/`](src/dp_short_read/) | Stub. Deferred — see [`docs/_parked/`](docs/_parked/) |

Engine subpackages: `ast_labels` · `download` · `embedding` · `finetune` · `splits` · `gene_lr` ·
`segment_amr_lr` · `concat` · `ref_catalogues` · `plots` · `config.py` · `scripts`.

### ⚠ Dead paths — these no longer exist

They are still referenced by older docs and by every memory written before 2026-07-11.

| Referenced as | Actually |
|---|---|
| `src/tl/` (`tl.train.*`, `tl.embed.*`) | `src/bacpredict/engine/` — `finetune/`, `embedding/` |
| `src/tb_ast/`, `src/kleb_ast/` | `src/bacpredict/apps/{tb,kleb}/` + the shared engine |
| `src/pangena_predict/` | Split across `engine/gene_lr/` and `_archive/tb_snp_diagnostic/` |
| `src/predict_hgt/` | Retired. Milestones parked in [`docs/_parked/`](docs/_parked/) |
| `src/admixture/` | Never created. Deferred — [`docs/_parked/`](docs/_parked/) |
| `snp_vs_esm_prediction.py` | Deleted. Its holdout resolver is now `engine/finetune/holdout.py` |

---

## 3. Layers

### 3.1 AMR prediction — fine-tune, catalogue ceilings, concat ladder

**Status.** All 32 fine-tunes are **trained and deployed**: 22 Kp + 10 TB, k-fold fold 0 / seed 1,
`n_folds=5`, `evaluate_seed=1`, base model `macwiatrak/bacformer-large-masked-complete-genomes`, all
**bf16**, run 15–21 July 2026. Kp ceilings are complete and current for all 22 drugs. **TB ceilings
are provisional** (§3.1 Caveats). The concat ladder is correct in code but its rendered outputs
predate the fixes and need regenerating.

**Numbers of record.**

*Kp fine-tune* — `$R/bac_ast_prediction/processed/train_kleb_ast/models/finetune/klebsiella_pneumoniae_<drug>_lr_0.00015_finetuned_fold00_seed1/results.json`, AUROC / AUPRC / balanced accuracy / n:

| Drug | AUROC | AUPRC | Bal.acc | n | Drug | AUROC | AUPRC | Bal.acc | n |
|---|--:|--:|--:|--:|---|--:|--:|--:|--:|
| amikacin | 0.9501 | 0.8786 | 0.8777 | 708 | imipenem | 0.9578 | 0.9543 | 0.9116 | 509 |
| ampicillin-sulbactam | 0.9463 | 0.9847 | 0.8027 | 450 | levofloxacin | 0.9636 | 0.9889 | 0.9165 | 482 |
| **azithromycin** | **0.7993** | 0.9544 | 0.7070 | 384 | meropenem | 0.9713 | 0.9494 | 0.9334 | 924 |
| aztreonam | 0.8630 | 0.9454 | 0.8191 | 602 | piperacillin-tazobactam | 0.9384 | 0.9515 | 0.8848 | 690 |
| cefazolin | 0.9861 | 0.9968 | 0.9353 | 594 | tetracycline | 0.8694 | 0.9112 | 0.8428 | 423 |
| cefepime | 0.8724 | 0.9249 | 0.7393 | 557 | tobramycin | 0.9743 | 0.9800 | 0.9389 | 570 |
| cefotaxime | 0.9861 | 0.9885 | 0.9330 | 464 | trimethoprim-sulfamethoxazole | 0.9614 | 0.9721 | 0.9056 | 848 |
| cefoxitin | 0.9207 | 0.9503 | 0.8606 | 619 | ceftazidime | 0.9852 | 0.9928 | 0.9485 | 951 |
| ceftriaxone | 0.9852 | 0.9958 | 0.9177 | 682 | cefuroxime | 0.9804 | 0.9981 | 0.8975 | 485 |
| ciprofloxacin | 0.9717 | 0.9863 | 0.9328 | 877 | colistin | 0.9094 | 0.8330 | 0.8312 | 282 |
| **ertapenem** | **0.9882** | 0.9937 | 0.9786 | 424 | gentamicin | 0.9707 | 0.9394 | 0.9468 | 984 |

*TB fine-tune* — `$R/bac_ast_prediction/processed/train_tb_ast/checkpoints/mycobacterium_tuberculosis_<drug>_.../results.json`:

| Drug | AUROC | AUPRC | Bal.acc | n | Drug | AUROC | AUPRC | Bal.acc | n |
|---|--:|--:|--:|--:|---|--:|--:|--:|--:|
| ethambutol | 0.8861 | 0.6984 | 0.7989 | 5841 | pyrazinamide | 0.9163 | 0.7644 | 0.8229 | 3166 |
| ethionamide | 0.8097 | 0.5962 | 0.6736 | 2535 | rifabutin | 0.8381 | 0.7798 | 0.7890 | 2427 |
| isoniazid | 0.8922 | 0.8723 | 0.8294 | 6909 | **rifampin** | **0.9642** | 0.9160 | 0.9214 | 7127 |
| kanamycin | 0.8431 | 0.6338 | 0.7180 | 3493 | streptomycin | 0.8727 | 0.7770 | 0.7943 | 1792 |
| levofloxacin | 0.8262 | 0.5929 | 0.6531 | 2761 | **moxifloxacin** | **0.7945** | 0.5002 | 0.6478 | 3460 |

*Catalogue ceilings* — [`visualisations/kp/catalogue_ceiling_panel.csv`](src/bacpredict/visualisations/kp/catalogue_ceiling_panel.csv)
(22 drugs, CARD, **current**) and [`visualisations/tb/catalogue_ceiling_panel.csv`](src/bacpredict/visualisations/tb/catalogue_ceiling_panel.csv)
(9 drugs, WHO/TB-Profiler, **provisional**). Each row carries its own `ceiling_estimator` and
`ceiling_status`.

**In flight.** Nothing. No AMR jobs are queued or running.

**Next.**
1. Regenerate the concat ladders and the `*_card_cause_histogram_*` figures — the checked-in copies
   predate the presentation fixes, and the untracked ladder CSVs predate the leak fix entirely. The
   *cluster* ladder tables are current; it is the checked-in mirror that is not.
   **⚠ While doing it, fix where the ladder's RED ceiling comes from.** `build_amr_ladder._catalogue_csv`
   resolves it to `visualisations/kp/<drug>/card_determinant_lr_<drug>_family.csv` — a **checked-in
   mirror of the retired k-fold probe** (all 44 of those files carry a non-zero `mut_auroc_sd`, and
   e.g. ertapenem reads 0.9905/155/1992 against the current 0.9828/168/2121). So the ladder's ceiling
   line is **not** the ceiling in `catalogue_ceiling_panel.csv`, and `tests/docs/` does not see those
   44 files. Point it at the panel, or regenerate the mirror from the current artifacts.
2. Add the head-vs-mean comparison across all drugs (a summary column plus a scatter). The one
   measured point is rifampin: LR-on-mean 0.958 vs deployed head 0.964 (Δ −0.006, ~2 SE), i.e. the
   head's LayerNorm is **not** shedding signal.
3. Rebuild the TB ceiling — all 10 drugs including rifabutin, through `load_splits` +
   `score_onehot_frame`, into a Kp-mirroring `who_ceiling/` layout. This is the first task of any
   TB work.

**Caveats.**
- **The TB ceiling is not comparable to the TB fine-tune.** It came from the retired whole-cohort
  k-fold probe, on a different evaluation set, and is missing rifabutin. A TB ceiling-vs-FT gap
  cannot yet be read in either direction. Full detail: [`visualisations/PROVENANCE.md`](src/bacpredict/visualisations/PROVENANCE.md).
- The `bacformer/` embedding store is **empty for both organisms** on CSD3.
- Kp and TB use **different checkpoint directory layouts** (see the paths above).
- Kp azithromycin **0.799** is the deployed head (its `results.json`); **0.816066** is the ladder's
  `ft_mean` re-probe (`…/train_kleb_ast/pangena_predict/amr_ladder/azithromycin/azithromycin_amr_ladder_table.csv`,
  rung 1). Both are honest — they are different estimators. Always say which.

**Owns.** `src/bacpredict/engine/`, `src/bacpredict/apps/`, `src/bacpredict/visualisations/`.

---

### 3.2 Invasion / isolation source

**Status.** Stage C **done** on all three KPSC-clean cohorts. The pooled, country-controlled cohort
is the headline **and, as of 2026-08-18, the settled model of record** — the `all_samples_2` retrain
collapsed against it on a shared holdout (below). The signal survives country control and holds
*within* every major clone.

**Numbers of record** — eval-holdout AUROC, each from its own run's `results.json` under
`<root>/processed/train_iso_source/<cohort>/models/…`:

| Cohort | n | AUROC | Note |
|---|--:|--:|---|
| `sampled_country_2_1_all` (pooled) | 14,211 | **0.786** | **The headline.** The cohort the GWAS ran on |
| `sampled_country_2_1_stratified` | 9,866 | 0.762 | |
| `all_samples` | 21,533 | 0.827 | **Country-confounded — do not quote.** Sits *below* its own linear metadata baseline (0.857). Pre-dates the frozen test set, so **not** comparable to the row below |
| `all_samples_2` (bf16) | 21,420 | **0.765** | **The country-confound test, and it collapsed.** Scored on the *identical* frozen 2,822-genome holdout as the pooled row |

Comparators, all on the pooled cohort: strongest linear metadata stack 0.731 (Bacformer +0.055);
all-Kleborate 0.640; Kleborate virulence+AMR 0.638; virulence one-hot 0.552; **virulence_score alone
0.489 — chance.** Per-sublineage: SL258 0.858, SL15 0.841, SL307 0.815, SL17 0.806, rare-SL 0.759,
SL147 0.738. Lineage identity is not what the model reads.

**`all_samples_2` — the country-confound test, resolved 2026-08-18.** Job `33615516` COMPLETED (28h46m
of a 36h wall, early stopping fired), bf16, read from
`processed/train_iso_source/blood_faeces/all_samples_2/kpsc_human/models/results.json`: eval AUROC
**0.7652**, auprc 0.7634, sens 0.772, spec 0.621, bal-acc 0.697, n 2,822, prevalence 0.5198.

The comparison is like-for-like and was verified rather than assumed, from
`all_samples_2/kpsc_human/split_manifest.json` and both split CSVs: the evaluate sets are **identical
genome-for-genome** (2,822 of 2,822), **zero** holdout genomes appear in `all_samples_2`'s train or
validate, and the manifest records the freeze explicitly (`n_frozen_test_requested` 2,822, present
2,822, missing 0). Cohort 21,420 (train 16,552 / validate 2,046) against the pooled cohort's 14,119
(train 9,885) — **~67% more training data, 2.1 AUROC points lower.**

Per the §6 model-choice rule this is a **collapse, which is clean evidence**; the near-duplicate audit
that rule requires applies only to a *win*, so this branch closes without one.

**In flight.** bf16 re-runs of all three cohorts (the originals were fp32; the bf16 cast landed later
in `a817ac2`).

**Next.**
1. ~~Land the bf16 A/B and the `all_samples_2` retrain, then apply the model-choice rule in §6.~~
   `all_samples_2` **DONE** and the rule applied — see above. The bf16 A/B on the other cohorts stands.
2. ~~Unitig honest re-run~~ — **DONE.** The leakage-free re-fit (selection on train+validate only)
   landed; its numbers are in §3.4, and they change the verdict — see §3.3.
3. ~~Score the lab collection for predicted invasiveness and hand over the ranking.~~ **DONE** — the
   ranking and the three-way model comparison are in §3.4.
4. bac-LM forward pass → per-gene LR → concat ladder. **Demoted to the last job.**

**Caveats.**
- Every published iso-source number is **fp32**, not bf16. Absence of `run_config.precision` in a
  pre-`a817ac2` `results.json` means fp32.
- ~~A win for `all_samples_2` is not conclusive~~ — **moot, it collapsed** (2026-08-18). The asymmetry
  rule in §6 still governs any future "more data" retrain.
- **The `all_samples_2` comparison rests on the frozen holdout being genuinely shared.** It was checked
  (identical evaluate sets, zero leakage). Any future cohort claiming comparability must be checked the
  same way — a matching `n` and prevalence is *not* proof, it was what first looked suspicious here.
- Mechanism is an **open hypothesis**. Do not assert it.

**Owns.** `src/kleb_iso_source/`.

---

### 3.3 Mechanism-agnostic baselines — the unitig GWAS

This is deliberately a **layer, not a package**: the same yardstick serves both §3.1 and §3.2, and
that convergence is the point. It answers "how good is the fine-tune, *really*" without assuming any
mechanism.

**Status — AMR (`src/bac_pyseer/ast_gwas/`).** Pilot complete on 2 of 22 Kp drugs. The Kp cohort,
unitig matrix and lineage clusters are **built and shared**, so the remaining 20 drugs are read-out
only. TB is not started.

**Numbers of record** — identical holdouts (**set membership verified identical**, not merely equal
counts), paired bootstrap CI. Unitig arm from `…/pyseer_ast/kp/<drug>/lr/results.json`; FT arm from
the checkpoint's **`eval_scores.npz`** re-score, *not* its `results.json`:

| Drug | FT AUROC/AUPRC | Unitig-LR AUROC/AUPRC | Bal.acc @Youden | Δ (unitig − FT) | Verdict |
|---|---|---|--:|---|---|
| ertapenem | 0.9878 / 0.9937 | 0.9775 / 0.9853 | 0.9804 vs 0.9530 | **−0.0103** [−0.0187, −0.0031] | separates from zero |
| colistin | 0.9100 / 0.8333 | 0.9188 / 0.8077 | 0.8444 vs 0.8477 | **+0.0088** [−0.0171, +0.0347] | **a tie** |

**⚠ Why these FT values differ from §3.1's.** §3.1 quotes the training-time `results.json`
(ertapenem 0.9882, colistin 0.9094); the table above quotes the `engine.finetune.evaluate` re-score
of the *same model on the same genomes* (0.9878, 0.9100). Both are honest — the ~5e-4 gap is
inference-time non-determinism, not a different model or a different holdout. **The re-score is of
record here** because the paired CI must come from the same `eval_scores.npz` the deltas do. Per §0
Rule 2, always say which pass a number came from.

**⚠ Ertapenem's unitig balanced accuracy (0.9530) is not a Youden number and not from the model
beside it.** It is `extra.pinned_C_metrics.balanced_accuracy` — the **pinned `C=1.0`** model at
**threshold 0.5** — whereas the 0.9775/0.9853 AUROC/AUPRC in the same row come from the swept-C model
(`C=0.001`). The same file also holds `operating_point.balanced_accuracy = 0.9255`
(`selected_on: "validation"`) and `metrics.balanced_accuracy = 0.9494` (threshold 0.5). Colistin's
0.8477 *is* its headline model's holdout-Youden operating point, so **the two rows of this table are
not measuring the same thing** and the "@Youden" column header is wrong for ertapenem.

**Fix before the fan-out standardises on this table:** re-score ertapenem's swept-C model at
Youden-on-holdout. AUROC and AUPRC are unaffected. The two pilot drugs were produced by different
code versions — the holdout-Youden convention (§6, 2026-08-13) landed between them.

Cohort: 7,080 of 7,088 genomes resolved; 5,829,181 unitigs → 3,760,582 features (27 GB matrix); af
filters min 71 / max 7009. ertapenem λ=4.198, 31,856 significant of 3,371,827 tested; colistin
λ=1.232, 9,277 of 2,486,812. **Ertapenem's λ and hit count are in its `lr/results.json`; colistin's are not — they are in `…/pyseer_ast/kp/colistin/gwas/colistin_gwas_summary.json`.** The two pilot drugs were written by different code versions and do not have the same fields.

**✅ The sublineage gap is CLOSED — the genomes are LIN-typed. Awaiting the swap on Isambard.**
Two separate causes, both now fixed: a **join failure** (the `Sample` key is a BioSample accession,
so long-read genomes deposited under a **GCA** accession were invisible), and genomes with no
`metadata_v2` row at all, which no join could reach and only LIN-typing could label.

**⛔ Sublineage is NOT derived from ST. They are different types and must never be conflated.**
`Sublineage`/`LINcode`/`Clonal group`/`Phylogroup` come from **Pasteur BIGSdb LIN-typing** — a
specific algorithm over a 629-locus cgMLST profile. **Kleborate v3.2.4 has no LIN-coding module**, so
no flag or mode makes it emit a Sublineage; `METADATA_v2_README.md` §6/§12 states this. Re-running
Kleborate on the unlabelled genomes would return the ST they already have. The stand-in
`sublineage_from_metadata --cluster-source st` remains available and remains clearly labelled, but is
**no longer needed for the Kp AST cohort**.

- **432 genomes LIN-typed** with MiST 1.3.0 against the Pasteur `scgMLST629_S` scheme (scheme 18),
  `…/david/lin_typing/{results,mist_lin_new.tsv}`. 432/432 typed, zero failures. **366 pass the
  quality gate** (`≤30` mismatched loci — the scheme's own `max_missing`), 114 distinct sublineages,
  led by SL258 (82) and SL307 (28). Produced by `src/genome_prep/lin_typing/`.
- **The inherited MiST index was repairable, not lost.** A Feb 2026 deletion took only the four
  top-level files of `seb/LIN_codes/scgMLST629_index`; all 629 locus directories survived. Three
  rebuild from disk (`rebuild_index_toplevel.py`) — no re-download, no re-clustering.
- **No BIGSdb credentials were needed.** The colleague's tokens are dead (two independent OAuth1
  implementations both mint a session token, then get 401 on every route — server-side). The
  **unauthenticated** `profiles_csv` stops at 2024-12-31 and misses the exact profile for 51% of
  these genomes, yet nearest-profile sublineage still agreed with the full database on **312/312** at
  `≤30` mismatched loci and on 100% out to 60. Every disagreement lay beyond that (median **442**
  loci), so failures are detectable via `pct_match` rather than silent
  (`…/lin_typing/profile_coverage_public.json`).
- **Two independent validations, both perfect.** Merging the 8,167 archived MiST calls against
  metadata_v2: **7,193 agreements, 0 conflicts**. And 48 already-labelled genomes re-called from
  scratch on real assemblies with the public-only table: **48/48 agree, 0 conflicts, 48/48 pass the
  gate**. The first tests the parser and merge; the second tests the profile table end to end.
- **178 recovered by fixing the join** — the long-read GCA-keyed rows, the **best-assembled genomes
  in the cohort** (median 60 contigs vs 122), so excluding them biased the clusters toward drafts.

**⚠ The no-label set is phenotypically non-random, and so is `other`.** Resistance differs
significantly on **9 of 16 drugs** between labelled and unlabelled (ertapenem — a pilot drug — 0.820
vs 0.648, p=4.5e-5), and the two subgroups diverge from each other (ceftazidime 0.486 GCA-keyed vs
0.752 unmatched). `other`, which §6 drops from the permutation null, is far less resistant than the
retained clusters (ciprofloxacin 0.393 vs 0.887; ceftazidime 0.422 vs 0.888). **Dropping it removes
the less-resistant 45%, not a random 45%.** Mechanism is **open** — recoverable-join artifact,
assembly quality, and study/provenance structure are all live readings.

**⚠ Lineage-cluster coverage — two different numbers, do not conflate.** *Label coverage* is how
many genomes carry any Sublineage at all; *named-cluster coverage* is how many land in a cluster
big enough to survive `min_size=100`. Since §6 **drops** `other` from the permutation null, the
second is the one that says what the null actually runs on. Conflating them once understated the
exclusion fivefold.

| | live | join fix only | **proposed** (+ LIN) |
|---|--:|--:|--:|
| label coverage | 91.2% | 93.7% | **99.1%** |
| named clusters at `min_size=100` | 10 | 11 | **11** |
| in a named cluster | 3,890 | 4,061 | **4,277** |
| named-cluster coverage | 54.9% | 57.4% | **60.4%** |
| `n_in_other` | 3,190 | 3,019 | **2,803** |

Live figures from `…/david/bac_ast_prediction/processed/pyseer_ast/kp/structure/lineage_clusters.manifest.json`
— **on CSD3, not Isambard**; the deprecated `…/david/processed/` tree has no `pyseer_ast` and looking
there suggests, wrongly, that the GWAS lives elsewhere. The other two columns are from
`…/david/lin_typing/{joinfix_only,proposed}/lineage_clusters.manifest.json`.

**The diff against the live file is strictly additive.** Same 7,080 samples both sides; **387
assignments change and every one is `other` → named**. Zero genomes leave a named cluster, and zero
move between named clusters — so no genome's existing lineage assignment is revised. Attribution:
**171** from the join fix, **216** from the new LIN labels. The one new cluster is **SL3010** (111
genomes, just over `min_size`) and it comes from the join fix, so LIN typing changes the null's
*coverage*, not its *structure*.

**⚠ `lineage_clusters.tsv` is still the OLD file.** The replacement sits at
`…/david/lin_typing/proposed/` and has **not** been swapped in. Because the change is additive, the
two pilot drugs' results are not *contradicted* — but their λ and permutation p-values were computed
against a null over 3,890 genomes and 10 clusters, so they are **not comparable** to anything run
after the swap. Re-run the pilots alongside the fan-out rather than quoting them across the change.

**Status — invasion (`src/bac_pyseer/kleb_iso_source/`).** Complete and written up.
[`PROGRESS_UNITIGS.md`](src/bac_pyseer/docs/PROGRESS_UNITIGS.md) ·
[`unitig_IGR_bias.md`](src/bac_pyseer/docs/unitig_IGR_bias.md).

- **Calibration settled.** The LMM with core-SNP kinship ablates under a within-lineage permutation at
  **both** sublineage and clonal-group resolution; the MDS fixed-effects model does **not** (λ_perm
  stays ~3.5–4 at common af). **The LMM is the method of record on both axes.** The common-af
  inflation is genuine signal, not structure — there is no af ceiling.
- **Hits mapped exhaustively** — all 33,039 significant unitigs placed in all 13,171 carriers
  (108.8M placements, ASM-recall 1.0). Invasion signal ~82% chromosomal / ~17% plasmid; faeces signal
  ~67% plasmid + ~13% prophage. **IS elements are not the hidden home** of the chromosomal fraction
  (0.01% IS overlap on the blood side).
- **IGR bias.** Hit unitigs touch intergenic DNA at ~2.3–2.5× a uniform-placement null, flat across
  thresholds, with an af-matched non-significant control sitting *at* the null. Holds on both
  chromosome and plasmid.
- **Head-to-head vs Bacformer — the tie does not survive a leakage-free unitig arm.** The original
  comparison (unitig **0.781** vs Bacformer **0.787**, a tie) gave the unitig model a **selection
  advantage**: its hit set was chosen by an LMM that had seen the holdout. Re-fitting that selection
  on train+validate only gives unitig **0.7655** [0.7472, 0.7824] against Bacformer **0.7858**, a
  delta of **+0.0210 [+0.0041, +0.0385] that separates from zero** — n=2,715, artifact
  `…/sampled_country_2_1_all_trainval/gwas_unitig_lmm/presence_model/unitig_cohort_scores.npz`
  (`"hit set selected on train+validate only"`). Full comparison in §3.4.
  **Quote the leakage-free number.** The selection-advantaged 0.7810 is a footnote, not the result.

**Next.**
1. **Kp AMR fan-out** — the remaining 20 drugs, in batches of ~5. Per drug ~57 min wall, ~55 core-h,
   ~12 min GPU, 1.6 GB; 20 drugs ≈ 1,100 core-h and ~5 GPU-h. LD-deduped control on all 20; FT
   re-score on all 20 (without `eval_scores.npz` there is no paired CI, and a gap cannot be told from
   a tie). Permutation null on the pilot plus any surprise only.
2. Invasion: DefenseFinder mapping; the faeces↔respiratory run and blood↔resp concordance;
   locus-level annotation.
3. **Blocked externally** — hotspot rates by isolation source (per-source rates against the
   whole-population background mutation rate at each locus, χ² for hotspots associated with invasive
   disease). Waiting on Aaron uploading the hotspots to HPC. Also blocked: the faeces vs
   liver/abscess contrast, pending recuration of the mixed `isolation_source_category` in BacHGT.

**Caveats.**
- **Only 2 of 32 drugs have `eval_scores.npz`.** Every other drug needs an `evaluate.py` pass before
  it can carry a paired CI.
- Threshold convention is **Youden on the holdout**, one convention for both arms. Sensitivity,
  specificity and balanced accuracy are therefore **optimistically biased** and must be reported as
  "at the optimal operating point". AUROC and AUPRC are unaffected.
- Hits are **LD-redundant, not independent**. Report at the pattern/locus level, never per-unitig.
- Mechanism readings in both write-ups are **hypotheses**, explicitly flagged as such.

**Owns.** `src/bac_pyseer/`.

---

### 3.4 Downstream applications

**Status.** `amr_over_time` has run; results mixed. The lab-collection ranking has an interim output —
`lab_collection_invasion_predictions.csv`, 677 genomes ranked with unitig and Kleborate comparators.
The `all_samples_2` gate in §3.2 is **resolved** (2026-08-18, pooled wins), so the ranking is final on
the pooled model — which is also the model the comparison below was already ordered by. The head-to-head
comparison is **built and published**.

**Numbers of record.** ⛔ **No lab-collection AUROC is of record** — withdrawn 2026-08-18 (§6). The
former 0.719 (n=44, 6 positives) and 0.903 (inflated, 176 of 251 fitted on) are **not** quotable and
must not be reintroduced; the split composition is 154 train / 22 validate / 31 unseen / 44 evaluate.
Accuracy figures come from the cohort holdout, n=2,822. SL258 predictions span 0.013–0.997.

**Next.** Finalise the ranking for *Galleria*; then the queued models below.

**Caveats.** The two candidate models agree only ρ=0.68 on the 673 lab genomes, with **top-20 overlap
2/20** (SL258: ρ=0.53, top-5 overlap 2/5). Different tubes go into the animal model depending on the
choice, so it is worth getting right.

**Model comparison — Bacformer vs unitig vs annotation (2026-08-14).** Built by
`kleb_iso_source.build_model_comparison_report` (`thresholds` → `compare` → `shortlists`); every figure
below is in `processed/train_iso_source/lab_collection/model_comparison_summary.json` with its source
path. The module re-scores both npz archives on load and refuses to run if they are not the files of
record — the leakage-free and selection-advantaged unitig cohorts differ only by a `_trainval`
directory suffix and hold an identically named `unitig_cohort_scores.npz`.

- **Numbers of record, pooled cohort holdout** (the only scope where all three are comparable):
  Bacformer **0.7858 [0.7695, 0.8024]** n=2,822 · unitig leakage-free **0.7655 [0.7472, 0.7824]**
  n=2,715. **Genome-only annotation comparators** (the like-for-like set): all-Kleborate **0.6396 —
  this is the Kleborate ceiling** · virulence+AMR 0.6384 · AMR-classes 0.6171 · virulence one-hot
  0.5522 · Kleborate `virulence_score` **0.4885** (below chance). **Not comparable — carry non-genomic
  metadata:** country+sublineage 0.6940 · country+sublineage+k_locus+virulence+amr 0.7307 (labelled
  "richest linear stack" in `BASELINE_LABELS`, which is what made it misread as an annotation model).
  Footnote only: selection-advantaged unitig 0.7810, whose hit set saw the holdout genomes.
- **Paired delta — the framing depends on which unitig model.** vs leakage-free:
  **+0.0210 [+0.0041, +0.0385], separates from zero.** vs selection-advantaged: +0.0055
  [−0.0110, +0.0230], a tie. Quote the first as the honest comparison and name the second as the
  handicap it removes.
- **Agreement on the 671 lab genomes** with both scores, at each model's own Youden point
  (Bacformer 0.4349, unitig 0.5272): both-invasive 220 · both-faeces 285 · unitig-only 124 ·
  Bacformer-only 42 · concordance **0.753**, κ **0.508**. Sensitivity: at 0.5, κ 0.479; at median
  split, κ 0.559. The unitig model calls far more genomes invasive (344 vs 262).
- **Correlation on the lab collection** (n=671): r² 0.430 logit / 0.435 prob, ρ **0.683**, slope 0.257,
  sd-ratio **0.392** — the unitig log-odds are ~0.4× as wide, so a shared cut-point would manufacture
  disagreement. Holdout figures recomputed identically to `model_agreement_holdout.json`.
- **Denominators.** 677 rows → 673 with a Bacformer score, 671 with a unitig score, 251 labelled
  (154 train / 22 validate / 31 unseen / **44** fully held out, 6 positives). No AUROC is reported on
  any of these scopes — see the withdrawal above and the §6 decision.
- **Commonest sublineages in the collection:** SL258 49, SL3010 40, SL307 37 — SL3010 has no cohort
  per-SL AUROC, so its ranking has no local validation.
- **Published report** (2026-08-18): <https://claude.ai/code/artifact/87cbaae7-d1f0-4e47-be98-910e3fd198b3>
  — six sections plus the register below, built from the seven `model_comparison_*` files only.

**Open questions from the comparison (2026-08-14/18) — observations and hypotheses, not findings.**
None blocks anything; they are what the queued models below are pointed at.

- **O1 — the SL3010 contradiction.** Four ST3010 genomes sit in Bacformer's SL3010 bottom ten at
  0.079–0.144 while the unitig model scores them 0.863–0.922 (`VRES0604` 0.127/0.922, `VRES0606`
  0.125/0.903, `VRES0611` 0.079/0.889, `VRES0565` 0.144/0.863). Not scale compression — opposite sides
  of *both* models' Youden points. All four `unseen`, so neither is reciting a label, and SL3010 is the
  only top-3 sublineage with **no cohort per-SL AUROC** to arbitrate.
- **O2 — it generalises beyond SL3010, and there the labels favour Bacformer.** Every genome in
  Bacformer's SL258 bottom ten carries a unitig probability 0.25–0.68, straddling the unitig threshold
  0.527; 7 of those 10 are outright disagreements and **all 10 are truly faeces** — so within the
  best-characterised lineage (n=49, cohort per-SL 0.858) the unitig model is close to uninformative.
  **But SL307 behaves well** (18/20 agree, bottom ten unanimous and all truly faeces), so it is not
  uniform across lineages — which constrains any explanation.
- **O3 — the collection sits low in Bacformer's distribution.** Median pooled probability **0.194** vs
  unitig 0.542 and cohort holdout prevalence 0.520; Bacformer calls 262/671 invasive at its Youden
  point, unitig 344. Genuine property of a carriage-heavy collection, or cohort→collection
  distribution shift — open.
- **O4 — the disagreement is asymmetric ~3:1.** Of 166 disagreements, 124 are unitig-invasive /
  Bacformer-faeces against 42 the reverse. Consistent with O3 and with the compressed unitig log-odds
  (sd-ratio 0.392), but consistency is a description, not an explanation.

**Queued models — not started.** Hypothesis under Q1/Q2: *the unitig model's deficit is a
lineage-representation deficit*, because GWAS significance filtering keeps phenotype-associated unitigs
and discards the lineage-defining ones. Reference points on the same holdout: sublineage alone 0.6032,
country alone 0.6326, country+sublineage 0.6940.

1. **Q1 — unitig + one-hot sublineage. RAN 2026-08-19 (job `33943231`, 12 min): NO LIFT.**
   Evaluate AUROC **0.76570** with 1,345 one-hot sublineage columns stacked onto the 19,622 unitig
   columns, against **0.76547** plain — **+0.0002**. Same `C`=0.01 chosen by the sweep; paired delta
   vs Bacformer essentially unmoved (+0.0208 [+0.0039, +0.0384] vs +0.0210 [+0.0041, +0.0385]).
   *Source:* `…/sampled_country_2_1_all_trainval/gwas_unitig_lmm/presence_model_sublineage/unitig_model_results.json`;
   split identical to the plain model (9,521 / 1,366 / 2,715).
   - **The two models are the same model.** On the 2,715-genome holdout their scores correlate at
     Pearson **r=1.0000** (Spearman 0.9999), mean |difference| **0.0012**, max 0.0148.
   - **Within lineage, nothing moved either**: SL258 (n=458) 0.8453 → 0.8454; mean delta over the ten
     sublineages with n≥40 and both classes **+0.0004**, 6/10 nominally up — a coin flip.
   - ⛔ **This design could not have answered O2, and that is a lesson not a result.** A sublineage
     **main effect** adds a constant to every logit in that lineage, which cannot reorder genomes
     *within* it — within-SL AUROC is invariant to it except through the indirect effect of refitting
     the unitig coefficients. Testing whether within-lineage failure is a representation deficit needs
     lineage **interactions** (per-lineage unitig weights) or a within-lineage-stratified fit. That is
     a materially bigger job, and it is the honest cost of answering O1/O2.
   - **It measured a floor** — L2 penalised the sublineage columns too, so the lift is a lower bound.
     Against "the penalty simply hid it": the sweep was free to choose a weaker penalty to exploit
     lineage and did not, picking the same C=0.01, with every higher C strictly worse on validate.
   - **Reusable:** `--with-sublineage` on `unitig_presence_model fit`, `WITH_SUBLINEAGE=1` on
     `run_unitig_presence_model.sh` (defaults to a separate out-dir and refuses to overwrite the
     plain model). ±country was **not** run.

   *(Superseded scoping note, kept because it was wrong in a useful direction: this was DEFERRED
   2026-08-18 as complex, on a claim that `fit` needed a general `--covariates` framework. It did
   not — it was ~20 lines, and it ran in 12 minutes.)*

   **Original deferral note —** (David: complex, and mostly
   supplementary proof of Bacformer's power). **Deferred with a correction attached:** an earlier note
   here called it complex and said `fit` needs a general `--covariates` framework. **It does not.** The
   cheap path is a `--with-sublineage` flag, ~20 lines plus a test:
   `align_to_split` **already returns `Sublineage`** (it appends the column when the split CSV has it,
   and the split CSV does); the design is sparse CSR so the block goes on with one
   `sp.hstack([X, sl_onehot]).tocsr()`; both blocks are 0/1 so there is no scaling mismatch; and the
   `C` sweep, scoring path and npz writer are all column-agnostic — only `save_model` needs the
   extended name list, which it already takes as an argument. **Do not re-scope this as a project.**
   - **What the cheap version gives up:** SL columns take the same L2 penalty as the unitig columns.
     That is the *conservative* direction (the penalty shrinks SL effects), so it answers "does lineage
     close the gap?" fine. It is the wrong estimator only for a precise **decomposition** claim — "how
     much do unitigs add *beyond* lineage" — which needs SL as unpenalised fixed effects and a custom
     solver. That, not the plumbing, is where the complexity lives.
   - Re-run the `C` sweep if it is ever built (`C=0.01` was tuned on a 33k-feature space). Bacformer has
     no explicit country feature, so **unitig+SL without country is the like-for-like comparator**.
   - **If it is revived, the reason will be O2**, not the headline: it is the direct test of whether the
     unitig model's within-lineage failure in SL258 is a lineage-representation deficit.
2. **Q1b — Bacformer + country + sublineage.** The symmetric experiment, and what makes the 0.731 row
   interpretable: nobody has given Bacformer those two variables, so 0.731 is a **floor** for a combined
   model, not a ceiling Bacformer is straining against. Shares Q1's estimator question.
3. **Q2 — relaxed unitig inclusion. Blocked on a decision.** The GWAS tested **6,280,612** unitigs and
   kept 33,039 (0.53%, Bonferroni 8.23e-09). ⚠ **p-value sweep vs a single all-unitigs fit is an open
   decision for David + pp** — nothing queued until settled. Feasibility established: `sample_nonsig_unitigs.py`
   already streams the full 603 MB `.assoc` selecting by p-value with af-matching, and presence comes
   from one streaming pass over the 82 GB `unitigs.pyseer.gz`. At 6.28M features p/n ≈ 460, so a null
   result would be ambiguous between "no signal" and "wrong regularisation" — and if the solver will not
   hold the design, that is a **scale problem to solve for the chosen estimator**, not grounds to
   substitute another. Free by-product: no selection step means **no selection leakage**, so the
   `trainval_only` / `full_cohort` distinction dissolves for that arm.

**MKP103 / KPNIH1 near-isogenic set (2026-08-14).** Seven ST258/SL258 genomes found by sweeping every
identity column of `metadata_v2` (an earlier narrower search found only two — quote seven). None is in
any cohort or in the 677, so all are out-of-sample. Artifact:
`processed/train_iso_source/lab_collection/mkp103_kpnih1_natural_experiment.csv`.

- **Numbers of record** (pooled P(blood) / all_samples): MKP103 W/t `SAMN22863586` **0.970** / 0.996 ·
  KPNIH1 parent `SAMN07312724` 0.849 / 0.927 · ramR CHD `SAMN22863596` 0.446 / 0.996 · CHD
  `SAMN22863587` 0.354 / 0.996 · ramA CHD `SAMN22863595` 0.258 / 0.996 · smvA CHD `SAMN22863597`
  0.207 / 0.996 · ex-mouse `SAMN17524437` 0.099 / 0.040.
- **Bears on the §3.2 model choice.** `all_samples` returns **0.9959–0.9960 for all five**
  chlorhexidine-study genomes — spread 1e-4, i.e. no within-lineage discrimination, which is the job.
  Pooled spreads the same genomes 0.21–0.97. Independent of AUROC; points the same way as clean
  interpretation. One series, and the four CHD derivatives are likely sibling clones, not replicates.
- **KPC-3 deletion test (AMR model).** MKP103 is KPNIH1 with KPC-3 deleted — confirmed from our own
  Kleborate calls (`Bla_Carb_acquired` KPC-3 vs `-`, SHV-11 and both Omp mutations identical). The AMR
  model predicts meropenem/imipenem/ertapenem **1.0000 for all seven**, KPC+ and KPC− alike: on this
  pair it is not reading the carbapenemase. Reading the ST258 background or the shared OmpK35/36
  defects is a **hypothesis**; porin loss alone can raise ertapenem MICs, and there are no measured
  MICs for any of the seven.
- **Do not repeat the tidy colistin story.** W/t → CHD is 0.023 → 0.924, but ramR CHD 0.094 and smvA
  CHD 0.211 go the other way; the one-pair result does not survive the series.
- Which physical stock a collaborator holds **cannot** be resolved from databases — only by sequencing
  the working isolate.

**Owns.** `src/amr_over_time/`.

---

### 3.5 Clustering / homology

**Status.** `gene_array_lasso` migrated from groupyr to skglm; Phase-1 build order A–D. The
`syntology` synteny-map work is a **sibling repo**, not this one.

**Next.** Phase-1 remainder.

**Caveats.** groupyr's prox built a dense O(n_groups²) mask — 367 GB at 5% prevalence, 4.4 TB at 1% —
which is why it was abandoned. The >5% prevalence filter is **scaffolding only** and biologically
wrong long-term: target AMR genes are individually rare but collectively penetrant.

**Owns.** `src/gene_array_lasso/`.

---

## 4. Artifact dependency table

Every shared artifact, once. **This is the table whose absence caused the drift this file exists to
correct.**

| Artifact | Produced by | Consumed by | Regenerating it invalidates |
|---|---|---|---|
| `<drug>_split.csv` | `engine/splits` | ladder, `ast_gwas`, ceilings, per-gene LR | **Everything** — it defines the holdout |
| FT `results.json` | `engine/finetune/finetune_amr` | every quoted FT number | Every FT AUROC in §3.1 |
| `eval_scores.npz` | `engine/finetune/evaluate` | paired CI in `collect_comparison` | The CI only |
| per-drug ceiling CSVs (`card_ceiling/…`, `tbprofiler_gene_lr_…`) | `apps/kleb/card_determinant_lr` (CARD) · the retired TB probe (WHO) | `catalogue_ceiling_panel.csv` | The ceiling column |
| `catalogue_ceiling_panel.csv` | `engine/ref_catalogues/build_ceiling_panel` | **`ast_gwas/collect_comparison` only** | The comparison table's ceiling column |
| `visualisations/kp/<drug>/card_determinant_lr_<drug>_family.csv` | ⚠ a **k-fold-probe-vintage mirror**, producer unclear | **the concat ladder's RED ceiling rung** (`build_amr_ladder._catalogue_csv`) | The ladder's ceiling line |
| `unitigs.pyseer.gz` | `ast_gwas/build_cohort_once` | every Kp drug | All Kp unitig GWAS |
| `lineage_clusters.tsv` | `ast_gwas/sublineage_from_metadata` | `--lineage`, permutation null | Calibration only |
| `mge_hits.parquet/` | `kleb_iso_source/map_unitig_hits_genomad` | the MGE/IS/IGR write-ups | §3.3's invasion mapping |

**`catalogue_ceiling_panel.csv` is reproducible.** It was hand-assembled once, which was itself the
defect this file exists to prevent; `engine/ref_catalogues/build_ceiling_panel` now regenerates it
byte-for-byte from the per-drug CSVs, and the checked-in panels are that command's output (verified:
re-running is byte-identical, and the values differ from the hand-made ones by at most one float64
ULP). It **requires** `--estimator` to be declared and refuses a `deployment_holdout` claim over data
carrying a non-zero spread — so the TB ceiling cannot be relabelled `current` to make a comparison
look like-for-like:

```
python -m bacpredict.engine.ref_catalogues.build_ceiling_panel \
  --ceiling-dir <…>/train_kleb_ast/card_ceiling --catalogue card --grain allele \
  --estimator deployment_holdout --status current \
  --out-csv src/bacpredict/visualisations/kp/catalogue_ceiling_panel.csv
```

**Split-table ↔ deployed-holdout equivalence: verified for all 32 drugs, 0 mismatches.** The weak
form is `n_samples` == the split-table holdout row count. For ertapenem and colistin — the two the
§3.3 comparison rests on — **set membership was verified identical**, which is the form that actually
licenses a paired test. Count equality is necessary, not sufficient: extend the membership check to
each drug as its `eval_scores.npz` lands in the fan-out.

---

## 5. Shared infrastructure

**Cluster.** Both CSD3 (UoHPC) and Isambard are available; **the user says which each session.**
Recent AMR and GWAS work is on **CSD3**. Per-cluster detail is in `~/.claude/cluster_uohpc.md` and
`~/.claude/cluster_isambard.md` — do not duplicate it here.

**Data roots.** Code resolves the root through `bacpredict.engine.config.resolve_data_root()`
(`--data-root` → `$BACPREDICT_DATA_ROOT` → `$SCRATCHDIR` → CSD3 path → error).

| Cluster | Root |
|---|---|
| CSD3 | `/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david` (referred to as `$R`) |
| Isambard | `$SCRATCHDIR` = `/scratch/u6fp/dca36.u6fp` |

**Cohorts — never write a bare count.** Three different numbers exist per organism and they mean
different things:

| Organism | Canonical AST cohort | Deprecated (May) | Embedding superset |
|---|--:|--:|--:|
| Kp | **7,088** | 6,838 | 9,724 |
| TB | **36,692** | 36,684 | 38,257 |

The canonical cohort lives under `$R/bac_ast_prediction/`. **`$R/processed/` is the deprecated May
tree** — the TB ceiling was built there, which is one of its four defects.

**Assembly resolution on CSD3.** TB resolves unchanged. **Kp does not** — there is no flat BioSample
directory and `raw/assemblies` is GCA-keyed whole-*Klebsiella*. Use
`raw/assemblies_file_list.tsv` (`Sample<TAB>path`, already the right format) → 7,080 of 7,088.

**Tests.** `uv run pytest tests/` is the gate — run it rather than trusting a number here. (A count was quoted twice in this file and was wrong both times; it moves with every commit, which makes it exactly the kind of fact that does not belong in a state document.) `tests/docs/` additionally enforces the conventions in §0.

---

## 6. Decisions of record

Dated, and harvested from the agent-memory snapshot in
[`docs/_archive/memory_snapshot_2026-08-14/`](docs/_archive/memory_snapshot_2026-08-14/) before that
directory was consolidated. Several of these existed nowhere else.

### Data-safety guards — losing these causes destructive mistakes

| Date | Decision |
|---|---|
| 2026-08-13 | **`min_size` stays 100** for the invasion lineage clusters. The `min_size=50` variant (17 clusters; moved `other` from 45% to 38%) was measured and **dropped as not worth the complexity**. `lineage_clusters_min50.*` was deleted — **do not regenerate it**. |
| 2026-07-24 | **git-rm scope for contaminated ladder tables.** Untracked auto-generated `visualisations/**/*_amr_ladder_table.csv` **regenerate freely**. The tracked curated publication tables — Kp `azithromycin_card_ladder_table_family.csv`, `azithromycin_card_ladder_family.png`, the `card_esm_vs_ft_*`/`esm_vs_ft_*` chain; TB `rif_ladder_table.csv`/`.md`, `rifampicin_ladder_barplot.png` — are hand-assembled from other code paths. **Confirm the scope with David before deleting any of them.** The label-blind rungs are clean — keep. |
| 2026-08-13 | **Keep `mash_lineage_clusters.tsv`** for the methods comparison. |

### Statistical method

| Date | Decision |
|---|---|
| 2026-07-19 | **Do not net out lineage.** The exercise is **raw AUROC recovery**. |
| 2026-07-19 | **Non-coding rung = core region** (prevalence ≥ ~0.9), carrier / zero-imputed embedding, **not** presence. Select the best across `upstream ∪ per_unit`; parcel = `baclm_reembed`. The selection band must include core up to 1.0, plus an `n_pos` floor (~20–50) to kill low-n CRISPR artifacts. |
| 2026-07-21 | **The gene rung was mathematically wrong, not merely suboptimal** — selected by carrier-only AUROC but zero-imputed into a *linear* head, so a low-penetrance gene becomes a mostly-zero block the model reads as "no contribution", and concat can *degrade* against FT-mean. Fixed by hard-failing without an imputed ranking. |
| 2026-07-21 | **Screens run imputed AND non-imputed**, in separate plots, with a carrier-vs-imputed density comparison. Not one or the other. |
| 2026-07-16 | **The plain LR fits on train+validate** (it needs no early-stop set) and tests on the untouched **evaluate** split — for both coding and non-coding, so the figure carries one consistent metric. |
| 2026-08-05 | **Keep the carrier-only catalogue comparison; relabel it as a within-carrier question.** The panel had compared an all-sample presence one-hot against a carrier-only embedding LR, so penetrant HGT genes went single-class and showed a blank bar — an artifact, not model failure. |
| 2026-07 | **CARD is the default Kp ceiling; Kleborate is a retained comparator.** This reversed the earlier "Kleborate alone" decision. |
| 2026-08-13 | **Youden on the holdout, one convention for both arms** — with the optimistic-bias caveat stated wherever sens/spec/balanced accuracy appear. |
| 2026-08-13 | **Exclude the `other` cluster from the permutation null** (drop those samples rather than un-shuffling them). The paired real run must be scored on the same subset. |
| 2026-06-26 | **`gene_array_lasso`: groupyr → skglm**, on a user-authored plan. Absence encoding = **zero embedding block** (sparse CSR, never stored); do **not** mean-impute. **Grouping must be a swappable input** — never hard-wire Panaroo into the estimator. Memory levers ranked: CSR sparse → float32 (validate) → warm-started decreasing-alpha path → regrouping → out-of-core last. |
| — | **Kept as a warning:** an earlier unilateral swap to *celer* silently changed the estimator to squared-loss group lasso. That was **wrong and was reverted.** A package change *is* a change of statistical method. |

### Embedding and representation

| Date | Decision |
|---|---|
| 2026-07-16 | **Embed regulatory regions both ways** — fragments at bakta-call granularity **and** whole_igr. It is a comparison, not a choice. **Pooling stays mean**, not max or concat. |
| 2026-07-16 | **Name a regulatory region by the gene it sits 5′ of** (`upstream:<gene>`), not by the flanking pair — keeps regions next to hypotheticals and names them as the catalogues do. |
| 2026-07-09 | **Re-embed all non-coding rather than salvaging the current IGR build.** RNA regions embedded **separately**; all other regions as blocks. Un-truncate long regions via overlapping windows and pool; emit a named-RNA index so `rrs`/`rrl` are locatable by name; keep `protein_embeddings`; do **not** re-run ESM or Bacformer. |
| 2026-07 | **The fragment channel needs its own keying scheme and reader — it is not a relabel.** `fragment_*` regions are IGRs split by adjacent RNA or truncation, so they are not cleanly CDS-flanked and the flank-pair namer drops them. Deferred; whole-IGR parity delivered instead. |
| 2026-07-15 | **IGR identity = the ordered 5′→3′ flanking-gene pair.** Best gene / best IGR = single top by own-LR AUROC. |

### Scope and sequencing

| Date | Decision |
|---|---|
| 2026-07-21 | **GBDT accessory concat is deferred to its own plan** — its own subpackage (not `engine/concat`) and its own output folder. |
| 2026-07-30 | **Cluster migration: keep the laptop bridge** (no Globus). **Prove the headline first**, in parallel with the transfer. **Move baclm-coding too.** The frozen Bacformer store is **not** tape-only, so transferring it directly saves ~50 GPU-h. |
| 2026-08-11 | **Do not use BacLM's SDPA fallback — too slow.** flash-attn can never load on CSD3 (glibc 2.28 against wheels built for 2.32). Port to Isambard, embed there, bring the store back. A stop-gap, **demoted to the last job**. |
| 2026-07-24 | **Target architecture: a clean-core `segment_amr_lr`** — one segment-type parameter, routed through the single `holdout.py` resolver and one fit/score primitive, then delete the duplicates. **Not a greenfield rewrite.** The leaks recurred because of accreted merge/split code and overloaded "gene" vocabulary. |
| 2026-07-21 | **Short CPU array jobs take a tight `--time`**, not the "be generous" default. |

### Invasion / lab collection

| Date | Decision |
|---|---|
| 2026-08-13 | **All 251 labelled lab genomes stay out of training** (207 forced to validate, 44 already in the frozen test set). They were deliberately *not* put in evaluate: the pooled model trained on 154 of them, which would break the pooled-vs-new comparison. Quote the 44 for a fully held-out read. |
| 2026-08-13 | **Model-choice rule.** If the `all_samples_2` advantage collapses, use the country-controlled pooled model for clean interpretation; if `all_samples_2` is still somewhat better, use that. **Asymmetry that must survive: a collapse is clean evidence, but a win is *not* conclusive** — the bigger training pool may hold near-identical outbreak siblings of the frozen test genomes, so a win still needs a **near-duplicate audit** to separate "more data helps" from clonal leakage. |
| 2026-08-18 | **RESOLVED — the country-controlled pooled model is the model of record.** `all_samples_2` scored **0.765** against pooled **0.786** on the identical frozen 2,822-genome holdout, with ~67% more training data (§3.2). The rule's collapse branch fired, so no near-duplicate audit was needed. **David's conclusion, and the framing to use: the best learning for genomes from unlabelled countries is generalist.** Deliberately kept simple — do not elaborate it into a mechanism, and do not reopen it without new evidence. |
| 2026-08-18 | **No AUROC is reported for the lab collection.** Only 44 labelled genomes are fully held out, with 6 positives; the interval runs from near-chance to near-perfect and cannot separate a good model from a mediocre one. The 0.719 and the inflated 0.903 are both **withdrawn from reporting** — quote cohort-holdout figures (n=2,822) instead. The lab collection's role is application, not validation. |
| 2026-08-18 | **Compare models by what they were allowed to see, not by which number is larger.** The `country+sublineage+k_locus+virulence+amr` baseline is **not** a Kleborate comparator — ~1,192 of its 1,360 features are country and sublineage one-hots, and country is not in the genome at all. Against annotation on equal terms Bacformer is 0.786 vs **0.640**, not vs 0.731. Any linear stack carrying non-genomic metadata must be labelled as such and set apart from genome-only models. |

### Framing rules

| Date | Rule |
|---|---|
| 2026-07-21 | **Never call a non-catalogue high-penetrance AMR gene a "lineage correlate."** Causal-vs-lineage is an open hypothesis, not a conclusion. Flagged repeatedly as a mistaken belief that kept returning. |
| 2026-08-01 | **The ladder deliverable is descriptive** — which genes and weights, comparing CARD → LR → ladder. It is **not** a causal-vs-phylogeny verdict on the model's worth. |
| — | **Frame GWAS results as invasion** — orient to the invasion allele (`invasive_af`, `\|β\|`) and lead with direction-free `var_explained`. Do not read by reference-allele β sign. |
| — | **Gather the data, then stop.** Present the statistics, offer hypotheses, and ask for direction. Do not leap to "the model is unusable" or "this is real effect X". |

### Open questions — not decisions

- **Head vs mean, across all drugs.** One point measured (rifampin: 0.958 vs 0.964); a full column and
  scatter were wanted.
- **Does the chromosomal-blindness hypothesis survive?** It was motivated by a TB rifampicin gap that
  the re-runs have largely closed, while the chromosomal / rRNA / promoter drugs remain weakest in
  both organisms. For discussion — see the banner in `docs/Bacformer_FT_DEFICITS.md`.
- **RNA blocks nuance, never confirmed at launch:** do the non-coding *blocks* exclude RNA, or include
  it as the code currently emits?

---

## 7. Retired documents

| Document | Where it went | Why |
|---|---|---|
| `ToDo.md` | [`docs/_retired/`](docs/_retired/) | Superseded by this file. Its Task 2 claimed Kp was "not formally evaluated" when 22 checkpoints existed, and it had no entry at all for the work at HEAD |
| `~/.claude/PROGRAM_PLAN_2026-05-30.md` | Kept as a dated historical record | Superseded |
| `visualisations/{kp,tb}/amr_summary_panel.csv` | [`visualisations/_superseded/`](src/bacpredict/visualisations/_superseded/) | The physical origin of the wrong FT numbers |
| Task 6 `predictHGT`, Task 4 admixture | [`docs/_parked/`](docs/_parked/) | Never started; milestones preserved |

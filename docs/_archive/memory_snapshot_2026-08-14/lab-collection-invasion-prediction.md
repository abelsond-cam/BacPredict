---
name: lab-collection-invasion-prediction
description: Lab-collection (677 genomes) invasion prediction deliverable + the pooled-vs-all_samples model choice that is being settled by the all_samples_2 retrain
metadata: 
  node_type: memory
  type: project
  originSessionId: 65fba51f-bd25-41d2-813a-24d9abb4255d
  modified: 2026-08-14T09:15:47.051Z
---

★ The **lab-collection deliverable**: predict P(blood) for a collaborator's 677 physical isolates,
rank them globally and within sublineage, so strains can be picked for **Galleria/animal testing**.
Spreadsheet `src/kleb_iso_source/docs/Kleborate_labcollection.xlsx` (tabs `Files` + `Kleborate`),
copied to `<data>/processed/train_iso_source/lab_collection/`. Plan:
`~/.claude/plans/invasion-prediction-we-have-cheerful-cat.md`.

## Facts that shaped it

- **673/677 already had ESM embeddings** in `processed/klebsiella_esm_embeddings` — the expected
  embedding pipeline was unnecessary.
- **Sublineage is NOT in the spreadsheet and cannot be derived from ST.** It comes from Pasteur
  LIN-typing via metadata_v2 (656/677 covered). No ST→SL map exists in-repo.
- **220 of 677 are in the pooled cohort** (154 train / 22 val / 44 eval); **251 in all_samples**
  (169/37/45). Manifest carries `pooled_split`, `all_samples_split`, `true_label`.
- Join the two xlsx tabs on **(sample_accession, strain)** — accession alone fans 680 → 684 because
  3 accessions appear twice, and can pair a genome with its sibling assembly's Kleborate calls.

## ⚠ OPEN: which model is "Bacformer" — the all_samples_2 retrain (job `33612953`)

`all_samples` beats pooled within **every** major sublineage on its OWN holdout (SL258 0.897 vs
0.865, SL147 0.804 vs 0.725, SL307 0.889 vs 0.819, SL15 0.893 vs 0.843, SL17 0.839 vs 0.810) — but
**ties exactly on genomes held out by both** (Δ −0.004, CI [−0.045,+0.036] at n=552;
Δ −0.004, CI [−0.031,+0.023] at n=1,203, per-SL signs mixed).

**David's hypothesis: the model has learnt country of origin, and country is a strong proxy for
isolation source.** all_samples holdouts carry systematically higher within-lineage blood prevalence.

**His chosen test — LIVE job `33615516`** (first submission `33612953` cancelled): freeze the
country-controlled `evaluate` split (n=2,822) as the test set, retrain on every other labelled
genome → cohort **`all_samples_2`**. Final config: **train 16,552 / validate 2,046 / evaluate 2,822**,
**bf16** (David: equivalent to fp32 and cheaper — measured Δ is −0.003, so read any loss under ~0.01
as precision noise), `OUTPUT_SUBDIR=models`, `EVAL_STEPS=1000`, patience 12.
**All 251 labelled lab genomes are kept OUT of train** (`--keep-out-of-train-csv <lab manifest>`):
207 forced to validate, 44 already in the frozen test set. Validate drives early stopping so it is
only *relatively* clean — quote the 44 for a fully held-out read. They were NOT put in evaluate on
purpose: the pooled model trained on 154 of them, which would break the pooled-vs-new comparison.
He predicts the advantage collapses.
**Decision rule he gave:** if it collapses, use the **country-controlled (pooled)** model for clean
interpretation; if `all_samples_2` is still somewhat better, use that.

**Asymmetry to remember:** a collapse is clean evidence. A *win* is NOT conclusive — the bigger
training pool may hold near-identical outbreak siblings of the frozen test genomes, so a win still
needs a near-duplicate audit to separate "more data helps" from clonal leakage.

**The choice matters a lot:** on the 673 lab genomes the two models agree only ρ=0.68, **top-20
overlap 2/20**, SL258 ρ=0.53 with top-5 overlap 2/5. Different tubes go in the animal model.

## Results so far (interim, ranked on pooled)

- `lab_collection_invasion_predictions.csv` — 677 ranked, with unitig + 4 Kleborate comparator
  columns, `split_provenance`, `true_label`.
- **Backup AUROC on the 251 labelled lab genomes is weak once split honestly**: all-labelled 0.903
  but that is inflated by 154 fitted-on genomes; **evaluate-only 0.719, CI [0.494, 0.906], n=44 with
  only 6 positives** — the CI includes chance. Do not quote 0.90.
- **SL258 (n=49) spreads 0.013–0.997** (SD 0.32, IQR 0.49) — ideal for picking extremes. Bottom:
  ME140475, ME120029, ME140075. Top: KP36, KP72, KPJNU-67.
- Bacformer-vs-unitig agreement on the 2,715 holdout: **r²=0.416, ρ=0.645**; unitig log-odds are
  **0.46× as wide** (slope 0.29) — a confidence-scale difference that does NOT affect r². Related
  but not redundant. Figures in `src/kleb_iso_source/docs/`.

## Reusable machinery built

`kleb_iso_source.build_lab_collection_manifest` · `.predict_lab_collection`
(bacformer/kleborate/assemble) · `.compare_cohort_models` · `.build_all_samples_2_split` ·
`bac_pyseer.kleb_iso_source.unitig_presence_from_assemblies` (Aho-Corasick presence from raw
assemblies) · `unitig_presence_model` now **persists coefficients + intercept + a unitig-order
SHA-256** and has `--score-all-splits` and a `predict` subcommand ·
`bacpredict.engine.finetune.linear_baselines.fit_and_predict_new` ·
`engine.plots.plot_model_agreement` · `engine.plots.plot_lab_sublineage_spread`.

**Gates that passed:** unitig refit reproduced **0.7655** exactly; assembly scanner vs the GWAS
matrix **294,330 cells, 100.0000% agreement, 0 discrepancies**; lab Bacformer probs match
cohort_scores at **1.1e-16**; all_samples cohort scoring reproduced its **0.8267**.

## MKP103 (asked 2026-08-13) — NOT in the lab spreadsheet; found in metadata_v2

**`SAMN22863586` = MKP103**, **`SAMN22863587` = "MKP103 CHD"**; both SL258/ST258, both have ESM
embeddings, both absent from *every* cohort → all predictions are genuinely out-of-sample.

**Polarity (asked, verified 2026-08-13): `blood_vs_faeces_label` 1 = BLOOD, 0 = faeces**, so every
probability in this file is P(blood) = P(invasive); higher = more invasive. Chain:
`prepare_…isolation_source.py:75` `(isolation == resolved_1)`, `resolved_1` = token1 = `blood`
(all scripts pass `--isolation-sources blood faeces`, order preserved by
`isolation_source_cli_parsing.py:87`); it is logged at build time as
`Filtering to 'blood' (label=1) and 'faeces' (label=0)`; `predict_proba` returns class-1 prob.

| model | MKP103 | MKP103 CHD |
|---|--:|--:|
| pooled (country-controlled) | **0.970** (blood) | **0.354** (leans faeces) |
| all_samples | 0.996 | 0.996 |
| unitig (trainval_only) | 0.401 | 0.506 |

**★ They are a DESIGNED PAIR, not two assemblies of one isolate** (metadata checked 2026-08-14).
Both from **PRJNA777533 / SRP344357, UKHSA Porton Down — "Analysis of Klebsiella pneumoniae strains
after exposure to chlorhexidine"**. `SAMN22863587` description: *"strain adapted to the biocide
Chlorhexidine"* → **CHD = chlorhexidine**, a lab-evolved derivative of the MKP103 wild-type.
Separate BioSamples/runs (SRR16761273 vs SRR16761272), 226 vs 179 contigs, 5,654,569 vs 5,655,703 bp.
**Neither is in any cohort because `host_category` AND `isolation_source_category` are both null** —
"Laboratory Derived" sits in `host` for the W/t and in `isolation_source` for the CHD. **No measured
`EBI_*` AST for either** (0/22).

⚠ **Earlier framing in this file was wrong and is corrected:** the near-identity measurements are
real (hit-unitig Jaccard **0.9955**, mean-pooled ESM cosine **0.999992**, both 5,367 proteins) but
"identical input, so movement = instability" is the wrong read — an adaptive-mutation pair is exactly
what this experiment produced, and a model *should* be able to move on it.
**Do not compare per-protein embeddings positionally across assemblies — gene order differs, so a low
median cosine there is an artifact.**

**★ Unplanned positive control — our AMR model recovers the chlorhexidine→colistin cross-resistance.**
Of all 22 drugs, **exactly one call flips**: colistin **0.023 (S) → 0.924 (R)**; gentamicin moves
0.404 → 0.004 but stays S; the other 20 calls are unchanged. Chlorhexidine adaptation conferring
colistin cross-resistance in *Klebsiella* is published, from this same Porton Down group (Wand et al.
~2017 AAC — confirm citation). The model never saw these genomes and there is no measured AST for them.

*Invasion swing — HYPOTHESES ONLY, awaiting David:* pooled 0.970 → 0.354, all_samples pinned 0.996
for both. Chlorhexidine adaptation touches *smvA* efflux and *phoPQ*/*pmrAB* (same LPS-modification
axis as colistin, plausibly serum-survival relevant), so a real shift is conceivable — as is
coincidence. Suggestive that pooled responds to within-lineage change while all_samples does not, but
**n=1 pair = anecdote, not evidence**; do not let it drive the pooled-vs-all_samples_2 model choice.

→ [[invasion-comparators-2026-08]] · [[invasion-live-run-state]]

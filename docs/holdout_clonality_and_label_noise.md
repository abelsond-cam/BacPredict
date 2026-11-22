# What the *P. aeruginosa* AMR holdout actually measures

Two properties of the held-out set change how its AUROCs should be read: the genomes in it are
clonally near-identical to training genomes, and half the labels sit where a single doubling dilution
would flip them. Both are measured here. Neither invalidates the model — it beats a memorisation
baseline on both holdouts, and it beats it by more where memorisation is impossible — but both set a
ceiling on what the headline numbers can mean.

> State, run status and current numbers of record live in [`PROJECT_STATE.md`](../PROJECT_STATE.md) §3.6.
> This document holds the argument and the evidence; where the two disagree, `PROJECT_STATE.md` wins.

**Cohort.** 3,677 QC-passing *P. aeruginosa* genomes from the EBI AMR Portal (release 2026-07) plus the
AMR-GNN supplement, labelled against EUCAST v15 breakpoints for ten drugs. Genome-level split: train
2,290 / validate 574 / random holdout 716 / ST155 holdout 82. The ST155 holdout is a sequence type
removed from training entirely, so it is lineage-disjoint; the random holdout is not.

---

## Part 1 — The random holdout is clonally saturated

### 1.1 Half of it has a near-identical training genome

Every held-out genome was compared against all 2,290 training genomes with `skani` (798 × 2,290 =
1.83 M pairs). The table is each genome's **maximum** ANI to any training genome.

| | random holdout (n = 716) | ST155 holdout (n = 82) |
|---|--:|--:|
| min / p25 / median / p75 / max | 99.13 / 99.94 / **99.98** / 99.99 / 100.00 | 99.58 / 99.77 / **99.86** / 99.97 / 99.99 |
| ≥ 99.99 | **351 (49.0 %)** | 13 (15.9 %) |
| ≥ 99.95 | 520 (72.6 %) | 25 (30.5 %) |
| ≥ 99.90 | 585 (81.7 %) | 30 (36.6 %) |
| ≥ 99.50 | 682 (95.3 %) | 82 (100 %) |

⚠ `skani` prints two decimals and scores a genome **against itself at 99.99**, so the ≥ 99.99 row is the
resolution limit, not a measurement: within it a clone and the same genome are indistinguishable.

⚠ The low end of this range is not interpretable as relatedness. Core ANI between *unrelated*
*P. aeruginosa* strains is already ~99.0–99.5, which is why 95.3 % of the random holdout clears 99.50 and
100 % clears 99.00. Only the top 0.1 % of the scale separates clonal from merely conspecific.

### 1.2 A lookup baseline recovers most of the random holdout's AUROC

For each held-out genome, take its highest-ANI **labelled** training genome and copy that genome's label
(all genomes tied at that ANI vote, so the score is continuous). No model is involved. Metrics come from
the same function that wrote the models' `results.json`, so the two are computed identically.

| Drug | model, random | lookup, random | model, ST155 | lookup, ST155 |
|---|--:|--:|--:|--:|
| tobramycin | 0.9326 | 0.7988 | 0.9819 | 0.5985 |
| imipenem | 0.8845 | 0.6199 | 0.7649 | 0.4926 |
| levofloxacin | 0.8227 | 0.7101 | 0.6865 | 0.4167 |
| ciprofloxacin | 0.8173 | 0.6994 | 0.6615 | 0.4732 |
| meropenem | 0.8098 | 0.6760 | 0.6673 | 0.5723 |
| amikacin | 0.7900 | 0.7316 | 0.5924 | 0.5000 |
| piperacillin-tazobactam | 0.7529 | 0.6461 | 0.6796 | 0.4681 |
| ceftazidime | 0.7484 | 0.6599 | 0.6856 | 0.5155 |
| aztreonam | 0.7456 | 0.6590 | 0.6899 | 0.5704 |
| cefepime | 0.7138 | 0.6427 | 0.6664 | 0.5491 |
| **mean** | **0.802** | **0.684** | **0.708** | **0.516** |

**Roughly 0.68 AUROC of the random holdout is obtainable with no biology at all.** On the
lineage-disjoint holdout the same lookup is at chance (0.516), because nothing clonally close remains.

### 1.3 What this does and does not license

The model beats the lookup on **both** holdouts, and by **more** where lookup fails: mean margin +0.117
on the random holdout, **+0.192** on ST155. On ST155 there is nothing clonally close to memorise and the
model still scores 0.59–0.98. That is generalisation, not recall.

So the honest reading is: the random holdout's headline numbers are **real signal plus a clonal lookup
bonus**, and the ST155 numbers are the ones that carry no such bonus. The model's own gap between the two
holdouts (mean −0.114) is *smaller* than the lookup baseline's (−0.169), which suggests its
representation is partly lineage-invariant and does not cash in the full clonal bonus.

⚠ **The lookup baseline is a lower bound on memorisation, not a ceiling.** Whole-genome ANI is dominated
by the core genome while most acquired AMR determinants are accessory, so a lookup keyed on accessory
gene content, or a k-NN in the embedding space itself, would score higher. A *low* number here would not
have licensed "the model generalises"; the high one does convict the random holdout.

---

## Part 2 — Half the labels sit where one dilution would flip them

### 2.1 The binarisation discards the Intermediate zone, by construction

The breakpoint table carries two thresholds per drug — `s_le_mg_l` (S ≤) and `r_gt_mg_l` (R >) — so
EUCAST's Intermediate zone is present in the data. **`s_le_mg_l` is read by no code in the pipeline.**
The label rule is *resistant iff MIC > r_gt*, so everything below R becomes "susceptible", and a reported
phenotype string is consulted only when a test has no MIC at all. Of 922 source-reported *intermediate*
PsA tests, 865 were silently re-derived from their MIC (578 → R, 287 → S); only the 57 without an MIC
were dropped.

Nothing records that this happened: the label-count artifact has no intermediate counter, and the
call-agreement artifact excludes every intermediate test by construction.

### 2.2 EUCAST v15 gives *P. aeruginosa* no susceptible category for 7 of the 10 drugs

`S ≤ 0.001` is EUCAST's formal device for "no susceptible-at-standard-dosing category — the wild type is
I (susceptible, increased exposure)".

| S ≤ 0.001 — no S category at all | genuine I band | no I zone |
|---|---|---|
| ciprofloxacin (R > 0.5), levofloxacin (R > 2), ceftazidime (R > 8), cefepime (R > 8), imipenem (R > 4), aztreonam (R > 16), piperacillin-tazobactam (R > 16) | meropenem S ≤ 2, R > 8 | amikacin 16/16, tobramycin 2/2 |

Verified rather than assumed: if this reading is right, the non-I-zone negatives for those seven drugs
must be genomes with no mg/L MIC at all. Checked against the label-provenance table — **true for all
seven** (e.g. ciprofloxacin's 227 holdout negatives = 200 in the I zone + 27 with no MIC).

Pooled, **7,093 of 14,038 "susceptible" test labels (50.5 %) sit in the EUCAST I zone**, and for seven
drugs the entire negative class does.

### 2.3 But the I *category* is the wrong handle — distance is the right one

For those seven drugs "I-zone negative" is a tautology for "negative", so it carries no information about
which labels are contestable: a cefepime MIC of 0.5 and one of 8 are both I and both labelled 0, but only
one is at the cut. The quantity that does carry information is

> **d = log₂(MIC / R breakpoint)**

and the band `−1 < d ≤ 1` is exactly and only the set of label states a one-dilution measurement error can
reverse. ISO 20776-2 essential agreement is ±1 dilution, so this band is derived from a property of the
measurement rather than chosen.

Decisively, **the two drugs with no I zone at all show the effect anyway** (§2.4). Whatever is happening
is binarisation of a continuous measurement at a cut, not the EUCAST I category.

### 2.4 Model error rises monotonically as the MIC approaches the breakpoint

Saved holdout predictions re-scored by band. No retraining. Error is **class-conditional** — false
positives among genomes labelled susceptible, false negatives among those labelled resistant — so
prevalence cannot drive it. Threshold is prevalence-matched per drug.

| band | susceptible: n / error | resistant: n / error |
|---|--:|--:|
| far (\|d\| > 3) | 203 / **0.108** | 321 / **0.159** |
| mid (1 < \|d\| ≤ 3) | 1,147 / 0.219 | 477 / 0.321 |
| **near (\|d\| ≤ 1)** | 449 / **0.419** | 425 / **0.454** |

Error is **3.9×** higher at the cut than far from it in the susceptible class and **2.9×** higher in the
resistant class. Near-breakpoint predictions are close to coin flips in both.

Per drug, 19 of 20 drug × class contrasts go the same way, and 11 of 20 have a Newcombe difference
interval excluding zero on their own. ⚠ Read the difference intervals, not the overlap of the per-band
intervals: two intervals can overlap substantially while their difference excludes zero.

`Spearman(p̂, d)` is **positive within every drug and class** (0.03–0.53), so the model is reading the
MIC gradient the binary label threw away — it is not approximating the label categorically.

### 2.5 The confirmatory number: 2.5× the odds of error, within the same genome

> `error ~ near + drug`, conditional on genome
> **odds ratio 2.52 (95 % CI 2.01–3.16)**, 368 informative genomes, 2,196 informative rows

Clonal similarity to the training set, lineage, assembly source and QC are all **constants within a
genome**, so conditioning on genome differences them out exactly — nothing has to be measured or
modelled. The `near` effect is identified only from drug-to-drug variation in breakpoint distance within
the same genome. It is robust to the operating point (OR 2.47 at a flat 0.5 threshold).

The error-concentration ratio — share of errors in the near band over share of genomes in it, bootstrapped
clustered on genome — is **1.535 (95 % CI 1.433–1.648)** over 646 genomes.

### 2.6 Per drug — and what actually predicts a drug's AUROC

Balanced error `(FPR + FNR) / 2` per band, which is prevalence-free and so comparable across drugs whose
%R runs 18–63. Sorted by headline AUROC.

| Drug | AUROC | % of holdout near | n near | **error near** | n far | **error far** | gap |
|---|--:|--:|--:|--:|--:|--:|--:|
| tobramycin | 0.9326 | 8.9 | 45 | 0.393 | 245 | 0.128 | +0.265 |
| imipenem | 0.8845 | 23.2 | 95 | 0.361 | 213 | 0.142 | +0.219 |
| levofloxacin | 0.8227 | 13.4 | 39 | 0.537 | 103 | 0.270 | +0.267 |
| ciprofloxacin | 0.8173 | 15.2 | 94 | 0.488 | 339 | 0.293 | +0.195 |
| meropenem | 0.8098 | 20.3 | 107 | 0.539 | 299 | 0.257 | +0.282 |
| amikacin | 0.7900 | 14.2 | 66 | 0.349 | 237 | 0.311 | +0.037 |
| piperacillin-tazobactam | 0.7529 | 25.1 | 97 | 0.447 | 152 | 0.330 | +0.117 |
| ceftazidime | 0.7484 | 22.8 | 124 | 0.482 | 311 | 0.319 | +0.162 |
| aztreonam | 0.7456 | 30.0 | 107 | 0.413 | 173 | 0.311 | +0.102 |
| cefepime | 0.7138 | **43.7** | 100 | 0.451 | 76 | 0.310 | +0.142 |

> **The intuitive version of the hypothesis is wrong, and the data says so clearly.**
> A drug's AUROC does **not** fall as its near-band error rises: Spearman ρ = **−0.08, p = 0.83**.

Near-band error is close to constant across the panel — 0.348 to 0.539, sd 0.067. **Near the cut every
drug is bad, and bad by about the same amount.** It is a floor, not a differentiator, so it cannot
explain why one drug scores 0.93 and another 0.71.

Two things do predict AUROC, and they are close to independent of each other (ρ = +0.52 between them,
p = 0.13; each survives partialling out the other):

| Predictor of a drug's AUROC | Spearman ρ | p | partial, controlling for the other |
|---|--:|--:|--:|
| **far-band** error — how good the drug is *away* from the cut | **−0.79** | 0.006 | −0.72 (p = 0.018) |
| **% of the holdout in the near band** — how much of the cohort is in the noisy zone | **−0.76** | 0.011 | −0.67 (p = 0.034) |
| near-band error | −0.08 | 0.83 | — |
| near-minus-far gap | +0.66 | 0.038 | — |

So **label noise does depress AUROC, through exposure rather than severity**: not how badly the model
does once it is at the cut, but how much of the cohort sits there. cefepime has 43.7 % of its holdout in
the near band and the lowest AUROC of the ten; tobramycin has 8.9 % and the highest. The positive sign on
the gap is the same fact seen from the other side — the *better* drugs have the *bigger* near-vs-far
gap, because they have further to fall.

⚠ n = 10 drugs, so |ρ| must reach 0.648 for p < 0.05, and the effective n is smaller: cefepime and
ceftazidime share AmpC-plus-porin biology, ciprofloxacin and levofloxacin share *gyrA*/*parC*, imipenem
and meropenem share OprD. Read this as descriptive. The genome-conditional model in §2.5 is the
confirmatory number, not this table.

### 2.7 What the holdout would score without the near band

Deleting the near-band genomes and recomputing AUROC. ⚠ This rises **mechanically** — those genomes also
sit near the model's decision boundary — so the gain is reported against two nulls: deleting the same
number at random, and deleting the same number drawn from the rest to match the removed set's predicted-
probability distribution by decile. That second null removes genomes the model found *equally hard* but
whose labels are not contestable.

| Drug | AUROC, placed genomes | n removed | without near | gain | beats both nulls |
|---|--:|--:|--:|--:|:--|
| meropenem | 0.7619 | 107 | 0.8728 | **+0.111** | yes |
| levofloxacin | 0.7217 | 39 | 0.8064 | **+0.085** | yes |
| imipenem | 0.8576 | 95 | 0.9226 | **+0.065** | yes |
| piperacillin-tazobactam | 0.6996 | 97 | 0.7568 | **+0.057** | yes |
| aztreonam | 0.6785 | 107 | 0.7347 | **+0.056** | yes |
| cefepime | 0.6512 | 100 | 0.7063 | +0.055 | no — random null p95 +0.083 |
| ciprofloxacin | 0.7327 | 94 | 0.7871 | **+0.054** | yes |
| ceftazidime | 0.6918 | 124 | 0.7422 | **+0.050** | yes |
| tobramycin | 0.8934 | 45 | 0.9141 | +0.021 | no |
| amikacin | 0.7839 | 66 | 0.7844 | +0.000 | no |

**Seven of ten drugs clear both nulls, mean gain +0.068.** That is the size of the AUROC currently being
spent on labels a single dilution could flip.

⚠ The baseline column is **not** the headline AUROC: it is computed on the genomes that have a placed
band, excluding `no_mic` and `undetermined` (34 % of genome × drug pairs), so it runs lower. Compare
within the row, never against §1.2.

The three that do not clear their nulls are informative rather than awkward. **amikacin gains nothing**
(+0.0005) — and amikacin is one of the two drugs with no Intermediate zone at all, whose near band is
purely a measurement-distance artifact. tobramycin, the other, gains only +0.021. cefepime's near band
is so large (43.7 %) that deleting 100 genomes at random moves its AUROC by +0.083 on its own, so nothing
can be concluded from deleting its near band specifically.

---

## Part 3 — The two effects do not explain each other

Clonality is the obvious confound: if near-breakpoint genomes were also clonally unusual, §2 would just be
§1 again. **It points the wrong way.**

| | ≥ 99.99 ANI to a training genome |
|---|--:|
| genomes near the breakpoint for ≥ 1 drug | **51.4 %** |
| genomes never near the breakpoint | 37.6 % |

Near-breakpoint genomes have *more* near-identical training neighbours, so clonal lookup should make them
**easier**. The model still does two to four times worse on them. And the genome-conditional model in
§2.5 removes clonality by construction regardless.

---

## Limits

- **ST155 cannot test the label-noise hypothesis.** Its 82 genomes are one clone and mutually correlated,
  and per-drug near-band cells hold 1–22 genomes; 11 of 20 contrasts are positive, which is noise. It is a
  direction check that excludes only large effects, and here it does not even do that.
- **Every ST155 AUROC has a wider interval than its *n* implies**, for the same reason. Two drugs'
  ST155 sets have a minority class of 7 (amikacin) and 12 (levofloxacin) and are close to
  uninterpretable alone.
- **The measured effect is a floor.** Genome × drug pairs whose tests disagreed were averaged to a
  fractional label and dropped before training *and* scoring, so the genomes most obviously affected by
  label noise are already absent from the holdout.
- **Not controlled: panel-edge effects.** A laboratory whose MIC panel tops out near the breakpoint
  reports censored values while a wide-panel laboratory reports exact ones, so `d` is confounded with
  platform → study → lineage. The columns exist and this is not yet stratified.
- **No per-drug significance claims.** The full table runs to roughly 840 cells, of which ~42 would clear
  a nominal 95 % threshold by chance. §2.5 is the single confirmatory number; everything per drug is
  descriptive with an interval.
- **Breakpoint-definition variability is small for PsA**, so it is not an alternative explanation:
  CABBAGE's ECOFF-based calls agree with our EUCAST v15 on 100 % of PsA tests (that divergence is
  Enterobacterales-only), and against the portal's own mostly-CLSI calls the re-derivation flips 282 of
  7,285. The variability is *inside* one definition, at the cut.

## Reproducing

| What | How |
|---|---|
| ANI of each held-out genome to training | `skani dist`, then `engine/splits/holdout_lookup_baseline.py` |
| The lookup baseline | `python -m bacpredict.engine.splits.holdout_lookup_baseline --ani-tsv … --split-dir …` |
| MIC distance bands | `python -m bacpredict.apps.pa.mic_distance_bands --tests ast_tests_called.parquet --out-dir …` |
| Error by distance, and the confirmatory model | `python -m bacpredict.apps.pa.holdout_error_by_mic_distance --models-dir … --bands-dir … --out-dir …` |

Artifacts (project tier, not scratch): `$AMR/processed/paeruginosa/ani/` and
`$AMR/processed/paeruginosa/mic_bands/`. The 599 MB pairwise ANI TSV is deliberately not kept; it
regenerates in ~10 min on 32 cores.

## Open — not decided here

Three downstream options follow from Part 2; none is taken in this document, which measures the scope of
the problem rather than choosing the fix.

1. **Exclude** the near-breakpoint genomes. §2.7 bounds what that buys on the holdout: +0.068 mean AUROC
   over the seven drugs that clear both nulls, nothing for amikacin and tobramycin. ⚠ It also *removes
   the genomes a clinician most needs a call on*, so a gain here is partly a gain from declining the hard
   cases, and the resulting number is not comparable to the current one.
2. **A multi-class head** that can say "intermediate". ⚠ For 7 of the 10 drugs EUCAST assigns no
   susceptible category at all, so a three-class head would have an empty S class for them; the classes
   would have to be defined on measurement distance rather than on the EUCAST category.
3. **Regress log₂ MIC directly** instead of the binary call, thresholding afterwards if a call is needed.
   This is the option §2.6 points at most strongly: near-band error is a near-constant floor across
   drugs, which is what a model forced to cut a continuous quantity at one threshold would produce, and
   `Spearman(p̂, d)` is already positive in every drug and class — the model is reading the gradient the
   binary label discards. The MIC is retained per test in `ast_tests_called.parquet` (value, sign, units)
   and never reaches the model today. ⚠ It changes the task, the loss and every downstream comparison:
   roughly a third of tests are censored (`≤`/`>`), which a plain regression cannot consume honestly —
   it needs interval-censored handling — and no existing result would remain comparable.

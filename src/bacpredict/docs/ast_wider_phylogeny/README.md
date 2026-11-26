# AST available above the family level, balanced to *P. aeruginosa*

How many AST labels the ten PsA drugs would have if the cross-species arm were drawn from a wider
phylogenetic level than Enterobacteriaceae — **after** every added species is thinned to PsA's own
resistance rate, as the 2026-11-25 decision of record requires.

> **Mechanics, argument and the full tables only.** Status, the decision of what to run and when,
> and what gates it live in [`PROJECT_STATE.md`](../../../../PROJECT_STATE.md) §3.8. There is no
> status block in this file, by design.

Measured 2026-11-26. Artifacts and the manifest: `$AMR/processed/ebi_amr_portal/scoping_wider_phylogeny_20261126/`
on CSD3, snapshotted here as [`labels_by_tier.csv`](labels_by_tier.csv),
[`labels_by_tier_drug.csv`](labels_by_tier_drug.csv), [`labels_by_species_drug.csv`](labels_by_species_drug.csv),
[`genomes_by_species.csv`](genomes_by_species.csv) and [`manifest.json`](manifest.json).

## What was measured

| | |
|---|---|
| Input | `cabbage_v2_20261121/phenotype.parquet`, md5 `98a46302…` — the same table §3.7 labelled |
| Scope | every species in it, no genus filter; the ten PsA drugs |
| Rule | [`engine/ast_labels`](../../engine/ast_labels/) **unchanged** — EUCAST v15 from the MIC where a breakpoint row exists, the source's own R/S where there is no MIC, censored MICs straddling R dropped, calls averaged per (genome, drug), only 0/1 kept |
| Balancing | each species thinned to PsA's %R for that drug by dropping at random from whichever class is in excess. Only the *count* is needed, so nothing is sampled: `kept = ⌊min(n_R / p, n_S / (1 − p))⌋`, and a species with no R or no S collapses to 0 |
| Target `p` | PsA's own rate, from the decision of record: MEM 43.5, IPM 64.2, TZP 52.7, AMK 21.8, CAZ 48.4, CIP 61.8, FEP 48.8, ATM 44.1, TOB 35.0, LVX 56.2 |
| Reproduce | [`count_ast_by_species_balanced_to_psa.py`](count_ast_by_species_balanced_to_psa.py), one short CPU job (`icelake-himem`, 8 cores, 30 min wall) |

**Harness check.** Restricted to the 21 genera the Enterobacterales build uses, this run gives
**493,801 clean labels** — exactly `enterobacterales_cabbage_v2/label_counts.csv`. Difference 0. The
widening numbers are therefore on the same footing as the arm now training, not a parallel count.

## The breakpoint table is the constraint, and it makes two different numbers

[`eucast_v15_breakpoints.csv`](../../engine/ast_labels/eucast_v15_breakpoints.csv) holds rows only for
Enterobacterales and *Pseudomonas*. Outside those two an MIC resolves to `no_breakpoint` and the test
is **dropped — it is not rescued by the source's own call** (145,581 of the 782,871 test rows read
here). So every table below carries two columns, and they are not interchangeable:

| variant | what it is |
|---|---|
| **today** | what the pipeline labels as it stands. Added taxa keep only bare R/S calls and disk tests. A count. |
| **ceiling** | what they would label if the breakpoint table were extended to the added taxa. The MIC-derived call does not exist until those rows do, so the source's own call stands in for it — §3.7 measured those two agreeing on 95.5–99.7 % of tests carrying both. **An estimate, not a count.** |

⚠ **Reaching the ceiling is a statistical-method decision, not a transcription job.** EUCAST has no
row at all for several of these taxon × drug pairs — *Acinetobacter* has no ceftazidime, cefepime,
aztreonam or piperacillin-tazobactam breakpoint — so those labels would have to rest on the source's
(largely CLSI) call while the rest of the cohort is re-derived against EUCAST. That is a second
estimator inside one cohort, and it has to be agreed before it is built.

## Tiers

Each tier is what that widening **adds**, relative to the Enterobacteriaceae arm now training.
Moraxellaceae sits inside Pseudomonadales, so ***Acinetobacter* enters at PsA's own order**, not at
class — which is why the order tier is the one that carries a ten-drug panel.

| tier | species | genomes | labels (today → ceiling) | **balanced (today → ceiling)** | balanced per genome |
|---|--:|--:|--:|--:|--:|
| 0 *P. aeruginosa* — reference, never thinned | 1 | 8,151 | 52,990 → 53,125 | **52,990 → 53,125** | 6.52 |
| 1 family: Enterobacteriaceae — **the arm now training** | 73 | 116,938 | 483,274 → 485,943 | **265,537 → 266,153** | 2.28 |
| 2 genus: other *Pseudomonas* | 19 | 126 | 967 | **214** | 1.70 |
| 3 **order**: rest of Enterobacterales + Moraxellaceae | 47 | 4,070 → 6,024 | 22,373 → 34,578 | **9,329 → 16,415** | 2.72 |
| 4 class: rest of Gammaproteobacteria | 18 | 239 → 499 | 288 → 1,121 | **38 → 192** | 0.38 |
| 5 phylum: rest of Pseudomonadota | 17 | 4,495 → 21,306 | 5,212 → 22,360 | **2,845 → 15,106** | 0.71 |
| 6 domain: all other bacteria | 77 | 19,546 → 46,230 | 28,908 → 74,251 | **12,686 → 29,380** | 0.64 |
| *6b — the same, excluding* M. tuberculosis | 76 | 24,828 | — | *6,883 → 12,866* | 0.52 |
| X excluded: not a bacterial taxon | 9 | 209 → 506 | 1,454 → 1,752 | *428 → 565* | — |

Chiefly `metagenome` (755 labels), `bacterium` (373), `Homo sapiens` (299) and `mixed sample`
(273). They are kept in the output under tier **X** rather than dropped, so the exclusion is
countable rather than invisible, and they are excluded from every other row here.

**The density column is the finding.** The order tier is the only widening that holds the family
arm's labels-per-genome (2.73 against 2.28). Class, phylum and domain run at 0.38–0.71: many
genomes, one or two drugs each, and most of them collapse under balancing.

## Per drug

Balanced labels, **ceiling** variant. The *today* variant is the same shape at roughly half the
size — [`labels_by_tier_drug.csv`](labels_by_tier_drug.csv) carries both.

| tier | AMK | ATM | FEP | CAZ | CIP | IPM | LVX | MEM | TZP | TOB |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **1 family (now training)** | 24,344 | 24,262 | 26,965 | 35,173 | 58,714 | 12,480 | 21,228 | 22,107 | 21,177 | 19,703 |
| 2 genus | 51 | 14 | 29 | 18 | 20 | 6 | 23 | 10 | 12 | 31 |
| **3 order** | 1,701 | 432 | 1,218 | 1,318 | 2,688 | **3,577** | 1,530 | 1,654 | 866 | 1,431 |
| 4 class | 27 | 15 | 28 | 66 | 27 | 1 | 16 | 5 | 7 | 0 |
| **5 phylum** | 4 | 2 | 16 | 69 | **14,909** | 13 | 33 | 49 | 1 | 10 |
| **6 domain** | **10,198** | 1 | 67 | 123 | 9,561 | 152 | **8,568** | 505 | 51 | 154 |
| *6 domain, excl.* M. tuberculosis | *428* | *1* | *67* | *123* | *9,350* | *152* | *2,061* | *479* | *51* | *154* |

Everything wider than the family, excluding *M. tuberculosis*, as a share of the family arm:
**ciprofloxacin +46 %**, imipenem +30 %, levofloxacin +17 %, meropenem +10 %, amikacin +9 %,
tobramycin +8 %, cefepime +5 %, ceftazidime +5 %, piperacillin-tazobactam +4 %, aztreonam +2 %;
**+17 % overall**, or **+7 %** without extending the breakpoint table.

## By species added

Every species outside Enterobacteriaceae with ≥ 20 balanced labels. `drugs` is how many of the ten it
is actually tested on. Full table, all 252 species in the seven live tiers and both variants, in
[`labels_by_species_drug.csv`](labels_by_species_drug.csv).

| tier | species | drugs | genomes | balanced today | **balanced ceiling** |
|---|---|--:|--:|--:|--:|
| **order** | ***Acinetobacter baumannii*** | **10** | 4,096 | 3,757 | **10,645** |
| order | *Providencia* sp. | 10 | 266 | 1,550 | 1,550 |
| order | *Serratia marcescens* | 10 | 399 | 1,273 | 1,275 |
| order | *Morganella morganii* | 10 | 261 | 894 | 895 |
| order | *Proteus mirabilis* | 10 | 224 | 888 | 888 |
| order | *Yersinia enterocolitica* | 10 | 288 | 269 | 269 |
| order | *Serratia* sp. | 10 | 110 | 197 | 197 |
| order | *Acinetobacter pittii* | 9 | 90 | 19 | 185 |
| order | *Providencia rettgeri* / *stuartii* | 10 | 49 | 249 | 256 |
| order | *Acinetobacter haemolyticus* | 10 | 32 | 115 | 115 |
| order | *Serratia ureilytica*, *Acinetobacter* sp. | 10 | 67 | 47 | 66 |
| class | *Aeromonas* | 9 | 47 | 0 | 90 |
| class | *Stenotrophomonas maltophilia* | 8 | 43 | 0 | 35 |
| class | *Vibrio parahaemolyticus* | 3 | 37 | 27 | 27 |
| **phylum** | ***Neisseria gonorrhoeae*** — CIP only | 3 | 20,305 | 2,743 | **14,872** |
| phylum | *Achromobacter xylosoxidans* | 6 | 56 | 0 | 106 |
| phylum | *Neisseria meningitidis* | 1 | 593 | 16 | 42 |
| phylum | *Burkholderia cenocepacia* | 4 | 34 | 28 | 28 |
| phylum | *Brucella melitensis*, *Laribacter hongkongensis* | 2–6 | 66 | 47 | 47 |
| **domain** | ***Mycobacterium tuberculosis*** — AMK, LVX, CIP | 4 | 21,402 | 5,803 | **16,514** |
| domain | ***Staphylococcus aureus*** — CIP, LVX | 9 | 7,954 | 4,781 | **6,058** |
| domain | ***Campylobacter jejuni*** — CIP | 4 | 6,744 | 992 | **3,625** |
| domain | *Campylobacter coli* — CIP | 3 | 2,239 | 261 | 1,378 |
| domain | *Streptococcus pneumoniae* | 8 | 3,940 | 97 | 576 |
| domain | *Enterococcus faecium* | 9 | 831 | 302 | 524 |
| domain | *Helicobacter pylori* — CIP | 1 | 289 | 131 | 181 |
| domain | *S. pseudintermedius*, *S. haemolyticus*, *E. faecalis* | 2–9 | 967 | 112 | 212 |
| domain | *Ornithobacterium*, *Mycobacteroides*, *Aliarcobacter*, *S. agalactiae*, *C. diphtheriae* | 2–4 | 1,259 | 143 | 198 |

**Four species carry 86 % of everything gained above the family level:** *A. baumannii*,
*N. gonorrhoeae*, *M. tuberculosis* and *S. aureus*.

## What the shape of the data says

Offered as hypotheses. None of it is a result about whether cross-species training helps — that is
what the arm now training is for.

1. **The order tier is a different kind of addition from the ones above it.** *A. baumannii* is the
   only species added anywhere that is tested on all ten drugs at scale, sits in PsA's own order, and
   shares its mechanism vocabulary — OXA carbapenemases, efflux, porin loss, aminoglycoside-modifying
   enzymes. It is also the only tier that moves imipenem, which is one of the two drugs balancing
   costs most (v2 keeps 39 %).
2. **The phylum and domain gains are one drug wide.** *N. gonorrhoeae* brings 20,305 genomes to buy
   ciprofloxacin and essentially nothing else; *C. jejuni* / *coli* likewise. Together they would
   raise ciprofloxacin's cohort by ~40 % and leave the other nine untouched — and ciprofloxacin
   already has the largest cohort and the longest wall of the four drugs in flight.
3. ***M. tuberculosis* is most of the amikacin and levofloxacin gain, and is probably a trap.** Its
   amikacin resistance is *rrs* / *eis* and its levofloxacin resistance *gyrA*, called against
   critical concentrations rather than clinical breakpoints — and TB is already trained separately
   in §3.3. Including it would put a 21,402-genome single-mechanism block into exactly two drugs.
4. **The Gram-positive β-lactam rows are intrinsic resistance, and balancing already says so.**
   *S. aureus* is 96–97 % R for meropenem, cefepime and piperacillin-tazobactam; the rule collapses
   those to 44–58 genomes on its own. The labels were never going to arrive.
5. **Cost shape.** Everything above the family tier is ~74k new genomes for ~+17 % balanced labels.
   The order tier alone is 6,024 genomes for +6 % — the only slice whose exchange rate beats the
   cohort already being trained.

## What this does not cover

- **These are labels, not a cohort.** No genome here has been fetched, assembled, QC'd, annotated or
  embedded. §3.6 lost ~6 % of labelled PsA genomes to QC and §3.7 ~0.9 % of Enterobacterales; the
  same attrition would apply, and species with no AllTheBacteria coverage would attrite harder.
  `n_genomes_with_route` says only that the portal names an assembly or a reads accession — which it
  does for every labelled genome in every tier.
- **The species unit of the arm is the MLST scheme, not the taxon** (§3.8). Added taxa need a scheme
  before `holdout_st` is defined for them: `abaumannii` / `abaumannii_2`, `ngonorrhoeae`,
  `campylobacter`, `saureus` all exist in `mlst`, but a species with no scheme can only join the
  residual unit, and a residual unit's per-unit number is not reportable as a species result.
- **Balancing is applied per species over the whole set here**, not per split as the arm does. Splits
  are random within a unit, so the totals agree to rounding — but `holdout_st` is an ST-based group
  and can carry its own prevalence, so a per-split build will not reproduce these counts exactly.
- **The AMR-GNN supplement is not included.** PsA's 53,125 labels here are portal-only; §3.7's
  64,281 includes the supplement. The reference rate used for balancing is the decision of record's,
  which does.
- **Taxonomy is a hand-written genus → family → order → class → phylum map in the script**, covering
  every genus that carries a label for these ten drugs. 50 genera were left unmapped, 1,833
  genome × drug pairs in all, 1,617 of them non-taxon strings; the rest are small and listed in
  `unmapped_genera.csv` beside the artifacts on CSD3.

# baclm embedding pipeline — progress report

**Date:** 2026-07-09 · **Branch:** `dev` · **Scope:** *M. tuberculosis* (TB) + *K. pneumoniae* (Kp)

A run-through of the pipeline that turns EBI antibiograms into baclm genome embeddings and validates
that those embeddings carry AMR signal. Structured as the four stages of the pipeline: **parse →
download → audit → validate.** Companion docs: the baclm store schema is in
[`Baclm_embeddings.md`](Baclm_embeddings.md); the earlier TB SNP-signal-loss diagnostic is in
[`PROGRESS_REPORT.md`](_archive/PROGRESS_REPORT.md) (a prior phase, not superseded by this).

> **Reproducibility.** Every number below is regenerable. Stage 1 parser:
> [`parse_ebi_ast_to_binary.py`](../engine/ast_labels/parse_ebi_ast_to_binary.py). Stage 3 audit:
> [`audit_noncoding_regions.py`](../engine/embedding/non_coding_segment_audit.py) +
> [`scripts/audit_noncoding_regions.sh`](../engine/embedding/scripts/) (organism-agnostic —
> point `--input-csv` at any cohort's `embedding_input.csv` to run E. coli etc.). Audit JSON output:
> `…/processed/train_{tb,kleb}_ast/pangena_predict/audit_noncoding/audit_{tb,kp}_<jobid>.json`.

---

## 1. Parsing the AMR table from EBI

EBI AMR records (one row per sample × antibiotic × test) are parsed to a binary resistance matrix by
the organism-agnostic [`parse_ebi_ast_to_binary.py`](../engine/ast_labels/parse_ebi_ast_to_binary.py): `resistant → 1`,
`susceptible → 0`, `intermediate → NaN`; MIC → log scale with censoring adjustments; **repeat tests
per sample × antibiotic are averaged** (a sample whose DSTs disagree becomes a fractional label,
dropped downstream as ambiguous); antibiotics with < 1,000 tests are dropped.

| | EBI records (unique BioSamples) | → with ≥1 binary AST label | drugs kept |
|---|---|---|---|
| **TB** | 41,724 | **40,021** | ~20 |
| **Kp** | 10,250 | **7,440** | 33 |

---

## 2. Downloading the assemblies (Bakta / BakRep)

Short-read assemblies + Bakta GFF3 annotations were downloaded for the labelled BioSamples. Not every
labelled sample has a public assembly, and the trainable cohort is the intersection of *labelled* ∩
*assembled* ∩ *ESM-embedded*.

| | in AMR table | assemblies downloaded | % of table | trainable (∩ ESM) |
|---|---|---|---|---|
| **TB** | 41,724 | **38,257** | 91.7% | **36,692** |
| **Kp** | 10,250 | **9,724** | 94.9% | **7,088** |

(Kp was assembled more broadly than it is AST-labelled — 9,724 assembled vs 7,440 labelled — so the
binding constraint on the Kp cohort is *labels*, not assemblies.)

Each assembly is embedded by **baclm-350m-masked** into per-CDS (coding) and per-non-coding-region
vectors; see [`Baclm_embeddings.md`](Baclm_embeddings.md).

---

## 3. Coding / non-coding region audit

Genome-wide, GFF-only audit of every assembled genome (TB 38,257 · Kp 9,724). "Non-coding run" = a
**maximal contiguous non-CDS stretch** (only protein-coding `CDS` is treated as occupying), which is
the unit baclm embeds on the DNA side — so it includes unannotated intergenic DNA **and** any
tRNA/rRNA/ncRNA/tmRNA and other features it spans.

### 3.1 Per-genome averages

| per genome | TB | Kp |
|---|---|---|
| coding CDS | 4,136 | 5,212 |
| non-coding runs (incl. unannotated IGR + all RNA + other features) | 2,584 | 3,391 |
| **non-coding runs over the 2,048-char context window** | **1.93** | **3.34** |
| RNA bodies (tRNA+rRNA+ncRNA+tmRNA) inside non-coding runs | 62 | 167 |

Over the whole cohort, runs exceeding the window are a **tiny tail — 0.075% (TB) / 0.098% (Kp)** of
all runs — adding only ~0.1% extra forward passes when windowed. ~80% of runs are < 300 bp.

### 3.2 Does splitting RNA from IGR remove the over-window pieces?

If each over-window run were split into its **RNA bodies** vs its **non-RNA** (IGR / CRISPR / other)
segments, how many pieces would *still* exceed the window?

| per genome | TB | Kp |
|---|---|---|
| over-window runs (merged, as embedded today) | 1.93 | 3.34 |
| …still over-window as **RNA-body** pieces after split | 0.98 | 0.46 |
| …still over-window as **non-RNA (IGR)** pieces after split | 0.90 | 2.43 |

**Conclusion: splitting does not remove the windowing need.** In TB roughly half the over-window
pieces are the **23S rRNA (`rrl`, ~2.9 kb)** — irreducibly longer than the window even in isolation —
and the other half are genuinely long IGR/CRISPR stretches. In Kp the over-window tail is *dominated*
by long non-RNA stretches. So windowing (tile + pool) is required regardless of whether RNA is
embedded separately from IGR. (16S rRNA `rrs`, ~1.5 kb, *does* fit one window.)

### 3.3 RNA-type adjacency (the "embed together vs separately?" question)

For each RNA type: count per genome, and how it sits in its non-coding run.
- **adj-IGR** = the run holds ≥ 30 bp of **genuinely unannotated** DNA beside the RNA (i.e. the RNA is
  fused with real intergenic DNA, *not* embedded alone).
- **adj-RNA** = the run also holds **≥ 1 other RNA** (e.g. an rRNA operon, a tRNA array).
- **solo** = neither — the run is essentially just this RNA, tightly CDS-flanked.

**TB**

| RNA type | /genome | adj-IGR | adj-RNA | solo |
|---|---|---|---|---|
| tRNA | 46.8 | 100.0% | 41.7% | 0.004% |
| ncRNA | 11.4 | 99.8% | 9.1% | 0.2% |
| rRNA | 3.2 | 99.3% | **95.7%** | 0.7% |
| tmRNA | 1.0 | 100.0% | 0.2% | ~0% |

**Kp**

| RNA type | /genome | adj-IGR | adj-RNA | solo |
|---|---|---|---|---|
| tRNA | 82.2 | 99.8% | 65.8% | 0.1% |
| ncRNA | 75.8 | 82.9% | 22.0% | 17.0% |
| rRNA | 7.8 | 82.7% | 37.9% | 16.4% |
| tmRNA | 1.0 | 100.0% | 0.3% | ~0% |

**Takeaway.** Under the annotation-agnostic (only-CDS-occupying) scheme, **nearly every RNA body is
embedded together with flanking intergenic DNA** — and rRNA is additionally fused with the rest of its
*rrn* operon (16S+23S+5S+spacers+tRNA) 96% of the time in TB. Clean "solo" RNA are rare (TB), somewhat
more common in Kp (likely short-read contig fragmentation isolating features). This is very likely how
baclm was trained — it consumes raw genomic DNA stretches, blind to Bakta. **Open question for the
team:** whether AMR-relevant RNA (esp. `rrs`/`rrl`) should be encoded *individually* rather than fused
with the operon/IGR. baclm is not built for per-feature encoding today; to enable the comparison the
re-embed additionally emits standalone `rna_embeddings` (each RNA body alone) alongside the merged
non-coding runs, so merged-vs-separate can be tested without re-running.

### 3.4 Other non-CDS features sharing the runs (per genome)

Not just RNA — the runs also carry CRISPR, Bakta `regulatory_region`, oriC, oriT:

| feature | TB /genome | Kp /genome |
|---|---|---|
| tRNA | 46.8 | 82.2 |
| ncRNA | 20.9 | 79.7 |
| regulatory_region | 13.4 | 53.1 |
| CRISPR (repeat) | 33.2 | 7.9 |
| CRISPR (spacer) | 33.2 | 7.9 |
| CRISPR (array) | 2.1 | 0.5 |
| rRNA | 3.2 | 7.8 |
| oriC | 2.0 | 3.8 |
| tmRNA | 1.0 | 1.0 |
| oriT | ~0 | 0.9 |
| gap | 0.5 | 0.1 |

(TB is CRISPR-rich — that is the direct-repeat / spoligotyping locus.) Each of these is currently
absorbed into whatever non-coding run it falls in; **whether to encode any of them individually is a
question for the team.**

### 3.5 The 2,048-char context window

2,048 is **baclm's own `max_seq_length`** (in its model config), which the embedder mirrors — not a
value we chose. Architecturally baclm uses **RoPE with XPos scaling** (rotary positions), *not* a
learned absolute-position table, so there is **no hard cap**: it *can* forward > 2,048, but that is
out-of-distribution vs its training context. Windowing keeps every tile ≤ 2,048 (in-distribution);
running a full 2.9 kb `rrl` directly would rely on XPos extrapolation — a decision to take to the
baclm developers.

---

## 4. Validation — does baclm carry the coding AMR signal?

Head-to-head learning curves of **baclm coding** vs **ESM-C coding** per-gene vectors (identical
logistic-probe harness, fixed evaluate holdout), at matched training size.

- **TB (6 genes):** baclm ≈ ESM across the board — at full N, Δ(baclm−ESM) is ≈ 0 for rpoB (−0.001),
  katG (+0.001), gyrA (+0.005), pncA (+0.006), embB (+0.004). Only pncA is data-hungry (baclm behind
  at n=500, caught up by n≈3,000).
- **Kp (2 genes):** gyrA → ciprofloxacin ESM 0.929 / baclm 0.933 (Δ +0.005); parC → ciprofloxacin ESM
  0.914 / baclm 0.922 (Δ +0.008).

**baclm's coding channel matches ESM-C in both species** — the coding embeddings are validated. This
gates the non-coding work.

- **Promoter IGR (TB):** fabG1 promoter → ethionamide **AUROC 0.823** (a real non-coding hit — inhA
  overexpression); build audit clean for the probed promoters (CDS-flanked, un-truncated).

---

## 5. What's next

- **TB + Kp driver panels** — per driving mutation (from the TB-Profiler / Kleborate driver lists):
  one-hot ceiling vs baclm vs ESM vs Bacformer, AUROC + AUPRC, grouped column chart per drug.
- **Non-coding re-embed (2d)** — regenerate baclm's non-coding channel with the fixes above (only-CDS
  occupying runs, standalone RNA bodies, windowed long regions); the rRNA driver rows (`rrs`, `rrl`)
  backfill after it.
- **rRNA probes (2e)** — `rrs` → streptomycin/kanamycin, `rrl` → linezolid, once re-embedded.

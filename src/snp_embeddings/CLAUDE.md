# Task 7 — `snp_embeddings`: why is TB AST so poor? Diagnosing SNP-level signal loss in ESM-C → Bacformer

**Status: active — diagnostic.** The SNP-vs-ESM linear probes (`snp_vs_esm_prediction.py`) are
the first build. No new model training for the diagnostic itself. Runs against the existing TB
ESM-C embeddings + protein-sequence parquets, on the **same canonical holdout the deployed
Bacformer model used**; the geometry probe (Step 3b) runs fully local on CPU.

See the root [CLAUDE.md](../../CLAUDE.md) for §0 global conventions. Cross-task status lives in
[ToDo.md](../../ToDo.md). The authoritative spec for this task is **this file**; the approved
implementation plan is `~/.claude/plans/i-d-like-to-start-crystalline-allen.md`.

## Aim

TB AST prediction underperforms badly (rifampin Stage C val AUROC ~0.88 vs a WHO-catalogue
ceiling ≥0.97), while *Klebsiella* AST is strong. The central programme hypothesis is that
**Bacformer is strong on HGT/gene-acquisition resistance but blind to chromosomal point
mutations** — exactly TB's regime. This task tests *why*, and points to the remedy.

**Suspected mechanism — a chain of averaging that dilutes a single causal residue:**

1. **ESM-C residue → protein pool.** ESM-C mean-pools ~1,178 rpoB residues into one protein
   vector (≈1/L dilution). A single RRDR substitution moves one residue out of ~1,178.
2. **Bacformer protein → genome pool.** Bacformer mean-pools ~4,000 protein tokens into a
   genome vector (≈1/N dilution).

A causal mutation may be clear *per-residue* yet near-invisible in the *pooled* token Bacformer
consumes. Because the ESM-C pool is **frozen and non-invertible**, no amount of Bacformer
fine-tuning can recover signal lost at step 1. Where the model still predicts resistance, it
may be reading **lineage/phylogeny** (an accessory-genome shortcut) rather than the causal SNP.

**Positive control:** *M. tuberculosis* rpoB / rifampicin (US spelling **`rifampin`** in
`binary_ast.csv`). RRDR mutations used as the panel: **S450L, H445Y, D435V, S441L** (Mtb
numbering — note the ~80-residue offset vs the older *E. coli* numbering; always **assert** the
WT residue identity at each codon before scoring).

## Diagnostic gate — Stage 1 (run cheapest-first; stop as soon as the picture is clear)

Stage 1 gathers information to decide between two worlds and to refine *which* remedy and
*which layer*:

- **Representational** — signal is present per-residue but pooled away (expected). → Remedies
  A then B; **no retrain needed**.
- **Absent** — the signal is not in ESM-C at all. → Remedy C only (the sole lever for the
  variable-site regime).

### The three-step test — `snp_vs_esm_prediction.py` *(the headline numbers)*

Every step is a `sklearn.LogisticRegression` (`C=1.0`, L2, lbfgs, no `class_weight`) fit on the
**train** split and scored on the **evaluate** split of the *same canonical holdout the deployed
Bacformer model used* — `binary_ast_with_split.csv` via `tl.train.evaluate.resolve_holdouts` —
so every number, including Bacformer's own ~0.9, sits in one comparable table. `validate` only
picks the Youden operating point. Metrics/JSON reuse `tl.train.metrics`. Steps are ordered as the
story is told (note **Step 2 and 3 were swapped** vs the original plan):

| key | step | features | standardise | compute |
|---|---|---|---|---|
| `onehot_rrdr` | 1 | one-hot RRDR codon genotype (parquet) — the SNP **ceiling** (~0.95–0.97) | no | CPU |
| `pooled_esmc_rpob` | 2 | frozen ESM-C mean-pooled rpoB 960-vector (store, mmap one row) — the suspected loss | yes | CPU |
| `masked_marginal_llr` | 3a | ESM-C masked-LM LLR at the RRDR codons — is the residue in ESM-C *at all*? | yes | GPU |
| `bacformer_rpob_token` | 2b | frozen Bacformer contextualised rpoB token (`frozen_bacformer_rpob_vectors.py`) — *bonus* | yes | GPU |

**Head-line:** `AUROC(Step 1) − AUROC(Step 2)` on the **intersection** of the samples every step
covers = information lost to ESM-C's residue→protein mean. `3a` high while `2` low ⇒ the residue
survives in ESM-C pre-pool (recoverable). `2b ≈ 2` ⇒ Bacformer's cross-protein attention adds
nothing — the loss was sealed at the ESM-C pool. Step 2b rides with the Step-3 GPU pass (it needs
a Bacformer forward), so it is a bonus, not strictly necessary.

**No lineage holdout.** Every probe sees only the rpoB locus, so there is no accessory / phylogeny
shortcut to block — the canonical random split suffices (Step 2's pooled vector *could* still
inflate via lineage structure; noted in the JSON).

### Step 3b — embedding-geometry probe *(`geometry_probe.py`; mechanistic, labels-free, local CPU)*

In-silico single-residue WT→mutant rpoB pairs (each RRDR substitution applied one at a time to
H37Rv; optional real resistant isolates as confirmation). Per layer ℓ: `d_site` (the mutated
residue), `d_window` (±k neighbours), `d_pool` (the **production** mean-pool, via
`production_mean_pool`), `d_max`, `d_cls` — in cosine + euclidean; plus the per-position masked-LM
LLR profile (expect the causal residue as a single sharp outlier).

**Read-out:** `d_site ≳ d_window ≫ d_pool` ⇒ represented per-residue, crushed by the mean ⇒ an
attention pool could recover it; report the best-preserving layer (max `d_site / d_pool`). This is
the cheapest analysis and explains where Steps 1–2 land.

### Stage 1.3 — causal ablations *(deferred, more complex; only if gene-vs-tree stays open)*

A genuinely later, heavier step, engaged only if 1.1/1.2 don't settle whether the defect is
representational. Brings in the **genome-wide** Bacformer predictor: counterfactual SNP
injection (edit a susceptible isolate's rpoB codon to the R allele in silico, re-embed,
re-predict — flips ⇒ causally SNP-sensitive); gene-masked ablation (`AUROC(full) −
AUROC(rpoB-masked)` = real dependence on the causal gene); out-of-lineage transfer. **This is
where lineage-blocked splits + the existing `tb_ast` predictor are required** — hence it is left
as subsequent work, not part of establishing the top-level defect.

### Gate

- **Representational** (per-residue present, pooled/counterfactual blind, model leans on
  phylogeny → expected) → **Remedy A then B; do NOT retrain.**
- **Absent** (counterfactual-insensitive AND `d_pos` small AND high entropy) → **Remedy C.**

## Remedies (provisional — selected by the gate; least-invasive the evidence supports)

- **A — explicit pre-pooling channel (cheapest, leakage-free).** Inject the `s_i`-derived
  channel `[max(s_i), mean(top-3 s_i), count(s_i > τ)]` (τ ≈ 90–95th pct, held-out tuned) —
  and/or population-homoplasy — concatenated onto each protein token; **fine-tune Bacformer
  only**, ESM-C frozen. Bypasses both averages. Channel is label-blind ⇒ no leakage.
  Masked-marginal reserved for a candidate-locus list (rpoB, katG, gyrA, gyrB, embB, pncA,
  rpsL, rrs, ethA, inhA + Kp equivalents); cheap unmasked single-pass as the global default.
- **B — residue-level attention pooling (if A leaves signal on the table).** Replace mean-pool
  **at residue → protein** (not genome level) so the variant can dominate the protein token;
  ascend attention-pool → LoRA through top ESM-C layers → full end-to-end. Watch
  family-invariance (a two-channel stable-family ⊕ variant-residual token) and catastrophic
  forgetting.
- **C — domain-adaptive pretraining (GATED to "absent").** Continued species MLM via
  adapters/LoRA over frozen ESM-C; **conditional surprise** (P(residue | rest of genome))
  detects convergence by unpredictability-given-context. Counterproductive for conserved sites
  (naive MLM learns the site is polymorphic, shrinking `s_i`). **Acceptance = out-of-lineage
  transfer, not in-distribution AUROC.**
- **Pyseer / homoplasy — oracle + variable-site track, NOT a predictor feature.** LMM+kinship
  on the full collection → achievable-association ceiling + which loci the model should attend
  to; population-level + label-derived, so quarantined from the predictor. Runs in parallel
  from the start (compute-bound). Conservation-surprise covers conserved sites, homoplasy covers
  variable sites — complementary.

## Cross-cutting validity (genome-wide predictive steps only — 1.3 and the remedies)

Lineage-/cluster-blocked splits, never random; label-derived features mined train-fold-only or
from the external WHO catalogue; report AUROC + balanced accuracy and **within- vs cross-lineage
transfer separately** (the gap = shortcut reliance). **Locus-restricted probes (1.1, 1.2) are
exempt** — they cannot see the background, so there is no shortcut to block.

## Build order (stepwise, by dependency)

| Increment | What | Where it runs |
|---|---|---|
| **0** | scaffold + docs (this file, `__init__.py`, `ToDo.md` block, root-doc entries) | — |
| **1** | Steps 1 + 2 (CPU): `locate_gene` + `rpob_genotype` + `snp_vs_esm_prediction` on the canonical split | HPC CPU sbatch |
| **2** | Step 3 GPU pass: 3a masked-marginal LLR + 2b frozen Bacformer token (`frozen_bacformer_rpob_vectors.py`) | HPC GPU |
| **2′** | Step 3b geometry probe: per-residue states + bundled rpoB reference + probe | local / HPC CPU |
| later | 1.3 ablations + remedies (gated); pyseer oracle in parallel | HPC |

**Read Steps 1 + 2 before spending the GPU.** If Step 1 ≈ 0.95–0.97 and Step 2 ≈ baseline, the
core hypothesis (the residue→protein mean throws the signal away) is already supported; the GPU
pass (3a/2b) then decides whether it is *recoverable* (in ESM-C pre-pool) vs *absent*.

## Data paths & repo facts

- **ESM-C model:** `Synthyra/ESMplusplus_small` (ESM++ reimplementation), loaded via
  `bacformer.pp.load_plm` / `AutoModel.from_pretrained(..., trust_remote_code=True)`; tokenizer
  is `model.tokenizer`. The production forward returns only `.last_hidden_state` — logits / the
  MLM head are never requested. Stage 1.1 needs the `ESMplusplusForMaskedLM` variant for
  masked-marginal logits (**verify the class name / return fields against the cached
  `trust_remote_code` modeling file day one**; if the encoder-only `AutoModel` has no LM head,
  tie logits from hidden states via the output embedding).
- **Bacformer model:** `macwiatrak/bacformer-large-masked-complete-genomes` (refreshed
  complete-genomes weights; HPC cache pinned to the 2026-05-15 snapshot).
- **Loader idiom:** `dtype="auto"`, **not** a manual `.to(torch.bfloat16)` cast — the cast pegs
  Stage A on CPU. (Same memory idiom as the task train entrypoints.)
- **Mean-pool location:** the einsum over residues with the attention mask in
  `generate_protein_embeddings` (`.venv/.../bacformer/pp/embed_prot_seqs.py`);
  `max_prot_seq_len=1024` truncates — rpoB (~1,178 aa) exceeds it, but the RRDR codons (~430–450)
  survive. The geometry probe must run **without** that truncation.
- **Embedding store (read-only, shared):** the TB ESM-C store + protein-sequence parquets live
  under `project_k/david/processed/train_tb_ast/` (`tb_esm_embeddings/`, `binary_ast.csv`; the
  protein-sequence parquet dir is set by the embedding-prep step — confirm before the first run).
  The TB `.pt` (verified 2026-06-12) is the **plain per-protein** layout: `protein_embeddings`
  (`[1, n_proteins, dim]`, one row per protein in flat order, e.g. `[1, 4055, 960]`), `contig_ids`,
  `attention_mask` — **no** interleaved CLS/SEP/PROT_EMB tokens, so the rpoB flat index maps
  directly to a row (`attention_mask == 1` drops any padding). `snp_vs_esm_prediction._real_protein_indices`
  also handles the alternative Bacformer-input bundle (`special_tokens_mask == 4` = PROT_EMB;
  CLS/SEP/PAD/END = 2/3/0/5) for stores written that way. A protein-count guard (rows vs the
  parquet's flat count) skips any sample where the two disagree, so a flat-order misalignment can't
  pass as "signal pooled away". **No labels, no per-residue states, no logits.** (Kp equivalent:
  `processed/klebsiella_esm_embeddings/{sample_id}_esm_embeddings.pt`.)
- **rpoB → embedding-index recovery:** protein order is preserved GFF → parquet → `.pt`.
  `{sample_id}_protein_sequences.parquet` (from
  [preprocess_assemblies_to_protein_sequences.py](../tl/embed/preprocess_assemblies_to_protein_sequences.py))
  retains nested `gene_name` / `protein_sequence` / `start` / `end` / `protein_id` / `contig_idx`.
  Flattening the nested lists in `contig_idx` order maps a gene to its flat index into
  `protein_embeddings`. This is how predictors (2) and (3) find rpoB.
- **TB AST table + canonical split:** the probes read `binary_ast_with_split.csv` (drug column
  `rifampin`, US spelling; `Sample`/`phenotype-BioSample_ID`, `train_val_eval`) under
  `processed/train_tb_ast/` — the **same 70/10/20 holdout `tb_ast/train_amr.py` trained on**, so the
  probe AUROCs are directly comparable to the deployed model's. (`binary_ast.csv` is the pre-split
  source the prepare step derived it from.) The 16 ambiguous `0.5` labels are dropped in code.
- **Genotype source (decided 2026-06-12):** Stage 1.1 reads the RRDR allele **from the parquet
  protein sequence**, not from a variant caller — the mutation is in the sequence ESM-C saw.
  The annotations name the rpoB gene but do *not* carry mutation calls; we don't need them to,
  because the translated CDS does.
- **TB-Profiler — parallel ground-truth + lineage, not on the critical path.** The de facto
  gold standard (maps to H37Rv, annotates against the WHO v2 catalogue, emits per-drug calls +
  the specific mutation + lineage). We have **assemblies only (no reads)**, so it runs in
  `--fasta` mode. Use it to (a) validate our sequence-derived RRDR calls, (b) supply **lineage**
  for the deferred genome-wide steps (1.3) and cross-lineage transfer reports, (c) catch isolates
  where rpoB is fragmented/truncated in the assembly. `bioconda::tb-profiler`. Mykrobe is the
  alternative.
- HPC root: `/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/`.

## Three-stage testing protocol (recap of root §0.2)

Diagnostic, so largely N/A for the headline. Stage A discipline still applies:

| Stage | Scale | Where |
| :-- | :-- | :-- |
| **A. Smoke** | tiny panel | local CPU — the 1.2 geometry probe is itself the CPU smoke (`--device cpu`) |
| **C. Full** | full RIF cohort | 1.1 ceiling ladder on HPC / login |

## Reporting

Versioned results JSON per the §0.4 / `tl/train/metrics.py` convention:

- **Stage 1.1:** AUROC + balanced accuracy per predictor; `AUROC(1) − AUROC(3)` (info lost to
  pooling); placement of (2) (loss-at-pooling vs absent-from-model).
- **Stage 1.2:** per-mutation `s_i`/LLR, masked entropy, `d_pos`, `d_pool`, `d_max`, `<cls>`
  distance, and `d_pos`-by-layer; the best-preserving layer; plots (`d_pos` vs `d_pool`;
  `d_pos`-by-layer).
- **Decision point:** explicit Representational-vs-Absent gate call and the implied remedy.

## Files in this folder

- `__init__.py` — package stub.
- `CLAUDE.md` — this spec.
- `locate_gene.py` — gene → flat embedding index (rebuilds the flat protein order GFF→parquet→`.pt`).
- `rpob_genotype.py` — RRDR allele from the parquet protein sequence; **provenance docstring**
  (reference, no-minimap rpoB location, alignment, WHO catalogue) + **rpoB-copy QC**
  (`build_genotype_table` keeps single-copy genomes only; 0-copy and >1-copy counted, printed,
  written to the QC log, excluded).
- `snp_vs_esm_prediction.py` — the three-step linear probes (Steps 1, 2, 3a, and 2b when its NPZ
  is supplied) on the canonical `resolve_holdouts` split; reuses `tl.train.metrics`; writes the
  schema-2.0 JSON + an `*_eval_probs.npz` plotting sidecar. `scripts/run_snp_vs_esm_prediction.sh`
  (CPU sbatch Steps 1+2; GPU variant commented for 3a+2b).
- `frozen_bacformer_rpob_vectors.py` — Step 2b (GPU): frozen Bacformer forward → contextualised
  rpoB token NPZ. Imports `load_bacformer_model` / `bacformer_last_hidden_state` from
  `../tl/embed/generate_embeddings.py` (extracted there, no behaviour change).
- `geometry_probe.py` + `scripts/smoke_geometry_probe.sh` — Step 3b; extends
  `../tl/embed/esm_residue_level.py` with `residue_states`, `production_mean_pool`,
  `apply_point_mutation` (unit-tested in `tests/tl/embed/test_esm_residue_level.py`).
- `reference_gene/rpoB_H37Rv.faa` — the H37Rv rpoB reference (UniProt P9WGY9), a biological
  reference (not a test fixture); `REFERENCE_RPOB_H37RV` in `rpob_genotype.py`.
- Shared: `../tl/embed/esm_residue_level.py` (MLM loader + `masked_marginals` + residue-level ops).

## Running notes

<!-- Agent appends here as work proceeds. -->
- 2026-06-12 — Increment 0: package scaffold + this spec created. Plan approved
  (`~/.claude/plans/i-d-like-to-start-crystalline-allen.md`). Decisions: stay on branch `dev`;
  Stage 1.1 genotype is **sequence-derived from the parquet** (TB-Profiler `--fasta` in parallel
  for validation + lineage; assemblies only, no reads); TB store + parquets under
  `project_k/david/processed/train_tb_ast/`.
- 2026-06-12 — Increment 1 (Stage 1.1) built. Day-one checks resolved against the HPC caches:
  - **ESM++ MLM head confirmed:** `ESMplusplusForMaskedLM` (load via
    `AutoModelForMaskedLM.from_pretrained("Synthyra/ESMplusplus_small", trust_remote_code=True)`)
    returns `ESMplusplusOutput.logits` `[B, T, vocab]`. Tokeniser wraps `<cls> A <eos>`, so AA
    position `p` → token `p+1`; `<mask>` is a real token.
  - **Pooled-vector layout:** the stored `.pt` selects real proteins via `special_tokens_mask == 4`
    (PROT_EMB); `protein_embeddings_to_inputs` interleaves CLS/SEP/pad rows (bacformer
    `SPECIAL_TOKENS_DICT`).
  - **rpoB numbering offset:** UniProt P9WGY9 positions are **+6** vs the standard Mtb RRDR codon
    numbering (D435/S441/H445/S450). `rpob_genotype.py` anchors on the conserved core motif
    `DQNNPLSGLTHKRR` (leading D = codon 435) and **asserts** the WT residue at every panel codon,
    so a wrong/ swapped reference fails loudly. Per-sample RRDR alleles are read by global-aligning
    the assembled rpoB to the H37Rv reference (no dependence on absolute counts).
  - Modules lint clean (ruff B/BLE/C4/D/E/F/I/RUF100/TID/UP/W; empty `__init__.py` D104 matches
    the repo-wide convention).
  - **TB data verified on HPC:** `rifampin` = the canonical RIF column, 38,758 labelled
    (26,147 S / 12,595 R; 16 ambiguous `0.5` dropped in code). Sample-ID column is
    `phenotype-BioSample_ID` (SAMEA… = parquet stems). 38,248 parquets + 38,248 esm `.pt`
    under `processed/train_tb_ast/{tb_protein_sequences,tb_esm_embeddings}/`.
  - **Login-node smoke (200 samples) PASS:** pipeline runs end-to-end; one-hot RRDR AUROC 0.969,
    pooled ESM-C rpoB 0.868 (resistant-enriched subset — numbers not yet meaningful). The
    protein-count guard did not trip → flat-order alignment of the pooled rpoB row is correct.
    Measured ~0.3 s/.pt read.
  - **Full predictors 1+3 submitted:** SLURM job **30485091** (icelake CPU, 12 h),
    `--skip-masked-marginal`, output
    `processed/train_tb_ast/snp_embeddings/ceiling_ladder_30485091.json`. (Superseded — see below.)
- 2026-06-13 — **Re-planned + rebuilt** (the first cut loaded all embeddings and used an ad-hoc
  `train_test_split`, so its numbers weren't comparable to the deployed model). The approved plan
  (`~/.claude/plans/i-d-like-to-start-crystalline-allen.md`) reuses the repo infra and renames
  everything. Job 30485091 (the old `ceiling_ladder.py`) is **dead/superseded** — no action needed.
  - `ceiling_ladder.py` → **`snp_vs_esm_prediction.py`**, rebuilt on `tl.train.evaluate.resolve_holdouts`
    (the deployed model's canonical `binary_ast_with_split.csv` holdout) + `tl.train.metrics`
    (`compute_full_metrics` / `youden_threshold`) + a schema-2.0 JSON. Per-step kept-id alignment +
    an **intersection-restricted** head-line so Steps 1/2 are strictly comparable; an optional
    `--reference-results-json` asserts the deployed model's split source / `n_evaluate` match and
    records its AUROC. Lazy mmap one-row pooled reads kept (optional `--pool-workers`).
  - **Steps 2 and 3 swapped** (per the user): Step 2 = frozen pooled ESM-C, Step 3 = "is it still
    inside ESM-C" (3a LLR / 3b geometry); **Step 2b** bonus = frozen Bacformer rpoB token
    (`frozen_bacformer_rpob_vectors.py`, GPU), which **imports** `load_bacformer_model` /
    `bacformer_last_hidden_state` (extracted into `tl/embed/generate_embeddings.py`, no behaviour
    change — shared touch).
  - **Bacformer genome head pooling corrected:** it is a **straight mask-normalised mean**
    (`einsum("ijk,ij->ik", …)/mask.sum`), **no** learned attention — strengthens the "chain of two
    plain means" framing.
  - `rpob_genotype.py`: **rpoB-copy QC** (single-copy only; 0-copy and >1-copy counted + logged to
    `rpob_copy_qc.log` + excluded — replaces the old take-the-longest fallback) + a full provenance
    docstring (NCBI `NC_000962.3` / GenBank `AL123456.3`, `Rv0667`, UniProt `P9WGY9`; Bakta
    annotation, no minimap; BLOSUM62 global align; WHO 2nd-edition catalogue; TB-Profiler as a
    deferred `--fasta` validation/lineage fast-follow).
  - `fixtures/rpoB_H37Rv.faa` → **`reference_gene/rpoB_H37Rv.faa`** (`REFERENCE_RPOB_H37RV`).
  - `esm_residue_level.py` extended: `residue_states` (all layers, no 1024 truncation, strips
    `<cls>`/`<eos>`, optional `<cls>` return), `production_mean_pool` (exact einsum; unit-tested),
    `apply_point_mutation`. `geometry_probe.py` + `scripts/smoke_geometry_probe.sh` written.
  - All new/modified modules lint clean (ruff) and byte-compile; the model-free pool/mutation unit
    tests are in `tests/tl/embed/test_esm_residue_level.py` (run on HPC — no local torch). **Next:**
    HPC login-node smoke (`--max-samples`) of Steps 1+2, then the CPU sbatch, then the GPU pass.

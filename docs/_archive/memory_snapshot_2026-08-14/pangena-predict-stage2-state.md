---
name: pangena-predict-stage2-state
description: Current plan + state for pangena_predict Stage 2 — predicting AMR from baclm embeddings (coding validation → non-coding)
metadata: 
  node_type: memory
  type: project
  originSessionId: f3c1d41f-8ba4-45e4-98f4-8a0db0e1b64a
---

**Task:** `src/pangena_predict/` (renamed from `snp_embeddings` this session), branch `dev`. Goal:
predict AMR from **baclm** genome embeddings — validate the **coding** channel vs ESM first, then
exploit the **non-coding** (IGR/RNA) channel. Approved plan lives at
`~/.claude/plans/we-are-blocked-from-radiant-gem.md` (Stages 2a–2e). See [[baclm-build-defects]],
[[ebi-ast-parser-canonical]], [[isambard-cpu-jobs-need-gpu]], [[baclm-embedding-layout-doc]].

**Done (all committed+pushed on `dev`, HEAD ~`d8d6c5e`; live on Isambard):**
- Stage 1: full `snp_embeddings → pangena_predict` rename (package, imports, tests, scripts,
  data-path segments, docs). Root `CLAUDE.md` task map updated.
- Canonical EBI AST parser moved kleb_ast→`pangena_predict/parse_ebi_ast_to_binary.py`.
- TB labels regenerated on Isambard: `binary_ast.csv` + `binary_ast_with_split.csv` (seed 1, 36,692
  samples). rifampin column byte-exact to canonical (12,595 R / 26,147 S / 16 ambiguous).
- Stage 2a: `coding_amr_lr.py` = per-gene LR pulling each sample's pooled gene 960-vec from BOTH ESM
  (`load_pooled_gene_vectors`) and baclm (new `load_baclm_gene_vectors`, reads plain `[n_cds,960]`),
  scored head-to-head via `run_kfold_probe` (k=5×s=3). Plus `build_multi_gene_presence` (one parquet
  sweep for all panel genes). Tests pass (4).
**RESULTS (2026-07-08, both CPU-only jobs COMPLETED on Isambard, ~14 min each):**
- **Coding ladder (`coding_amr_lr.py --ladder`, job 5583104 → `coding_amr_lr/ladder_tb_5583104.json/.png`):
  baclm coding ≈ ESM across all 6 genes — VALIDATED.** The n=500 rpoB Δ−0.035 preview was a subset
  artifact: at matched n=500 both = 0.959; full-N Δ≈0 (rpoB −0.001, katG +0.001, gyrA +0.005, pncA
  +0.006, embB +0.004, rpoC +0.073 on a tiny 1129 pool). Only pncA is data-hungry (baclm −0.027 at
  n=500, overtakes by n=3000). Non-coding work is now on solid ground.
- **Promoter-IGR probe (`igr_amr_lr.py`, job 5583276 → `igr_amr_lr/promoter_tb_5583276.json/.png`):**
  build audit CLEAN for all 3 promoters (100%/99% CDS-flanked, 0% RNA-abutting, 0% truncated → these
  loci do NOT need the 2d re-embed). Full-N AUROC: **fabG1 promoter→ethionamide 0.823** (real non-coding
  hit — inhA overexpression), fabG1→INH 0.642 (katG dominates), eis→kanamycin 0.629 (rrs missing),
  pncA→PZA 0.560 (PZA is gene-body LoF; coding pncA=0.885). Curves flat from n=500 = low-dim promoter
  point-mutation signal. IGR anchors on flank-gene 5′ via GFF strand (no parquet); ladder generalised to
  `coding_amr_lr.ladder_over_frames`.

**PROGRESS (2026-07-09, post-compact):**
- **Kp AST labels DONE.** A Kp-specific EBI records table already existed at
  `$SCRATCHDIR/raw/kleb_ast/ebi_kleb_amr_records.csv` (133k tests, all K. pneumoniae, 10,250
  BioSamples). Ran `parse_ebi_ast_to_binary.py` → `binary_ast.csv` then `kleb_ast/prepare_esmc_...py`
  (`add_splits` seed 1, prune to the 9,724 esm/) → `train_kleb_ast/binary_ast_with_split.csv`:
  **7,088 samples** (train 4956/val 711/eval 1421), 33 drugs (cipro 2926R/1462S→gyrA/parC, gentamicin,
  meropenem, ceftazidime…). Wrapper: `pangena_predict/scripts/prepare_kp_ast_labels.sh` (job 5594659).
  TB splits confirmed already done+reproducible (add_splits seed 1 → 36,684).
- **Stage 2d re-embed CODE built+tested+committed** (shared `tl/embed`, sole agent on dev):
  `extract_intergenic_from_gff_fna` now only-CDS-occupying → maximal non-CDS runs + named-RNA index +
  standalone RNA bodies; keys renamed `intergenic_*`→`noncoding_*` (+`rna_*`). `baclm_embed` windows
  long regions (non-overlapping MAX_LEN tiles, token-weighted pool = exact whole-region mean-pool;
  `--window-overlap` for boundary context); proteins byte-identical; saves `protein_embeddings`+
  `noncoding_embeddings`+`rna_embeddings`. `igr_amr_lr` reads `noncoding_*` w/ legacy fallback. 10
  tests pass, ruff clean.
- **User decisions on the re-embed:** (a) window-pool scheme — DECIDE AFTER the audit (user wants the
  count of >MAX_LEN regions to take to the baclm devs); (b) store = write to a **NEW dir**, keep old
  until validated; launch approved AFTER audit + n=10 protein-identity smoke.
- **Step 4 audit built+running.** `audit_noncoding_regions.py` (GFF-only, fast) +
  `scripts/audit_noncoding_regions.sh`; jobs TB 5595590 / Kp 5595591. Reports: # runs & RNA bodies
  >MAX_LEN (windowing load), % runs that fuse IGR+RNA, run-length histogram, RNA-type counts.
  → `train_{tb,kleb}_ast/pangena_predict/audit_noncoding/audit_{tb,kp}_<jid>.json`. **THIS gates the
  re-embed launch + the separate-RNA-channel decision.**

**AUDIT RESULTS (2026-07-09, full cohort — jobs 5595590 TB / 5595591 Kp):**
- **Windowing is negligible.** Runs >MAX_LEN(2048): TB 74,026/98.8M = **0.075%**; Kp 32,455/33.0M =
  **0.098%**. Extra forward-passes from windowing +0.12% (TB) / +0.10% (Kp). Longest run TB 43,931 bp /
  Kp 12,565 bp. ~80% of runs <300 bp. → the number for the baclm devs: truncation drops the tail on
  only ~2 runs/genome; un-truncating costs ~0.1% compute. Non-overlapping (exact) pool is fine.
- **RNA bodies >MAX_LEN:** TB 37,503 (longest 4,002) / Kp 4,501 (longest 3,082) — this is the **23S
  (`rrl`, ~2.9kb → linezolid) needing 2 windows**; **16S (`rrs`, ~1.5kb) fits one window**.
- **IGR↔RNA fusion (architecture Q):** runs containing an RNA = TB **1.82%** / Kp **2.87%**. RNA/genome
  TB 72 / Kp 171 (tRNA≫ncRNA>rRNA). → merging RNA into its run touches only ~2–3% of the channel; we
  now ALSO emit standalone `rna_embeddings` so merged-run-vs-separate-body is testable without re-running.
- **CAVEAT — SR assemblies collapse rRNA operons.** TB ~3.15 rRNA/genome (= its 1 rrn operon ✓) but Kp
  only ~7.8 (Kp has ~8 operons ⇒ should be ~24) — repetitive rrn collapses in short-read assembly, so
  `rrs`/`rrl` copy number is unreliable on this cohort (matters for 2e rrs probing).
- JSON: `train_{tb,kleb}_ast/pangena_predict/audit_noncoding/audit_{tb,kp}_<jid>.json`.

**Kp CODING LADDER DONE (job 5595657):** baclm coding ≈ ESM in Kp too — gyrA→cipro ESM 0.929/baclm
0.933 (Δ+0.005, pool 3475), parC→cipro ESM 0.914/baclm 0.922 (Δ+0.008, pool 3480). Coding channel now
validated in BOTH species. (Kp IGR deferred: no canonical Kp promoter panel + rides on pre-2d build.)

**baclm context = 2048 (its `max_seq_length` config), architecture = RoPE + XPos (rotary), NOT a learned
position table** → no hard cap; model CAN forward >2048 but that's OOD extrapolation vs training. So
"just raise the limit" is a devs question; windowing keeps each tile in-distribution.

**Per-RNA-type adjacency (2597154 TB / 5597155 Kp) — the architecture answer:** nearly EVERY RNA sits
adjacent to IGR under only-CDS-occupying. TB: tRNA 100% / ncRNA 99.8% / rRNA 99.3% / tmRNA 100% adj-IGR;
rRNA also 95.7% adj-another-RNA (operons). Kp: tRNA 99.8% / ncRNA 82.9% / rRNA 82.7% / tmRNA 100%
adj-IGR (Kp has more "solo" ncRNA/rRNA ~16%, likely SR contig fragmentation). Solo (clean CDS-flanked)
RNA are RARE (TB rRNA solo 0.7%). ⇒ merged-run embedding ALWAYS fuses RNA+IGR (+rRNA with its whole rrn
operon) — likely how baclm was trained (annotation-agnostic DNA stretches), BUT means a clean rrs signal
needs the standalone `rna_embeddings` we now emit. **Other non-CDS features fused into runs:** TB CRISPR
huge (repeat+spacer ~1.27M each — the DR/spoligotype locus), regulatory_region ~513k (~13/genome),
oriC ~76k, oriT; Kp regulatory_region ~516k, CRISPR ~77k, oriC/oriT. So runs also carry CRISPR/regulatory
/oriC, not just RNA+IGR.

**SPLIT ANALYSIS (does separating RNA from IGR remove the >window pieces? NO):** of the >window runs,
pieces STILL >window if split RNA-vs-non-RNA — TB 0.98 RNA-body (the irreducible 23S rrl) + 0.90 non-RNA
(long IGR/CRISPR) /genome; Kp 0.46 RNA + 2.43 non-RNA /genome (Kp tail is non-RNA-dominated). ⇒
windowing is needed regardless of the RNA/IGR architecture choice. 16S rrs fits one window; rrl doesn't.

**PIPELINE NUMBERS (for report):** TB — EBI records 41,724 BioSamples → 40,021 binary-labelled → 38,257
assembled (91.7%) → 36,692 trainable (∩ESM). Kp — 10,250 → 7,440 labelled → 9,724 assembled → 7,088
trainable (Kp assembled>labelled; labels are the constraint). Per-genome: CDS TB 4,136/Kp 5,212;
non-coding runs TB 2,584/Kp 3,391; runs>window TB 1.93/Kp 3.34.

**DRIVER PANEL + BACFORMER SWEEP BUILT (2026-07-09, committed dev):** `driver_panel.py` — per drug,
table [one-hot (from tbprofiler/kleborate CSV) | baclm | ESM | Bacformer] AUROC+AUPRC per driver via
run_kfold_probe; coding drivers filled, non-coding/rRNA left BLANK (one-hot only) per user; grouped
column chart/drug. `bacformer_gene_panel_vectors.py` — ONE Bacformer forward/genome extracts every
panel gene's token (drug-agnostic, all split samples) → per-gene NPZ backfilling the Bacformer column.
Scripts `driver_panel.sh` (CPU) + `bacformer_gene_panel_vectors.sh` (GPU, gpu-venv, --gres=gpu:1). 6
tests. **TB CSVs exist (10 drugs); Kp per-drug determinant CSVs DON'T yet — must run
kleborate_determinant_lr.py first (needs metadata_v2 + Kp split).** Bacformer runs in
`$SCRATCHDIR/envs/bacpredict-gpu-venv` (has bacformer+transformers).

**CARD GENE LOCATOR built** (`card_gene_locator.py`, committed): reads kleb_ast AMR sidecar
`{Sample}_amr.parquet` (flat protein index + amr_gene_family from minimap CARD) → SAME presence-table
schema as build_multi_gene_presence, so acquired Kp genes (AAC(6')/bla_KPC/ArmA — Bakta under-annotates)
get baclm/ESM/Bacformer vectors. Wired into driver_panel + bacformer sweep via `--amr-sidecar-dir`
(sidecar dir `train_kleb_ast/amr_annotation`) with `sidecar_dir_available` graceful fallback to Bakta.
Single-copy only; flat_index<0 (Bakta-missed) ignored. Kp CSVs = kleb_ast/docs/visualisations/
amr_per_abx/kp_<drug>/card_determinant_lr_<drug>_family.csv (schema: category/n_determinants/__ALL_CARD__).
**Kp PREREQ: if train_kleb_ast/amr_annotation sidecars ABSENT on Isambard, run annotate_amr_sidecar.py
(minimap2/pixi + metadata_v2) before the Kp panel.** Launcher reports sidecar presence on recovery.

**AUDIT extended to CRISPR/oriC/regulatory** (feature_breakdown replaces rna_breakdown; cols total,
/genome, adj-IGR, adj-feat[any other annotated feature], adj-RNA[RNA rows only], solo). Answers "embed
CRISPR/oriC separately from regulatory?". Needs re-run for the numbers.

**RE-EMBED ARCHITECTURE DECIDED by user:** RNA regions embedded SEPARATELY + all other regions as BLOCKS
(= current baclm plan). My 2d code already emits both (noncoding_embeddings blocks incl. RNA + standalone
rna_embeddings). ONE nuance to confirm at launch: do the blocks EXCLUDE RNA (user said "RNA separate to
the other regions") or include it (current code)? "We can redo variants." Window-pool still open (devs).

**ISAMBARD SSH DOWN 2026-07-09 (~evening)** — transient login-node connection-refused. Background
launcher (scratchpad/launch_when_up.sh, task broxhbtmb) will git-pull + submit on recovery: bacformer-
panel-tb (GPU), driver-panel-tb (CPU), audit-nc-tb/kleb (CPU re-run for CRISPR/oriC numbers).

**PROGRESS REPORT WRITTEN:** `src/pangena_predict/docs/baclm_pipeline_report.md` (new file; parse→
download→audit→validate; does NOT touch the SNP-diagnostic PROGRESS_REPORT.md). User may later want it
promoted to THE PROGRESS_REPORT.

**OPEN — two team decisions gate the re-embed:** (1) window-pool scheme (non-overlapping-exact vs
overlapping); (2) whether to encode AMR RNA (rrs/rrl) + other features individually vs annotation-
agnostic merged runs (baclm not built for per-feature today; we emit standalone rna_embeddings anyway
to test). Re-embed → NEW dir, after n=10 protein-identity smoke. Re-embed launch (to a NEW dir; after an n=10 protein-identity
smoke) waits on that call.

**AGREED PLAN (2026-07-09, user; reordered — step 4 audit now runs BEFORE the step-1 re-embed launch):**
1. **Re-embed ALL non-coding (Stage 2d) — kick off first.** Decision made: re-embed everything (safest),
   don't try to salvage the current IGR build. Spec (see [[baclm-build-defects]]): treat only `CDS` as
   occupying → each region = maximal contiguous non-CDS run (RNA embedded WITH adjacent IGR, no
   fragmentation); emit a named-RNA index (`rna_gene_name/seqid/start/end/type`) so `rrs`/`rrl` are
   locatable by name; un-truncate long regions via overlapping windows + pool; KEEP `protein_embeddings`;
   re-run baclm TB(38257)+Kp(9724); do NOT re-run ESM/Bacformer. Requires editing
   `tl/embed/extract_intergenic_from_gff_fna.py` (+RNA index) + `baclm_embed.py`, then a GPU sbatch.
2. **TB driver panel — all driving mutations (big deliverable).** Per-drug driver lists live in
   `src/pangena_predict/docs/visualisations/tb_<drug>/tbprofiler_gene_lr_<drug>.csv` (see
   [[tbprofiler-driver-csvs]]). Build the final table PER DRIVER: **one-hot (=CSV `mut_auroc`/`mut_auprc`),
   baclm, ESM, Bacformer** AUROC + AUPRC, then a grouped column chart (AUROC + AUPRC) per drug. Coding
   drivers → baclm+ESM (have) + Bacformer (GPU, `bacformer_genome_vectors`); non-coding/promoter → baclm
   IGR; rRNA drivers → after 2d. `__ALL_WHO_one_hot__` row = full WHO ceiling.
3. **Extend to Kp — kick off immediately.** Coding ladder (gyrA/parC…) + IGR for Kp; needs Kp AST-column
   + `train_kleb_ast` path confirm.
4. **Audit report: IGR-next-to-RNA (genome-wide, architecture question).** User's independent interest:
   across ALL intergenic regions (not just the 3 promoters, which were 0% RNA-abut), how often does an
   IGR abut an RNA (tRNA/rRNA/ncRNA)? Because baclm embeds an RNA+adjacent-IGR stretch as ONE vector —
   so the number says how often 2d's merge matters, and whether IGR could instead be embedded SEPARATELY
   from RNA (viable only if annotation boundaries are clean/reliable). Classify each IGR by flank types
   (CDS-CDS / CDS-RNA / RNA-RNA); report cohort fractions. Reuse `igr_amr_lr._parse_gff` + feature typing.
5. **Write up 1–4 into PROGRESS_REPORT.md** (+ figures).

Stage 3 plot consolidation still PENDING (blocked on CSD3).

**Isambard paths:** `$SCRATCHDIR/processed/train_tb_ast/{esm,baclm,protein_sequences,intergenic}/`
(38,257 each) + `binary_ast_with_split.csv`. venv `$SCRATCHDIR/envs/bacpredict-gpu-venv/bin/python`,
`PYTHONPATH=$HOME/BacPredict/src`. Run: `coding_amr_lr.py --species tb --panel --n-folds 5 --seeds 1,2,3 --pool-workers 16 --output …`.

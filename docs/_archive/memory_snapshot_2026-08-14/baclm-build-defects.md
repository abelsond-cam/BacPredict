---
name: baclm-build-defects
description: Two defects in the existing baclm non-coding embedding build that require a re-embed (Stage 2d)
metadata: 
  node_type: memory
  type: project
  originSessionId: f3c1d41f-8ba4-45e4-98f4-8a0db0e1b64a
---

The current baclm store (TB 38257 + Kp 9724) has two build defects, both driving Stage 2d re-embed
(see [[pangena-predict-stage2-state]], [[baclm-embedding-layout-doc]]):

1. **Missing all non-CDS RNA.** The protein pass keeps only `CDS`
   (`src/tl/embed/extract_proteins_from_gff_fna.py:463`, `parts[2] != "CDS"`); the DNA pass took only
   *gaps between all annotated features* (`extract_intergenic_from_gff_fna.py`,
   `_NON_OCCUPYING_TYPES={region,databank_entry,gap}`). So every non-CDS RNA gene body — rRNA
   (`rrs`/`rrl`/`rrf`), tRNA, tmRNA, ncRNA — is in **neither** store. → `rrs` (streptomycin, kanamycin)
   unprobeable today. ESM/Bacformer are protein models and can't represent RNA anyway, so ONLY baclm
   needs re-running (DNA side; fast flash-attn path).

2. **Fragmentation + truncation.** Because RNA features count as "occupying", an intergenic region
   abutting a tRNA/rRNA was split off from a contiguous non-coding stretch baclm likely saw whole →
   possibly out-of-distribution fragment. Long regions (>2048 chars) were truncated then mean-pooled.
   Promoters flanked by two CDS (inhA/eis/pncA) are clean; must audit (Stage 2b) which loci are affected.

**Fix (2d):** treat only `CDS` as occupying → each non-coding region = maximal contiguous non-CDS run
(includes RNA, no fragmentation); emit a separate named-RNA index (`rna_gene_name/seqid/start/end/type`)
so `rrs` is locatable by name; embed long regions in overlapping windows (no truncation). Re-run
extraction + baclm for TB+Kp; keep `protein_embeddings`; do NOT re-run ESM/Bacformer. Kp AMR is
overwhelmingly enzymatic/protein (AAC/ANT/APH, armA/rmtB) → RNA gap barely affects Kp; it bites TB.

**SUPERSEDED by the two-pass re-embed (2026-07-16, plan the-cambridge-hpc-and-dreamy-thacker).** User
wants BOTH granularities + all feature types. **CP-A DONE + pushed `d62cea5`** (branch
refactor/consolidate-engine): `extract_intergenic_from_gff_fna` now emits THREE channels —
`noncoding_*` (whole CDS-to-CDS runs = **whole_igr**), `fragment_*` (runs split at every named non-CDS
feature = **per_unit** promoter fragments), `feature_*` (named bodies + `feature_type`/`feature_name`:
rRNA/tRNA/tmRNA/ncRNA + **CRISPR** [the whole array] + **regulatory_region** + **oriC**). `baclm_embed`:
`_windowize` now splits >2048 into ⌈L/2046⌉ **EQUAL** segments (user: a tiny tail window over-weights /
poorly contextualises); `.pt` keys `{protein,noncoding,fragment,feature}_embeddings` (drops `rna_*`;
`noncoding_*` kept so `build_per_igr_lr_store` reads it unchanged). setup/isambard/*.sbatch stale
`src/tl/embed`→`engine/embedding` (via `REPO`, default worktree) + smoke schema fixed. 47 tests green;
**real-genome check on 3 Kp: feature_type MATCHES the GFF exactly, rrl (~2900bp) captured as a windowed
feature body.** **CP-B code DONE** (`613e35e`): `FRESH=1` env on extract_proteins.sbatch (intergenic) + embed_baclm.sbatch
drops `--skip-existing` → overwrite the stale stores IN PLACE (no destructive delete; proteins stay
skipped/unchanged). **Smoke `5675824` SUBMITTED (PENDING for a GPU)** — validates the new 3-channel .pt
end-to-end. NB the new `.pt` metadata is native-python (via `_col`) so `torch.load(weights_only=True)`
(the smoke's step-6 assert) is safe (the old July-6 smoke failed there on numpy scalars). **When smoke
passes → launch full embed:** `FRESH=1 sbatch --export=ALL,REPO=$WT,FRESH=1 --array=0-7 embed_baclm.sbatch`
(Kp, TASK=kleb) + `--array=0-31` (TB, TASK=tb), preceded by the intergenic regen (`extract_proteins.sbatch`
FRESH=1, or run extract_intergenic_to_parquet directly). Maciej's flash-attn env for baclm.

**CP-C code DONE + VALIDATED ON REAL DATA** (`537f392`, out-dir default → repo `visualisations/` `34507f2`):
`engine/plots/plot_igr_lr_ranking.py` (organism-agnostic; causal via `--causal-genes`/`--causal-csv`).
Rendered TB rif + Kp cipro from the real overnight per_igr rankings — **all four figures correct**: Kp cipro
top10 = pptA→lamB 0.973 leading, presence one-hot ≈0.5 (grey) for every hit, embedding bars PALE (low-prev
Kp), no hatch; density = bulk ~0.55 with the 10th-best (0.923) far out in the right tail. TB rif = weak (cap
0.71, coding-mutation control), bars more saturated (higher prev). Review artifact published
(claude.ai/code/artifact/6b8f6680). Plots land `visualisations/<species>/<display_drug>/<method>/{top10,density}.png`.
**KEY INTERPRETATION (user, corrects my first read — the density SHAPE is the diagnostic, not peak AUROC):**
Kp cipro density has a **heavy right tail** (a large fraction of all 4,757 IGRs score >>0.6) → that breadth is
**pervasive phylogenetic signal** in Kp's varied, lineage-linked intergenic sequence (and cipro is itself
lineage-structured), so the top hits sit in a POPULATED tail, NOT as clean outliers → **causality unresolved**.
TB rif density is **extremely tight (near chance) with a small well-separated bump ~0.65** → that isolated bump
is statistically striking despite low AUROC. BUT TB top-region prevalences are still ~0.5 → could STILL be
phylogeny. **Phylogeny is a live confounder in BOTH** (presence one-hot ≈0.5 only rules out carriage-alone, not
lineage-linked sequence). ⇒ **CP-D / downstream MUST add an explicit lineage control** (within-lineage
permutation null — cf [[gwas-calibration-reliability-protocol]] / [[bac-pyseer-unitig-lambda-investigation]] — or
a clade covariate) before any causal claim; the per_unit RNA/CRISPR/regulatory hatch is where a real catalogue
mechanism (rrl→azithromycin, rrs→streptomycin) would show, distinguishing mechanism from lineage.
**CP-C batch driver DONE + full plot set generated (2026-07-16).** Two thin per-app drivers committed
(`dbaa2b5`): `apps/tb/plot_tb_igr_rankings.py` (causal via committed `visualisations/tb/<display>/tbprofiler_gene_lr_<drug>.csv`
`gene_name`) + `apps/kleb/plot_kp_igr_rankings.py` (causal via `card_label.causal_genes_for_drug`, guarded).
Both take `--method` (per_igr | whole_igr | per_unit) → names the ranking-file prefix + output subdir, so
**CP-D re-runs them verbatim** on the re-embedded stores. Engine plotter stays app-agnostic (TB CSV / Kp
list passed in). **Full current per_igr set rendered on disk: 10 TB + 22 Kp × {top10,density} = 64 PNGs under
`src/bacpredict/visualisations/<sp>/<drug>/per_igr/` — GITIGNORED (`.gitignore:44 *.png`), NOT committed**
(regenerate any time from the ranking CSVs via the drivers). Verified: TB streptomycin weak/prevalence-coloured,
Kp azithromycin density shows the heavy right-tail (10th-best 0.791) — the weak-model drug where CP-D per_unit
rRNA (`rrl`) testing matters. Ranking CSVs pulled from `train_{tb,kleb}_ast/pangena_predict/per_igr_lr_ranking/<drug>/per_igr_lr_<drug>.csv`.
**CP-C presence fan-out LAUNCHED (2026-07-16): TB `5676672` (array 0-9) + Kp `5676674` (0-21), FEATURE=presence,
CPU, idempotent (skips rif/cipro, backfills the rest).** Needed the shared `build_per_igr_lr_ranking.sh`
PYTHONPATH fix (`ce1c9db`): was hardcoded `$HOME/BacPredict/src` (stale `dev` checkout, no bacpredict pkg) →
now `REPO`-resolved (default worktree). Both presence arrays PENDING on cluster load (837 nodes alloc).

**CP-D (gated on CP-B embed finishing):** whole_igr LR = re-run existing `build_per_igr_lr_store` on the
re-embedded `noncoding_embeddings` (new namespace); NEW `engine/gene_lr/build_per_unit_lr_store.py` (reads
`fragment_embeddings`+`feature_embeddings`, body-anchored keys `rrna:16S`/`crispr:<seqid>`/`left→right#k`,
relaxed single-copy gate, `unit_type` column); then re-run the CP-C plotter for the final two-method figures.
Real-genome CP-A check confirmed rrl (~2900bp) is a windowed feature body → the azithromycin/streptomycin RNA test.

**CAPTURE BUG FOUND + FIX (2026-07-16) — the inhA-promoter case (David: "the way we chose to embed
regions might not be capturing it").** Traced on a real TB genome (SAMEA10029749): the mabA-inhA operon
promoter (ethionamide/isoniazid −15) is the **59 bp region 5′ of fabG1** (contig1 106042–106100, − strand;
inhA 104470–105279, fabG1 105298–106041, both −). Chain: (1) it **IS embedded** — a CDS–CDS gap ≥30 bp, so
it's a `noncoding` row in the store (verified: row 49 of the stale .pt = 106042–106100). (2) But it's
**dropped from the per-IGR ranking**: its far flank is an **unnamed AbiEi antitoxin CDS** (product-only, no
`gene=`), and `_flank_pair` requires BOTH flanks named within `boundary_tol=3` → the region fails the filter.
Confirmed: **`fabg1`/`inha` NEVER appear as a flank in the ethionamide OR isoniazid IGR rankings.** (3) Even
the surviving generic-`fabg` IGRs score only **0.56** vs the catalogue's **0.826** for inhA-promoter — mean-pool
dilutes the −15 SNP. Also naming: fabG1 lowercases + collapses with 8 generic `fabG` paralogs.
**FIX (David-chosen): name a regulatory region by the GENE IT SITS 5′ OF (`upstream:<gene>`), not the flank
pair** — keeps regions next to hypotheticals, names them as the catalogues do. NEW module
`engine/gene_lr/build_upstream_region_lr_store.py` (`be11e57`): per named gene, take the region abutting its
5′ end (− strand→region just above `gend`; + strand→just below `gstart`, within boundary_tol), key
`upstream:<gene>`, single-copy gate, reuse `fit_per_gene` + `_read_intergenic` (reads legacy `intergenic_*`
AND re-embed `noncoding_*`) verbatim. **Runs on the CURRENT stale store — no GPU/re-embed needed for the
proof** (the region is already embedded, just needs synteny-location not flank-naming). **Pooling stays MEAN**
(David: not max/concat). Decisions: embed regulatory regions BOTH as fragments (bakta-granularity) AND
whole_igr — important comparison. **Sibling project `nuna`→`syntology`** = bakta-independent homology+synteny
classification, the long-term home of this anchoring — see [[syntology-synteny-map-project]].
**PROOF DONE (login, no GPU, subsample 200): eth `upstream:fabg1` 0.80 (cat 0.826), inh 0.62 (cat 0.646)** — the
top anchor in both; recovers the dropped signal. (Login node has a ~2.5-min process killer → login smokes ≤~200
genomes, one drug per run; full fan-outs → sbatch.)

**FAN-OUT PLAN APPROVED + CP-U LAUNCHED (2026-07-16 pm, plan the-cambridge-hpc-and-dreamy-thacker, now pruned to
the non-coding-capture focus).** 3 workstreams: **U** upstream fan-out NOW on stale store · **P** catalogue-vs-baclm
comparison plot · **B** re-embed (GPU, smoke-gated) · **D** re-run U+P on re-embed (whole_igr-vs-fragment "both") +
lineage control.
- **U DONE + COMPLETED:** NEW driver `engine/scripts/build_upstream_region_lr_ranking.sh` (`33ff9d8`, sibling of
  per_igr); **arrays `5679704` (TB 0-9) + `5679705` (Kp 0-21) all COMPLETED ~2026-07-16 19:00** (2-8 min each,
  subsample 2000, n_jobs 32) → wrote all 10 TB + 22 Kp `upstream_lr_ranking/<drug>/per_upstream_lr_<drug>.csv`.
- **P GENERATED (2026-07-16, THE deliverable + inhA-promoter recovery CONFIRMED on real data).** Ran the batch
  plotter (gpu-venv python, `BACPREDICT_DATA_ROOT=$SCRATCHDIR`, worktree @ 15b71e5) → **9 TB** (rifabutin skipped,
  no catalogue CSV) **+ 22 Kp** figures + tidy join CSVs at `src/bacpredict/visualisations/<sp>/<disp>/{tb_profiler,card}_vs_bac_lm.{png,csv}`.
  **Ethionamide is the headline:** `inhA (promoter)` → `upstream:fabg1` baclm **0.797** vs catalogue **0.826** — the
  determinant the flank-pair IGR screen DROPPED (far flank = unnamed AbiEi antitoxin CDS) recovered near-parity via
  the `CATALOGUE_ANCHOR={("tb","inha"):"fabg1"}` bridge. Isoniazid corroborates: katG coding 0.899≈cat 0.893;
  inhA-promoter→upstream:fabg1 0.619 vs 0.646; ahpC-promoter 0.570 (baclm BEATS cat 0.510). Streptomycin: rpsL
  0.775≈0.775 but `rrs (promoter)`→rRNA→BLANK ("per_unit re-embed pending", n/a) — fills only after CP-D's feature
  channel. **Kp gaps (expected, catalogue-only bars now):** gentamicin/tobramycin/amikacin 0-matched (aminoglycoside
  aac/aph ACQUIRED genes below the coding prevalence gate); azithromycin 0/7 (rrl=rRNA blank + acquired mph/erm) —
  both fill after CP-D. Figures NOT committed (PNGs gitignored; CSVs regenerable, superseded by the re-embed run).
- **P DONE (code, `d2fb42f`):** `engine/plots/plot_catalogue_vs_baclm.py` — pure JOIN (NOT driver_panel, which
  re-fits live + skips non-coding): catalogue CSV (`parse_driver_csv` reuse, `--catalogue-kind tbprofiler|card`) ×
  per_gene (coding) × per_upstream (noncoding). Per drug → `tb_profiler_vs_bac_lm.png` / `card_vs_bac_lm.png`,
  grouped bars (catalogue solid vs baclm hatched), mechanism-coloured, `__ALL__` ceiling line. Match: coding/chromo/
  acquired gene→`per_gene[base]` (`_base_symbol` strips ` (mut)`/` (WT)` so Kp `GyrA (mut)`+`GyrA (WT)`→`gyra`);
  TB promoter(G)→`upstream:<anchor>` via `CATALOGUE_ANCHOR={("tb","inha"):"fabg1"}` (catalogue-facing sibling of
  `IGR_PANEL`, igr_amr_lr.py:69); rRNA→blank (drawn "n/a") until per_unit re-embed. Batch `main()` loops species×drug
  (AST drug lists hardcoded; folder=display_name, file+ranking-dir=AST drug). Stage-A test green (anchor bridge +
  mut/WT collapse + render + missing-ranking). **Figure generation gated on CP-U rankings (0 on disk; arrays PENDING).**
  per_gene baclm rankings live at `<root>/processed/train_<task>/pangena_predict/per_gene_lr_ranking_baclm/<drug>/per_gene_lr_<drug>.csv`
  (confirmed on disk); catalogue CSVs committed at `visualisations/{tb,kp}/<disp>/{tbprofiler_gene_lr_<drug>,card_determinant_lr_<drug>_family}.csv`
  (TB folder=UK `rifampicin` but filename=AST `rifampin`).
- **CP-B smoke `5675824` COMPLETED + VALIDATES the re-embed schema:** a `.pt` carries `noncoding`(3589,960)+
  `fragment`(3690,960)+`feature`(242,960)+`protein`(5263,960); stale `intergenic_*` gone; `feature_type`=
  {rrna:12,trna:86,ncrna:85,tmrna:1,crispr:2,regulatory_region:52,oric:4}. Presence fan-out TB `5676672`(0-9)+
  Kp `5676674`(0-21) COMPLETED. **CP-B is ready to launch.**
- **CP-B LAUNCHED to a SEPARATE store (David: "save CP-B to a new store, keep both") — race eliminated.** The
  re-embed writes to a NEW `baclm_reembed/` dir (new `STORE` env on `embed_baclm.sbatch`, `15b71e5`; default stays
  `baclm`), so the stale July-7 `baclm/` (intergenic-only) is untouched → CP-B runs CONCURRENTLY with CP-U (no
  shared-file race) AND we keep a stale-vs-re-embed before/after. Fresh empty dir → `--skip-existing` embeds all,
  no FRESH/overwrite. **Chain:** intergenic regen TB `5680688`/Kp `5680689` (extract_proteins.sbatch FRESH=1 → new
  3-channel parquets IN PLACE at `$PROC/intergenic`, overwriting the regenerable stale ones — safe, nothing kept
  reads them) → embed TB `5680691_[0-31]`/Kp `5680693_[0-7]` (GPU, `--dependency=afterok`, `STORE=baclm_reembed`).
  Sizes: stale .pt 12.6 MB (2-ch) vs re-embed 24.7 MB (4-ch); new store ≈1.0–1.2 TB; scratch 2.57/5 TiB → fits.
  **CP-D LR fan-outs MUST pass `--baclm-dir <root>/processed/train_<task>/baclm_reembed`; CP-U + all stale-store
  rankings stay on `baclm/`.** (Count files with `find | wc -l`, NOT `ls *.pt | wc -l` — 38k TB files blow ARG_MAX.)

**CP-R — "REAL NUMBERS" held-out-test upgrade (2026-07-16, David: "full numbers on the full set of training
samples AND test set … then same for new embedded").** The LR screens NEVER touched the test set: `lr_auroc`
was 5-fold OOF-CV on a 2k train subsample (validate/evaluate never loaded). CP-R adds a real held-out-test
number. **Decisions (David):** fit on **train+validate** (a plain LR needs no early-stop set), test on the
untouched **evaluate** split; upgrade **BOTH** coding (`per_gene`) + non-coding (`upstream`) so the CP-P figure
is one consistent metric. **Engine (committed `41a45a5`+`54dde19`+`b9fd0a5`, SHARED — all opt-in, default byte-
identical):** `fit_one_gene(...,eval_ids)` splits each region's genomes into fit(¬eval)/eval, fits OOF+full on
fit, `eval_auroc`=full-fit scored on eval (present-conditioned like OOF); `eval_ids=None`→original exactly.
`run(...,eval_holdout)` on both stores: fit_pool=train+val, core-gene discovery+prevalence stay on fit only
(evaluate never selects regions); new cols `eval_auroc_<drug>,n_eval,n_eval_pos`. **Memory wall:** full ~34k-
genome design matrices ≈250–520 GB float32 > 460 GB/node (BOTH coding AND upstream — upstream holds ~1-2k
regions/genome, same order, NOT 90 GB) → added `store_dtype=float16` to both collectors (halves storage;
`fit_one_gene` upcasts→float32 so the estimator/loss/penalty are UNCHANGED — precision loss confined to the
stored embedding, a benign engineering choice not a method change). CP-P prefers `eval_auroc_<drug>` col
(fallback `lr_auroc`), axis label "held-out test", `--ranking-suffix _eval`. sbatch envs `EVAL/SUFFIX/MAX_TRAIN
(={} =full via ${VAR-def} not :-)/STORE_DTYPE/BACLM_DIR`; fixed coding script's stale `$HOME/BacPredict/src`
PYTHONPATH→REPO. 36 tests+ruff green. **PROBES LAUNCHED (TB rifampin, float16, --mem=440G whole node):** coding
`5684433` + upstream `5684434` (both PENDING, workq backed up) — measure MaxRSS + rifampin eval-AUROC BEFORE
fanning out the full 64-task array (TB 0-9 + Kp 0-21 × {coding EMBEDDING_STORE=baclm, upstream}, SUFFIX=_eval,
STORE_DTYPE=float16). Then regenerate CP-P with `--ranking-suffix _eval`. **CP-R-reembed** repeats the upstream
eval on `baclm_reembed/` (BACLM_DIR=…/baclm_reembed) once CP-B lands.

**CP-R PROBE RESULTS + FANOUT (2026-07-17):** Coding probe `5684433` COMPLETED, **MaxRSS 285.7 GB**
(float16, full cohort) → coding fan-out **LAUNCHED** TB `5688403`(0-9) + Kp `5688404`(0-21),
EMBEDDING_STORE=baclm EVAL=1 SUFFIX=_eval float16 --mem=400G. **REAL NUMBER:** rifampin rpoB
**eval_auroc=0.9699** (held-out test, n_train=28110/n_eval=6944; OOF 0.9718 → no overfit; n_core=1949,
n_evaluate=7071). Upstream probe `5684434` **OOMed (MaxRSS 421.7 GB @440G)** — the collector did
`results=[_genome_upstream_records(...) for ...]`, materialising EVERY genome's records + every named
gene's region (mostly rare accessory) >440 GB. **FIX committed `8c19b29`:** two-pass STREAMED collector
(pass 1 keys-only → core >min_prev; pass 2 vectors for CORE anchors only, eval scored-not-selected) —
mirrors coding's discover-then-collect; peak ~440→~60 GB; identical output (run() filtered to same core
anyway). Re-probe `5688736` (rifampin, two-pass, 300G) queued → confirm fit then fan out upstream
TB(0-9)+Kp(0-21). **CP-B FULLY COMPLETE** (regen 5680688/9 + embeds 5680691_[0-31]/5680693_[0-7] all
COMPLETED ~04-08:00; `baclm_reembed/` 4-channel store built) → **CP-D + CP-R-reembed now UNBLOCKED**.
Worktree @ 8c19b29.

Isambard SSH login threw intermittent `Permission denied (publickey)` + cluster→GitHub `git fetch` connection
resets mid-session (both transient; recover on retry). Worktree `$SCRATCHDIR/worktrees/consolidate` @ be11e57.
Branch refactor/consolidate-engine.

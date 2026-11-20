# Enterobacterales AST (`apps/entero`)

The Enterobacterales adapter for the multi-species AMR work (PsA first, then Enterobacterales). **Status,
numbers and next steps live only in [`PROJECT_STATE.md`](../../../../PROJECT_STATE.md) §3.6**; this file holds
the mechanics and the agreed build design. The pipeline is the engine; this folder holds only run scripts and
Enterobacterales data files.

## Labels

[`scripts/build_entero_ast_labels.sh`](scripts/build_entero_ast_labels.sh) runs
`engine/ast_labels/build_binary_ast_labels` on a CSD3 login node (~1 min) into
`<root>/processed/train_entero_ast/`. Same ten drugs and rule as PsA (see `apps/pa/CLAUDE.md`): EUCAST v15 from
the MIC, the portal's own call where there is no MIC, unitless MICs as mg/L. Portal only (AMR-GNN is PsA-only).
The genus list is every genus whose family maps to the order Enterobacterales in
`engine/ast_labels/breakpoints.py`, so the label filter and breakpoint resolution cannot disagree. Of those 21
genera, 9 have labelled genomes.

## Genome build — design (agreed 2026-11-19, not yet built)

Scale: ~67.8k labelled genomes, 96.5 % of them in AllTheBacteria (ATB). The PsA engine
(`engine/genomes/`) is reused; PsA outputs must stay byte-identical (regression-check every engine change by
re-running PsA collate/plan into `$TMPDIR` and md5-ing against the live files).

**Annotation is Bakta 1.9.4 + full DB v5.1, not Prodigal** (decision of record, PROJECT_STATE §6):
- BakRep v2 (<https://bakrep.computational.bio>) serves ATB's own Bakta run for ATB release 0.2 + the 2024-08
  increment. Its GFF3 embeds the contigs, and its contigs are byte-identical to ATB's (Bakta ran with
  `--keep-contig-headers`), so its coordinates sit on our ATB contigs.
- Per-sample files need one API GET to find the host bucket:
  `https://bakrep.computational.bio/api/v2/datasets/<BioSample>`. The full index is the export at
  `$R/raw/bakrep/v2_export_20250612/bakrep-export.tsv.gz`. BakRep serves ~0.6 MB/s per stream (ATB's S3 ~6
  MB/s); keep concurrency modest (a 76-way pull hit SSL EOF throttling).
- On ingest, refuse any GFF3 whose `# Software:` / `# Database:` header is not `v1.9.4` / `v5.1, full`: a
  BakRep re-run with Bakta 1.12 is being prepared upstream.
- Genomes BakRep lacks get our own Bakta run of the same version and flags, after a reproduction gate against
  BakRep-held genomes. Pin 1.9.4: Bakta ≥ 1.10 writes pseudogenes as `pseudogene=` rather than `pseudo=True`,
  and the protein extractor filters on `pseudo` only.
- Already on CSD3: BakRep GFF3s for 8,710 of the 9,478 *Klebsiella* (`$R/raw/klebsiella_gff3`, from the Kp
  pipeline; also Bakta 1.9.4 / DB v5.1).
- Guard the protein step: every CDS contig must be found in the assembly and the protein count must equal the
  non-pseudo CDS count. A contig-name mismatch otherwise yields empty parquets that every existing check passes.
  (Built for PsA: `engine/genomes/bakta.py` `ingest`/`run`/`collate`, and the count check in
  `apps/pa/scripts/build_pa_embeddings.sh`; mechanics in `apps/pa/CLAUDE.md` *Annotation*. Reuse them.)
- ⚠ **ATB contig numbering can have gaps** (`contig00130`, `contig00132`, …) that our store renumbers
  consecutively, so a BakRep GFF3 can carry our sequences under other names: ingest refuses it (the PsA count is
  in PROJECT_STATE §3.6). The same applies to the 8,710 Kp files in `$R/raw/klebsiella_gff3`. Decide at Entero scale between our-run
  fallback (~1.3 core-h each) and mapping seqids by sequence.

**Stages** (each `sbatch` prints its resource line; checks count against the manifest, never against files that
exist):
1. *Plan* sources (`plan_genome_sources`, login node; ~2,400 ENA queries). ATB rows are fixed by priority; only
   the portal/shovill subset depends on genome size and `--prefer-reads`.
2. *Build* contigs + `mlst` (ATB download array first: ~650 tasks × 100 genomes; then portal and shovill).
   Shovill needs a per-genus genome size (one scalar mis-states depth by 15–40 % across these genera).
3. *Build gate*: completeness against the manifest; failures grouped by cause with rerun task indices.
4. *CheckM2*, sharded (~2,000 genomes per 32-core task; one job would take 42–56 h). CheckM2's DIAMOND works in
   250-genome chunks and each genome is predicted independently, so concatenated shards equal one run.
5. *Species check* with skani against a reference panel, then *collate*. The rule is a decision to make with
   the data (label × call crosstab): *Shigella* sits inside *E. coli* at species-level ANI, and the KpSC, the
   *S. enterica* subspecies and the *E. cloacae* complex need their own references.
6. *Annotation*: BakRep GFF3 where held; our Bakta otherwise.
7. *Proteins* → *ESM-C* (bf16, `--max-proteins 9000`, FLOTO-SL2-GPU; ~250 A100-h at the measured 2.8 ms per
   protein).

**Traps already found** (design review, 2026-11-19):
- ST numbers collide across MLST schemes: count and hold out by scheme:ST. `mlst` 2.35.0 has schemes for every
  genus here except *Morganella*.
- Chunk indices are positional over the manifest's sorted rows: never re-plan between a submit and its reruns.
- The CSD3 checkout is shared: array tasks run whatever `HEAD` is when they start. Pin long arrays to a SHA
  (an `EXPECT_SHA` check, or a dedicated worktree).
- The ENA cache never expires `__NO_RUNS__`: use a fresh cache per cohort.
- Logging in both embedding modules is silently lost (a `bacformer.pp` import configures the root logger first),
  so read throughput from tqdm or `sacct`.

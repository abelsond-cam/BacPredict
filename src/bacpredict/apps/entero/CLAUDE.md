# Enterobacterales AST (`apps/entero`)

The Enterobacterales adapter for the multi-species AMR work (PsA first, then Enterobacterales). **Status,
numbers and next steps live only in [`PROJECT_STATE.md`](../../../../PROJECT_STATE.md) §3.6**; this file holds
the mechanics and the agreed build design. The pipeline is the engine; this folder holds only run scripts and
Enterobacterales data files.

## Labels

[`scripts/build_entero_ast_labels.sh`](scripts/build_entero_ast_labels.sh) runs
`engine/ast_labels/build_binary_ast_labels` on a CSD3 login node (~1 min) into
`processed/enterobacterales/` under the AMR root (layout: `apps/pa/CLAUDE.md` *Where the data lives*). Same ten drugs and rule as PsA (see `apps/pa/CLAUDE.md`): EUCAST v15 from
the MIC, the portal's own call where there is no MIC, unitless MICs as mg/L. Portal only (AMR-GNN is PsA-only).
The genus list is every genus whose family maps to the order Enterobacterales in
`engine/ast_labels/breakpoints.py`, so the label filter and breakpoint resolution cannot disagree. Of those 21
genera, 9 have labelled genomes.

## Genome build — design (agreed 2026-11-19)

Scale: ~67.8k labelled genomes, 96.5 % of them in AllTheBacteria (ATB). The PsA engine
(`engine/genomes/`) is reused; PsA outputs must stay byte-identical (regression-check every engine change by
re-running PsA collate/plan into `$TMPDIR` and md5-ing against the live files).

**Where it goes** (layout: `apps/pa/CLAUDE.md` *Where the data lives*): contigs, Bakta, protein parquets and ESM-C
into the same flat per-BioSample stores as PsA (`raw/contigs/`, `raw/bakta/`, `processed/protein_sequences/`,
`processed/esm/`); manifests, shards, QC and the annotation table under `processed/enterobacterales/`. ⚠ Before
the first Enterobacterales protein parquet lands, give `esm_embeddings_array.sh` a sample list: it slices the
store's whole listing, so it would otherwise embed PsA and Enterobacterales together.

**Annotation is Bakta 1.9.4 + full DB v5.1, not Prodigal** (decision of record, PROJECT_STATE §6):
- BakRep v2 (<https://bakrep.computational.bio>) serves ATB's own Bakta run for ATB release 0.2 + the 2024-08
  increment. Its GFF3 embeds the contigs, and its contigs are byte-identical to ATB's (Bakta ran with
  `--keep-contig-headers`), so its coordinates sit on our ATB contigs.
- Per-sample files need one API GET to find the host bucket:
  `https://bakrep.computational.bio/api/v2/datasets/<BioSample>`. The full index is the export at
  `raw/bakrep/v2_export_20250612/bakrep-export.tsv.gz` under the AMR root. BakRep serves ~0.6 MB/s per stream (ATB's S3 ~6
  MB/s); keep concurrency modest (a 76-way pull hit SSL EOF throttling).
- On ingest, refuse any GFF3 whose `# Software:` / `# Database:` header is not `v1.9.4` / `v5.1, full`: a
  BakRep re-run with Bakta 1.12 is being prepared upstream.
- Genomes BakRep lacks get our own Bakta run of the same version and flags, after a reproduction gate against
  BakRep-held genomes. Pin 1.9.4: Bakta ≥ 1.10 writes pseudogenes as `pseudogene=` rather than `pseudo=True`,
  and the protein extractor filters on `pseudo` only.
- Already on CSD3: BakRep GFF3s for 8,710 of the 9,478 *Klebsiella* (`project_k/david/raw/klebsiella_gff3`, from the Kp
  pipeline; also Bakta 1.9.4 / DB v5.1).
- Guard the protein step: every CDS contig must be found in the assembly and the protein count must equal the
  non-pseudo CDS count. A contig-name mismatch otherwise yields empty parquets that every existing check passes.
  (Built for PsA: `engine/genomes/bakta.py` `ingest`/`run`/`collate`, and the count check in
  `apps/pa/scripts/build_pa_embeddings.sh`; mechanics in `apps/pa/CLAUDE.md` *Annotation*. Reuse them.)
- ⚠ **ATB contig numbering can have gaps** (`contig00130`, `contig00132`, …) that our store renumbers
  consecutively, so a BakRep GFF3 can carry our sequences under other names: ingest refuses it (the PsA count is
  in PROJECT_STATE §3.6). The same applies to the 8,710 Kp files in `project_k/david/raw/klebsiella_gff3`. Decide at Entero scale between our-run
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
   *S. enterica* subspecies and the *E. cloacae* complex need their own references. Stages 4–5: *QC* below.
6. *Annotation*: BakRep GFF3 where held; our Bakta otherwise. Built: *Annotation (Bakta)* below.
7. *Proteins* → *ESM-C* (bf16, `--max-proteins 9000`, FLOTO-SL2-GPU; ~250 A100-h at the measured 2.8 ms per
   protein).

## QC: CheckM2, skani, collate

The stages of [`scripts/build_entero_genomes.sh`](scripts/build_entero_genomes.sh) after the build:

| Stage | What it does |
|---|---|
| `refs` | Login node. Fetches every accession of [`species_reference_panel.tsv`](species_reference_panel.tsv) from NCBI Datasets into `raw/reference_genomes/<acc>.fna.gz` (as `build_pa_genomes.sh setup`); skips files already there; exits 1 unless every reference is on disk |
| `qc` | Needs `ARRAY_REPO` (a detached worktree at the SHA to run), exactly as `submit` does; there is no fallback to the shared checkout. Refuses while any `entero-build-genomes` or `entero-skani` job is queued or running, if `squeue` itself fails (never read as "none queued"), or while any buildable manifest genome has no shard row. Writes `genomes/qc/checkm2_list.tsv` (`Sample`, contigs path; every `ok` genome by its newest shard row, sorted, no header) and makes it read-only. Submits the CheckM2 array (`entero-checkm2`, `%40`) and the skani job (`entero-skani`), each with its resource line |
| `qc checkm2 [indices]`, `qc skani` | Rerun from the frozen list, e.g. `qc checkm2 3,17`. `qc` alone refuses once the list exists: task k takes rows by position. `qc skani` refuses while another `entero-skani` job is queued or running |
| `collate` | Refuses unless every CheckM2 task wrote its report and the skani job wrote both `qc/skani_dist.tsv` and `qc/skani_dist.refs.tsv`. Runs `collate_genomes --species-mode report --cohort cohort.csv` over every report, `qc/skani_dist.tsv`, and the references skani ran with: `qc/skani_dist.refs.tsv`, the job's own output (never the panel, nor `qc/skani_refs.tsv`, which is only the job's input). Writes `genome_qc.csv`, `species_crosstab.csv`, `st_counts_by_drug.csv` and **`embedding_input_unfiltered.csv`, not `embedding_input.csv`** (see *Collate in report mode*) |
| `status` | Also: CheckM2 tasks written and missing indices, listed genomes reported (CheckM2 `Name` normalised as collate reads it), genomes with a skani hit, and a skani result with no reference list beside it |

**The two engine scripts** (`engine/genomes/scripts/`; `run_genome_qc.sh` stays PsA's one-job QC):
- `run_checkm2_shard.sh`: array task k takes rows k·2,000+1 … of the list (the 2,000 is fixed in the driver, never
  overridden between `qc` and a rerun), links them as `<Sample>.fna.gz` on `/local`, and runs CheckM2 from
  `CHECKM2_BIN` (the `checkm2` pixi env's `bin/`, resolved at submit: no pixi in a task). It writes
  `qc/checkm2/checkm2_quality_report_<k, 5 digits>.tsv` atomically. Exits 3 if the checkout moved (`EXPECT_SHA`),
  4 on an empty slice or a bad list, 5 on an unreadable contigs file, and 6 if CheckM2 reported fewer genomes
  than it was given (that report is still written; the missing ones are named after normalising CheckM2's `Name`
  as collate does, a trailing `.fna`/`.fna.gz` dropped). Shape: 32 cores, 100 GB, 4 h on `icelake`, from PsA's
  ~1.6 core-min per genome; `QC_PARTITION`, `QC_MEM`, `QC_TIME` override it. CheckM2's memory has never been
  measured: read sacct MaxRSS on the first tasks.
- `run_skani.sh`: one job, `skani dist` of every listed genome against every panel reference, in slices of
  5,000 genomes (`QCHUNK`) so memory stays bounded, concatenated under one header into `qc/skani_dist.tsv`. It
  copies `REFS_TSV` when it starts and publishes that copy as `qc/skani_dist.refs.tsv` in the same step as the
  result (old list removed, then result and list moved in), so the result always carries the reference list it
  was run with; a job killed between the moves leaves no list, which collate refuses. `collate_genomes` stops if
  the skani table names a reference that is not in the list it was given.
- Both run under `ARRAY_REPO`, which `qc` requires exactly as `submit` does (the shared checkout's `HEAD` moves
  under queued tasks); neither runs repo code.

**Collate in report mode** (`engine/genomes/collate_genomes.py`; PsA's target mode is unchanged):
- No species rule: species never fails QC, so the QC pass set is not species-filtered. It is written to
  `genomes/embedding_input_unfiltered.csv`; **`genomes/embedding_input.csv` (the embedding step's input) is not
  written until a species rule is chosen** and collate runs in target mode (`--species-mode target
  --target-species …`, as PsA).
- `genomes/species_crosstab.csv`: the label species (`cohort.csv` `species`) × the called species, over every
  built genome: `n`, `n_qc_pass`, `n_species_conflict` (the portal's two tables disagree), `n_no_skani_hit`,
  quantiles of the best ANI (`best_ani_min`, `_q01`, `_q05`, `_q25`, `_median`, `_q75`, `_max`), and
  `gap_to_runner_up_median` (best species' ANI minus the next species').
- `genome_qc.csv` gains `species`, `species_genotype`, `species_conflict` from `cohort.csv`, and one `ani_<name>`
  column per panel species name, the best over that name's references (*E. coli* has two).
- `st_counts_by_drug.csv` keys `st` as `<scheme>:<ST>` whenever more than one `mlst_scheme` is present; an untyped
  genome is `<scheme>:-` (`-:-` where mlst found no scheme), never a holdout candidate.
- A built genome with no CheckM2 row stops collate, naming it (`--allow-incomplete` to inspect).
- ⚠ A genome that becomes `ok` after the list is frozen (a rerun build) has no CheckM2 row; there is no top-up
  stage yet.

**The panel** ([`species_reference_panel.tsv`](species_reference_panel.tsv)) is a **DRAFT for David to decide**:
39 references, each an NCBI reference genome or, for subspecies (which NCBI does not give reference genomes),
an NCBI type-material assembly. Its header holds the NCBI Datasets commands each row was verified with and the
gaps it leaves.

**Traps already found** (design review, 2026-11-19):
- ST numbers collide across MLST schemes: count and hold out by scheme:ST. `mlst` 2.35.0 has schemes for every
  genus here except *Morganella*.
- Chunk indices are positional over the manifest's sorted rows: never re-plan between a submit and its reruns.
- The CSD3 checkout is shared: array tasks run whatever `HEAD` is when they start. Pin long arrays to a SHA
  (an `EXPECT_SHA` check, or a dedicated worktree).
- The ENA cache never expires `__NO_RUNS__`: use a fresh cache per cohort.
- Logging in both embedding modules is silently lost (a `bacformer.pp` import configures the root logger first),
  so read throughput from tqdm or `sacct`.

## Annotation (Bakta)

[`scripts/annotate_entero_genomes.sh`](scripts/annotate_entero_genomes.sh) runs `engine/genomes/bakta.py` as PsA
does, into the same flat store `raw/bakta/` (keyed by BioSample). The release check, what "current" means, the
reproduction gate and the AMRFinderPlus DB are in `apps/pa/CLAUDE.md` *Annotation*. Order: `plan` → `ingest`
(with `submit planned` beside it) → `status` → `submit refused` → `collate`.

| Stage | What it does |
|---|---|
| `plan` | Reads the build shards, so it runs before QC. Writes `genomes/bakta/built_genomes.csv` (every built genome: `Sample`, `sr_assembly_file`, `source`; where a stale shard disagrees, the manifest's source wins) and `bakrep_roster.tsv` (built ATB genomes listed in `in_bakrep_ids.txt`). Re-run it as more shards land |
| `ingest` | One CPU job (icelake, 8 cores, 16 GB, 24 h; 8 BakRep streams) from `ARRAY_REPO`. Puts BakRep's GFF3s in `raw/bakta/` and writes every roster genome's outcome to `genomes/bakta/ingest_status.tsv` (`Sample`, `status`, `detail`). Resumable: a current ingest record is kept, not re-fetched. Exits 1 only if any outcome is `error:*` |
| `submit [all\|planned\|refused]` | Writes a new, dated, read-only `genomes/bakta/run_roster_<YYYYmmdd_HHMMSS>.tsv`, then runs the array `engine/genomes/scripts/run_bakta_array.sh` from `ARRAY_REPO`: 4 genomes per task, at most 100 tasks at once, 8 cores, 52 GB, icelake-himem, 3 h, keeping gff3, faa and txt |
| `collate` | CPU job (icelake, 16 cores, 32 GB, 2 h). Writes `genomes/bakta_annotation.csv` for `COHORT`, but only if every genome has exactly one current annotation (checked by md5). Otherwise it writes `bakta_annotation_problems.csv` and exits 1 |
| `status` | Counts by ingest outcome; roster genomes without an outcome; `COHORT` genomes that are ingested, have our run record, have both, or have neither; the dated rosters; any `entero-bakta*` job queued or running |

**Ingest outcomes** (`INGEST_STATUSES` in `bakta.py`; exactly one per roster genome):

| Status | Meaning | Then |
|---|---|---|
| `ingested` | A current ingest record stands. The detail says `fetched` (this run) or `current record kept` (an earlier run) | annotated |
| `absent` | BakRep has no GFF3: an API 404, or no `gff3` among its files | our run |
| `refused:release` | The GFF3 header is not `# Software: v1.9.4` / `# Database: v5.1, full` | our run |
| `refused:names` | The embedded contigs are ours under other names (the same multiset of sequences). This is the ATB contig-numbering-gap case | our run, unless seqids are later mapped by sequence |
| `refused:content` | The embedded sequences are not ours | our run |
| `skipped:run` | Our own `.bakta.run.json` exists. Ingest never fetches over, or replaces, our run's GFF3 (it checks before the download and again before the file goes in place) | already ours |
| `error:transient` | BakRep's API or the download failed after retries, or BakRep served a truncated or corrupt GFF3 | re-run ingest |
| `error:other` | Anything else: our contigs missing or unreadable, contigs changed during ingest, a bug (a `KeyError` from a changed API shape lands here, not in `absent`) | investigate, re-run |

The `refused:names` count is what decides between running our Bakta on those genomes (~1.3 core-h each) and mapping
BakRep's seqids to ours by sequence (PROJECT_STATE §3.6 *Next*).

**Submit modes.** Job names are `entero-bakta-planned`, `-refused` and `-all`.
- `planned`: `COHORT` genomes BakRep cannot cover (not ATB-sourced, or ATB but not in `in_bakrep_ids.txt`). It does not
  read the ingest status, so it can run while ingest does.
- `refused`: BakRep-roster genomes that ingest did not take (`absent`, `refused:*`, `skipped:run`). It refuses to
  submit while ingest is queued or running; if any `COHORT` genome on the roster has no outcome (re-run `plan`, then
  `ingest`); and if any outcome is `error:*` (re-run `ingest`).
- `all` (the default): both sets.
- Every mode leaves out genomes whose run record has this env's fingerprint and is newer than their contigs. This
  costs one directory listing plus one JSON read per such genome, with no md5 pass on the login node; `collate`
  re-checks every genome by md5.
- No mode submits unless `bakta fingerprint` of the env and database equals the fingerprint in a PsA run record (the
  first genome in PsA's `run_roster_with_refused.tsv`). The two species share one annotation of record.
- Overlap guards: a submit refuses while an overlapping array is queued or running. `planned` and `refused` are
  disjoint, so they may run together. `ingest` refuses while a `refused` or `all` array is live.

**`COHORT`** (default `genomes/embedding_input.csv`; needs `Sample`, `sr_assembly_file`) says which genomes `submit`
and `collate` cover. Before species QC exists, use `COHORT=$GEN/bakta/built_genomes.csv`. That spends Bakta on genomes
QC will later fail (a few per cent in PsA: PROJECT_STATE §3.6 *genome QC*). Re-collate on `embedding_input.csv` once
QC lands: `collate` overwrites `genomes/bakta_annotation.csv`, which is the protein stage's input.

**`ARRAY_REPO`** is required for `ingest` and `submit`: a detached worktree at the SHA to run, with `.venv` symlinked
to the shared env (as for PsA). Each `--wrap` checks at start that the worktree is still at the submitted SHA and puts
`$ARRAY_REPO/src` first on `PYTHONPATH`. The array passes `EXPECT_SHA` to `run_bakta_array.sh`. `collate` uses
`ARRAY_REPO` if it is set, otherwise this checkout, with the same guard.

Ingest downloads every GFF3 from BakRep, including those for *Klebsiella* genomes whose BakRep files are already on
project_k (`david/raw/klebsiella_gff3`).

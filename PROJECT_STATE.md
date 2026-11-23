# BacPredict — project state

> **Last verified: 2026-11-22 @ `19065b7`** (branch `refactor/consolidate-engine`).
> **Verification scope, this pass (2026-11-22, thirteenth): §3.6 label noise at the breakpoint.** Every PsA AST
> test re-banded by `d = log2(MIC / r_gt)` from `ast_tests_called.parquet`, reading the `s_le_mg_l` column no
> other code reads; the reading that EUCAST v15 gives PsA no S category for 7 drugs checked against
> `label_provenance.csv` (`n_mic == 0`) and confirmed for all 7; the banding validated by regenerating the
> previously-uncommitted `near_breakpoint_by_drug.csv` exactly. Saved holdout predictions then re-scored by band
> — no retraining — through `engine.finetune.stratified_metrics`' own helpers. All 10 PsA models now landed.
>
> **Twelfth pass (2026-11-22): §3.6 the PsA holdout's clonal similarity and the lookup
> baseline.** Every held-out PsA genome measured against all 2,290 training genomes with `skani dist` (job
> `41644484`), then the nearest-labelled-neighbour lookup scored on the same held-out sets through
> `engine.splits.load_splits` and `engine.finetune.metrics.compute_full_metrics` — the same readers and the same
> metric function the fine-tunes used, so baseline and model are comparable. Model AUROCs read from the six
> `paeruginosa/models/<drug>/results.json` that exist; four drugs are still training.
>
> **Eleventh pass (2026-11-22): §3.6 the ESM-C completion, duplicate isolates across all
> three cohorts, and what the
> splits builder does with `--exclude`/`--groups`.** Read on CSD3: `enterobacterales/genomes/genome_qc.csv` for the
> `(total_length, n_contigs)` collisions, then every candidate pair's contigs re-read from `raw/contigs/` and
> hashed sequence-only (headers stripped, orientation canonicalised); the ten `paeruginosa/splits/<drug>_split.csv`
> for where the two PsA candidate pairs actually sit. The builder's re-draw behaviour read from
> `build_amr_holdout_splits.py`. Both cohorts then re-run through
> `engine/genomes/find_duplicate_assemblies.py` at `c94df7f` from a detached worktree, reproducing the hand
> result exactly — and the run is what caught the first version's key being wrong (it returned nothing on PsA).
> **No list was written and no split table was built or rebuilt**; the duplicate policy is still open with David.
>
> **Verification scope, the pass before (2026-11-22, tenth): §3.7 the PsA v2 genome collate and the BakRep
> coverage of both v2 plans only.** Read on CSD3: `paeruginosa_cabbage_v2/genomes/{genome_qc,embedding_input,
> reused_genomes,genome_manifest}.csv` (collated at `d6c3e05`), `reuse_built_genomes status` for both cohorts, sacct of
> the v2 CheckM2 and skani jobs, and BakRep's export joined to both v2 `genome_manifest.csv`. **Nothing else was
> re-verified this pass.**
>
> **Verification scope, the pass before (2026-11-21, ninth): §3.6 the species units and the Klebsiella-only arm.**
> `apps/entero/species_units.py` run against `genomes/genome_qc.csv` on CSD3 (the seven-way unit table, and the
> 9,370 `kpsc` genomes); ST37's per-drug counts read from `genome_qc.csv` × `binary_ast.csv`; the fp32-vs-bf16
> question settled by loading one genome (`SAMD00497419`) from both the linked store and the ESM-C smoke output and
> finding all 5,330 protein rows bit-identical after a bf16 cast; 25 tests pass. **The Kp arm has not been built or
> trained**, and the per-drug split figures quoted for it are an in-memory preview, not tables of record.
> **Nothing else was re-verified this pass.**
>
> **Verification scope, the pass before (2026-11-21, eighth): §3.6 the *E. aerogenes* MLST correction only.**
> `genome_qc.csv` re-collated on CSD3 after job `41608859` and checked species by species for a single MLST scheme
> (1 trips it, *E. chengduensis*, 7 typed + 1 untyped — no species carries two real schemes).
> **Nothing else was re-verified this pass.**
>
> **Verification scope, a pass by the v2 agent (2026-11-21, seventh): §3.7 the CABBAGE v2 labels only** — a pass by the v2 agent.
> Read on CSD3: `$AMR/processed/{paeruginosa,enterobacterales}_cabbage_v2/{manifest.json,label_counts.csv,cohort.csv}`
> and their `checks_vs_2026-07/`. **Nothing else was re-verified this pass.**
>
> **Verification scope, the pass before (2026-11-21, sixth): §3.6 the PsA ST155 switch only** — the ten new split tables
> and `check-splits` read live on CSD3, the preserve-move of the ST235 arm verified by md5 and a per-drug file
> listing, and the retrain array submitted (no results yet). **Nothing else was re-verified this pass.**
>
> **Verification scope, a pass by the v2 agent (2026-11-21, fifth): §3.7 CABBAGE v2 only** — a separate pass by the v2 agent.
> Read on CSD3: `$AMR/raw/ebi_amr_portal/cabbage_v2_20261121/md5sums` and `$AMR/processed/ebi_amr_portal/scoping_cabbage_v2_20261121/`
> (`report.txt`, its CSVs, `puffin/`). **Nothing else was re-verified this pass.**
>
> **Verification scope, the pass before (2026-11-21, fourth): §3.6 the per-species MLST re-type and re-collate, and the
> re-typed held-out-ST candidates, only.** Read on CSD3: sacct of jobs `41604641`/`41604643`/`41604644`/`41604645`,
> `genomes/retype_mlst/shards/*/`, and `genomes/genome_qc.csv` diffed against `genomes/_pre_retype/genome_qc.csv`.
> **Nothing else was re-verified this pass.**
>
> **Verification scope, the pass before that (2026-11-21, third): §3.6 Enterobacterales annotation, proteins, the Kp ESM-C
> link, the ESM-C submission, and the MLST scheme audit only.** Read on CSD3: `genomes/{bakta_annotation_embedding_input_unfiltered.csv,
> esm_linked_kp.tsv,esm_samples_to_embed.txt}`, the collate / check-proteins / link-kp / smoke / check-smoke logs,
> sacct of the Bakta, protein, link and smoke jobs. **Nothing else was re-verified this pass.**
>
> **Verification scope, the pass before (2026-11-21, at `9a813f0`): §3.6 the Enterobacterales build, QC, species
> calls and BakRep ingest only.** Read on CSD3: `build_entero_genomes.sh status`, `genomes/{genome_qc,embedding_input_unfiltered,
> species_crosstab}.csv`, `genomes/bakta/ingest_status.tsv`, sacct of the CheckM2/skani/Bakta jobs.
> **Nothing else was re-verified this pass.**
>
> **Stamp before this one: 2026-11-20 @ `82ca4f2`.**
> **Verification scope, this pass (2026-11-20, fifth): §3.6 PsA ESM-C completion and the Enterobacterales jobs in
> flight only.** Read on CSD3: sacct of array `41526902`; `build_pa_embeddings.sh check` at `4ff29d7`; the store
> count and one file's size; squeue/sacct of the Enterobacterales arrays. **Nothing else was re-verified this pass.**
>
> **Stamp before this one: 2026-11-20 @ `4ff29d7`.**
> **Verification scope, this pass (2026-11-20, fourth): §3.6 Enterobacterales genome sources and build shape, PsA
> proteins, and the ampere torch crash only.** Read on CSD3: `enterobacterales/genomes/genome_manifest.csv` (plan at
> `6369d50`) joined to `bakrep_coverage_20261119/in_bakrep_ids.txt`; `shards_smoke/atb+portal_gff_00000.tsv`; sacct of
> arrays `41519203` and job `41519198`; the `check-proteins` output at `4ff29d7`; the logs of jobs `41520932`,
> `41525227`. **Nothing else was re-verified this pass.**
>
> **Stamp before this one: 2026-11-20 @ `0f5cc5d`.**
> **Verification scope, this pass (2026-11-20, third): §3.6 PsA Bakta annotation completion and the move of every
> §3.6 artifact to `$AMR` only.** Read on CSD3: sacct of array `41499745`; `bakta_annotation.csv` re-collated at
> `0f5cc5d` (md5 `53bb2fa5…`); the migration log `/rds/user/dca36/hpc-work/logs/amr_data_migration/`
> (16 directories renamed, 48,623 paths rewritten in 4,203 records, 0 left, 11,244 of 11,244 pre-move paths
> present after). Paths in the stamps below are pre-move (§2 maps them). **Nothing else was re-verified this pass.**
>
> **Stamp before this one: 2026-11-20 @ `b7cd555`.**
> **Verification scope, this pass (2026-11-20): §3.6 Bakta reproduction gate, PsA BakRep ingest and the Bakta
> environment/database only.** Read on CSD3 from `processed/bakta_vs_bakrep_validation/bakta_gate_summary.csv`
> (md5 `0ed46bbf…`; array `41495079` at `be9891b`, compared at `b72b500`) and its empty `bakta_gate_diffs.csv`,
> sacct of `41495079`, the ingest log `/rds/user/dca36/hpc-work/logs/pa_bakta_ingest_20261120_103009/`
> (job `41497655`), `train_pa_ast/genomes/bakta/*.tsv` and `raw/bakta_db/v5.1_full.md5`.
> **Nothing else was re-verified this pass.**
>
> **Verification scope, this pass (2026-11-20, second): §3.1 the first cohort-scale attribution
> result only**, at `23ebf7d`. Every number read from the four tables under
> `$B/train_tb_ast/attribution/stats_ig_cohortmean/` written by that commit's
> `explain.attribution_statistics` over `rifampin_ig_cohortmean_annotated.parquet` (399,691 rows,
> 98 genomes), and from `attribution/rpoB_{ser450leu,wt}_substitution.csv`. A separate pass from the
> §3.6 one above, by a different agent. **Nothing else was re-verified this pass.**
>
> **Stamp before this one: 2026-11-19 @ `480ce9d`.**
> **Verification scope, that pass (2026-11-19, fourth): §3.6 Enterobacterales labels, BakRep coverage, the
> CheckM2 scikit-learn check and the PsA ESM-C smoke only.** Read on CSD3 from
> `train_entero_ast/{label_counts,cohort,label_provenance}.csv` (built at `480ce9d`; md5 `2bbb7558…`,
> `84b67e1f…`), `processed/bakrep_coverage_20261119/join_output.txt` (the join re-run on CSD3 against the
> staged BakRep export, md5 `95e43aa1…`), `train_pa_ast/genomes/qc/checkm2_sklearn_check/` and the logs of
> jobs `41397962`, `41399821`, `41405814`. **Nothing else was re-verified this pass.**
>
> **Stamp before this one: 2026-11-19 @ `e46c81e`.**
> **Verification scope, that pass (2026-11-19, second): §3.6 genome build, QC and ST counts only.** Read on
> CSD3 from `train_pa_ast/genomes/`: `genome_qc.csv`, `embedding_input.csv` and `st_counts_by_drug.csv`
> (written by `build_pa_genomes.sh collate` at `e46c81e`; md5 `b5a8b03f…`, `f6de3cfa…`, `65c09abe…`), the
> 439 build shards and `qc/{checkm2_quality_report,skani_dist}.tsv`. Every number was recomputed from the
> raw inputs by four independent agents and reconciled by a fifth (0 mismatches in QC pass/fail or in the
> 4,936 ST × drug rows). **Nothing else was re-verified this pass.**
>
> **Verification scope, this pass (2026-11-19, third): §3.1 gene-level attribution only**, at
> `9906110`. Every number re-read from the artifact named beside it — the four
> `attribution/diag/method_comparison/*_method_comparison.json`, the
> `attribution/diag/class_baselines/class_baseline_report.json`, and the job logs
> `attr-degeneracy-diag-41324596.out` / `attr-method-compare-41326249_[0-3].out`. Independently
> audited against those artifacts by a second agent, which corrected six errors in the first draft
> (an n=50/n=200 value mix, the ablation sign convention, the bf16 ULP, the scope of the gradient
> degeneracy, and two over-claims). The bf16 ULP line in §3.1 was **wrong and is corrected here**.
> **Nothing else was re-verified this pass.**
>
> **Stamp before this one: 2026-11-19 @ `c03e93c`.**
> **Verification scope, that pass (2026-11-19): §3.6 genome sources only.** Read on CSD3 from
> `train_pa_ast/genomes/genome_manifest.csv` (plan at `1f049c4`) and the download array's shards in
> `genomes/shards/`. The shovill observations come from two smoke genomes and one debug job; their logs
> were not kept. **Nothing else was re-verified this pass.**
>
> **Stamp before this one: 2026-11-18 @ `7e47486`.**
> **Verification scope, that pass (2026-11-18): §3.6 only** — the new multi-species AMR layer. Every
> §3.6 figure was read on CSD3 from `$R/bac_ast_prediction/processed/train_pa_ast/` (`label_counts.csv`,
> `cohort.csv`, `call_agreement.csv`), written by `apps/pa/scripts/build_pa_ast_labels.sh` at `7e47486`;
> the same build run locally on md5-identical inputs gave byte-identical outputs. **Nothing outside §3.6
> was re-verified this pass** — every other section carries the scope of the stamps below.
>
> **Stamp before this one: 2026-08-30 @ `034edb5`.**
> **Verification scope, that pass (2026-08-30): §3.3 TB only, Phase 1.** Every TB figure below was read
> on CSD3 this session from the artifact named beside it — `tbprofiler_parse_manifest.json`, the two
> cluster manifests, `assembly_refs.txt`'s resolution manifest, the ten `<drug>_split.csv` re-counted,
> the `drugs` column of `tbprofiler_variants.parquet`, `sacct` for job `34597335`, and md5 against the
> June parse. **This pass corrected two figures the previous stamp asserted** — the cohort is 36,692 and
> nothing fails to resolve; "36,389/36,390" was read off the deprecated tree. Treat that as the standing
> hazard here: there are two `train_tb_ast` trees with different cohort sizes. Phase 0's code claims are
> backed by the test suite (900 passing; 14 pre-existing `tests/docs/` failures belong to another agent's
> in-flight package deletions).
> **NOT re-verified this pass** — carried forward from the stamps below, and each still carrying
> whatever caveat it carried then: all 32 fine-tune numbers, both catalogue ceilings, the invasion
> GWAS results, §3.4 and §3.5. The Kp §3.3 numbers were verified on 2026-08-28 in the preceding
> session (see that stamp).
>
> **Earlier verification scopes, retained as the audit trail:**
> **2026-08-14 @ `23bf069`:** repo tree read directly; all 32 fine-tune numbers read from each
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
> Added 2026-09-04 (§3.4 cohort composition): computed by `sublineage_composition` from the
> cohort split CSV joined to `models/cohort_scores.npz`, join asserted total with label==y_true and
> identical splits across all 14,119 genomes; threshold read from `model_comparison_thresholds.json`.
> Added 2026-09-04 (§3.4 O1 correction): per-sublineage evaluate AUROCs recomputed from
> `cohort_scores.npz` + the cohort split CSV, with the recipe validated by reproducing the two
> figures already of record (SL258 0.858, SL307 0.815) before the SL3010 figure was believed.
> Added 2026-09-01 (§3.4 pick list): per-sublineage counts read from the regenerated
> `model_comparison_shortlist_meta.json`; `model_comparison_thresholds.json` re-checked unchanged
> (md5 d4542259…), so the Youden points and the reproduce-before-use gate still stand; Section 5's
> `model_comparison_shortlist_overall.csv` diffed byte-identical against the published version.
> Added 2026-08-19 (§3.4 Q1): unitig+sublineage numbers read from that run's own
> `unitig_model_results.json`; the plain-vs-+SL score agreement and per-sublineage AUROCs
> recomputed from both `unitig_cohort_scores.npz` archives, which were checked to cover the same
> genomes, labels and splits before being compared.
> Added 2026-08-28 (§3.3 holdout scoring rule): the rule (A)-vs-(B) discordance numbers were measured
> on CSD3 this pass, not carried over — 40 ertapenem genomes scored by both rules and diffed against
> GGCAT's own colouring in `hits_submatrix.tsv`, read from that run's `rule_discordance.json`. The
> 22/22 `.assoc` and 21/22 `lr/results.json` counts were listed on disk. Not re-verified this pass:
> every fine-tune number, both catalogue ceilings, and the invasion GWAS results.
> Added 2026-08-28 (§3.3 TB): every TB figure below was listed or read on CSD3 this pass — the ten
> split tables and their holdout counts, the assembly resolution (36,389/36,390), and the 36,684
> TB-Profiler `results.json` (one of which was opened to confirm it carries `main_lineage`/
> `sub_lineage`). The memory bracket is an *extrapolation*, explicitly labelled as such, and is
> the thing wave 3a exists to replace. No TB compute has been spent.
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
- **Multi-species AMR** (from 2026-11) — every genome in the EBI AMR Portal: *P. aeruginosa* first,
  then all Enterobacterales, then all species, on one drug panel of ten (§3.6). A **second experiment** runs the
  same procedures on **CABBAGE v2**, an extension of the portal's data at about twice the genomes (§3.7).

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
`segment_amr_lr` · `concat` · `ref_catalogues` · `explain` · `plots` · `config.py` · `scripts`.

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

**Data paths moved 2026-11-20** (multi-species AMR, §3.6; David: one root on project_k). Stamps and job logs
older than that name the old paths; this is the map. `$AMR` = `project_k/david/amr_prediction`.

| Referenced as (`$R` = `project_k/david`) | Now |
|---|---|
| `$R/raw/ast_cohort_genomes/{contigs,prodigal,bakta}/` (also spelled `$R/bac_ast_prediction/raw/…`) | `$AMR/raw/{contigs,prodigal,bakta}/` |
| `$R/raw/{bakta_db,bakrep,ebi_amr_portal,allthebacteria,reference_genomes}/` | `$AMR/raw/` + the same name |
| `$R/raw/amr_gnn_nguyen2026/` | `$AMR/raw/paeruginosa/amr_gnn_nguyen2026/` |
| `$R/bac_ast_prediction/processed/train_pa_ast/` | `$AMR/processed/paeruginosa/` |
| `…/train_pa_ast/{protein_sequences_prodigal,esm_smoke_prodigal}/` | `$AMR/processed/_superseded_prodigal/paeruginosa_{protein_sequences,esm_smoke}/` |
| `$R/bac_ast_prediction/processed/train_entero_ast/` | `$AMR/processed/enterobacterales/` |
| `$R/bac_ast_prediction/processed/{bakrep_coverage_20261119,bakta_vs_bakrep_validation}/` | `$AMR/processed/` + the same name |
| `$R/processed/ebi_amr_portal/` | `$AMR/processed/ebi_amr_portal/` |

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

**In flight.** **Gene-level attribution** of the two single-determinant fine-tunes — TB rifampin and
Kp ciprofloxacin — to test whether the models read `rpoB` and `gyrA`/`parC` rather than lineage.
Stages 1–5 written and unit-tested (63 tests under `tests/engine/explain/`). **TB rifampicin has a
cohort-scale result at n=98 and its lineage control is read** (below). **Kp ciprofloxacin has
nothing** — no attribution value, no `gyrA` number, no test. Plan:
`~/.claude/plans/i-want-to-run-polymorphic-oasis.md`.

**★ First cohort-scale attribution result — TB rifampicin, 98 genomes, 2026-11-20.** IG against the
cohort-mean baseline at `n_steps=200`, fp32, `checkpoint-31000`. All four tables under
`$B/train_tb_ast/attribution/stats_ig_cohortmean/`, from
`rifampin_ig_cohortmean_annotated.parquet` (399,691 rows). The cohort file asks for 100; **2 genomes
are missing** — both their fill jobs died with `Illegal instruction` (exit 132) at process start on
two different `ampere` nodes, before Python ran.

**(a) Ranking — which genes the model attends to**, mean within-genome normalised |attribution|,
one-sided Wilcoxon, BH-adjusted, `min_genomes=20`, 2,196 genes tested
(`gene_attribution_ranking.csv`):

| gene | n_genomes | mean z | median z | median copies | p_adj |
|---|--:|--:|--:|--:|--:|
| **rpoB** | 98 | **38.19** | 39.97 | 1 | 3.34e-15 |
| pncA | 92 | 23.87 | 22.59 | 1 | 8.41e-15 |
| ethA | 98 | 17.05 | 16.37 | 1 | 3.34e-15 |
| rpoC | 98 | 6.57 | 5.14 | 2 | 3.34e-15 |
| pirG | 93 | 5.62 | 3.56 | 3 | 9.20e-14 |
| cobH | 93 | 4.47 | 3.48 | 1 | 6.32e-15 |
| — negative controls — | | | | | |
| rpoA | 98 | **−0.049** | | 1 | 1.00 |
| katG | 96 | **−0.055** | | 1 | 1.00 |
| gyrA | 98 | **−0.058** | | 1 | 1.00 |
| inhA | 97 | **−0.070** | | 1 | 1.00 |

`rpoB` sits ~38 SD above its genome's average gene. The four controls — the other polymerase
subunit, and the determinants of three *other* drugs — sit at zero. **36.2% of total |attribution|
is in genes Bakta could not name** (genome-specific locus tags, unpoolable across genomes); that
share is the gap only an annotation-free gene family can reach.

**(b) The resistant-vs-susceptible contrast must be run SIGNED. `|·|` inverts it.** Decision of
record 2026-11-20, taken with David. Split on the **AST label**, never on variant carriage — the
label is the only thing known for a phenotype without a catalogue. Signed scores are used **raw**:
the cohort-mean baseline's `f(baseline)` is genome-independent to within 0.004 logits, so genomes
already share an origin. From `gene_resistant_vs_susceptible_signed.csv` and
`gene_resistant_vs_susceptible_abs_z.csv` (31 R / 67 S):

| gene | median R | median S | **signed** Δmean | rank-biserial | signed p_adj | `\|·\|` Δ | `\|·\|` p_adj |
|---|--:|--:|--:|--:|--:|--:|--:|
| **rpoB** | **+3.85** | **−10.17** | **+13.48** | **0.958** | <1e-15 | **−11.41** | 8e-4 |
| pncA | +0.85 | −6.67 | +7.13 | 0.795 | <1e-15 | −9.88 | 0.024 |
| ethA | −0.34 | −4.67 | +4.39 | 0.786 | <1e-15 | −6.76 | 0.004 |
| rpoC | +0.99 | +1.24 | −0.33 | −0.274 | 0.043 | +3.19 | 0.001 |
| katG | −0.004 | +0.009 | −0.013 | −0.869 | <1e-4 | −0.015 | 0.012 |
| rpoA | −0.001 | −0.004 | +0.002 | 0.213 | 0.121 | +0.015 | 0.011 |

Wild-type `rpoB` carries a **large negative** attribution and mutant `rpoB` a smaller positive one,
so the absolute value ranks susceptible genomes higher and the contrast comes out **confidently
backwards** on all three top genes. This is an inversion, not a weakened effect, and it is pinned by
a regression test (`test_absolute_value_inverts_the_contrast_the_signed_score_recovers`). The
ranking in (a) is unaffected and unchanged — direction is irrelevant there.

**(c) The lineage control — read, and the signal survives it.** `gene_resistant_vs_susceptible_
within_lineage_signed.csv`; only lineage2 and lineage4 have ≥5 genomes per class (lineage3 n=12 is
not testable at that floor):

| gene | lineage | n R / n S | median R | median S | Δ | disjoint | p_adj |
|---|---|---|--:|--:|--:|---|--:|
| **rpoB** | lineage2 | 14 / 13 | +4.08 | −8.37 | **+12.80** | **yes** | 1.0e-4 |
| **rpoB** | lineage4 | 13 / 36 | +3.20 | −10.14 | **+12.81** | no | 1.0e-4 |
| pncA | lineage2 | 11 / 13 | +0.75 | −3.99 | +4.31 | no | 1.9e-3 |
| pncA | lineage4 | 11 / 35 | +1.04 | −9.86 | +10.24 | no | 2.0e-4 |
| ethA | lineage2 | 14 / 13 | +0.07 | −3.91 | +4.10 | no | 7.0e-4 |
| ethA | lineage4 | 13 / 36 | −0.53 | −6.06 | +5.01 | no | 1.0e-4 |

`rpoB`'s pooled effect of +13.48 is reproduced almost exactly inside each lineage (+12.80, +12.81),
and in lineage2 the two classes **do not overlap at all**. This is the control the whole analysis
turns on, and it is the first time it has been run on anything in this programme's attribution arm.

**⚠ Read `lineage_control_summary.csv` by effect size, never by the flag.** 1,410 of 2,191 testable
genes carry `survives_stratification = True`, which sounds like a failure of specificity and is not:
at n=98 a consistent shift of 0.006 logits reaches BH significance. Among those 1,410 the **median
|effect| is 0.0063 logits**, only **13 exceed 0.1**, and **exactly 3 exceed 1.0** — `rpoB` (13.48),
`pncA` (7.13), `ethA` (4.39). `katG` is the clean illustration: it "survives" in both lineages at an
effect of **−0.013 logits**. `main()` now logs this distribution beside the count so the number
cannot be quoted without its scale.

**⚠ Multi-copy genes are collapsed to one value per genome before any test** (summed — IG
attributions are additive; `--multi-copy max|mean` available). Counting rows had made `n_genomes`
read 190 for `rpoC` and 331 for `pirG` across 98 genomes, and two annotated copies in one genome are
not two independent observations. 26,267 (genome, gene) pairs were collapsed.

**⚠ What this does NOT establish.** `pncA` and `ethA` are the determinants of *pyrazinamide* and
*ethionamide*, and they were contrasted on the **rifampicin** label. Rifampicin resistance is
usually MDR, so co-resistance and genuine reading are both live and untested; David's reading is
that the model is still partly using population association. Distinguishing them needs the same
signed contrast run against each gene's own drug label. Attribution remains **descriptive** — it
says where the model's output is sensitive, not why.

**WT-substitution counterfactual — `rpoB` Ser450Leu, measured 2026-11-20**, n=100 TB genomes,
`$B/train_tb_ast/attribution/rpoB_{ser450leu,wt}_substitution.csv`. One protein embedding is
replaced and the genome re-predicted; nothing else changes. The full 2×2:

| donor | recipient | n | median donor distance | median Δlogit | flipped |
|---|---|--:|--:|--:|--:|
| **Ser450Leu** | **non-carrier** | 67 | 0.00719 | **+4.804** | **28/67** |
| **wild-type** | **carrier** | 33 | 0.00713 | **−0.204** | **4/33** |
| Ser450Leu | carrier (already has it) | 33 | 0.00370 | +0.018 | 1/33 |
| wild-type | non-carrier (already WT) | 67 | 0.00026 | −0.008 | 0/67 |

The two informative cells are the first two: **matched donor distance** (0.0072 vs 0.0071), opposite
directions. The bottom two are null controls — substituting a protein for one the genome effectively
already has — and both land at ~0, as they must. ⚠ **A previous note in this file claimed donor
distance was constant at 0.00719 across both arms. It is not** (0.00026 for WT→non-carrier), and
that cell's null result is near-tautological; the WT→carrier row is the real converse test.

**⚠ The asymmetry is a hypothesis, not a result.** Adding the mutation flips 42% of susceptible
genomes; removing it flips only 12% of resistant ones. Consistent with resistant genomes carrying
further rifampicin determinants, or with the model reading other correlates — untested either way.
Flippers and non-flippers occur in every lineage; what separates them is the genome's starting logit
(p=1.7e-07) and its response magnitude (p=7.9e-10, rho=+0.49), **not** its lineage.

**⛔ Input-space zero-baseline gradient attribution is INVALID on this architecture — measured
2026-11-19, and 100 TB + ~600 Kp genomes of output were deleted because of it.** The checkpoints'
bundled `modeling_bacformer.py` applies `nn.LayerNorm` as the **first** operation on the raw protein
embeddings (`:642`, `:666`), so the model is degree-0 homogeneous in its input: `f(alpha*x)` equals
`f(x)` to within 6.4e-4 across `alpha` from 1e-3 to 10. Read from
`/rds/user/dca36/hpc-work/logs/attr-degeneracy-diag-41324596.out` (genome `SAMEA10030293`):

| quantity | value | should be |
|---|--:|--:|
| logit at `alpha` = 0 / 1e-5 / 1e-3 / 0.01→10 | +1.600 / +2.658 / +6.1524 / +6.1518 | — |
| IG sum, zero baseline, `n_steps`=50 | −0.0105 | +4.5516 |
| IG sum, zero baseline, `n_steps`=200 | **−10.51** | +4.5516 |
| input×gradient, summed over the genome | −6.0e-08 | — |

More steps is *worse*: Euler's theorem makes input×gradient identically zero, and IG's integrand goes
as `1/alpha`, concentrating the integral below the smallest quadrature node at any affordable
`n_steps`. **Scope this correctly — it is specific to the INPUT space.** `LayerIntegratedGradients`
at the post-LayerNorm activation, with the same zero baseline, satisfies completeness
(relative error 0.0000–0.0064 over four genomes). ⚠ The pip-installed `bacformer` package is
`linear(x)` *then* LayerNorm — a **different architecture** from what these checkpoints run via
`trust_remote_code`, and not affected. Testing against it does not describe the deployed model.

**Method comparison — 4 TB genomes, full zero-baseline ablation as ground truth.** Each from
`$B/train_tb_ast/attribution/diag/method_comparison/<Sample>_method_comparison.json`; variant status
from `$B/train_tb_ast/tbprofiler/tbprofiler_variants.parquet`:

| genome | rpoB variant | AST | ablation Δ (rank) | IG cohort-mean n200 (rank) |
|---|---|---|--:|--:|
| SAMEA10030293 | `p.Ser450Leu` | R | 1.231 (**1**/4071) | **+3.874** (1) |
| SAMEA10030311 | `p.Ser450Leu` | R | 2.382 (3/4097) | **+1.935** (2) |
| SAMEA1019317 | none catalogued | S | 0.016 (135/4056) | **−9.869** (2) |
| SAMEA104362366 | none catalogued | S | 0.950 (8/4068) | **−15.002** (1) |

Ablation Δ is `base − f(ablated)`, so a positive score means removing the protein pushed the model
towards susceptible. ⚠ **Confounded and n=4**: both resistant genomes are MDR, both susceptible ones
pan-susceptible, so R-vs-S here is confounded with total resistance burden. A positive control for
the method, **not a result**. Cost: ablation 1,250 s/genome against 162 s for IG at `n_steps=200`
(7.7×). `n_steps=50` is not usable — completeness error 4–76% on the real 30-layer model.

**Decision of record 2026-11-19 — IG against a cohort-mean baseline** (one vector: the mean of every
protein embedding over every cohort genome, broadcast to every token). `f(baseline)` is near
genome-independent: **−2.7569, −2.7571, −2.7586, −2.7607** over genomes of 4,056–4,097 proteins, both
labels, different lineages (`/rds/user/dca36/hpc-work/logs/attr-method-compare-41326249_[0-3].out`),
so attributions share an origin across genomes and already compare without renormalisation. ⚠ This
constant is from a 100-genome mean and will shift with the cohort. ⚠ The adopted method agrees only
weakly with ablation (Spearman of |·| 0.036–0.151; top-20 overlap 3–7 of 20) — do not describe it as
reproducing ablation.

**A class-contrastive baseline buys nothing — measured, job 41393159, and the idea is closed.**
Contrasting each resistant genome against the susceptible mean (and vice versa) was tested against
the global mean. From
`$B/train_tb_ast/attribution/diag/class_baselines/class_baseline_report.json`: `cosine(mean_R,
mean_S)` = **0.99994**, their L2 gap is 1.15% of the mean vector's own norm, and `f(baseline)` moves
only ~0.017 logits between them. rpoB's rank is **identical under all three baselines in all four
genomes**, and whole-ranking Spearman against the global-mean baseline is **0.991–0.999**. The effect
is in the intended direction but is ~2–3% of the score. R and S genomes share their core genome too
completely for their means to differ.

**⚠ 41 Kp genomes fail `verify_protein_count` and will block the planned 877-genome rerun.** The
cancelled ciprofloxacin array was 2 shards COMPLETED, **5 FAILED**, 9 CANCELLED — the failures are
genomes whose embedding and Bakta parquet disagree on protein count (e.g. `SAMEA8948660` 4,922 vs
4,919; `SAMN35159602` 6,000 vs 6,020, the truncation cap). Fix before committing GPU-h.

**⚠ The only cohort file on disk is n=100** (`$B/train_tb_ast/attribution/attribution_cohort_rifampin_n100.csv`),
and several strata hold 1–3 genomes. No ~1,500-genome cohort file exists.

**⛔ bf16 cannot carry this measurement — and fp32 costs 3.8×.** Same genome, same code, four
configurations, each read from its own
`$B/train_kleb_ast/attribution/{ciprofloxacin,precision_*}/SAMEA104414830_attribution.parquet`
and that run's `attribution_manifest_all.json`:

| dtype | perturbations/eval | seconds | ms/protein | distinct values | genes at exactly 0 |
|---|--:|--:|--:|--:|--:|
| bf16 | 1 | 659.8 | 122.4 | **13** | **96.2%** |
| bf16 | 16 | 584.4 | 108.4 | 13 | 96.2% |
| fp32 | 4 | 2,528.4 | 469.1 | **4,903** | **0.0%** |
| fp32 | 16 | 2,491.9 | 462.3 | 4,902 | 0.0% |

- **fp32 resolves the signal completely** — 4,903 distinct values against bf16's 13, no zeros at all.
- **Batching buys nothing.** ppe=16 vs ppe=4 differ by at most 4.77e-07 (correctness confirmed), and
  bf16 ppe=16 is only 11% faster than ppe=1. One 5,390-token sequence already saturates the A100, so
  **throughput is not the lever — genome count is.**
- bf16 keeps the largest effects (top-20 overlap 19/20, sd 0.0169 vs 0.0171) but destroys the body of
  the distribution: of the 5,185 genes bf16 called exactly zero, fp32 gives median |a| = 0.0058.
  Spearman over all 5,390 genes is only 0.320, because 96% of bf16's values are tied at zero.
- **Cost at fp32: ~607 GPU-h for the 877-genome Kp holdout alone**, against a plan estimate of
  80–100 GPU-h for both organisms. The TB arm would add more. **This is unaffordable as planned.**

**⚠ bf16 was the original finding and still stands:** One Kp ciprofloxacin
genome (`SAMEA104414830`, 5,390 proteins), read from
`$B/train_kleb_ast/attribution/ciprofloxacin/SAMEA104414830_attribution.parquet`: only **13 distinct
attribution values**, **96.2% of genes exactly 0**, every value a multiple of 1/32. The genome's base
logit is **7.28125**, where **bf16's ULP is 0.03125** (the next representable value is 7.3125; verified with `torch.nextafter`), so any gene contributing less than that is not representable and reads as a clean zero. *(Corrected 2026-11-19: this line previously said ~0.016, which is the HALF-ULP — the maximum rounding error, not the spacing. The 13 observed values are all integer multiples of 1/32, which is the ULP.)* The signal is not absent — flat indices 3161/3162 reach
+0.71875 — but 96% of the gene universe is unresolvable, which no amount of downstream statistics
repairs. fp32 and batched-throughput runs (jobs 41291284/41291285/41291286, all COMPLETED) are on
disk at `$B/train_kleb_ast/attribution/precision_{fp32_ppe4,fp32_ppe16,bf16_ppe16}/` and **have not
been read**.

**⚠ One genome only — NOT a result, and the genome is maximally confounded.** `SAMEA104414830` is
**ST512 / SL258**, the KPC high-risk clone, carrying `GyrA:p.Ser83Ile; ParC:p.Ser80Ile` plus KPC-3,
SHV-11, OXA-9 and TEM-150 (`final/metadata_v2_all_samples_and_columns.tsv`). Its base logit is
**7.2776** (p = 0.99931). In the fp32 run, **gyrA ranks 3,580 of 5,390** genes by |attribution| and
**parC ranks 5,010**; the median rank of all 17 AMR-annotated genes is 2,131 (39.5th percentile), and
the top-attributed genes are a contiguous block at flat indices 3159–3163 (`abgB`/`abgT` operon) plus
an `aac(6')-Ib` copy CARD did not call. **No conclusion may be drawn from n=1 on a clonal genome**
where resistance is predictable from the lineage alone. Hypotheses are open and listed in the
session hand-off; the cohort is 448 Flq-mutant / 343 wild-type, so the contrast is runnable.

Two decisions already of record for it (see §6):

- **The stored quantity is the logit shift plus each genome's base logit**, not the probability
  shift. The probability target saturates on a confident model — the published Bacformer run lost
  1,895 of 4,242 Salmonella genomes to zero attribution variance — and
  `explain.gene_ablation_attribution.delta_probability` recovers the bounded −1..1 score exactly, so
  nothing is given up.
- **Cohort is a stratified subsample of the holdout**, resolved only through `load_splits`:
  ~1,500 TB genomes stratified on phenotype × lineage, and all 877 Kp ciprofloxacin holdout genomes.

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
- **⚠ `finetune_amr`'s resume path crashes, unfixed — found 2026-09-04 by the invasion sweep, NOT by
  an AMR run.** `finetune_amr.py:405` calls `trainer.train(resume_from_checkpoint=...)` on a model
  built with `problem_type="binary_classification"` (line ~293). HF's
  `Trainer._load_from_checkpoint` reads the checkpoint's `config.json` through the **generic**
  `PretrainedConfig`, which rejects that value outright — so **any** `--resume-from-checkpoint` on the
  stock-head path dies immediately with *"The config parameter `problem_type` was not understood"*.
  The attention-pooling path may differ and was not tested. **The fix already exists and is tested**:
  wrap the call in `bacpredict.engine.finetune.checkpoints.tolerate_custom_problem_type()` (one import,
  one `with`). Deliberately **not applied here** — `finetune_amr.py` is the TB arm's live file and this
  is the invasion agent's finding; whoever owns that file should apply it. Do not "fix" it by changing
  `problem_type`: that value selects the loss (`binary_cross_entropy_with_logits`,
  `bacformer/modeling/modeling_large.py:1375`), so an HF-legal value is a different estimator.

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

**In flight — the publication k-fold x seed sweep (started 2026-09-04).** Every headline above and in
§3.4 rests on **one** fine-tune and **one** unitig refit. The Bacformer-minus-unitig delta is
+0.0210 [+0.0041, +0.0385], but a bootstrap CI measures sampling noise on a *fixed pair of models* and
says nothing about how far either number moves when you refit. David's question: is the edge bigger
than refit noise? Root §0.2 reserves folds x seeds for exactly this.

- **Bacformer arm — 15 runs**, `FOLD = task % 5`, `SEED = task // 5 + 1`, `evaluate_seed=1`, on the
  pooled cohort, bf16, patience 12, `max_steps=100000` untouched. Output
  `…/sampled_country_2_1_all/kpsc_human/models_kfold_fold{NN}_seed{S}/`.
- **Unitig arm — exactly ONE model, not fifteen.** `evaluate_seed` alone fixes the holdout, so
  `train ∪ validate` is one invariant set of **11,296** genomes across all 15 runs. Selecting hits and
  fitting on that set therefore has **no fold-to-fold variance by construction** — its SD is zero
  because it never uses folds. Stage 2c is a distribution of 15 Bacformer fits against a fixed point.
- **⛔ No sweep number will be comparable to 0.7858 / 0.7655.** At `evaluate_seed=1` the k-fold holdout
  (n=2,823) overlaps the deployed single-split holdout by only **21.5%** (608 of 2,822). David chose the
  standard k-fold holdout knowingly — the sweep is a fresh estimate, not one conditioned on the original
  arbitrary partition. What *is* comparable is Bacformer vs unitig *within* the sweep, since both arms
  share this holdout.

**Operational facts, measured 2026-09-04 — both cost a cluster round-trip and neither was predictable
from the plan.**
- **Chained resume works, and is verified end-to-end**, not assumed: link 2 of a two-link test resumed
  at **step 708** from `checkpoint-700` with learning rate **1.0635e-05**, which is exactly what the
  10,000-step warmup prescribes for step 709 (`1.5e-4 x 709/10000`). Checkpoint choice, global step and
  scheduler state all restored.
- **It did not work on the first attempt, and a one-job smoke could never have shown that.** Link 2
  died on HF's generic config reader rejecting Bacformer's `problem_type` (fixed in `caa32d6`; see
  §3.1's caveat, where the same defect still sits unfixed in `finetune_amr`). The failure is invisible
  to a fresh start — it exists only in the second link — so the plan's prescribed "smoke one job first"
  would have passed and released all 75 jobs to die one link in.
- **A HELD job freezes an `afterany` chain forever, and it RECURS.** 52 of the 64 unitig shards hit
  `user_env_retrieval_failed_requeued_held`, and after a hand release **29 of the survivors did it
  again** — so it is a standing condition, not a one-off. The mechanism is CSD3's
  `PrologFlags=…,ForceRequeueOnFail,…`: a prolog-side env-retrieval failure requeues the job and then
  holds it, and because the requeue is *forced*, **`--no-requeue` (`75a6d57`) does not reliably prevent
  it** — keep the flag, but do not trust it alone. A held job never *ends*, so the dependency never
  fires. Recovery is `scontrol release <jobid>`; `scontrol update JobId=<id> ArrayTaskThrottle=N`
  reduces the recurrence by cutting simultaneous starts (set to 10 on the unitig array).
  **⚠ Consequence for the unattended sweep: a GPU chain that lands held will sit still until someone
  releases it.** Nothing is lost — the next link resumes from the last checkpoint — but the run does
  not progress. **Whoever picks this up should check `squeue -u dca36 -o '%i %j %R' | grep held` and
  release anything found**, rather than assuming a quiet queue means the sweep is running.

**Split table of record for the sweep** —
`…/sampled_country_2_1_all/kpsc_human/kfold_sweep/{kfold_selection_split.csv, kfold_fold_assignments.csv,
kfold_splits_manifest.json}`, from `kleb_iso_source.materialise_kfold_splits`: universe **14,119**,
fitting **11,296** (prevalence 0.5348), holdout **2,823** (prevalence 0.5204), holdout digest
`5515932873d063d3`. Task 0's training log independently derived train 9,036 / validate 2,260 /
holdout 2,823 — two code paths, checked to agree rather than assumed to.

**Compute, and what is actually queued.** `FLOTO-SL2-GPU` had **288 h** left (18,812 of 19,100 used)
against the ~570 h the sweep needs — each run costs its full wall at a **measured 3.1 s/step** and
~40,000 steps to peak + patience. David's call: run all 15 on the free **`FLOTO-SL3-GPU`**, whose QOS
caps walltime at **12 h**, as chains (`submit_kfold_sweep.sh`, `--dependency=afterany` because every
link but the last is *expected* to TIMEOUT, and `--no-requeue` so a node fault cannot freeze one).

**All 15 released 2026-09-04**, ~73 chain links, one independent chain per run so no failure can stop
another: tasks **1-4, 6-14** as 5 SL3 links each; **task 0** part-trained on SL2 (36 h) + 3 SL3 links;
**task 5** part-trained on SL2 + 5 SL3 links. Expect ~10 days on a low-priority queue — David is away
and accepted that. Unitig arm: one GWAS on the carved cohort
`pyseer_iso_source/blood_faeces/sampled_country_2_1_all_kfold_trainval` (**n=10,869**, 5,759 blood /
5,110 faeces, `subset_manifest.json`), **~516 core-h** on `FLOTO-PROJECT-K-SL2-CPU` — below the plan's
560-610 because prep reused the matrix chunks (they are matrix-keyed, not cohort-keyed) and took 14 min
rather than the 2h54m budgeted; only the 902 MB LMM cache had to be rebuilt.

**Unitig arm — the GWAS is DONE (2026-09-05).** From
`…/sampled_country_2_1_all_kfold_trainval/gwas_unitig_lmm/blood_vs_faeces_unitig_gwas_summary.json`:
**6,284,709 variants tested, 5,660,757 unique patterns, 19,897 significant** at Bonferroni
8.8327e-09; genomic inflation **lambda 2.946**; phenotype variance 0.2491. All 64 shards COMPLETED.
For scale, the single-split `_trainval` cohort gave 6,264,286 / 19,622 / lambda 3.181 on a differently
carved 80% — the two agree closely, which is the sanity check that the new carve behaved.

**⚠ Vocabulary caveat, unchanged from the existing leakage-free arm.** Only the *selection* is
holdout-free. The GGCAT unitig vocabulary was built over all cohort genomes, so the feature
*representation* was shaped by holdout sequence — the same `full_cohort` vocabulary caveat §3.3 records
for the AMR arm. This matches the convention the deployed 0.7655 was produced under, so the sweep's two
arms stay comparable to each other; it is not a new exposure.

**Unitig arm — the model is DONE (2026-09-05). This is the sweep's fixed baseline.** Read from
`…/sampled_country_2_1_all_kfold_trainval/gwas_unitig_lmm/presence_model/unitig_model_results.json`:

| | |
|---|---|
| **Evaluate AUROC** | **0.7691** (n=2,733 — the 2,823 k-fold holdout genomes that have a unitig row) |
| `C` | **0.01**, by 5-fold CV *inside* train+validate; inner-CV AUROC **0.7686 ± 0.0059** |
| Fit on | all **10,869** train+validate rows (`n_validate=0` — the selection table labels the whole 80% `train`) |
| Features | 19,897 unitigs · `selection_scope` `trainval_only` · `c_selection` `inner_cv` |
| Cohort scores | all **13,602** genomes scored; train 0.8604 / evaluate 0.7691 |

Two things worth noting. The `C` sweep is cleanly unimodal (0.7218 → **0.7686** at C=0.01 → 0.7203 at
C=10) and picks **the same optimum the validate-tuned protocol picked**, so the new protocol did not
move the model — it only removed the reason to distrust the number. And the inner-CV estimate (0.7686)
lands within 0.0005 of the holdout (0.7691), which says the CV was a well-calibrated estimate here
rather than an optimistic one.

**⛔ 0.7691 is NOT comparable to the deployed 0.7655** — different holdout (21.5% overlap), different
`C`-selection protocol, and a hit set re-derived on a differently carved 80% (19,897 unitigs vs
19,622). It is comparable only to the 15 Bacformer fits, which is the whole point.

**Reading the sweep when it lands** — `kleb_iso_source.compare_kfold_sweep` writes
`kfold_sweep_per_run.csv` + `kfold_sweep_comparison.json` into the sweep dir. It pairs each fit with
the unitig model on the genomes they both cover, so the verdict is the sign of 15 deltas, not a contest
between two averages, and it **refuses to score a run whose `eval_scores.npz` carries no `sample_ids`**
— without them neither the holdout check nor the paired bootstrap is possible, and a matching row count
proves nothing.

**Also in flight.** bf16 re-runs of all three cohorts (the originals were fp32; the bf16 cast landed
later in `a817ac2`).

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

**Status — AMR (`src/bac_pyseer/ast_gwas/`).** **All 22 Kp drugs have a complete `.assoc`**; 21 have
`lr/results.json`, with ceftazidime's read-out the last outstanding (its `.assoc`, `patterns.txt` and
450,950-row hit table are already written). The Kp cohort, unitig matrix and lineage clusters are
**built and shared**. **TB's Phase 0 has landed (code only, no compute — see *TB* below);
no TB GWAS has run.**

**⚠ This 22-drug set is the `full_cohort` arm, and its vocabulary saw the holdout.** No *label* ever
leaked — the GWAS phenotype carries `train,validate` only and its intersection with the holdout is 0 —
but the GGCAT unitig vocabulary and the lineage clusters were built over all 7,080 cohort genomes, so
the *feature representation* was shaped by holdout sequence. That was agreed as acceptable at the time
(`ast_gwas/docs/PLAN.md`'s leakage table) and is not in dispute; it is being closed now because the
results are strong enough that the claim is only worth making if it survives the check — and because
**the invasion arm already shows the cost is real** (0.7810 → 0.7655 once selection was re-fitted on
train+validate, a delta that separates from zero). A `trainval_vocab` rebuild across all 22 drugs is in
progress; until it lands these numbers are of record but carry that qualifier. Plan:
`~/.claude/plans/tb-uses-the-tb-shimmering-fountain.md`.

**✅ Rebuild stage C3 complete 2026-08-28 — all 22 vocabularies are built and asserted clean.** Read
from `…/processed/pyseer_ast/kp_trainval_vocab/build_summary.tsv` (mirrored at
`src/bac_pyseer/docs/trainval_vocab_build_summary.tsv`), produced by
`bac_pyseer.ast_gwas.summarise_vocab_build`, which reports **22/22 ok · 0 FAIL · 0 unchecked**. Per
drug, from GGCAT's own `color_names.jsonl` rather than the reflist we passed it:
`n_colors == n_reflist` and **`n_holdout_coloured == 0`**, plus `n_missing_from_graph == 0` and
`n_extra_in_graph == 0`. A missing audit section counts as *unchecked*, never as passed.

| | value |
|---|---|
| reflists (train+validate) | **1,128–3,932 genomes, mean 2,454 = 34.7%** of the 7,080 cohort |
| `MIN_SAMP` (1% floor) | **12–40, per drug** — was a flat 71; confirmed in each build log, not just the audit |
| lineage clusters at `min_size=100` | **3–7 per drug**, vs 11 at cohort scale; `other` rises accordingly |
| GGCAT build time | 3:41 (colistin) → 27:45 (gentamicin) on 32 cores, `-m 90G` |
| unitig matrices | **117.6 GiB** total (0.40–12.65 GiB); colormaps a further 60.7 GiB |
| GGCAT scratch spill | **~1 GiB per build** — 20 concurrent builds moved hpc-work 267 → 272 GB |

**Two of the plan's estimates were wrong and are corrected here.** Matrix size does *not* scale as
`27 GB × N/7,080`: that linear model over-predicts by ~43% overall (206 GB predicted vs 117.6 GiB
actual) and badly for small drugs (colistin 4.3 GB predicted, 0.40 GB actual), because a smaller
cohort shortens every carrier list *and* thins the feature set. And `uprep` was budgeted at 6 h on
the assumption that per-drug matrices force a real split — they do, and colistin's took **2 min 44 s**.

**✅ Rebuild stage C4 complete 2026-08-28 — all 22 GWAS run on the new vocabularies.** Read from
`…/kp_trainval_vocab/gwas_arm_comparison.csv` (mirrored at
`src/bac_pyseer/docs/trainval_vocab_gwas_comparison.csv`), produced by
`compare_vocab_arms --stage gwas`. **`pheno_var` is identical across the two arms on all 22 drugs**,
which is the control that says the rebuild changed the vocabulary and not the labels.

| trainval / full_cohort | range | median |
|---|---|---|
| variants tested | 0.46 – 0.87 | **0.69** |
| unique patterns | 1.01 – 1.09 | 1.03 — **above 1 on 22/22** |
| significant hits | 0.52 – 0.83 | **0.67** |
| λ | — | 4.54 → **4.18** (lower on 17/22) |
| **variants per pattern** | — | **1.73 → 1.15** |

**The redundancy number is the informative one.** The full-cohort vocabulary carried 1.73 variants
per distinct presence pattern against 1.15 now: it was substantially more LD-redundant, consistent
with more genomes creating more branch points, hence more and shorter unitigs whose carrier sets
coincide. Unique patterns rise on *every* drug despite ~30% fewer variants, which the `MIN_SAMP`
rebase (rare unitigs have distinct patterns) plausibly compounds. **Bonferroni is set by pattern
count, so the rebuild is tested slightly more stringently and still returns ~a third fewer hits** —
the drop is not a threshold artefact. ⚠️ **Hit count is not AUROC**: a drug can shed redundant hits
and predict identically, so nothing about predictive performance follows from this table. That is
C5's question. These mechanisms are **hypotheses pending David's comment**, not conclusions.

**✅ C5/C6 COMPLETE 2026-08-28 — the unitig results SURVIVE a leakage-free vocabulary.** All 22
drugs rebuilt, scored and compared. Read from
`…/pyseer_ast/kp_trainval_vocab_comparison/` (headline CSVs mirrored at
`src/bac_pyseer/docs/trainval_vocab_auroc_comparison.csv` and `unitig_vs_finetune_kp.csv`).

| paired Δ = full_cohort − trainval_vocab | value |
|---|---|
| median / mean | **+0.0002 / −0.0001** |
| range | −0.0206 to +0.0095 |
| CI excludes zero | **3 of 22** — cefotaxime +0.0027, t-sulfamethoxazole +0.0012, ceftazidime +0.0005 |
| direction | 13 favour full_cohort, 9 favour trainval_vocab |
| LD control (`lr_dedup`) | median −0.0000, 2 of 20 separating |

**19 of 22 drugs are statistically indistinguishable between arms, and the three that separate do so
in the third decimal place.** The largest single movement (tetracycline, −0.0206) favours the
*rebuild* and its CI spans zero. The vocabulary changed substantially — variants tested fell to a
median 0.69× — and the predictions did not.

**Verification behind that claim, all measured:** `n_holdout_coloured == 0` on all 22 (GGCAT's own
`color_names.jsonl`); mash `max_abs_diff == 0.0` on all 22 with full reflist overlap; and the
sequence scanner reproduced GGCAT's colouring across **5,594,790,713 cells with 0 mismatches**,
8/8 shards per drug. Gate table: `kp_trainval_vocab_comparison/readout_gates.tsv`.

**Unitig vs Bacformer fine-tune, and it does not depend on the leak.** FT AUROCs read from each
checkpoint's own `results.json`, never a panel:

| | leakage-free unitig | full-cohort unitig |
|---|--:|--:|
| mean Δ (unitig − FT) | **+0.0067** | +0.0065 |
| median Δ | +0.0066 | +0.0075 |
| unitig wins | **15 / 22** | 15 / 22 |
| exact binomial p | **0.134** | 0.134 |

**Consistently very slightly ahead; NOT demonstrated.** p = 0.134 does not reach conventional
significance, and it is **optimistic** besides: the drugs are not independent (overlapping genomes,
correlated phenotypes, one carbapenemase moving ertapenem/imipenem/meropenem together), so the
effective number of comparisons is below 22. Two further caveats — these are **point estimates**
(FT training saves no per-sample scores, so no per-drug CI against the FT exists without re-running
`engine.finetune.evaluate`; **David declined that run 2026-08-28**), and the holdouts are
**near-paired, not paired**: FT scores 0–3 more genomes per drug (max 0.444%) because it scores from
embeddings, which exist for genomes with no assembly that the unitig scanner cannot reach.

*Hypothesis, not established:* the FT wins where both are near ceiling (ceftazidime 0.985, ertapenem
0.988) and unitigs win biggest where the FT is weakest (azithromycin 0.799→0.845, cefepime +0.024),
suggesting the arms are complementary rather than interchangeable and that the mean hides an
interaction with drug difficulty.

**Still outstanding from the plan:** the AMR ladders have **not** been re-rendered against the
rebuild (one command: `NESTED_DRUG_DIR=1 PYSEER_ROOT=<vocab root>
bash src/bacpredict/engine/scripts/render_amr_ladders.sh`), and the λ calibration protocol
(af-stratified λ + within-lineage permutation null) is unrun — λ is high in **both** arms (median
4.18–4.54, up to ~12 on levofloxacin and ceftazidime), so **no claim about which unitigs matter
should be made until it is**. AUROC is far less exposed, since the LR is fitted and scored on
splits rather than on p-values.

**⚠ The `full_cohort` arm scores a few holdout genomes that have no features at all.** Found
2026-08-28 when the rebuild's holdout id-set guard fired on cefazolin (594 vs 593). `SAMEA3357403`
is in cefazolin's split table and its `design/samples.txt`, but **not in
`kp/unitigs/assembly_refs.txt`** — the reflist saying which genomes became GGCAT colours — so it has
an **entirely zero feature row** (nnz 0 against a median of 16,103 carriers) and was scored from the
model intercept alone. Surveyed across all 22 comparator designs by loading each `presence.npz`:
**105 all-zero rows of 67,584 (0.16%)**, in every drug except colistin, 1–8 per drug (worst:
tobramycin 8/2,852 = 0.28%). These are genomes present in the split tables with no assembly to
resolve. **The rebuild does not have them** — its scanner needs an assembly — which is why the two
arms' holdout sets differ slightly and why `compare_vocab_arms` pairs on the intersection, counting
and naming the exclusions rather than intersecting silently. Effect on the published `full_cohort`
AUROCs is small (~1–2 holdout genomes per drug) but real, and belongs in any write-up.
`engine`-side, `unitig_design_matrix.check_holdout_coverage` counts `n_holdout_all_zero` and would
have caught this when those designs were built, had it existed then.

**⚠ ertapenem's `gwas/` summary is stale and must not be quoted.** It predates the combine-phase fix
and left `pheno_var` at the 0.249 default (`pheno_var_source: "default"`); its regenerated sibling one
level up — `kp/ertapenem/ertapenem_gwas_summary.json`, written with `--phenotype-tsv` — is the
authority, and the two also disagree on `n_variants`, `n_patterns`, `n_significant` and λ.
`compare_vocab_arms.resolve_summary` encodes that precedence. ertapenem is the only drug with both.

**The mash re-sketch is now a measurement, not an argument.** `leakage_audit mash` asserts each
drug's freshly-sketched `similarity.tsv` against the comparator's, and returns **`max_abs_diff = 0.0`
exactly**, with `n_shared` equal to the drug's full reflist. Subsetting the cohort triangle and
re-sketching per drug are bit-identical, so mash kinship is closed as a leakage path by number.
(That assertion existed since C2 but was called by nothing until 2026-08-28 — an implemented,
unrun check that made the verification table read as passed.)

**⚠ Storage bug fixed 2026-08-21 — it had silently killed a whole batch.** `unitig_lmm_sharded_job.sh`
keyed `CHUNK_DIR` on `$PAIR`, and `run_drug.sh` sets `PAIR=$DRUG`, so all 22 Kp drugs — which share
ONE `unitigs.pyseer.gz` — each re-split it into a private ~26.5 GB copy. Twelve copies filled
`/rds/user/dca36` and five batch-2 preps died mid-write with "Disk quota exceeded". `CHUNK_DIR` now
derives from the matrix path, so sharing is structural; prep dropped from minutes-plus-26.5 GB to
**9 s and 0 GB** (`unitig_lmm_shard_34159281_*.out`: "reusing 64 existing chunks"). Duplication was
content-verified before deleting (decompressed `chunk_07.gz` md5 identical across five drugs).
**162 GB reclaimed** — `/rds-d6/user/dca36` 407.9 → 245.5 GB — via
`src/bac_pyseer/ast_gwas/scripts/reclaim_unitig_chunks.sh` (dry-run by default); all seven drugs'
`.assoc` results survived at 64/64. Two hazards fixed in the same pass: concurrent preps now take an
atomic `mkdir` lock instead of racing to rebuild one shared directory, and the split stages into
`.building/` and publishes only after verifying all 64 chunks are non-empty — a killed prep used to
leave a SHORT chunk that the name-only count accepted as present.

**The AUROC ladder now carries the unitig arm** (`engine/plots/plot_amr_ladder.py`, 2026-08-21):
red catalogue → **purple unitig-LR** → blue Bacformer FT and FT⊕baclm concat heads. The join is at the
**plot layer**, not in `<drug>_amr_ladder_table.csv` — the two arms come from different pipelines, and
appending a rung would force a ladder re-run. No threshold reconciliation is involved: `build_amr_ladder`
writes only AUROC/AUPRC and never picks an operating point, so the unitig arm's Youden-on-holdout
threshold is irrelevant to this figure. What is guarded is *genome identity* — `--split-table` resolves
the holdout through `engine.splits.load_splits` and a `results.json` whose `split.n_evaluate` disagrees
is dropped rather than drawn. `engine/scripts/render_amr_ladders.sh` renders all 22 and reports which
arms each got, so a partial panel cannot be mistaken for a complete one.

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

**⚠ CORRECTED 2026-08-28 — the swap HAS happened on CSD3, and it landed between the pilots and the
fan-out.** This entry previously said `lineage_clusters.tsv` was "still the OLD file"; that is no
longer true. The live
`…/bac_ast_prediction/processed/pyseer_ast/kp/structure/lineage_clusters.tsv` was replaced
**2026-08-20 18:34** (mtime) and is the proposed 11-cluster file — read from the TSV itself, not the
manifest beside it: 7,080 rows, 12 distinct labels (11 named + `other`), `other` = 2,803, which
matches `lineage_clusters.manifest.json`'s `n_clusters: 11` / `n_in_named_cluster: 4277` /
`named_cluster_coverage: 0.6041` exactly. So the **live** column of the coverage table above is now
the *proposed* column. (Isambard is still unswapped and unverified.)

**Which drugs used which file — from the 22 `.assoc` mtimes.** Exactly the two pilots predate the
swap: **ertapenem** (2026-08-13 12:21) and **colistin** (2026-08-13 14:04) ran against the old
10-cluster / 3,890-named file; **the other 20** ran 2026-08-20 23:02 → 2026-08-24 12:28 against the
new one. The existing conclusion therefore stands and is now grounded: **re-run the two pilots
rather than quoting their λ or permutation p-values across the change.**

**This does not touch any AUROC, and here is why.** The clusters reach pyseer only at the per-shard
association step (`--lineage --lineage-clusters`); the LMM cache the shards load is built in a
separate prep step from `--similarity` alone, with neither `--lineage` nor `--distances`
(`unitig_lmm_sharded_job.sh:145`). β and p therefore come from a clusters-free LMM, so the hit set,
the design matrix and every read-out AUROC are identical either way. What differs for the two pilots
is the post-hoc `lineage` attribution column and any null computed against those clusters.

**The `trainval_vocab` rebuild is internally consistent on this axis**, unlike the comparator: all 22
of its cluster files are built in one pass by `build_trainval_vocab.sh`, with the same method and the
same two extra LIN-typing tables (`archived_lin.tsv`, `mist_lin_new.tsv`) that produced the live file.

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

**Decision of record — the holdout is scored by k-mer containment, not exact substring (2026-08-28).**
A train+validate-only vocabulary has no holdout rows, so the holdout must be scored from sequence, and
two rules were available: **(A)** every canonical 31-mer of the feature occurs in the genome, or
**(B)** the feature occurs as a contiguous substring (the Aho-Corasick scan in `unitig_placement`).
(B) implies (A), never the reverse, so mixing rule-(A) training rows with rule-(B) holdout rows would
have injected an assembly-fragmentation confound into the test set alone.

**(A) is what GGCAT colouring already means** — `ggcat_to_pyseer` emits each feature as
`seq[kmer_off : kmer_off + n_kmers + k - 1]` carrying one colour subset, i.e. exactly the samples
holding *every* k-mer of that span. Measured rather than assumed: on ertapenem, 40 genomes × 31,856
features = **1,274,240 cells, 0 mismatches in all three directions** (A vs GGCAT, B vs GGCAT, A vs B),
including **297,838 positive calls** reproduced exactly — agreement on presence, not just on absence.
Sampled genomes spanned 45–333 contigs (median 119); features 31–1,484 bp (mean 39.7). Read from
`…/pyseer_ast/kp/ertapenem/design/rule_discordance.json`.

Rule (A) is adopted on three grounds: it reproduces the operator that produced the existing train and
validate rows, so the scan is **self-verifying** — it must reproduce `hits_submatrix.tsv` exactly, and
does; the two rules are empirically indistinguishable here, so the choice costs nothing in fidelity;
and it is far cheaper — a sorted `uint64` k-mer array is ~44 MB per genome and ~36 MB for
ceftazidime's 450,950 features, against a multi-GB trie over 901,900 words. That collapses the planned
holdout-scan job from `mem=192G --array=0-15` to a small single job.

**Where the rules *would* diverge, for the record**: not at a contig break as such — a clean break
destroys the spanning k-mers, so both rules call the feature absent. They separate only when the break
is **overlap-repeated** (contig 1 ends with `feat[:35]`, contig 2 starts with `feat[5:]`), where every
k-mer survives but no contiguous occurrence does. Pinned as a fixture in
`tests/bac_pyseer/ast_gwas/test_unitig_kmer_presence.py`.

**Next.**
1. **The `trainval_vocab` rebuild, all 22 Kp drugs** — one GGCAT build per drug over train+validate
   only, then the identical GWAS → hits → design → LR chain, scored on the unchanged holdout by rule
   (A). Measured cost of the completed `full_cohort` fan-out was **1,269.3 core-h = £12.69**
   (`sacct CPUTimeRAW`); the rebuild adds per-drug builds, real prep splits and the holdout scan for
   **≈ 2,350 core-h ≈ £23**. Blocker cleared 2026-08-28: `run_unitig_lmm_sharded.sh` hardcoded
   `--time=24:00:00`, which *reserves* 12,288 core-h per drug — 270,336 across 22, against ~106,371
   available — so every chain would have sat on `AssocGrpCPUMinutesLimit` and none would have started.
   `PREP_TIME`/`ARRAY_TIME`/`COMB_TIME` are now knobs; at `ARRAY_TIME=2:00:00` (measured shard max
   6.2 min) the fan-out reserves ~22,500 core-h. `COHORT` is passed explicitly now too, because it keys
   the scratch shard dir and its old fallback would have written the rebuild's shards into the
   completed run's directory, where the empty- and runt-checks would both have passed on stale files.
2. Report the paired old-vs-new delta with bootstrap CIs, and attribute it three ways — representation
   advantage removed, out-of-vocabulary penalty, and the MAF-floor change, which points the *other*
   way: `MIN_SAMP` is 1% of the reflist, so the old build's 71 was an effective 4.2% floor on
   ertapenem's ~1,697 phenotyped genomes while pyseer was simultaneously told `--min-af 0.01`.
   Rebasing on a train+validate reflist gives 17, which finally matches. Control for it by refitting on
   columns with ≥71 train+validate carriers.
3. FT re-score on every drug (without `eval_scores.npz` there is no paired CI, and a gap cannot be told
   from a tie). Permutation null on the pilot plus any surprise only.
4. Invasion: DefenseFinder mapping; the faeces↔respiratory run and blood↔resp concordance;
   locus-level annotation.
5. **Blocked externally** — hotspot rates by isolation source (per-source rates against the
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
- **⚠ The shard memory gate's *unmeasured* bracket is not conservative — measured 2026-09-04.** Its
  first real observation, the invasion k-fold canary (`sacct -j 34911480_0`, MaxRSS 39,559,612 K =
  **37.7 GB** at n=10,869, cpu=8, 98,134 unitigs/shard, 23m22s), came in at **1.8× the top of the
  16.5–20.7 GB bracket the gate had printed for it**. The gate would therefore have passed a
  `--mem=32G` request (its stated floor was 25.8 GB at margin 1.25×) that would then have OOM'd the
  whole array. **`CANARY=1` is what stands between the estimate and that outcome — run it on any
  cohort size nobody has measured, and do not raise the estimate's authority above the canary's.**
  Recorded to `src/bac_pyseer/ast_gwas/shard_memory_observations.json`; note it is the *only*
  observation there, and `_fit_exponent` correctly declines to fit a scaling from one point, so the
  bracket stays "UNMEASURED" until a second cohort size is measured. That is honest behaviour, not a
  bug — but it means **nothing has yet narrowed the bracket for the AMR fan-out either.**

**Owns.** `src/bac_pyseer/`.

---

**Status — TB (`src/bac_pyseer/ast_gwas/`, organism `tb`).** **Phases 0–1 complete, 2026-08-30.**
The TB-Profiler call set is parsed into the canonical tree and both comparator cluster files are
written; the cohort reflist is resolved. **No GWAS has run** — Phase 2 (the cohort build) is next. Plan: `~/.claude/plans/tb-uses-the-tb-shimmering-fountain.md`. Approach
agreed with David: **one shared full-cohort vocabulary, subset per drug** (licensed by the C6 result
above — the leakage-free rebuild moved Kp by a median +0.0002), with the leakage-free arm deferred.
TB has an option Kp did not: a large minority of genomes sit in **no** drug's holdout, so a *single*
leakage-free shared build is possible later, against Kp's 22 per-drug builds. ⚠ The figure previously
recorded here (12,225 of 36,390, 33.6%) was computed against the **superseded 36,390 cohort** — see the
table below. **Recompute it against 36,692 before relying on it**; the qualitative point stands, the
number does not.

**Cohort facts, listed on CSD3 2026-08-28 — not carried over from any plan.**

| | value |
|---|---|
| GWAS cohort | **36,692**, and **36,692 resolve — 0 missing** (`pyseer_ast/tb/unitigs/assembly_refs.txt` + its manifest, 2026-08-30). Flat, BioSample-keyed, **no `--file-list`** |
| per-drug train+validate (the LMM's n) | **7,167 (streptomycin) → 28,507 (rifampin)**, read from the 10 `<drug>_split.csv` — 2.1× the largest cohort this pipeline has run |
| holdout n | **1,792 (streptomycin) → 7,127 (rifampin)**; matches each drug's fine-tune `n`, all 10 — so unitig-vs-FT pairs on identical holdouts, unlike Kp |
| TB-Profiler coverage | **36,624 of 36,692 (99.81%)**; **68 uncovered, named** in `tbprofiler_uncovered_samples.txt`; 60 called genomes sit outside the cohort |

> ⚠ **Two numbers previously in this table were wrong, and both came from the *deprecated* tree.** The
> cohort is **36,692, not 36,390**, and **nothing fails to resolve** (the "36,389 of 36,390, 1 missing"
> never existed). There really are two trees — `david/processed/train_tb_ast/` (deprecated, 36,684) and
> `david/bac_ast_prediction/processed/train_tb_ast/` (canonical, 36,692) — and `resolve_data_root()`
> returns the canonical one. `ast_gwas/CLAUDE.md` §layout records the distinction; the TB plan did not,
> which is how the wrong denominator propagated into the "69 missing" figure (the measured count against
> the right denominator is **68**, a near-coincidence that would have hidden the error).

**The TB-Profiler finding supersedes `PLAN.md` Decision 3** and matching claims in
`lineage_from_distances.py`, `sublineage_from_metadata.py` and `ast_gwas/CLAUDE.md`, all of which
said TB lineage labels could not exist "until TB-Profiler is run over ~39k assemblies". All four are
corrected. **Mash stays primary for TB** (David, 2026-08-28) on a resolution argument — a discrete
`sub_lineage` is constant *within* a large sublineage and so cannot correct the divergence inside
one — with TB-Profiler's calls written alongside as the comparator partition and as the permutation
null's strata. Note this changes no β or p either way: the LMM cache is built from `--similarity`
alone, so clusters reach pyseer only at the post-hoc attribution step.

**Phase 0 — what landed (7 commits, `33dd4f9` → `782d53f`).** All shared-engine edits are
behaviour-preserving with defaults unchanged, pinned by tests that run the real scripts with
`sbatch` stubbed.

- **`parse_triangle` vectorised, and parsed once per drug instead of two or three times.** Measured
  **3.6× faster and exactly 2× lower peak, output identical** (n=4,000: 14.18 s / 0.39 GB → 3.90 s /
  0.19 GB). Extrapolated to TB's 36,389-genome triangle: peak ~32 GB → ~16 GB.
- **A memory gate.** `estimate_shard_peak` reproduces the invasion calibration (n=13,602, cpu=8:
  10k/50k/100k unitigs/shard → 9/21/26 GB) with `peak ≈ 7.383·ln(u) − 59.0`, fitted on the outer two
  points and predicting the **held-out** middle one at 20.9 vs 21 measured.
- **⚠ The "peak ≈ cpu × n²" model is refuted by its own calibration.** One n×n rotation matrix at the
  anchor is 1.48 GB, so eight private worker copies would be 11.8 GB — *more than the 9 GB measured
  in total* at u=10k. pyseer does not hold a copy per worker. Cohort-size scaling is therefore
  **unknown**, and the estimator returns a bracket (linear → quadratic in n) rather than a number.
- **TB rifampin brackets at 54–114 GB against the `--mem=128G` default.** The default clears the
  upper bound by 1.12×, where the invasion run had ~5×, on a bound extrapolated from one cohort. The
  honest verdict is "nobody can currently say"; the gate returns that, and refuses to submit.
- `PYSEER_CPU` decouples pyseer's workers from the allocation (they were one variable, so buying
  memory with cores also multiplied per-worker memory). `CANARY=1` runs one shard first.
- **Resume:** a shard writes a completion sentinel *after* its `.assoc` is checked, so a blind array
  requeue is free for shards already done. Skipping on "non-empty `.assoc`" instead would have made
  the ceftazidime runt permanent.
- Cache fingerprinting on (phenotype, kinship); a dead-builder chunk-lock reclaim; `NSHARDS`
  required rather than defaulting to 16 in one file and 64 in the other; `QOS`/`LOGDIR` actually
  consumed; a `SLURM_JOB_ID` unbound-variable abort outside SLURM.
- **`--mem=128G` allocates 20 cores, not 8** (`MaxMemPerCPU=6,760 MB`), so the reservation line was
  understating itself 2.5× — the same class of error that held every Kp array in
  `AssocGrpCPUMinutesLimit`. Now computed from billed cores.
- **`chunk_%02d` was NOT changed.** The claim that it caps `NSHARDS` at 100 is wrong — `printf`
  treats it as a minimum width, and the combine iterates by index, not by glob. Verified before
  touching it; renaming would have invalidated every existing chunk set for nothing.

**Phase 1 — the TB-Profiler parse (job `34597335`, 17:51 wall, 2026-08-30).** Artifacts in
`bac_ast_prediction/processed/train_tb_ast/tbprofiler/`; numbers from
`tbprofiler_parse_manifest.json` unless stated.

| | value |
|---|---|
| result JSONs parsed | **36,684 of 36,684; 0 unparseable** |
| catalogue | **one commit for all 36,684 genomes** — `who_v2+` `7fe4364e`, db-schema 2.1.0, TB-Profiler **6.7.0** |
| variant rows / distinct variants | **77,585** / **4,647** |
| genomes with a lineage call | **36,604** (46 `sub_lineage` clusters, or 5 `main_lineage`, at `min_size=100`) |
| cluster files | `structure/tbprofiler_{sub,main}_lineage_clusters.tsv` — named-cluster coverage **94.7%** / **99.2%**, `other` = 1,954 / 308 |

**The re-parse reproduced the June output byte-for-byte** — all three artifacts md5-identical to
`snp_embeddings/tbprofiler_calls/parsed/`. That is the empirical form of the argument that licensed
re-parsing rather than re-calling 36k assemblies: a `results.json` is a deterministic function of one
assembly and one catalogue version. It is now checked rather than asserted — the parser reads
`pipeline.db_version` from **every** file and refuses a mixed catalogue, because a call set spanning two
commits is two one-hot vocabularies and nothing downstream could have detected it.

> **⚠ Rifabutin cannot have a WHO ceiling.** The `who_v2+` catalogue annotates **20 drugs and rifabutin
> is not one of them** (it annotates *rifapentine*, a different rifamycin) — read from the `drugs` column
> of `tbprofiler_variants.parquet`. That, not a parser bug, is why it scores **0 of 36,684**;
> `_norm_drug` passes the name through untouched. `tbprofiler_gene_lr` already skips a drug with no
> variants, so **Phase 5 yields 9 of 10 and does not error**. Rifabutin's *GWAS* is unaffected — it has a
> real phenotype (9,711 train+val, 2,427 holdout). Whether to give it rifampin's `rpoB` one-hot as a
> proxy feature set is **open, and David's call** (see the plan's Phase 5).

> **⚠ The parse peaked at 4.0 GB against a 4 GB request** (`sacct MaxRSS=4194536K`, job `34597335`). It
> completed cleanly, but my estimate was "<1 GB" and the driver was the one thing that estimate ignored:
> the **transient** per-file object graph. Each JSON carries `gene_name2locus_tag`, `other_variants` and
> the `lineage` support arrays, so a 108 KB file becomes multi-MB of Python objects, and CPython does not
> return freed arenas to the OS — so RSS tracks the high-water mark, not the live set. **Accumulators
> were never the cost.** Re-running this needs ≥8 G.

**Next (TB).** **Phase 2** — `build_cohort_once.sh ORGANISM=tb` (the reflist is already resolved and
will be reused), **measuring** matrix/colormap/triangle bytes and setting `NSHARDS` from them, then
comparing the mash partition against the two TB-Profiler ones · **Phase 3** the drug ladder,
**streptomycin alone first** (it is also the job that builds the shared chunk set every later drug
reuses in ~9 s) · Phases 4–6 read-out, WHO ceiling, λ calibration.

**Caveats (TB).**
- **Every memory figure above is an extrapolation, not a measurement.** No `sacct MaxRSS` for a
  unitig shard exists anywhere in the repo. Wave 3a's canary is what replaces it.
- **The WHO ceiling is still the provisional June one.** The layout now mirrors CARD and
  `discover_who` reads both shapes, but the rebuild itself (Phase 5) has not run — so a TB
  ceiling-vs-unitig gap still cannot be read in either direction. When it does run it will cover
  **9 drugs, not 10**, unless the rifabutin proxy above is agreed.
- **The mash-vs-TB-Profiler partition comparison has not happened.** Both TB-Profiler cluster files
  exist; the mash one is built by Phase 2, so "both written and compared" is only half done.
- **λ will be worse than Kp's 4.18–4.54 median**, TB being far more clonal. `permute_unitig_lambda.sh`
  has never been wired into `ast_gwas` despite `run_drug.sh` saying it is not optional. **No claim
  about *which* unitigs matter until Phase 6 runs.** AUROC comparisons are unaffected.

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
- ⚠ **The picks moved to the prevalence-matched cut, 2026-09-04.** Both models are now cut where
  they call blood as often as blood occurs — **Bacformer 0.2606, unitig 0.5558** (Youden was 0.4349 /
  0.5272). Rationale: at the Youden cut a genome like `SAMN18874767` (Bacformer 0.449) sat +0.014
  above the line and read as marginal when its true margin is +0.188, and the disagreement skew was
  3:1 rather than its true 1.3:1. **Agreement 505 → 522 of 671; concordance 0.753 → 0.778, κ 0.508 →
  0.555.** Per-sublineage agreeing counts: SL258 32→27, SL3010 16→**25**, SL307 30→31.
  - The gate is unaffected — it checks AUROC, which is threshold-free, and still passes at
    0.785816 / 0.765471.
  - `thresholds --mode prevalence` on `build_model_comparison_report`; both cut-points are always
    recorded in the JSON. Artifacts: `model_comparison_thresholds_matched.json` and
    `matched_model_comparison_shortlist_*` beside the Youden originals.
  - ⚠ **Base-rate caveat stands.** A matched cut carries the cohort's 52% blood rate onto a
    carriage-heavy collection whose true rate is likely lower, so it will over-call blood there. This
    was David's call, made knowing that.
- **Per-sublineage pick list rebuilt 2026-09-01 (`ee7e6b2`)** — it is a collaborator hand-off, so it
  now lists **only genomes both models call the same way**, never a forced quota. Counts from
  `model_comparison_shortlist_meta.json` → `per_sublineage`: SL258 32 of 49 agree (9 invasive / 23
  faeces), SL3010 16 of 40 (10 / 6), SL307 30 of 37 (11 / 19). Rows a known label contradicts are
  **withheld and counted** — 3 in SL258 (2 invasive, 1 faeces), 0 elsewhere. Listed: SL258 7 + 10,
  SL3010 10 + 6, SL307 10 + 10 = **53 rows, none contradicting its own band**. `unseen` genomes lead
  each block, because a confident score on a fitted-on genome is partly recall.
  - **The old 10 + 10 quota was unachievable**, not merely undesirable: SL258 has 9 agreed-invasive
    and SL3010 only 6 agreed-faeces. Filling it required disagreements and known-wrong rows.
  - **SL258's invasive picks are the weak ones.** All 9 agreed-invasive are `unseen`, but the only 2
    carrying a label are both wrong (`ME140270`, `ME140311` — called invasive, actually faeces).
  - **The faeces side is largely recall, the invasive side is prediction.** SL307 agreed-faeces is
    11/19 fitted on and SL3010's is 6/6 labelled, while the invasive lists are 9/9, 10/10 and 10/11
    `unseen`. Read the invasive picks as the novel ones.
  - Section 5's two global tables are **unchanged** (verified byte-identical) and still use the
    original top/bottom-by-probability rule over the agreeing set.
- **Published report**: <https://claude.ai/code/artifact/87cbaae7-d1f0-4e47-be98-910e3fd198b3>
  — seven sections plus the register below.

**Cohort composition by sublineage (2026-09-04, `1a25d95`).** Built by
`kleb_iso_source.sublineage_composition`; every figure is in
`sublineage_composition_{all,heldout}.csv` + `sublineage_composition.json` beside the other artifacts.
The join is guarded on more than row count — it requires a total join, `blood_vs_faeces_label ==
y_true`, and identical split assignment for all 14,119 genomes.

- **Numbers of record.** `all` n=14,119: **53.2% blood observed / 44.8% predicted**. `heldout`
  (validate+evaluate) n=4,234: **53.5% / 43.4%**. Top 15 sublineages by size, remainder pooled.
- ⛔ **The predicted-minus-observed gap is a threshold property, not biology.** Youden maximises
  sens+spec and does not preserve prevalence; here sens 0.633 < spec 0.790, so at the deployed cut the
  model under-calls blood by **−8.4 pp** (`all`) / **−10.2 pp** (`heldout`) *globally*. `compose()`
  returns the offset so it cannot be forgotten.
- **The charts therefore use a prevalence-matched cut, not the deployed one (2026-09-04).**
  `prevalence_matched_threshold()` picks the cut at which the model calls blood as often as it occurs:
  **probability 0.2606 (logit −1.04)** against the deployed Youden **0.4349 (logit −0.26)**. Residual
  offset falls to **−1.5 pp** (`all`) / **−1.4 pp** (`heldout`), so a per-lineage deviation now means
  something — e.g. SL15 −10.8 pp stands out, which was invisible when every lineage sat at −10.
  - **Cost: balanced accuracy 0.711 → 0.706** (sens 0.633→0.718, spec 0.790→0.694). **AUROC is
    unchanged** — it does not depend on a threshold. Plain 0.5 would be *worse* (39.8% predicted).
  - ⛔ **Only the charts** — but note the reason, because the first one given was wrong. It is *not*
    circular to use this on the lab collection: the cut is derived on the labelled cohort and applied,
    exactly as the Youden cut is. The real caveat is **base-rate transfer** — a prevalence-matched cut
    bakes in the cohort's 52% blood rate, and the lab collection is carriage-heavy so its true rate is
    likely lower, which would make the matched cut over-call blood there. **The strain picks keep the
    Youden cut** for both models, the conservative choice for a candidate list.
  - Artifacts: `prevalence_matched_sublineage_composition_{all,heldout}.csv` + `.json` beside the
    Youden-cut originals; the JSON records both thresholds and which was used.
- **The charts are Bacformer alone.** The unitig model contributes nothing to this section; it enters
  only at the agreement 2×2 and the pick lists.
- **"other" is the largest single bar** — 1,363 sublineages, **6,313 genomes = 45% of the cohort**,
  2.6× SL258. Pinned last rather than sorted first.
- **Scope matters and both are published.** At `all` scope 70% of rows are memorised training data
  (train AUROC 0.959 vs 0.786), so predicted bars there are largely recall; `heldout` is the honest
  scope. SL3010 reads 53.3 → 37.5 on `all` but 40.7 → 22.2 on heldout.

**Open questions from the comparison (2026-08-14/18) — observations and hypotheses, not findings.**
None blocks anything; they are what the queued models below are pointed at.

- **O1 — the SL3010 contradiction.** Four ST3010 genomes sit in Bacformer's SL3010 bottom ten at
  0.079–0.144 while the unitig model scores them 0.863–0.922 (`VRES0604` 0.127/0.922, `VRES0606`
  0.125/0.903, `VRES0611` 0.079/0.889, `VRES0565` 0.144/0.863). Not scale compression — opposite sides
  of *both* models' Youden points. All four `unseen`, so neither is reciting a label.
  - ⚠ **CORRECTED 2026-09-04: SL3010 does have a within-lineage AUROC.** O1 originally said it had
    "no cohort per-SL AUROC to arbitrate", and the published reports repeated it. It is computable
    from the same `cohort_scores.npz` + split CSV that yield the other two, and reproduces them
    exactly (SL258 0.8577 vs 0.858 of record, SL307 0.8148 vs 0.815). **SL3010 evaluate n=36,
    16 positive, AUROC 0.650 [0.438, 0.837]** — the interval includes chance, so it is *weak and
    imprecise* validation rather than none. That is the more informative statement and also the
    worse one: SL3010 is the least-validated of the three lineages *and* carries the most model
    disagreement (24 of 40). Found by an adversarial verification pass over the artifacts.
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
- **O4 — ~~the disagreement is asymmetric ~3:1~~ — LARGELY A THRESHOLD ARTIFACT (corrected
  2026-09-04).** As written: of 166 disagreements, 124 unitig-invasive / Bacformer-faeces against 42
  the reverse. **Re-cutting both models at their prevalence-matched points collapses the skew to
  1.3:1** (85 vs 64) and lifts agreement 75.3% → 77.8%. The asymmetry was mostly Bacformer's Youden
  cut under-calling blood, not a property of the models.
  - Prompted by David spotting `SAMN18874767` in the SL258 picks — Bacformer 0.449, unitig 0.742,
    which reads as Bacformer hesitating over a genome the unitig model is sure of. At the matched
    cuts the margins are **+0.188 and +0.186**: equally confident, and the hesitancy was the cut.
  - ⚠ **Raw probabilities are not comparable between the two columns.** Bacformer's are compressed
    low (matched cut 0.261 against Youden 0.435); the unitig model's barely move (0.556 vs 0.527).
  - **What survives:** O3's "the collection sits low" is *not* an artifact. Even at the matched cut
    Bacformer calls only 46% of lab genomes blood against the cohort's 52% (39% at Youden).
  *Source:* re-cut counts from `lab_collection_invasion_predictions.csv` with prevalence-matched
  thresholds derived on each model's own evaluate split.

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

### 3.6 Multi-species AMR — EBI AMR Portal (*P. aeruginosa* → Enterobacterales → all)

**Status.** Scope is **PsA and Enterobacterales only** for now (David, 2026-11-19); the all-species
cohort stays planned. *P. aeruginosa* (PsA) **labels are built** (`7e47486`) and its **genomes are built and
QC'd**: 3,677 pass. **Enterobacterales labels are built** (`480ce9d`); **its genomes are built, QC'd and annotated
(67,326 pass the non-species QC, all with Bakta), and their ESM-C array is queued** (2026-11-21). David, 2026-11-20: start
Enterobacterales as soon as possible — **a cross-species model is the aim**.
**Annotation is Bakta 1.9.4 with the full DB v5.1, not Prodigal** (David, 2026-11-19: "Bakta as standard").
Our run reproduces BakRep's exactly (the gate below), so a genome takes BakRep's GFF3 where BakRep holds it on
our contigs, and our run otherwise. **PsA annotation is complete** (3,677 genomes: 1,781 BakRep, 1,896 ours);
**PsA ESM-C embeddings are complete and checked (3,677 of 3,677).** **All §3.6 data now lives under `$AMR` = `project_k/david/amr_prediction`**
(David, 2026-11-20; map of the old paths in §2). The Prodigal protein parquets (job
`41397962`) and ESM-C smoke (`41399821`) are superseded and kept aside as `$AMR/processed/_superseded_prodigal/paeruginosa_protein_sequences/`
and `paeruginosa_esm_smoke/`; the Prodigal ESM-C array (`41401432`) was cancelled before any task started. QC
(CheckM2, skani), MLST, the 3,677 pass set and the held-out-ST candidates do not use gene calls and stand. The PsA
held-out ST and cohort drops are deferred to the splits stage. Splits and training are not started. Mechanics:
[`apps/pa/CLAUDE.md`](src/bacpredict/apps/pa/CLAUDE.md) (its *Genomes* section still describes Prodigal); the
Bakta stages are [`apps/pa/scripts/annotate_pa_genomes.sh`](src/bacpredict/apps/pa/scripts/annotate_pa_genomes.sh).

**Numbers of record — PsA labels.** Read from
`$AMR/processed/paeruginosa/label_counts.csv`; its `manifest.json` records the input
md5s (EBI AMR Portal release 2026-07 + the AMR-GNN supplement) and the code SHA.

| Drug | Labelled genomes | % R | From MIC | From the source's call |
|---|--:|--:|--:|--:|
| ciprofloxacin | 3,394 | 62.6 | 3,143 | 251 |
| ceftazidime | 3,039 | 47.1 | 2,791 | 247 |
| meropenem | 2,989 | 35.4 | 2,893 | 94 |
| tobramycin | 2,797 | 37.3 | 2,639 | 158 |
| amikacin | 2,598 | 15.9 | 2,381 | 217 |
| imipenem | 2,299 | 60.5 | 2,068 | 231 |
| piperacillin-tazobactam | 2,144 | 52.8 | 2,088 | 56 |
| aztreonam | 1,964 | 38.6 | 1,915 | 49 |
| levofloxacin | 1,615 | 60.8 | 1,401 | 214 |
| cefepime | 1,264 | 45.7 | 1,177 | 86 |

- **24,103 clean labels over 3,932 genomes** (`cohort.csv`, `n_drugs_labelled > 0`): `genome_route`
  assembly 2,150, reads 1,782. These are *labelled* genomes, not yet genomes we hold.
- **The breakpoint table reproduces AMR-GNN's own EUCAST v15 calls 16,596 / 16,596**
  (`call_agreement.csv`). Against the portal's own (mostly CLSI) calls on 7,285 tests carrying both,
  the re-derivation flips 129 S→R and 153 R→S (meropenem MIC 8 is CLSI R, EUCAST S).

**Numbers of record — PsA genome sources.** Read from
`$AMR/processed/paeruginosa/genomes/genome_manifest.csv`, written on CSD3 by
`build_pa_genomes.sh plan` at `1f049c4` (with `--prefer-reads`, min depth 10×, genome size 6.6 Mbp):

| Source | Genomes |
|---|--:|
| AllTheBacteria (shovill 1.1.0) | 2,187 |
| Portal submitter assembly (`portal_gff`) | 1,236 |
| Shovill on ENA reads | 353 |
| Dropped: RNA-seq reads only | 143 |
| Dropped: long reads only | 7 |
| Dropped: under 10× depth | 6 |
| **Total labelled** | **3,932** |

- All 3,423 download-sourced genomes built `ok` (Prodigal + `mlst`): read from the `atb+portal_gff_*.tsv`
  shards in `genomes/shards/`, array `41326140`.
- No `portal_gff` genome has usable reads at ≥10×, so `--prefer-reads` moved none: the 1,236 have no
  shovill alternative.
- Shovill-sourced genomes by read depth: 10–20× 6, 20–30× 12, 30–50× 39, 50–100× 89, ≥100× 207.

**Numbers of record — PsA genome QC.** Read from `$AMR/processed/paeruginosa/genomes/genome_qc.csv` (collate at `e46c81e`);
shovill array `41326674` plus rerun `41335542`, QC job `41336261` (CheckM2 3 h 6 min at 32 cores).

| Outcome | Genomes | 0/1 labels |
|---|--:|--:|
| QC pass (= `embedding_input.csv`) | 3,677 | 22,670 |
| QC fail, built | 96 | 752 |
| Build failure | 3 | 16 |
| Dropped before building | 156 | 665 |
| **Total labelled** | **3,932** | **24,103** |

- **QC failures among built genomes** (a genome can fail several): completeness 47, species 38, contigs 10,
  contamination 6, N50 3, length 0. By source: atb 55 of 2,187, portal_gff 24 of 1,236, shovill 17 of 350 built.
- **Species** (skani, closest of PAO1/PA7 at ANI ≥ 95): *P. aeruginosa* 3,735, *P. paraeruginosa* 37
  (ANI to PA7 98.6–99.5, to PAO1 93.5–93.9; 25 are ST2211), unknown 1 (SAMN04040115: GC 39 %, 3.9 Mbp, no
  skani hit to either reference). The portal's genotype table flags 6 of the 37.
- **Build failures:** SAMEA7290678 (13.6×, assembled to 5.9 kbp); SAMN11110420 and SAMN11110638, whose ENA
  fastq for one mate is an empty *directory* on `ftp.sra.ebi.ac.uk` (it serves an HTML listing; the
  filereport still lists the file's size and md5).
- **Labels per drug after QC** (`st_counts_by_drug.csv`, summed): CIP 3,150 · CAZ 2,800 · MEM 2,746 ·
  TOB 2,578 · AMK 2,493 · IPM 2,199 · TZP 2,052 · ATM 1,888 · LVX 1,554 · FEP 1,210.
- **CDS per QC-passing genome:** 5,085–7,954; 2,458 of 3,677 exceed 6,000, none exceeds 9,000.
- **mlst** agrees with the AMR-GNN paper's ST for 2,188 of 2,189 genomes where both are numbers
  (`cohort.csv` `st_reported`; the exception is one mutL allele). 205 QC-passing genomes are untyped (`-`).

**Held-out-ST candidates** (≥ 50 QC-passing labelled genomes; `st_counts_by_drug.csv`, `genome_qc.csv`).
"Scoreable" = drugs where both R and S have ≥ 10 genomes in the ST.

| ST | Genomes | Drugs n ≥ 50 | Scoreable drugs | Largest study share | Typed SLV genomes outside the ST |
|---|--:|--:|--:|--:|---|
| 235 | 286 | 10 | 10 | 21 % | 1 (+13 untyped) |
| 621 | 244 | 8 | 6 (no S for CIP, LVX) | 99 % no study recorded | 20 (ST4588 ×19) |
| 175 | 144 | 8 | 4 | 44 % (two Barrio-Tofiño papers) | 0 |
| 111 | 125 | 9 | 8 | 30 % | 9 (ST966 ×6) |
| 244 | 115 | 8 | 7 | 21 % | 1 (+6 untyped) |
| 155 | 82 | 9 | 9 | 45 % | 3 |
| 253 | 80 | 5 | 5 | 18 % | 3 |
| 274 | 77 | 5 | 7 | 26 % | 16 (7 STs) |
| 17 | 68 | 1 | 6 | 21 % | 14 (ST845) |
| 179 | 63 | 3 | 8 | 29 % | 1 |
| 395 | 63 | 3 | 6 | 35 % | 4 |
| 308 | 57 | 1 | 6 | 21 % | 0 |

**Numbers of record — Enterobacterales labels.** Read from
`$AMR/processed/enterobacterales/label_counts.csv`, built by
`apps/entero/scripts/build_entero_ast_labels.sh` at `480ce9d` (every Enterobacterales genus in
`engine/ast_labels/breakpoints.py`, the PsA ten drugs and label rule, portal only).

| Drug | Labelled genomes | % R | From MIC | From the source's call |
|---|--:|--:|--:|--:|
| ciprofloxacin | 61,385 | 27.5 | 46,560 | 14,741 |
| meropenem | 46,499 | 6.0 | 36,737 | 9,737 |
| ceftazidime | 26,382 | 34.9 | 17,183 | 9,191 |
| amikacin | 21,194 | 11.2 | 18,544 | 2,618 |
| piperacillin-tazobactam | 20,797 | 27.7 | 14,573 | 6,224 |
| imipenem | 16,340 | 11.9 | 13,712 | 2,615 |
| cefepime | 13,748 | 42.1 | 11,813 | 1,929 |
| tobramycin | 13,179 | 38.3 | 10,171 | 3,000 |
| aztreonam | 11,812 | 49.9 | 9,973 | 1,837 |
| levofloxacin | 11,237 | 49.1 | 10,198 | 1,031 |

- **242,573 clean labels over 67,763 genomes** (`cohort.csv`, `n_drugs_labelled > 0`), 9 genera, 44 species
  labels: *Salmonella* 32,253, *Escherichia* 21,905, *Klebsiella* 9,478, *Shigella* 2,275, *Enterobacter*
  1,333, *Serratia* 329, *Proteus* 105, *Morganella* 60, *Providencia* 25. `genome_route` assembly 67,077,
  reads 686. 25 cells are fractional (conflicting tests).
- **EUCAST re-derivation moves %R per genus.** *Salmonella* ciprofloxacin is 17.3 % R under EUCAST (31,691
  genomes; `label_provenance.csv` × `cohort.csv` species) against 9.0 % under the portal's own calls
  (`$AMR/processed/ebi_amr_portal/scoping_2026-07/species_drug_counts.csv`; not the same genome set, so a contrast,
  not a paired comparison). Other large shifts in the same read: *Enterobacter* imipenem 13.9 % vs 54.4 %,
  *E. coli* aztreonam 30.0 % vs 12.3 %.
- **6,919 of the 7,088 Kp AST samples (§3.1) are in this cohort.** Any comparison against the July Kp
  checkpoints must hold out none of their train/validate samples.

**Numbers of record — PsA split tables (the baseline arm, ST155).** `$AMR/processed/paeruginosa/splits/<drug>_split.csv`
+ `split_manifest.json` (md5 `908fc0af…`), `engine/splits/build_amr_holdout_splits` at `f595f0d` via
`train_pa_models.sh splits` from the pinned worktree `BacPredict-f595f0d`: eligible = the 3,677 of
`bakta_annotation.csv` (each with a `.pt`), minus the 15 Ecuador genomes (`apps/pa/excluded_genomes.tsv`); the 3 CDC
duplicate pairs grouped, none of them spanning ST155. Genome level: train 2,290, validate 574, holdout (random 20 %
of what is left) 716, holdout_st (`paeruginosa:155`, exact) 82. Per drug (`check-splits`, 0/1 labels only):

| Drug | Train | Validate | Holdout | %R | ST155 | %R |
|---|--:|--:|--:|--:|--:|--:|
| ciprofloxacin | 1,933 | 504 | 620 | 63.4 | 78 | 71.8 |
| ceftazidime | 1,732 | 434 | 543 | 46.2 | 77 | 42.9 |
| meropenem | 1,703 | 426 | 527 | 36.1 | 75 | 26.7 |
| tobramycin | 1,586 | 409 | 508 | 38.8 | 75 | 34.7 |
| amikacin | 1,554 | 385 | 464 | 18.1 | 75 | 9.3 |
| imipenem | 1,357 | 349 | 410 | 58.3 | 69 | 69.6 |
| piperacillin-tazobactam | 1,266 | 321 | 386 | 54.4 | 66 | 54.5 |
| aztreonam | 1,161 | 304 | 357 | 38.9 | 66 | 31.8 |
| levofloxacin | 967 | 226 | 292 | 62.7 | 54 | 77.8 |
| cefepime | 736 | 183 | 229 | 53.7 | 47 | 53.2 |

- **The holdout is 21.8 % of genomes in all** (716 + 82 of 3,662), against 26.2 % for the ST235 arm; training gains
  69–166 genomes per drug. ST155 is **2.1–3.9 % of each drug's labelled set**, inside David's 5 % cap.
- **8 of 10 drugs are scoreable inside ST155** (≥ 20 in each class) — the same count as ST235, but failing on
  different drugs: amikacin has 7 R of 75 and levofloxacin 12 S of 54, where ST235 was thin on ciprofloxacin (17 S)
  and levofloxacin (15 S). ST155 is far better balanced overall (9–78 % R, against ST235's 34–93 %).
- Stage A/B were **not** repeated: no trainer code changed between the ST235 run (`9a813f0`) and `f595f0d` — only
  `HOLDOUT_ST` and docs — and `check-splits` is the trainer's own `resolve_split_table` read of all ten new tables
  (0 genomes without a `.pt`).

> **⚠ SUPERSEDED 2026-11-21 — both PsA panels below are the ST235 arm, and are being replaced.** David capped the
> lineage holdout at **5 % of a drug's labelled set**; ST235 is 7.1–9.9 % (mean 8.6 %), which with the random 20 %
> put **26–28 % of genomes out of reach of training**. The held-out ST is now **ST155** (2.1–3.9 %, the same 8 of 10
> drugs scoreable). ⛔ **Every number in both panels is superseded, not just the ST column**: the builder permutes
> the whole non-held-out pool, so returning ST235 to it and removing ST155 re-draws train, validate *and* the random
> holdout. Of the 675 old random-holdout genomes only 136 stay in it (418 → train, 102 → validate, 19 → holdout_st),
> so the old and new random-holdout AUROCs are **on different genomes and cannot be paired**. The ST235 artifacts are
> preserved at `$AMR/processed/paeruginosa/_superseded_st235/` and stay quotable *as the ST235 arm*. Decision row in
> §6, 2026-11-21.

**Numbers of record — PsA split tables (the ST235 arm, superseded).** `$AMR/processed/paeruginosa/_superseded_st235/splits/<drug>_split.csv`
+ `split_manifest.json` (md5 `9e5e3063…`), `engine/splits/build_amr_holdout_splits` at `8b791d4` via
`train_pa_models.sh splits`: eligible = the 3,677 of `bakta_annotation.csv` (each with a `.pt`), minus the 15
Ecuador genomes (`apps/pa/excluded_genomes.tsv`); the 3 CDC duplicate pairs grouped. Genome level: train 2,159,
validate 542, holdout (random 20 %) 675, holdout_st (`paeruginosa:235`, exact) 286. Per drug (`check-splits`, 0/1
labels only):

| Drug | Train | Validate | Holdout | %R | ST235 | %R |
|---|--:|--:|--:|--:|--:|--:|
| ciprofloxacin | 1,846 | 463 | 591 | 57.7 | 235 | 92.8 |
| ceftazidime | 1,598 | 403 | 509 | 40.5 | 276 | 75.0 |
| meropenem | 1,590 | 396 | 489 | 33.3 | 256 | 62.1 |
| tobramycin | 1,482 | 381 | 491 | 33.0 | 224 | 68.3 |
| amikacin | 1,442 | 370 | 457 | 11.6 | 209 | 34.0 |
| imipenem | 1,271 | 338 | 420 | 58.3 | 156 | 78.2 |
| piperacillin-tazobactam | 1,198 | 311 | 376 | 47.9 | 154 | 85.1 |
| aztreonam | 1,100 | 280 | 347 | 35.4 | 161 | 65.8 |
| levofloxacin | 874 | 230 | 283 | 57.2 | 152 | 90.1 |
| cefepime | 679 | 190 | 221 | 44.8 | 105 | 76.2 |

- ST235 is mostly resistant (62–93 % R for nine drugs), so its holdout has few susceptible genomes for some drugs
  (ciprofloxacin ~17 S). That bounds how precisely a zero-shot AUROC can be read there.

**Numbers of record — PsA baseline models, the ST235 arm (superseded).**
`$AMR/processed/paeruginosa/_superseded_st235/models/<drug>/results.json`,
collected in `_superseded_st235/models/holdout_metrics.csv` (md5 `4af2fc6f…`): array `41548353` at `9a813f0` (`finetune_amr
--split-table`, bf16, deployed TB/Kp hyperparameters, early stopping on validate AUROC; 4.0–7.2 h per drug on one
A100), on the split tables above. AUROC (balanced accuracy at 0.5 in brackets):

| Drug | Random holdout (n, R) | AUROC | ST235 zero-shot (n, R) | AUROC | Validate AUROC (best) |
|---|---|--:|---|--:|--:|
| ciprofloxacin | 591, 341 | 0.827 (0.762) | 235, 218 | 0.783 (0.518) | 0.824 |
| ceftazidime | 509, 206 | 0.774 (0.708) | 276, 207 | 0.595 (0.582) | 0.778 |
| meropenem | 489, 163 | 0.807 (0.739) | 256, 159 | 0.703 (0.636) | 0.805 |
| tobramycin | 491, 162 | 0.944 (0.899) | 224, 153 | 0.872 (0.836) | 0.951 |
| amikacin | 457, 53 | 0.779 (0.684) | 209, 71 | 0.550 (0.539) | 0.793 |
| imipenem | 420, 245 | 0.889 (0.835) | 156, 122 | 0.764 (0.773) | 0.886 |
| piperacillin-tazobactam | 376, 180 | 0.709 (0.672) | 154, 131 | 0.562 (0.513) | 0.753 |
| aztreonam | 347, 123 | 0.668 (0.607) | 161, 106 | 0.514 (0.533) | 0.695 |
| levofloxacin | 283, 162 | 0.789 (0.694) | 152, 137 | 0.749 (0.582) | 0.796 |
| cefepime | 221, 99 | 0.671 (0.625) | 105, 80 | 0.476 (0.466) | 0.681 |

- No confidence intervals yet; ST235 has few susceptible genomes for several drugs (ciprofloxacin 17, levofloxacin 15).
- ⚠ `results.json` records `split.table` as an **absolute path** into `splits/`, which now resolves to the ST155
  table. `split.table_md5` is the only field that ties a result to the table it was actually scored on.
- The best validate checkpoint came early (step 500–2,250, of 3,360–6,000 run); validate AUROC then fell 0.01–0.05.
- Not yet scored: mechanism-stratified metrics (§0.4: HGT/acquired vs chromosomal), calibration read-out.

**Numbers of record — Enterobacterales genome sources.** Read from
`$AMR/processed/enterobacterales/genomes/genome_manifest.csv` (`build_entero_genomes.sh plan` at `6369d50`; per-genus
genome sizes, `--prefer-reads`, min depth 10×, a fresh ENA cache):

| Source | Genomes | In BakRep (`in_bakrep_ids.txt`) |
|---|--:|--:|
| AllTheBacteria | 65,363 | 63,700 |
| Portal submitter assembly (`portal_gff`) | 2,049 | 0 |
| Shovill on ENA reads | 214 | 0 |
| Dropped (below 10× 55, no runs 31, other strategy 28, single-end 22, RNA-seq 1) | 137 | – |
| **Total labelled** | **67,763** | **63,700** |

- No `portal_gff` genome has usable reads at ≥ 10×, so `--prefer-reads` moved none. No genome is shared with the PsA
  cohort (the contig store is). The ATB rows are frozen read-only as `genomes/genome_manifest_atb.csv`.
- Our Bakta will run on 3,926 genomes (1,663 ATB not in BakRep + 2,049 portal + 214 shovill) plus every BakRep file
  ingest refuses, less build/QC failures.
- **Build smoke** (`shards_smoke/`, job `41518648` at `fbc500a`): 1 ATB genome per genus + 2 portal, 11 of 11 ok;
  `mlst` autodetect chose `ecoli_achtman_4` (*E. coli*, *Shigella*), `salmonella`, `klebsiella`, `ecloacae`,
  `serratia`, `proteus`, `providencia`, and no scheme for *Morganella* (none exists). One portal genome labelled
  *Escherichia* typed as `salmonella` ST3529 — the kind of label/species conflict the species check is for.
- **ATB task shape, measured** (array `41519203`, 100 genomes per 8-core task, first 248 tasks): 2.0–4.4 min per task
  (median 2.7), MaxRSS 1.6 GB. The remaining tasks were cut from 45 min / 24 GB to 10 min / 6 GB, which made them
  backfill again on a busy queue.

**Numbers of record — Enterobacterales build, QC and annotation.**
- **Build:** 67,575 of 67,626 buildable genomes built (`build_entero_genomes.sh status`): atb 65,363, portal 2,049,
  shovill 163 of 214. The 51 failures are all shovill: 45 `ftp.sra.ebi.ac.uk` refused the connection (EBI throttling;
  transient), 4 HTTP 403 on the reads, 1 SPAdes, 1 Prodigal. Not rerun; they are outside QC.
- **Non-species QC** (`collate_genomes --species-mode report`; `genomes/genome_qc.csv` md5 `d0aa9ef8…`): 67,326 pass,
  300 fail (contamination 183, contigs 76, completeness 61, not built 51, N50 35, length 3; a genome can fail
  several). The pass set is `genomes/embedding_input_unfiltered.csv` (md5 `d255a91f…`): the cohort that is annotated
  and embedded (§6). CheckM2: 34 + 1 top-up tasks of ≤ 2,000 genomes at 32 cores, median 88 min, MaxRSS ≤ 67 GB.
- **Species calls** (skani vs the 39-genome DRAFT panel; `genomes/species_crosstab.csv` md5 `832ff16b…`): 66,277 of
  67,575 called at ≥ 95 % ANI. Patterns, not conclusions: 8,319 genomes labelled *E. coli* are called *S. boydii* /
  *S. dysenteriae* with a median 0.14 % ANI gap to the runner-up (labelled *Shigella* separate by ~1.4 %); most
  *Enterobacter* (*E. cloacae* 611 of 745, *E. hormaechei* 283 of 292) and *Serratia marcescens* (263 of 307) fall
  below 95 % to every reference; 4 labelled *K. pneumoniae* are called *Proteus mirabilis*; 207 labelled *K.
  pneumoniae* are other KpSC members. **Hypotheses:** the panel's *E. coli* references (phylogroups A, E) miss B2/D/B1;
  the *Enterobacter*/*Serratia* clinical lineages lack references. The species rule is David's (asked 2026-11-21).
- **BakRep ingest** (job `41530416`, 2 h 4 min, 8 streams; `genomes/bakta/ingest_status.tsv` md5 `ef3731b2…`):
  63,552 of 63,700 ingested, **148 refused:names** (0.23 %; PsA 0.61 %), no errors. Our Bakta: 2,903 (array
  `41530989`, all 726 tasks COMPLETED), the 148 refused (array `41541891`, 37 tasks COMPLETED) and the 972 genomes
  built after the plan (array `41561033`, 243 tasks COMPLETED).
- **Annotation of record** (`annotate_entero_genomes.sh collate`, `COHORT=embedding_input_unfiltered.csv`, job
  `41581536` at `9a813f0`; `genomes/bakta_annotation_embedding_input_unfiltered.csv` md5 `5f701793…`): **67,326 of
  67,326** QC-pass genomes have one current Bakta annotation — 63,323 BakRep, 4,003 our run. (63,552 were ingested;
  the other 229 failed QC.)
- **Proteins** (`build_entero_embeddings.sh proteins`, job `41582191`, 26 min, 32 cores): 67,326 parquets;
  `check-proteins` (job `41592749`) PASS, 0 problems (protein count = CDS − pseudogenes; contigs unchanged since
  annotation).
- **Kp ESM-C reuse** (`build_entero_embeddings.sh link-kp`, job `41594725`; `genomes/esm_linked_kp.tsv` md5
  `67d6351f…`): of the 9,370 cohort genomes labelled *Klebsiella*, **8,651 linked** (all fp32), 203 refused for a
  protein-count mismatch, 516 with no Kp `.pt`. The 719 unlinked are embedded by the array like any other genome.
- **ESM-C smoke** (job `41595953`, 8 min on one A100): 10 genomes, one per genus (*Salmonella, Escherichia, Klebsiella,
  Enterobacter, Proteus, Morganella, Providencia, Serratia, Shigella*), each bf16 `(1, n, 960)` with n equal to its
  parquet's protein count; `check-smoke` PASS. Written to `$AMR/processed/esm_smoke/`, not the store.
- **MLST re-typed with the scheme pinned per species — the 7.7 % artefact is gone** (jobs `41604641`,
  `41604643`, `41604644`, `41604645`, 2026-11-21; 88 tasks, every one COMPLETED). **5,142 of the 67,326 QC-pass
  genomes were on the wrong scheme** and were re-typed: 1,807 → `ecoli_achtman_4`, 1,751 → `salmonella`,
  1,472 → `klebsiella`, 112 → `ecloacae`. All 5,142 came back on the pinned scheme, **5,104 changed ST**, and
  5,020 (97.6 %) carry a real ST. **QC-pass genomes whose scheme genus disagrees with the label: 0** — was 5,164
  (7.7 %) (`genomes/genome_qc.csv`, re-collated 2026-11-21). The four largest artefacts are all at zero:
  `ecoli_achtman_4:102` 1,540 → 0, `salmonella:3529` 1,270 → 0, `ecoli_achtman_4:14464` 955 → 0,
  `ecoli_achtman_4:11307` 434 → 0.
- **⚠ That "0" first had a blind spot, since closed.** The check mapped scheme names to genera and had no entry for
  `kaerogenes`, so rows carrying it were skipped rather than tested. Underneath it, *Enterobacter aerogenes* — 28
  QC-pass genomes — was still split three ways by autodetect (21 `kaerogenes`, 4 `cronobacter`, 3 `ecloacae`): one
  species, three `st_key` namespaces. *E. aerogenes* was reclassified as *Klebsiella aerogenes* and has its own
  PubMLST scheme, so `kaerogenes` is correct, and our own skani call agrees (25 of the 28 are called
  *K. aerogenes*). The 7 stragglers were re-typed (job `41608859`, 49 s) and re-collated.
  **The check is now "no species carries more than one scheme", which needs no genus map**: 1 species trips it,
  *Enterobacter chengduensis* (7 `ecloacae`, 1 untyped), and an untyped genome is a failure to type, not a competing
  scheme — so **0 species carry two real schemes**. Untyped (`st` = `-`) among QC-pass: 603, nearly all of genera
  with no PubMLST scheme (*Morganella*: 0 of 60 type, and no Morganella scheme ships in our pixi db). The rise from
  600 is the 3 of those 7 that have no `kaerogenes` ST — no ST is better than a wrong-scheme one.
- **⛔ Which shard row wins is decided by FILE MTIME, and nothing encodes it in the filenames.**
  `collate_genomes.read_shards` sorts `shards/*.tsv` by mtime and keeps the last row per `Sample`
  (`collate_genomes.py:60-66`). The re-typed STs win only because the `retype-*` copies were `touch`ed after being
  copied in. Anything that rewrites mtimes in `genomes/shards/` — a restore, a copy without `-p`, an `rsync` without
  `-t` — would silently resurrect the pre-re-type STs, with no error anywhere. **Assert after every collate that no
  species carries more than one real scheme**; it is one `groupby` and it catches this immediately.
- **Nothing but the typing changed.** The re-run took per-scheme subsets of the *unchanged* `genome_manifest.csv`
  (`genomes/retype_mlst/genome_manifest_<scheme>.csv`), so each genome's contigs were reused on their matching
  `src=` header and the Prodigal GFF on its `contigs_md5=`. Diffing `genomes/genome_qc.csv` against
  `genomes/_pre_retype/genome_qc.csv`: **0 rows with a changed `n_cds`, `n_contigs` or `total_length`**, the same
  67,326 QC-pass genomes, and the same sample set in `embedding_input_unfiltered.csv`. `contigs_path`/`gff_path`
  differ in spelling only (`/rds/project/…` vs `/home/dca36/rds/…`) and resolve to the same file for all 5,142.
  **Cause, fixed in this run but NOT yet in the script:** `build_entero_genomes.sh` leaves `MLST_SCHEME` unset, so
  `run_mlst` omits `--scheme` and `mlst` auto-detects per genome (`engine/genomes/build_genomes.py:272-277, 340`).
  **PsA was never affected** — `build_pa_genomes.sh` pins `SCHEME=paeruginosa` (all 3,677 QC-pass carry it).
- **Held-out-ST candidates, re-typed** (`genomes/genome_qc.csv` × `binary_ast.csv`, 2026-11-21; share is of each
  drug's *labelled* set, which runs well above the share of the unit because high-risk clones are over-represented
  among drug-tested isolates; "scoreable" = ≥ 20 R and ≥ 20 S inside the ST):

| Unit | ST | n | mean % | max % | scoreable |
|---|---|--:|--:|--:|--:|
| *E. coli* | ST131 | 3,029 | 22.3 | 30.4 | 9/10 |
| *E. coli* | ST10 | 756 | 2.4 | 3.7 | 7/10 |
| *E. coli* | ST38 | 482 | 2.5 | 3.3 | 7/10 |
| *Klebsiella* | ST258 | 1,085 | 15.8 | 21.0 | 6/10 |
| *Klebsiella* | ST15 | 565 | 5.4 | 7.7 | 8/10 |
| *Klebsiella* | ST37 | 211 | 2.2 | 2.5 | 7/10 |
| *Salmonella* | ST32 | 1,586 | 4.3 | 7.6 | 2/9 |
| *Enterobacter* | ST171 | 69 | 6.6 | 9.2 | 2/10 |

  ST131 more than doubled on re-typing (1,796 → 3,029) and *Klebsiella* ST258 and ST307 grew likewise, which is the
  measure of how much the artefact was hiding. **There is no *E. coli* ST37** under any scheme.
  *S. sonnei* and *S. flexneri* admit no ST holdout at all: ST152 is 97 % of *S. sonnei* and ST245 is 84 % of
  *S. flexneri*, so holding either out is a species holdout.
- **ESM-C: COMPLETE.** Array `41596846` (58,675 genomes, `genomes/esm_samples_to_embed.txt` md5 `a5bd7804…` — the
  cohort minus the 8,651 linked; 196 tasks of ≤ 300, bf16, `--max-proteins 9000`): **all 196 COMPLETED**, longest
  task 00:42:53, peak 3,722 MB. `check_cohort_store` (job `41632894`) then found **all 67,326 cohort genomes have
  an embedding** (8,651 of them symlinks into the older Kp store) with **0 rows in `problems.csv`** — protein
  counts, contig counts and shapes all agree. Sampled dtypes are `bfloat16` for files and `float32` for links,
  which is the expected split: the linked store holds bf16 values in an fp32 container, and `datasets.py:54-65`
  casts everything to fp32 at load, so model precision comes from `--precision`, not from the store.
  **All 9,370 `kpsc` genomes now have a `.pt`**, so the Klebsiella arm's embedding gate is cleared.

**Numbers of record — BakRep coverage.** BakRep v2 serves AllTheBacteria's own Bakta run (Bakta 1.9.4, full
DB v5.1, over ATB release 0.2 + the 2024-08 increment). Read from
`$AMR/processed/bakrep_coverage_20261119/join_output.txt` (`join.py` beside it, re-run on CSD3): the BakRep export
staged at `$AMR/raw/bakrep/v2_export_20250612/bakrep-export.tsv.gz` joined to each cohort and the ATB inventory.

| Cohort | In BakRep | We must run Bakta |
|---|--:|--:|
| PsA, QC-pass | 1,792 of 3,677 | 1,885: 339 ATB 2026-09 batch, 1 ATB 2025-06, 1,212 `portal_gff`, 333 shovill |
| Enterobacterales, labelled | 63,700 of 67,763 | 4,063: 283 ATB Bakta failures (274 *Salmonella*), 1,380 later ATB batches, 2,176 non-ATB assemblies, 224 reads-only |

- For all 1,843 PsA genomes BakRep holds, its contig count, total length and N50 equal ours exactly
  (`cmp_pa.py`); sequence-level checks on 4 genomes match byte for byte (BakRep ran Bakta with
  `--keep-contig-headers`), so BakRep GFF3 coordinates sit on our ATB contigs.
- **⚠ The PsA species evidence has no published reference panel beside it.** `paeruginosa/genomes/qc/` holds
  `skani_dist.tsv` and nothing else: PsA's skani ran inside `run_genome_qc.sh` (`build_pa_genomes.sh qc`), which
  writes only the result, not through `run_skani.sh`, which publishes `skani_dist.refs.tsv` beside it (and, since
  `d6c3e05`, `skani_dist.list.tsv`). Enterobacterales has the refs file; PsA does not, and neither cohort has the
  list file, both having run before it existed. **The panel is recoverable but not published:** the only two
  `Ref_file` values in `skani_dist.tsv` are `$AMR/raw/reference_genomes/GCF_000006765.1.fna.gz` (PAO1,
  *P. aeruginosa*) and `GCF_000017205.1.fna.gz` (PA7, *P. paraeruginosa*), which are the two `--species-ref`
  arguments `build_pa_genomes.sh collate` passes. **No refs file has been written after the fact** — provenance
  reconstructed from a script is not provenance, and the result's own `Ref_file` column is the better evidence.
  Any consumer (CABBAGE v2 §3.7 reuses these results) must read the panel from those paths and tolerate both files'
  absence for a pre-`d6c3e05` run.
- **BakRep's MLST is not used.** It leaves 464 of the 1,843 PsA genomes untyped where ours leaves 132, and
  never disagrees where both type (`cmp_pa.py`). Our own `mlst` 2.35.0 runs on every genome.
- ⚠ Equal contig count, length and N50 did **not** guarantee equal contig *names*: see the ingest refusals below.

**Numbers of record — Bakta reproduction gate.** Read from
`$AMR/processed/bakta_vs_bakrep_validation/bakta_gate_summary.csv` (`engine/genomes/compare_bakta_to_bakrep.py`;
array `41495079` at `be9891b`, compared at `b72b500`). Our Bakta 1.9.4 + DB v5.1 full (`$AMR/raw/bakta_db/v5.1_full`,
AMRFinderPlus DB 2024-07-22.1, the `bakta` env of `engine/genomes/pixi.toml`), BakRep's flags
(`--keep-contig-headers`, plus `--skip-plot`), on 18 genomes: 4 PsA and 11 Enterobacterales that BakRep holds
(3 each of *E. coli*, *S. enterica*, *K. pneumoniae*; 1 *E. cloacae*, 1 *S. sonnei*; 31–3,010 contigs), and 3 of
the 283 whose ATB Bakta run failed.
- **On all 15 BakRep-held genomes our output is BakRep's:** the same CDS (4,295–6,269 per genome: coordinates,
  strand, phase, pseudogene flag), no feature on one side only, no attribute difference on any feature (products,
  Dbxrefs, locus tags), the same AMRFinderPlus calls (10–66 per genome), and identical extracted protein lists.
  `bakta_gate_diffs.csv` is empty.
- **The protein extractor equals Bakta's own `.faa`** on all 18, residue for residue, pseudogenes aside (4–40 per
  genome: Bakta writes them to its `.faa`, the extractor drops them by design).
- **The 3 ATB-failure genomes annotate normally** (4,267–4,700 CDS).
- **Cost (sacct):** 7.7–9.5 min per genome at 10 cores up to 513 contigs, 16.2 and 18.6 min at 1,677 and 3,010
  contigs; MaxRSS 8.7–42.6 GB (bimodal: 7 tasks ≤ 35 GB, 11 at 37.5–42.6 GB).

**Numbers of record — PsA Bakta annotation.** `annotate_pa_genomes.sh plan` at `503be9a`: of the
3,677 QC-passing genomes, 1,792 are ATB-sourced and in BakRep (`genomes/bakta/bakrep_roster.tsv`), 1,885 are not
(`run_roster.tsv`: 1,212 `portal_gff`, 340 ATB, 333 shovill). Ingest (job `41497655`, 4 min):
- **1,781 BakRep GFF3s accepted** into `$AMR/raw/bakta/`, each kept byte for byte and accepted only
  with BakRep's release header and its embedded contigs equal to ours, name for name and base for base.
- **11 refused** (SAMEA5312861, SAMN02360661, SAMN04901631, SAMN07291495, SAMN07291501, SAMN07291533, SAMN07291537,
  SAMN07291538, SAMN07291549, SAMN07291553, SAMN11110650): AllTheBacteria's contig numbering in these
  files has gaps, which our store renumbers consecutively, so BakRep's seqids name different contigs of ours. The
  sequences are the same (identical multisets on the 4 checked). They join our run: 1,896 genomes
  (`run_roster_with_refused.tsv`).
- **Our run: 1,896 of 1,896 annotated, 0 failed** (array `41499745` at `871bc69`, 474 tasks × 4 genomes, all
  `COMPLETED`; sacct: 33 min mean and 61 min max per task, MaxRSS 34.6 GB at most at 8 cores).
- **Collate: 3,677 of 3,677 genomes have exactly one current annotation** (bakrep 1,781, bakta_run 1,896):
  `$AMR/processed/paeruginosa/genomes/bakta_annotation.csv`, md5 `53bb2fa5…` (re-collated from the moved
  paths at `0f5cc5d`; byte-identical to the path-rewritten table). Proteins per genome (CDS − pseudogenes):
  5,065–7,445, median 6,209 (BakRep) and 6,127 (ours); none near the 9,000 cap. Pseudogenes per genome: ATB-sourced
  3–53 (99th percentile 33); `portal_gff` up to 766 (below).

**Numbers of record — CheckM2 scikit-learn check.** CheckM2 1.1.0 warns that its scaler was pickled under
scikit-learn 1.6.1 and is loaded under 1.9.1. Rerun on 50 PsA genomes (10 nearest the QC thresholds, 40
random) under both versions, in two envs that differ only in scikit-learn (job `41405814`): the two
`quality_report.tsv` are **byte-identical** (md5 `df9df788…`) and equal `genome_qc.csv` exactly. No pin; the
PsA QC numbers stand. Artifacts: `$AMR/processed/paeruginosa/genomes/qc/checkm2_sklearn_check/`.

**Numbers of record — PsA ESM-C embeddings.** `$AMR/processed/esm/<Sample>_esm_embeddings.pt`, 3,677 of 3,677:
array `41526902` at `4ff29d7` (16 × A100, FLOTO-SL2-GPU; 66–73 min per task, 18.6 GPU-h in all) over
`processed/paeruginosa/esm_samples.txt` (md5 `7944b380…`), from the proteins of job `41519198`
(`$AMR/processed/protein_sequences/`). `build_pa_embeddings.sh check` at `4ff29d7` passes: every cohort genome has a
readable parquet holding CDS − pseudogenes proteins, contigs unchanged since annotation, a readable `.pt` newer than
its parquet, and 10 sampled `.pt` of shape (1, proteins, 960), bf16, no NaN. ~11 MB per genome (~41 GB).

**Measured — ESM-C throughput on one A100** (PsA smoke, job `41399821`, Prodigal inputs, superseded but the
rate stands): ~18 s per genome at ~6,400 proteins (2.8 ms per protein) after warm-up, HuggingFace ESM-C (no
faESM), bf16, `--max-proteins 9000`; log `pa-esm-smoke_41399821_0.err`. 34 of 3,677 PsA genomes have more
than 1,000 contigs; Bacformer's input packing (`max_n_contigs=1000`) keeps all their proteins but gives every
contig past the 1,000th the same contig id.

**Numbers of record — the Enterobacterales species units.** `apps/entero/species_units.py` (`assign_units`,
`unit_samples`), read against `genomes/genome_qc.csv` 2026-11-21. A unit is only a `--species NAME` in the splits
builder, which **does not filter by species**: a unit's eligible CSV is its whole membership
(`build_amr_holdout_splits.py:390`) and the NAME seeds its shuffle, so this rule is the single definition every arm
reads — the single-species arms, the cross-species arm and CABBAGE v2 (§3.7).

| Unit | QC-pass genomes | Holds |
|---|--:|---|
| `salmonella` | 32,149 | genus *Salmonella* (serovar labels too: `Salmonella Infantis`, `Salmonella sp.`) |
| `ecoli` | 21,720 | *E. coli*, plus *S. boydii* 12, *S. dysenteriae* 6 and a bare `Shigella` 3 |
| `kpsc` | 9,370 | the 5 KpSC species named explicitly (here: *K. pneumoniae* 9,350, *K. quasipneumoniae* 20) |
| `ssonnei` | 1,851 | *S. sonnei* |
| `enterobacter` | 1,323 | genus *Enterobacter*, plus *K. aerogenes* under **both** spellings |
| `sflexneri` | 395 | *S. flexneri* |
| *no unit* | 518 | *Serratia* 329, *Proteus* 104, *Morganella* 60, *Providencia* 25 |

- **⛔ A unit NAMES its species; it never prefix-matches the genus.** *K. aerogenes* is not KpSC, and the two label
  sheets spell it differently (2026-07 `Enterobacter aerogenes` 28; v2 `Klebsiella aerogenes` 424), so a prefix rule
  puts one organism in two units depending only on which sheet a row came from. Pinned by a test.
- **⚠ Awaiting David's decision, and invisible from this cohort alone.** The *K. oxytoca* complex is *Klebsiella* but
  not KpSC, so it falls in no unit: none here, ~330 in v2 (`UNDECIDED_KOSC`). Nine genera exist only in v2
  (`UNDECIDED_GENERA_V2`): *Citrobacter* 399, *Yersinia* 296, *Kluyvera* 12, *Pluralibacter* 9, *Hafnia* 8,
  *Leclercia* 3, *Cronobacter* 2, *Erwinia* 1, *Pantoea* 1. With the four undecided genera (518 here, ~1,435 across
  both cohorts) that is three groups needing one unit decision for both cohorts.
- **⛔ The lists are documentation; nothing refuses yet.** An arm that filters to the units it knows still drops every
  unassigned genome silently. The guard belongs at the call site — assert that every QC-pass genome is in a used
  unit, an arm's explicit exclusion, or `UNDECIDED`, and print the counts. Agreed with the v2 agent as one shared
  helper, to be written once the units are settled; **not written yet**.

**Numbers of record — PsA holdout clonal similarity.** `$AMR/processed/paeruginosa/ani/psa_heldout_max_ani_to_train.csv`,
written by job `41644484` (`skani dist`, every held-out genome against all 2,290 training genomes, 798 × 2,290 =
1.83 M pairs, 32 cores, ~10 min). Each row is one held-out genome's **maximum** ANI to any training genome.
⚠ skani prints two decimals and scores a genome **against itself at 99.99**, so `≥ 99.99` is the resolution limit,
not a measurement — it cannot separate a clone from the same genome.

| | random `holdout` (n=716) | `holdout_st` = ST155 (n=82) |
|---|--:|--:|
| min / p25 / median / p75 / max | 99.13 / 99.94 / **99.98** / 99.99 / 100.00 | 99.58 / 99.77 / **99.86** / 99.97 / 99.99 |
| ≥ 99.99 to some training genome | **351 (49.0 %)** | 13 (15.9 %) |
| ≥ 99.95 | 520 (72.6 %) | 25 (30.5 %) |
| ≥ 99.90 | 585 (81.7 %) | 30 (36.6 %) |
| ≥ 99.50 | 682 (95.3 %) | 82 (100 %) |

**Half the random holdout has a training genome at the limit of what skani can tell apart from itself.** ⚠ A
thinning threshold cannot be read off the low end of this range: PsA core ANI between *unrelated* strains is already
~99.0–99.5, so a 99.5 cut deletes 95 % of the holdout as "clonal" when most of that is species-level core identity.
The usable band is 99.90–99.99, which is ~10 distinguishable skani values wide. Whether whole-genome ANI is the right
metric at all is open (§6) — it is core-dominated, and acquired AMR determinants are largely accessory.

**Numbers of record — the clonal-lookup baseline.** `$AMR/processed/paeruginosa/ani/psa_nn_lookup_baseline.csv`,
written by `engine/splits/holdout_lookup_baseline.py` from the same skani TSV and the ST155 split tables. For each
held-out genome it takes the highest-ANI **labelled** training genome (every genome tied at that ANI votes) and
predicts that label. **No model is involved**, and the metrics come from `engine/finetune/metrics.compute_full_metrics`
— the same function that wrote the models' `results.json`, so the two are computed identically.

| Drug | lookup AUROC, `holdout` | lookup AUROC, ST155 |
|---|--:|--:|
| tobramycin | 0.7988 | 0.5985 |
| amikacin | 0.7316 | 0.5000 |
| levofloxacin | 0.7101 | 0.4167 |
| ciprofloxacin | 0.6994 | 0.4732 |
| meropenem | 0.6760 | 0.5723 |
| ceftazidime | 0.6599 | 0.5155 |
| aztreonam | 0.6590 | 0.5704 |
| piperacillin-tazobactam | 0.6461 | 0.4681 |
| cefepime | 0.6427 | 0.5491 |
| imipenem | 0.6199 | 0.4926 |
| **mean** | **0.684** | **0.516** |

**Roughly 0.68 AUROC of the random holdout is available from clonal lookup alone; the ST155 holdout yields
chance (0.516).** That is the size of the inflation term in every random-holdout number in this layer.

**Numbers of record — PsA baseline models, the ST155 arm. COMPLETE, all 10 drugs** (array `41605929`, 4.3–7.0 h
per drug). Read from `$AMR/processed/paeruginosa/models/<drug>/results.json`, against the lookup baseline on the
same held-out sets.

| Drug | `holdout` AUROC | lookup | margin | ST155 AUROC | lookup | margin |
|---|--:|--:|--:|--:|--:|--:|
| tobramycin | 0.9326 | 0.7988 | +0.134 | 0.9819 | 0.5985 | +0.383 |
| imipenem | 0.8845 | 0.6199 | +0.265 | 0.7649 | 0.4926 | +0.272 |
| levofloxacin | 0.8227 | 0.7101 | +0.113 | 0.6865 | 0.4167 | +0.270 |
| ciprofloxacin | 0.8173 | 0.6994 | +0.118 | 0.6615 | 0.4732 | +0.188 |
| meropenem | 0.8098 | 0.6760 | +0.134 | 0.6673 | 0.5723 | +0.095 |
| amikacin | 0.7900 | 0.7316 | +0.058 | 0.5924 | 0.5000 | +0.092 |
| piperacillin-tazobactam | 0.7529 | 0.6461 | +0.107 | 0.6796 | 0.4681 | +0.212 |
| ceftazidime | 0.7484 | 0.6599 | +0.089 | 0.6856 | 0.5155 | +0.170 |
| aztreonam | 0.7456 | 0.6590 | +0.087 | 0.6899 | 0.5704 | +0.120 |
| cefepime | 0.7138 | 0.6427 | +0.071 | 0.6664 | 0.5491 | +0.117 |
| **mean** | **0.802** | **0.684** | **+0.117** | **0.708** | **0.516** | **+0.192** |

**The fine-tune beats clonal lookup on both holdouts, and by more out of lineage.** On ST155 the lookup is at
chance, so there is nothing clonally close to memorise, and the model still scores 0.59–0.98. The ST155 holdout is
below the random one for 9 of 10 drugs (mean −0.114); tobramycin alone rises. Hypotheses and limits: *Caveats*.

**Numbers of record — PsA label noise at the breakpoint.** `$AMR/processed/paeruginosa/mic_bands/`, written by
`apps/pa/mic_distance_bands.py` (bands) and `apps/pa/holdout_error_by_mic_distance.py` (the re-scoring). Both are
**re-scoring of saved predictions** — no retraining, no split touched. Argument and evidence:
[`docs/holdout_clonality_and_label_noise.md`](docs/holdout_clonality_and_label_noise.md).

- **The I zone is folded into S and nothing records it.** The label rule is *resistant iff MIC > `r_gt_mg_l`*;
  `s_le_mg_l` sits in `eucast_v15_breakpoints.csv` and is read by **no code in the repo**
  (`breakpoints.py:73`, `:102`). A phenotype string is used only when a test has no MIC (`ast_tests.py:148`), so
  of 922 source-reported *intermediate* PsA tests, **865 were re-derived from their MIC** (578 → R, 287 → S) and
  only the 57 without one were dropped. `label_counts.csv` has no intermediate counter and `call_agreement.csv`
  excludes every I test by construction (`ast_tests.py:186-189`), so the override is invisible in every artifact.
- **EUCAST v15 gives PsA no S category for 7 of 10 drugs** (`S ≤ 0.001`): ciprofloxacin, levofloxacin,
  ceftazidime, cefepime, imipenem, aztreonam, piperacillin-tazobactam. meropenem has a genuine band (S≤2, R>8);
  amikacin (16/16) and tobramycin (2/2) have none. **Verified**: the non-I-zone holdout negatives for those 7 are
  exactly the genomes with `n_mic == 0` in `label_provenance.csv` — MATCH for all 7. Pooled, **7,093 of 14,038
  "susceptible" test labels (50.5 %) are EUCAST I**.
- **⚠ The I *category* is the wrong handle; distance is the right one.** For those 7 drugs "I-zone negative" is a
  tautology for "negative". The quantity that carries information is `d = log2(MIC / r_gt)`, and `|d| ≤ 1` is
  exactly the set of label states one dilution of measurement error can flip (ISO 20776-2 essential agreement).
  **The two drugs with no I zone at all show the effect anyway.**
- **Error rises monotonically toward the cut** (`error_by_distance/error_by_band.csv`, random holdout,
  class-conditional, prevalence-matched threshold):

  | band | susceptible n / error | resistant n / error |
  |---|--:|--:|
  | far (\|d\| > 3) | 203 / **0.108** | 321 / **0.159** |
  | mid (1 < \|d\| ≤ 3) | 1,147 / 0.219 | 477 / 0.321 |
  | **near (\|d\| ≤ 1)** | 449 / **0.419** | 425 / **0.454** |

  **3.9× and 2.9×** far → near. 19 of 20 drug × class contrasts go the same way; 11 of 20 have a Newcombe
  difference interval excluding zero (`near_vs_far.csv`). `Spearman(p̂, d)` is positive in every drug × class
  (0.03–0.53), so the model reads the MIC gradient the binary label discarded.
- **Confirmatory, `conditional_logit.csv`: `error ~ near + drug` conditional on genome, OR 2.52 (95 % CI
  2.01–3.16)**, 368 informative genomes, 2,196 rows; OR 2.47 at a flat 0.5 threshold. Clonality, lineage,
  assembly source and QC are genome-level constants and are differenced out **exactly**. Error-concentration
  ratio **1.535 (1.433–1.648)** over 646 genomes (`error_concentration.csv`).
- **Clonality points the wrong way, so it is not the explanation**: genomes near the breakpoint for ≥1 drug have
  a ≥99.99-ANI training neighbour **51.4 %** of the time against **37.6 %** for the rest — they should be
  *easier*.
- **⚠ Per drug, the intuitive form of the hypothesis is WRONG** (`per_drug_vs_auroc.csv`,
  `per_drug_correlations.csv`). A drug's AUROC does **not** fall as its near-band error rises: Spearman
  **ρ = −0.08, p = 0.83**. Near-band balanced error is near-constant across the panel (0.348–0.539, sd 0.067) —
  near the cut every drug is bad by about the same amount, so it is a floor, not a differentiator. What predicts
  AUROC is **far-band error** (ρ = −0.79, p = 0.006) and **the share of the holdout in the near band**
  (ρ = −0.76, p = 0.011); the two are near-independent (ρ = +0.52, p = 0.13) and each survives partialling out
  the other (−0.72 and −0.67). **Label noise depresses AUROC through exposure, not severity**: cefepime has
  43.7 % of its holdout near the cut and the lowest AUROC; tobramycin 8.9 % and the highest. ⚠ n = 10 drugs
  (|ρ| ≥ 0.648 for p < 0.05) and the effective n is smaller — cefepime/ceftazidime share AmpC+porin,
  cipro/levo *gyrA*/*parC*, imi/mero OprD. Descriptive; §2.5's genome-conditional model is the confirmatory one.
- **Ceiling: removing the near band is worth a mean +0.068 AUROC** over the **7 of 10 drugs that clear both
  nulls** (`leave_out_ceiling.csv`): meropenem +0.111, levofloxacin +0.085, imipenem +0.065, down to
  ceftazidime +0.050. ⚠ The gain is reported against a random-deletion null **and** a `p̂`-decile-matched null,
  because deleting genomes near the label boundary raises AUROC mechanically. The three that fail are
  informative: **amikacin gains +0.0005 and tobramycin +0.021 — the two drugs with no I zone** — and cefepime's
  near band is so large (43.7 %) that random deletion alone moves it +0.083. ⚠ That table's baseline is over
  band-placed genomes only (excluding `no_mic`/`undetermined`, 34 % of pairs) so it runs below the headline
  AUROC; compare within a row, never against the model table above.
- **Reproduces the previously-uncommitted figure of record.** §6 cited "PsA 19.8–42.3 % within one dilution" from
  `checks_vs_2026-07/near_breakpoint_by_drug.csv`, produced by a script never committed. `mic_distance_bands.py`
  regenerates its v1 column **exactly for all 10 drugs** (18.1–54.4 %, matching `n_labelled` too).

**In flight.**
- ~~**PsA fine-tunes on the ST155 tables**: array `41605929`~~ — **complete**, all 10 drugs, 4.3–7.0 h each. The
  metrics are in *Numbers of record* above. Kept here only so a reader of an older copy sees it closed.
- **The Klebsiella-only arm** (`apps/entero/scripts/train_kpsc_models.sh`, `52e1311`): unit `kpsc`, 9,370 genomes,
  held-out ST `klebsiella:37` (211 genomes, **1.8–2.5 % of each drug's labelled set**), random 20 % of the rest,
  seed 1 / fraction 0.2 / 5 folds / fold 0, **no `--exclude` and no `--groups`** (neither file exists for this
  cohort). Per drug 3,601–7,817 labelled, against PsA's 1,195–3,136; ST37 scores 7 of 10 drugs (meropenem 12 R,
  imipenem 11 R and amikacin 15 R fall under 20). Outputs under `processed/kpsc/`. Now carries
  `--exclude apps/entero/excluded_genomes.tsv` (`21e3c78`), so its eligible set is 9,367, not 9,370 — ⚠ an exclusion
  re-draws the **whole** split, not just the listed genomes. **Embeddings no longer block it:** all 9,370 have a
  `.pt` (job `41632894`, above). Splits not yet built.
- ~~**Enterobacterales ESM-C**: array `41596846`~~ — **complete**; the counts are in *Numbers of record* above, and
  its check (job `41632894`) found 0 problem rows. Kept here only so a reader of an older copy sees it closed.

**Next.** Agreed with David 2026-11-19 ("even annotation sorted before embedding and models lock us in"):
1. **A cross-species model as soon as possible** (David, 2026-11-20). Needs: a split builder that holds out STs
   (keyed scheme:ST, grouping the 3 two-BioSample CDC strains) beside a random holdout, restricted to QC-passing
   genomes with 0/1 labels — `engine/splits` today makes random k-fold holdouts only; and a trainer task that reads
   labels from several species folders and the shared ESM store (`finetune_amr` maps only `tb_ast`/`kleb_ast`).
   **Built** (`986813e`…`8b791d4`): the split builder (`engine/splits/build_amr_holdout_splits`; `load_splits`
   gained `load_split_ids`, `load_split_species` and a `species` filter), `finetune_amr --split-table` (scores
   `holdout` and `holdout_st`, per species when the table has several), `apps/pa/scripts/train_pa_models.sh`, and the
   Kp ESM-C reuse stage. **PsA baseline: done** — Stage A smoke, Stage B overfit and the Stage C array
   (`41548353`) all ran; the ten models' metrics are the numbers of record above. Reported to David 2026-11-21 with
   hypotheses only; confidence intervals and the §0.4 mechanism split are not yet computed. **Open, for David:**
   which `validate` the cross-species arm uses for early stopping and model selection — pooled PsA + Enterobacterales, or PsA only (the baseline's).
2. **Enterobacterales** (stages in `apps/entero/CLAUDE.md`): when the builds finish, `status` (every genome built or
   failed by cause); BakRep ingest pre-QC (`annotate_entero_genomes.sh plan`, `ingest`); CheckM2 (34 × 2,000) and
   skani (`build_entero_genomes.sh qc`); collate in report mode → `species_crosstab.csv` and
   `embedding_input_unfiltered.csv`; our Bakta on the rest; proteins; ESM-C (~250 A100-h, SL2-GPU).
   **Done through the ESM-C submission (2026-11-21); the array is in flight. MLST re-typed per species and
   re-collated (2026-11-21).** Left before training: the species rule (David's), the holdout ST per species under
   the ≤ 5 % cap, the cross-species split tables (`--assert-same-as` the PsA tables, so the PsA rows are identical
   across arms), and the cross-species `validate` choice.
   Decided 2026-11-20 (§6): embed every genome passing the non-species QC; BakRep-refused genomes get our Bakta;
   existing Kp ESM-C embeddings are reused by symlink where they align (*Caveats*).
3. Splits (the ST holdout plus others, through `engine/splits`), then training. **PsA decisions made 2026-11-20**
   (§6): ST235 held out (exact ST), random 20 % holdout, the 15 Ecuador long-read assemblies dropped, Cortes-Lara
   kept, no per-class minimum.

Later: the AMR-GNN-cut benchmark, ANI thinning, and causal attribution of the predictions.

**Caveats.**
- **⚠ What the clonal-lookup baseline does and does not license.** It is a **lower bound on memorisation, not a
  ceiling.** Whole-genome ANI is core-dominated while most acquired AMR determinants are accessory, so a lookup keyed
  on accessory gene content — or a k-NN in the embedding space itself — would score higher than 0.684. A *low* number
  would therefore not license "the model generalises"; the *high* one we measured does convict the random holdout.
  Three further limits on the tables above: the ST155 holdout is **one lineage**, so its 47–78 genomes are mutually
  correlated and every CI in that column is wider than its *n* implies; amikacin's ST155 set has only **7 resistant**
  genomes and levofloxacin's only **12 susceptible**, so those two rows are close to uninterpretable alone; and
  "model minus lookup" compares two differently-sized sets at different prevalences, so the margins rank drugs but do
  not subtract cleanly. The *direction* — lookup at 0.684 on the random holdout against 0.516 on ST155 — is the
  finding; the per-drug arithmetic is not.
- **Hypotheses from the ST155-versus-random gap, offered to David 2026-11-22 and not resolved.** (1) The random
  holdout's headline is real signal **plus** a clonal lookup bonus, and ST155 is the honest number — thinning removes
  an inflation term rather than rescuing a broken result. (2) The model's own gap between the two holdouts (−0.085
  mean over 6 drugs) is *smaller* than the lookup baseline's (−0.168), suggesting its representation is partly
  lineage-invariant and does not cash in the full clonal bonus. (3) Tobramycin has the **strongest** clonal structure
  of any drug (lookup 0.7988, the highest) yet is the one drug the model scores **better** out of lineage (0.9819 vs
  0.9326) — consistent with acquired AME genes being read as gene presence, which transfers across lineages intact,
  against imipenem (*oprD* loss) and levofloxacin (*gyrA*/*parC*) being chromosomal-mutational and lineage-entangled.
  That is the §0.4 HGT-vs-chromosomal hypothesis appearing as a holdout-type contrast. ⚠ Amikacin is also an
  aminoglycoside and cuts the other way, but its ST155 set is the 7-resistant one. **The §0.4 mechanism
  stratification has not been computed**, so (3) is a hypothesis about a stratification we have not run.
- **Duplicate isolates — measured, policy still open with David.** The PsA arm's two filter files are different
  things and must not be read as one: `apps/pa/excluded_genomes.tsv` (15) is an **assembly-quality** exclusion (the
  Ecuador block below), and `apps/pa/duplicate_isolate_groups.csv` (3 groups) is the only duplicate artifact. Both
  were hand-written in `986813e`; **no script produced either and no document records the duplicate criterion.** It
  re-derives exactly as an exact match on the portal phenotype table's `isolate` column, which is consistent but
  unproven.
  - **The whole Enterobacterales cohort holds 5 pairs of bit-identical assemblies — 10 genomes of 67,326.**
    Screened at `73da407` in **13 s**: 42 candidate groups on `(total_length, n_contigs)`, of which the sequence
    hash confirms 5 and refutes 37. Three are `kpsc` (SAMN04014892/SAMN04448224, 197 contigs, 5,375,650 bp,
    `klebsiella:34`; SAMN12738924/SAMN12738928, 10, 5,737,703, `klebsiella:39`; SAMN11953933/SAMN11953937, 167,
    6,132,443, `klebsiella:11`), one is `ecoli` (SAMN04014925/SAMN05170083, 79, 5,142,992, **`ST131`**) and one is
    `salmonella` (SAMN19836102/SAMN20181416, 94, 5,018,812, **`ST32`**) — both of those STs were on the original
    holdout-ST shortlist. The KpSC pairs carry **conflicting names** (`FDA-CDC-AR_0051` vs `6`; `AR_0613` vs
    `AR_0617`; `Kp10` vs `Kp33`) and different WGS projects, so no name or provenance screen can see them.
  - **The rate tracks the source mix, not the species.** §3.6's cohort is `atb` 65,122 + `portal_gff` 2,049 +
    `shovill` 155 and holds **no `ncbi_gca` genome at all**, which is why it yields 5 pairs in 67,326 while v2's
    PsA yields 12 in 8,643 (§3.7): 11 of v2's 12 are NCBI GenBank deposits of one isolate under two GCAs. Any
    arm that takes NCBI assemblies should expect a different order of magnitude.
  - **What grouping them would be worth, measured — under one genome per drug.** For a pair, the chance of
    straddling a random 80/20 draw is 2 × 0.2 × 0.8 = 0.32. The three `kpsc` pairs are labelled on both members
    for 1–3 drugs each, so the expected number of holdout genomes whose twin is in train runs **0.32–0.96 per
    drug — 0.044 % (levofloxacin) to 0.109 % (imipenem) of that drug's holdout** (`binary_ast.csv` × the screen's
    report). **So grouping is hygiene, not a measurable correction**: it will not move any reported number. What
    makes it worth doing *now* rather than later is only that no `kpsc` table exists yet, so it is free today and
    costs a full re-draw after the tables are built.
  - ⚠ **An identical genome pair can carry contradictory labels.** `kpsc` SAMN11953933/SAMN11953937 are the same
    sequence and are labelled on all ten drugs, disagreeing on **meropenem and imipenem**; PsA's
    SAMN03105472/SAMN03105661 disagree on meropenem and levofloxacin; 4 of v2's 12 disagree somewhere (§3.7).
    **Hypothesis, untested:** either the AST record is noisy at roughly this rate, or a "duplicate" is sometimes
    two isolates whose assemblies were confused upstream — the two readings differ in what should be done about
    it, and nothing measured here separates them.
  - **The screen is `engine/genomes/find_duplicate_assemblies.py`**, so both experiments answer this one way. Its
    candidate key is `(total_length, n_contigs)` and the sequence hash decides. ⛔ **`n_cds` must not be added**:
    it is the weakest key that cannot miss a real hit (identical contig sets force both columns to agree), and
    precision there is free because the hash adjudicates — 4 candidate pairs in KpSC's 9,370, 1 in PsA's 3,677.
    With `n_cds` in the key the **PsA screen returns nothing at all**, dropping SAMN03105472/SAMN03105661, which
    are annotated 5,821 against 5,818 CDS because each carries its submitter's annotation. Run at `c94df7f`:
    KpSC 4 candidate groups → **3 confirmed, 1 refuted** (SAMN06437650/SAMN06438666, 5 of 142 contigs shared —
    refuted by the hash, which is where it should be refuted); PsA 1 candidate → **1 confirmed**.
  - **A file checksum can never find these.** `contigs_md5` and `atb_md5` hash the gzipped file, and every contig
    header embeds the BioSample (`>SAMD00061218.contig00001 … src=atb`), so all 9,095 KpSC `atb_md5` are distinct
    by construction (`build_genomes.py:179-184`).
  - **The two metadata screens do not discriminate in KpSC.** Shared strain/isolate names: 83 groups / 311 genomes,
    but **307 of the 311 are generic labels** — ST numbers used as a strain name (213), bare integers (75), the
    literal `Klebsiella pneumoniae` (19); only 4 genomes share a specific name and a single ST. Provenance keys
    (study × date × country × host × source): 93 % of the 6,034 genomes with a complete key fall in a shared group,
    and the 65 single-ST groups sit inside a study × year × country permutation null of 55.5 (95 % 48–65). Applied
    to PsA that key flags 400 genomes and co-groups **0 of the 3 known CDC pairs**.
  - **ANI cannot rule a pair *in*.** skani scores a genome against itself at 99.99, and 1,076 of 4,950 random
    KpSC ST258 pairs score ≥ 99.98. In a clonal cohort near-identity is the background, so ANI can only rule out.
  - **Two findings in §3.6's own PsA tables.** `MRSN 20176`/`MRSN20176` (SAMN03316892, SAMN09788414 — both ST316,
    24 vs 290 contigs) sit in **holdout and train respectively for five drugs** (aztreonam, ceftazidime,
    levofloxacin, meropenem, piperacillin-tazobactam) in the live ST155 tables: the exact-match criterion missed
    them on one space. At 1 genome of a 716-genome holdout this is below anything that moves a reported number, and
    rebuilding would re-draw all ten tables, so **array `41605929` was left to run**. Separately,
    SAMN03105472/SAMN03105661 are **identical sequence** (50 contigs, 6,380,431 bp) with **opposite AST calls for
    meropenem and levofloxacin**; both are in train, so no leak, but one of the two labels is wrong.
  - ⛔ **An `--exclude` or `--groups` list re-draws the whole split, not just the listed genomes** — the builder
    permutes the post-exclusion pool, so its contents and its length both change (`build_amr_holdout_splits.py`
    L396 → L430 → L311-313; the repo's own test asserts it at `test_build_amr_holdout_splits.py:212-215`).
    Measured on a synthetic 9,370-genome unit: ~0.50 of the rest keep their split after excluding one
    early-sorting genome, ~0.64 after adding three two-member groups. **So the policy must be fixed before an
    arm's tables are built.** `$AMR/processed/kpsc/splits` does not exist yet, so for the Kp arm it is still free.
- **Low-depth MiSeq 2×300 reads assemble to a fraction of a genome under shovill's defaults** — measured
  across the cohort, all in one study (Cortes-Lara et al. 2021): of the 18 shovill genomes under 30×, 12
  fail completeness and 1 fails to build; all 14 shovill completeness failures are this study, at
  13.6–32×. At ≥ 50×, 3 of 295 built shovill genomes fail QC (2 species, 1 contamination), none on
  completeness.
  - For a 17× genome the reads are sound: 93.8 % map to PAO1, at 96.5 % breadth and 15.2× mean depth.
    SPAdes assembled 6.34 Mbp; shovill then removed 800 of 853 contigs as "low coverage (< 2×)", because at
    k = 127 SPAdes reports about 1× coverage on these error-dense 300-bp reads.
  - **Hypothesis:** this is why these samples are absent from AllTheBacteria.
  - Rescuing them needs non-default shovill settings for a subset, against the even-surface decision.
- **Long-read portal assemblies pass QC on a different error profile.** 32 `portal_gff` genomes have only
  long reads in ENA (`genome_manifest.csv` `no_reads_reason`); 20 pass QC. 15 of those are one Ecuador
  BioSample block (SAMN338…, 2–16 contigs) at 951–1,140 CDS per Mbp (10 above 1,000) against the cohort's
  ~917, and 10 of the 15 get no ST. **Hypothesis:** unpolished indel errors splitting genes. The other 5
  are single contigs at normal density. Whether to keep them is decision (b) under *Next*.
  - **Bakta flags the same 15** (`bakta_annotation.csv` × `genome_qc.csv`): 125–766 pseudogenes each against at most
    53 in any ATB genome, 6,105–7,943 CDS, and CheckM2 completeness 91.3–99.8 % (every other QC-passing genome:
    median 100 %, 1st percentile 99.92 %). The same hypothesis; their protein inputs would carry gene fragments.
- **ST621 is one collection with almost no metadata.** All 244 are `portal_gff` with no reads; 240 sit in
  one BioSample block (SAMN29441777–SAMN29442029) with one assembly-accession block, `NCBI_antibiogram`
  labels, and no study, country or year. All 240 CIP and 99 LVX labels are R. Its single-locus variant
  ST4588 (19 genomes, all R for CIP) is a separate collection (atb, MBRL isolates, USA 2019).
- **Genome source is nearly collinear with study.** 315 of 333 QC-passing shovill genomes are Khaledi et
  al. 2021, labelled almost only for CIP/CAZ/MEM/TOB. All 244 ST621 genomes are `portal_gff`, and they
  hold 199 of that source's ~230 TOB-resistant labels: TOB %R is 70 % in `portal_gff` against 32 % in atb.
- **`st` `-` is not an ST.** It pools the 205 untyped QC-passing genomes in `st_counts_by_drug.csv`; 39
  of them match a typed cohort ST once mlst's novel/partial-allele flags are removed (ST235 3, ST179 3,
  ST308 3, …).
- **`cohort.csv` `study` is not a PubMed-ID column.** It joins the portal's `AMR_associated_publications`
  (PMIDs) with AMR-GNN's author-year citations; 557 QC-passing genomes have neither. AMR-GNN-only genomes
  (1,506 QC-passing) carry no country or year anywhere locally. 157 `st_reported` values are 40-character
  hex strings, not ST numbers.
- **Study 27216077 is lost entirely** (135 genomes, RNA-seq reads only), which is why pre-build drops sit
  almost wholly in CIP/CAZ/MEM/TOB.
- **Shovill is given the genome size** (`--gsize 6.6 Mbp`, `d7b9656`). Its own estimate counts R1 21-mers
  seen more than 10 times, which collapsed to 39 kbp at 17×. Shovill then believed the reads were 2,861×
  deep, kept 5 % of them, and returned an empty assembly with exit status 0. Given the true size, shovill
  subsamples only reads deeper than 150×, as its defaults intend.
- **3 CDC strains each sit under two BioSamples** (isolates S54485, U2504, X24509). The splits must
  group them, or they leak across the holdout.
- `finetune_amr` has a `pa_ast` task key and `--split-table` (`c56eca4`); `build_split_csv` still has none.
- **`engine/config.py` `PA` resolves under `$AMR`** (`0f5cc5d`): tables in `processed/paeruginosa/`, per-sample
  stores flat in `processed/{protein_sequences,esm}/`. The `pa_ast` trainer key, when added, should read it.
- **The ESM-C array takes a sample list** (`SAMPLES`, md5-pinned): PsA and Enterobacterales share
  `processed/{protein_sequences,esm}/` without embedding each other's genomes (mechanics: `apps/pa/CLAUDE.md`
  *Embeddings*; Entero driver `apps/entero/scripts/build_entero_embeddings.sh`). ⚠ A PsA array submitted before
  this change still slices the store's whole listing, so no Enterobacterales parquet may land while one is live:
  `build_entero_embeddings.sh proteins` refuses while any `pa-esm-*` job is queued or running.
- The `bakrep_coverage_20261119/{join,cmp_pa,miss}.py` scripts are kept as they ran and still build pre-move paths.
- **Reusing the Kp ESM-C store (`david/bacformer_processed/esm/klebsiella/`) for Enterobacterales *Klebsiella*,
  measured 2026-11-20** (`kp_reuse_probe.py`; 40 random genomes our BakRep ingest had taken): 8,917 of the 9,478
  labelled *Klebsiella* have a Kp `.pt`. Its proteins are ours in the same order and count except the three
  selenoproteins (fdnG ×2, fdhF), where our extractor writes selenocysteine as `U` and the Kp parquet had `*`; 39 of
  40 `.pt` have exactly our protein count, 1 lacks those three (so it would misalign positionally). The `.pt` are
  **fp32** (ours bf16), 960-dim, keys `protein_embeddings, attention_mask, contig_ids`. The Kp *parquets* on disk were
  regenerated later and some drop the three selenoproteins, so they are not what the `.pt` were made from. Link a Kp
  `.pt` only when its protein count and per-contig counts equal our parquet's; compute the rest.
- **A GPU job submitted from a cclake login node crashed `import torch` on ampere** ("Fatal Python error: Illegal
  instruction", exit 1, nothing on stderr): `sbatch --export=ALL` carried the login node's spack `LD_LIBRARY_PATH`
  (cascadelake-built libraries) to ampere's AMD CPUs. Jobs `41520932` (the first PsA smoke) and `41525227` (the
  diagnosis: a clean environment imports torch 2.10.0+cu128 with CUDA). `esm_embeddings_array.sh` now unsets the loader
  paths and runs Python with `-X faulthandler` (`4ff29d7`). CPU jobs on icelake (Intel) were unaffected.
- **The QC-pass Prodigal GFFs and `n_cds` in `genome_qc.csv` are not the annotation of record**: Bakta is.
- **`--mlst-scheme` is job-wide, and `st_key` is `<scheme>:<ST>`.** `run_mlst` takes one scheme for the whole
  chunk (`build_genomes.py:272-277, 340`; `build_genomes_array.sh:45`), so pinning a mixed-species cohort needs one
  array submission per species over its own subset manifest — there is no per-row scheme column, and
  `plan_genome_sources` keeps no species column (`plan_genome_sources.py:198-200`). A re-run of `build_genomes` is
  **effectively MLST-only when the manifest is unchanged**: contigs are reused on a matching `src=` header and the
  Prodigal GFF on a matching `contigs_md5=`, while stats, the CDS count and `mlst` are always recomputed
  (`build_genomes.py:290-313`). ⚠ If the manifest names a different source for a genome, its contigs are **deleted
  and rebuilt**. `collate_genomes.st_counts_by_drug` groups by `(drug, st)` and ignores the scheme
  (`collate_genomes.py:93-94`), so that file pools the same ST number across schemes.
- **The splits builder already supports what the cross-species arm needs** (read at `43d4b92`): species units are
  just the `--species NAME=LABELS,ELIGIBLE,QC` name (`build_amr_holdout_splits.py:326, 430-431`), so
  subgroup units (`ecoli`, `ssonnei`, `sflexneri`, …) need **no code change** — only one eligible-genome CSV per
  unit, which **nothing in the repo writes yet**. `--holdout-st NAME=SCHEME:ST` is repeatable and per species
  (`:137-140, 374-376, 431`), so one ST per unit works today. Units may not overlap (`:391-393`), each needs ≥ 1
  genome in train/validate/holdout (`:329-331`), and the unit name seeds its shuffle (`:160-163, 311`) — renaming a
  unit or moving a genome between units changes every split in it. Several STs held in one unit land in a single
  `holdout_st` pool that cannot be scored per ST.
  `n_cds` and the CDS-per-Mbp figures above are Prodigal counts.
- **Bakta's memory is sacct's to measure.** A run record's `max_child_rss_gb` (getrusage, the largest single
  child) read 8–12 GB on gate genomes whose sacct MaxRSS was 37.5–42.6 GB; size `--mem` from sacct.
- **Unverified, raised by a design review — check before training:** `finetune_amr` casts labels with
  `int()`, which would turn a fractional (conflicting) label into 0, and draws k-folds over every labelled
  genome, so a genome with no embedding would fail mid-run. PsA has 2 fractional cells and 255 labelled
  genomes without an embedding; Enterobacterales 25 and thousands. Splits must restrict to QC-passing genomes
  with 0/1 labels.
- **⛔ A species unit must name its species, never prefix-match the genus.** *K. aerogenes* is **not** in the KpSC
  (which is *pneumoniae*, *variicola*, *quasipneumoniae*, *africana*, *tropica*), so a unit written as
  `species.startswith("Klebsiella")` wrongly absorbs it. It is invisible in §3.6 alone — this cohort spells the
  organism *Enterobacter aerogenes* (28 genomes) and has no *Klebsiella aerogenes* row — but CABBAGE v2 (§3.7)
  spells the same organism *Klebsiella aerogenes* (424 genomes), so a prefix rule puts one organism in two
  different units depending only on which label source the row came from. **The KpSC unit names its five species
  explicitly; *K./E. aerogenes* goes in the *Enterobacter* unit under both spellings**, and §3.7 takes the same
  rule.
- **⚠ The skani call cannot adjudicate the *E. coli* / *Shigella* boundary, so the species rule there rests on the
  submitter label.** Of the 21,699 QC-pass genomes labelled *E. coli*, skani calls 13,010 *E. coli* and **5,199
  *Shigella boydii***; of the 12 labelled *S. boydii*, 8 come back *S. sonnei*. *Shigella* is phylogenetically
  embedded within *E. coli*, so ANI against `species_reference_panel.tsv` (a DRAFT) has little room to separate
  them, and it evidently cannot separate the *Shigella* species from one another either. **Read as a limit of the
  panel, not as evidence about any genome** — hypothesis, not a finding; raised with David 2026-11-21, not acted on.
  The panel is also thin for the smaller genera: *E. cloacae* 608 of 739 call `unknown`, *S. marcescens* 263 of 307.
- **ST numbers collide across MLST schemes** (ST11 is *Salmonella*, *K. pneumoniae* and *E. coli*). Any
  multi-genus ST count or ST holdout must key on scheme:ST, never `st` alone.

**Owns.** `engine/ast_labels/{breakpoints,ast_tests,build_binary_ast_labels}.py`,
`engine/ast_labels/eucast_v15_breakpoints.csv`, `engine/splits/holdout_lookup_baseline.py`, `apps/pa/`,
`apps/entero/`, [`docs/holdout_clonality_and_label_noise.md`](docs/holdout_clonality_and_label_noise.md).
⚠ `holdout_lookup_baseline` is **shared engine** — organism-agnostic by construction (it takes a
skani TSV and split tables and nothing else), and §3.7 runs it on v2's cohorts from the same file.
`apps/pa/{mic_distance_bands,holdout_error_by_mic_distance}.py` are app-level and touch no engine code; they move
to the engine only when a second organism needs them.

### 3.7 Multi-species AMR — CABBAGE v2 (the second experiment, same procedures at ~2× the data)

**Status.** **Labels built** (stage 1 of 4) for PsA and Enterobacterales, and reviewed by David (§6, 2026-11-21).
**PsA: genomes (stage 2) and ESM-C (stage 3) done** — 9,044 genomes pass QC, each with one Bakta annotation, its
proteins and an embedding (`check` PASS, below); splits (stage 4) not started. **Enterobacterales: genomes built,
QC'd and collated** (117,277 pass; 41 duplicate pairs listed for exclusion); Bakta (13,880 QC-pass genomes still
without an annotation) and ESM-C running — see *In flight*. CABBAGE v2 is an extension of the data behind the EBI AMR
Portal, given to the project as one CSV, with about twice the BioSamples. **It is a second experiment, not an
update of §3.6** (David, 2026-11-21: "second test, same procedures"): v2 labels are neither merged with nor
substituted for §3.6's, which continues on the portal's 2026-07 release ("v1"). Every stage repeats §3.6's procedure —
the label rule with the AMR-GNN supplement for PsA, genome sources, QC, MLST, Bakta, ESM-C, then **new splits,
retrained models and a second PsA baseline** — and the two experiments' models are then compared. **Both experiments
run on one branch (`refactor/consolidate-engine`) so they run the same code**; a change to shared code is additive
and announced to the §3.6 agent (§6). Owned by a separate agent from §3.6.

**Numbers of record — the v2 input.** `$AMR/raw/ebi_amr_portal/cabbage_v2_20261121/cabbage_v2.csv` (md5 `f608adf0…`;
unpacked from `cabbage_v2.csv.bz2`, md5 `9c374834…`, identical to the copy David received; `README.md` alongside):
3,345,275 rows, 307,652 BioSamples, 586 species labels, 248 drug names. v1 (`2026-07/phenotype.parquet`): 1,714,486
rows, 170,750 BioSamples, 58 species, 165 drug names.

**Numbers of record — v2 against v1.** Read from `$AMR/processed/ebi_amr_portal/scoping_cabbage_v2_20261121/report.txt`
and its CSVs (script `compare_cabbage_v1_v2.py` alongside). **Harness check:** its v1 dry run reproduces §3.6's
Enterobacterales labels exactly (242,573 clean labels over 67,763 genomes).

| BioSamples | v1 | v2 | both | v1 only | v2 only |
|---|--:|--:|--:|--:|--:|
| All | 170,750 | 307,652 | 170,446 | 304 | 137,206 |
| Enterobacterales (the build's 21 genera) | 72,043 | 139,931 | 71,976 | 67 | 67,955 |
| PsA | 2,518 | 8,575 | 2,378 | 140 | 6,197 |

- **Enterobacterales loses nothing:** all 67,763 §3.6-labelled genomes are in v2; the 67 "v1 only" are re-speciated
  there ("mixed sample" 39, "Enterobacteriaceae bacterium" 25, host names). **PsA loses study 27216077** (135 — already
  lost in §3.6, RNA-seq reads only) and 5 re-speciated to *P. paraeruginosa*.
- **AMR-GNN is only partly in v2:** 1,148 of its 2,515 BioSamples (v1: 820). Still absent: Stanton 2022 (628), Alfred
  Hospital (341), Barrio-Tofiño 2017 (42), most of Cortes-Lara and Teo — so the supplement is still needed. Where v2
  and the supplement hold the same test, 5,852 of 5,976 agree on sign and value (`pa_amr_gnn_*.csv`).
- **Same tests agree:** on shared (BioSample, drug) pairs values agree on 99.4 % of Enterobacterales pairs, phenotypes
  99.2 % (`shared_pairs_measurement_agreement.csv`).
- **Dry run of the §3.6 rule on v2** (ten drugs, PsA + Enterobacterales, a missing sign read as "="): 546,791 clean
  labels over 126,759 genomes (v1: 253,035 over 70,121). Where both hold a clean label: Enterobacterales 241,378 of
  241,853 agree (69 S→R, 406 R→S), PsA 9,888 of 9,888 (`dry_run_*.csv`). %R rises in v2 (Enterobacterales meropenem
  6.0 % → 12.1 %, imipenem 11.9 % → 19.5 %) — composition, not yet explained.
- **New in-scope genomes:** Enterobacterales 55,742 (37,074 with an assembly accession, 18,668 reads only); PsA 6,190
  (4,116 and 2,074) (`scope_genome_route.csv`). PsA, v2 ∪ AMR-GNN: about 9,800 genomes before genome QC — an
  estimate (8,418 v2 genomes with a ten-drug test + the 1,367 AMR-GNN BioSamples v2 lacks), not a count.

**Numbers of record — the `puffin` source.** Read from `…/scoping_cabbage_v2_20261121/puffin/` (`puffin_probe{,2,3,4}.txt`
and CSVs).
- **A new value of v2's `database` column, described nowhere public.** The EBI AMR Portal's about and methods pages do
  not name it, and list "expanding AMR profiles from text mining the literature" as future work; the CABBAGE preprint
  (bioRxiv 10.1101/2025.11.12.688105) could not be read (rate-limited). **Hypothesis: CABBAGE's literature extraction.**
  To be asked of the CABBAGE side.
- 188,761 BioSamples (128,545 not in v1) from 1,569 PubMed IDs — every row carries one; 6,968 BioSamples come from 2026
  papers. Puffin-only rows (`puffin_fill_rates.csv`): no sequencing platform and no standard year on any; testing
  standard missing on 71 %; 38 % carry an R/S call and no measurement (42 % in Enterobacterales); 51 % of in-scope
  Enterobacterales MICs have no sign.
- **Against v1 on the same (BioSample, drug):** puffin rows merged with a curated source (469,179 pairs) match on
  99.93 % of MICs and 99.99 % of R/S calls. The ~5 % left unmerged (24,026 pairs): MIC identical in 92.3 %, within one
  dilution in 99.9 %; their R/S calls disagree almost everywhere both sides have one, mostly puffin R against curated S
  **at the same MIC** — hypothesis: the paper's own interpretation, not a clinical breakpoint.
- **What rests on puffin alone** (ten drugs, `puffin_inscope_*.csv`): PsA 5,837 of 8,418 genomes (33,492 MIC pairs,
  7,527 bare-call pairs); Enterobacterales 53,420 of 123,926 genomes (145,687 MIC, 89,551 bare call). The rule
  recomputes MIC tests from the MIC, so puffin's interpretation reaches a label only through bare calls.

**Numbers of record — the v2 labels** (built at `20ad6f3`, the SHA in each `manifest.json`). CABBAGE v2 is first
written in the portal's `phenotype.parquet` columns by `engine/ast_labels/cabbage_v2_phenotype.py`
(`$AMR/processed/ebi_amr_portal/cabbage_v2_20261121/phenotype.parquet`, md5 `98a46302…`, counts in the `.json` beside it:
all 3,345,275 rows kept, 918,370 empty signs read as "=", 19,804 rows genus-only), then labelled by the **unchanged**
`build_binary_ast_labels` (`--phenotype` that table, `--portal-source cabbage_v2`, `--genotype` the 2026-07 table,
`--unitless-mic-as-mg-l` as in §3.6 — v2 has no unitless MIC, so it changes nothing). PsA's `amr_gnn_tests.csv` is
byte-identical to §3.6's (md5 `89be6150…`).

| `label_counts.csv` / `cohort.csv` | PsA (`paeruginosa_cabbage_v2/`) | Enterobacterales (`enterobacterales_cabbage_v2/`) |
|---|--:|--:|
| genomes in `binary_ast.csv` | 9,422 | 118,619 |
| genomes with a clean (0/1) label | 9,422 (assembly 6,046 · reads 3,376) | 118,608 (assembly 102,095 · reads 16,513) |
| clean labels, ten drugs | 64,281 | 493,801 |
| §3.6 at the same stage | 3,932 genomes | 67,763 genomes |

Checks against §3.6's labels (`checks_vs_2026-07/`, scripts alongside):
- **Agreement where both hold a clean label:** PsA 23,529 of 23,529; Enterobacterales 241,378 of 241,853 — the
  dry run exactly (`pair_agreement_by_drug.csv`). The 475 disagreements: 374 are ciprofloxacin where §3.6 has a
  PATRIC MIC of 1 (R, > 0.5) and v2 holds only a puffin row at 0.5 (one at 0.25; S) for the same genome — *E. coli*
  190, *K. pneumoniae* 157 (`discordant_pairs_mic_detail.csv`); 96 are labels §3.6 took from a bare call and v2 from an
  MIC; 5 are PATRIC against puffin-only MICs the other way (S → R). **Hypothesis, not checked:** puffin re-extracted the
  value from the paper, one dilution from PATRIC's.
- **§3.6 genomes v2 does not label** (`v1_labelled_genomes_not_labelled_in_v2.csv`): PsA 138 — study 27216077 (135,
  absent from v2; already lost in §3.6) and 3 re-speciated *P. paraeruginosa*. Enterobacterales 135, all present in v2:
  74 *Salmonella* (+2 *Shigella*) whose disk tests v2 carries with **no R/S call** (`PATRIC;puffin`, zone in mm), so no
  label; 17 *Salmonella* whose ten-drug tests (PATRIC disk + E-test in §3.6) v2 does not carry at all; 39
  *K. pneumoniae* that v2 calls "mixed sample"; 3 given non-bacterial species (*Macaca* ×2, *Saccharomyces*).
- **Labels resting on puffin alone** (every called test's `database` is exactly `puffin`; `puffin_alone_by_drug.csv`):
  PsA 39,091 of 64,281 (31,367 from an MIC, 7,722 from a bare call); Enterobacterales 245,085 of 493,801 (146,871 MIC,
  97,795 bare call). Where a puffin-only test carries both an MIC and its own R/S, that call matches the EUCAST v15
  re-call on 95.5–99.7 % of tests by stated standard (Enterobacterales CLSI 95.5 %, EUCAST 99.7 %, none 97.4 %; PsA
  98.5–99.5 %) (`source_call_agreement_by_database.csv`) — the closest available check on the bare calls, not a check of them.
- **Within one dilution of the breakpoint** (an `=` MIC with R/2 ≤ MIC ≤ 2R; `near_breakpoint_by_drug.csv`), reported not
  changed: PsA 19.8–42.3 % of labels by drug (§3.6 18.1–54.4 %); Enterobacterales 3.9–10.2 % (§3.6 2.0–12.0 %).
- 345 labelled Enterobacterales genomes carry a genus-only species (*Salmonella* 132, *Escherichia* 131, …); they take
  the genus- and order-level breakpoints. 220 have a species conflict with the 2026-07 genotype table (PsA 3).

**Numbers of record — the v2 genome plans.** Read from `$AMR/processed/{paeruginosa,enterobacterales}_cabbage_v2/genomes/`
(`genome_manifest.csv`, read-only; `reused_genomes.csv`; `by_scheme/genome_manifest_<scheme>.csv`, read-only), planned at
`c66a737` (`build_{pa,entero}_genomes_cabbage_v2.sh plan`, jobs `41609928`, `41609929`; logs
`/rds/user/dca36/hpc-work/logs/v2-{pa,entero}-plan_<job>.out`). Every labelled genome is either §3.6's (reused, with
§3.6's build decision) or planned here, never both (`check-disjoint`).

| | PsA | Enterobacterales |
|---|--:|--:|
| labelled genomes | 9,422 | 118,608 |
| reused from §3.6 (§3.6 sources: atb / portal / shovill / drop) | 3,794 (2,186 / 1,234 / 353 / 21) | 67,628 (65,234 / 2,049 / 210 / 135) |
| planned here | 5,628 | 50,980 |
| – ATB | 3,304 | 33,779 |
| – NCBI GenBank assembly (`ncbi_gca`) | 1,620 | 15,549 |
| – shovill | 183 | 712 |
| – portal assembly | 0 | 34 |
| – dropped | 521 | 906 |

- **Drops:** PsA `other_strategy` 487 (DNBSEQ paired runs; Illumina runs with no `_1`/`_2` fastq pair), single-end 21,
  long-read 8, RNA-seq 3, < 10× 2. Enterobacterales `other_strategy` 644 (DNBSEQ/BGISEQ; Illumina `WCS`), single-end
  172, long-read 75, < 10× 13, no runs and no assembly 2. **DNBSEQ/BGISEQ paired reads are now assembled** (§6,
  2026-11-22): a `replan` per cohort (PsA 414 drops → shovill; Enterobacterales 510 drops + 31 NCBI → shovill, once its
  build arrays drain), the table above being the plan before it.
- **NCBI listed every accession asked for** (1,620 of 1,620; 15,549 of 15,549; `ncbi_assembly_dirs.tsv`). Without the
  fallback the same plans dropped 2,141 PsA (38 %) and 16,455 Enterobacterales (32 %) genomes, carrying on average more
  labels than the kept (PsA 8.0 vs 6.4 drugs); those plans are kept as `genomes/_plan_pre_ncbi_20261121/`.
- **Enterobacterales by pinned scheme** (atb / ncbi / shovill / portal): `ecoli_achtman_4` 9,152 / 6,849 / 178 / 0;
  `klebsiella` 10,944 / 4,665 / 222 / 34; `salmonella` 11,515 / 2,716 / 75 / 0; `ecloacae` 671 / 501 / 191 / 0;
  `koxytoca` 285 / 44 / 1; `kaerogenes` 390 / 3 / 3; `cfreundii` 314 / 83 / 2; `yersinia` 283 / 6 / 0; `providencia`
  20 / 305 / 9; `serratia` 78 / 138 / 8; `proteus` 70 / 62 / 20; `cronobacter` 1 / 1 / 0; `auto` (no scheme) 56 / 176 / 3.
- **Canaries, one task per new path, all `ok` on their pinned scheme** (`shards/<scheme>/*_00000.tsv`): PsA atb 100/100,
  shovill 1/1 (ST773), ncbi 40/40; Enterobacterales ncbi (`ecoli_achtman_4`) 40/40, atb `koxytoca` 100/100,
  `kaerogenes` 100/100, `auto` 56/56 (53 untyped, 2 `ecloacae`, 1 `providencia` — auto-detection on scheme-less
  genera), portal (`klebsiella`) 34/34, shovill (`ecloacae`) 1/1. Peak memory 1.3–1.7 GB (download tasks), 4.1 GB
  (shovill); an NCBI task of 40 genomes ran in ~1.5 min.

**Numbers of record — PsA v2 genome QC.** Read from `$AMR/processed/paeruginosa_cabbage_v2/genomes/genome_qc.csv`
(md5 `ae5365c2…`) and `embedding_input.csv` (md5 `dbd01289…`), re-collated at `94938a3` after the DNBSEQ re-plan
(`build_pa_genomes_cabbage_v2.sh collate`, 2026-11-22) over CheckM2 lists `qc/checkm2_list_{01..04}.tsv`
(4,881 + 226 + 412 + 1) and `qc/skani/` (PAO1 + PA7). Supersedes the first collate (8,643; `genome_qc.csv` md5
`553c5988…`, at `d6c3e05`), which had no DNBSEQ genome.

| | built | pass QC | species | contamination | completeness | contigs | other |
|---|--:|--:|--:|--:|--:|--:|--:|
| built here | 5,520 of 5,521 | **5,367** | 57 | 50 | 17 | 14 | 16 |
| reused from §3.6 | 3,773 (+21 dropped by §3.6) | **3,677** | 35 | 3 | 44 | 5 | 9 |
| **v2 PsA** | 9,293 | **9,044** | | | | | |

- **Reused genomes pass exactly as in §3.6:** 3,677 is §3.6's QC-pass PsA cohort. The `check-schemes` gate passed —
  every genome built here is on `paeruginosa`, and all 3,773 reused carry §3.6's scheme and ST.
- **Built here, by source:** ATB 3,183 of 3,304 pass; NCBI GenBank assemblies 1,606 of 1,620; shovill on Illumina
  177 of 183; **shovill on DNBSEQ 401 of 414** (12 fail QC — contamination 5, contigs 2, species 2, combinations 3 — and
  1 never built: ENA now answers SAMN28697289's fastq URL with a directory redirect). DNBSEQ assemblies are more
  fragmented than Illumina's (median 246 contigs against 191). Fail counts are the single reason; combinations are
  "other".
- **The held-out ST155: 225 of 9,044 QC-pass genomes (2.5 %)**, under the 5 % cap.
- Build reruns: 82 NCBI 503s (`submit paeruginosa ncbi 9,10,21,22,24`, all rebuilt); 2 truncated fastq downloads
  (one rebuilt).

**Numbers of record — Enterobacterales v2 genome QC.** Read from
`$AMR/processed/enterobacterales_cabbage_v2/genomes/genome_qc.csv` (md5 `44e74d3b…`) and
`embedding_input_unfiltered.csv` (md5 `a96f3eae…`, exactly the QC-pass set), collated at `6936272`
(`build_entero_genomes_cabbage_v2.sh collate`, inline, 2026-11-23, 1 min 50 s, 0.7 GB; log
`/rds/user/dca36/hpc-work/logs/v2-entero-collate_20261123.log`) over CheckM2 lists `qc/checkm2_list_{01..04}.tsv`
(35,032 + 15,002 + 537 + 2) and `qc/skani/` (job `41692513`, 5.5 min: 50,206 of the 50,573 built genomes within
~80 % ANI of a panel reference), in report mode as §3.6 — no species rule, which is David's (`species_crosstab.csv`).

| | to build (dropped at plan) | built | pass QC |
|---|--:|--:|--:|
| built here (ATB / NCBI / shovill / portal) | 50,584 (396) | 50,573 | **50,083** (33,504 / 15,342 / 1,203 / 34) |
| reused from §3.6 | 67,493 (135, §3.6's) | 67,442 (51 failed in §3.6) | **67,194** |
| **v2 Enterobacterales** (118,608 labelled) | 118,077 (531) | 118,015 | **117,277** |

- **Reused genomes pass exactly as in §3.6:** each of the 67,493 has §3.6's own verdict (0 differ). `check-schemes`
  passed: every QC-passing genome built here is on its pin, and all 67,493 reused carry §3.6's scheme and ST.
- **Fails by rule** (a genome can fail several): contamination 536, completeness 219, contigs 190, N50 73, not built
  62 (51 of them §3.6's), length 6. Built here, the pass rate is 99.2 % for ATB, 98.9 % NCBI, 96.0 % shovill.
- **Builds:** the first round left 640 failed — 374 NCBI genomes on six nodes whose `/tmp` was full (mlst's
  temporary FASTA; fixed at `379d415`), 253 NCBI 503s, 13 fastq downloads, 1 Prodigal error — and the DNBSEQ re-plan
  (`379d415`: exactly 510 drop → shovill, 31 `ncbi_gca` → shovill) moved tasks; the moved tasks (`41641473`,
  `41641474`, `41641507`, `41641515`) and then every task still holding a failed genome reran (at `6936272`: NCBI
  `41676706`, `41677334`, `41678246`, `41679256`, at most 4 at a time; shovill `41676723`, `41677647`, `41678595`,
  `41689897`) → every NCBI genome built. **11 are not built:** 10 fastq URLs that ENA answers with a redirect to an
  empty directory (a 797–802-byte listing, on two tries hours apart) though its filereport still lists each file with
  its size and md5 — as PsA's SAMN28697289; and **SAMN13829618, which is not a genome**: its reads (SRR11267877)
  assemble to 15,499 bp in 8 contigs, which Prodigal refuses (exit 10).

**Numbers of record — BakRep coverage of the v2 plans.** BakRep's export
(`$AMR/raw/bakrep/v2_export_20250612/bakrep-export.tsv.gz`, 2,438,386 BioSamples) joined to each v2
`genome_manifest.csv` (inline on CSD3, 2026-11-22; `annotate_*_cabbage_v2.sh plan` writes the roster of record,
`genomes/bakta/bakrep_roster.tsv`, from the genomes actually built). §3.6's `in_bakrep_ids.txt` cannot be reused: it is
§3.6's cohorts joined to the export (66,482 rows), so it lists no genome only v2 has.

| planned here | ATB, in BakRep (ingest) | ATB, not in BakRep | NCBI | shovill | portal | our Bakta, before QC |
|---|--:|--:|--:|--:|--:|--:|
| PsA | 2,564 | 740 | 1,620 | 183 | 0 | ~2,540 |
| Enterobacterales | 28,075 | 5,704 | 15,549 | 712 | 34 | ~22,000 |

No NCBI genome is in BakRep (BakRep annotates ATB's assemblies, and these have none).

**Numbers of record — BakRep ingest** (`genomes/bakta/ingest_status.tsv`, `annotate_*_cabbage_v2.sh ingest` at
`5880518`): **PsA 2,529 of 2,564 ingested**, 35 `refused:names` (job `41621245`, 7 min); **Enterobacterales 27,547 of
28,075**, 528 `refused:names` (1.9 %; job `41624840`). Refused genomes go to our run. Every reused PsA QC-pass genome
already had exactly one §3.6 record (1,781 ingest + 1,896 run — §3.6's split).

**Numbers of record — v2 annotation tables and proteins** (`annotate_*_cabbage_v2.sh` / `build_*_embeddings_cabbage_v2.sh`
from worktrees at `86fd4f7`/`6cadddc`, 2026-11-22).
- **PsA:** every Bakta run done (array `41644190`, the 401 DNBSEQ genomes: 101 tasks, 0 failed). `annotate collate`
  (job `41646827`) → `genomes/bakta_annotation.csv` md5 `d4df2c0f…`: **9,044 of 9,044** with one current annotation
  (4,211 BakRep, 4,833 our run). Proteins (job `41647155`, 5,367 new parquets in 9 min; reused genomes' kept);
  **check-proteins PASS 9,044/9,044** (job `41647550`).
- **Enterobacterales, pre-QC:** `annotated` → `genomes/preqc_annotated.csv` md5 `50f52800…`: of 49,545 genomes built
  here, 27,547 ingested + 2,968 our run finished, 53 running, 18,977 none. Collate (job `41645790`, 8 min) →
  `genomes/bakta_annotation_preqc_annotated.csv` md5 `3a2ed689…`: **30,515 of 30,515**. Proteins (job `41646365`,
  30,515 in 18 min). **check-proteins FAIL on 27** (job `41647413`; `hpc-work/logs/v2-entero_esm_check-proteins_20261122_115914/problems.csv`):
  each is one protein short of CDS − pseudogenes. All 27 are NCBI GenBank assemblies with our Bakta run; their
  contigs carry IUPAC ambiguity codes (R, Y, M), and in each one CDS has an ambiguous codon that Biopython translates
  as a stop (e.g. `TRA`), so the extractor drops that CDS under its internal-stop rule. Bakta's own `.faa` writes
  `X` there and keeps the protein. §3.6 never met this (no NCBI assemblies). **David chose `X` (§6, 2026-11-22)**;
  both arrays (`41647810`, `41650067`) were cancelled before any genome was embedded (PsA task 0 ran 6 min, wrote no
  `.pt`), so nothing was embedded under the old reading. Until then the 27 were held back from the pre-QC cohort: `HOLD=genomes/esm_hold_ambiguous_codons.csv` (that problems.csv, copied to
  project_k) with `ANNOTATED_FROM` the first cohort (kept as `genomes/bakta/preqc_annotated_20261122_1121.csv`) →
  `preqc_annotated.csv` md5 `44be1fde…`, **30,488** → collate (job `41647929`) → `bakta_annotation_preqc_annotated.csv`
  md5 `c645abc7…` → **check-proteins PASS 30,488/30,488** (job `41649101`).
- **Enterobacterales Kp link** (job `41649303`, 10 min): of the 10,371 cohort genomes labelled *Klebsiella*, **10,036
  linked** from the Kp ESM-C store (fp32), 287 absent from it, 48 with a different protein count. Its report
  (`esm_linked_kp.tsv`, md5 `ee0f69fc…`) was overwritten by the next round's link-kp, which listed only the 343
  genomes still without a `.pt` (0 linked); the links themselves are re-listed from the store as
  `genomes/esm_kp_symlinks_20261122.tsv` md5 `fcf339d8…` (10,036, every target in the Kp store), and the driver now
  keeps one dated report per run (`esm_linked_kp_<time>.tsv`).
- **Ambiguous codons → `X`, applied** (§6; `reextract` at `9c81041`, which rewrites a parquet only where its
  proteins change). **PsA** (job `41656147`): of 5,367 genomes built here, 320 rewritten, all with the same protein
  count, 5,047 unchanged; check-proteins PASS 9,044/9,044 (job `41656464`); every residue in the 5,367 is one of the
  20, `X` or `U` (U 5,342, X 13,844 overall; job `41656465`). **Enterobacterales**, a fresh `annotated` with no hold
  → `preqc_annotated.csv` md5 `52c663e6…`: 31,888 (27,547 ingested, 4,341 our run; 48 running, 18,088 not yet) →
  collate (job `41656460`) → `bakta_annotation_preqc_annotated.csv` md5 `8307c6f8…` → proteins (job `41656953`) →
  reextract (job `41657884`): 218 rewritten (191 the same count; the 27 held genomes gain 1–10 proteins each),
  31,670 unchanged → **check-proteins PASS 31,888/31,888** (job `41658673`). No rewritten genome had a `.pt`, linked
  or embedded. `reextract` now also fails on any residue outside the 20, `X`, `U` and `O`.
- **PsA ESM-C — done.** Array `41656695` (`FLOTO-SL2-GPU`, from a worktree at `9c81041`): 18 of 18 tasks completed,
  1 h 24 min – 1 h 39 min each (≤ 299 genomes per task), embedding the 5,367 genomes built here. **`check` PASS**
  (job `41685615` at `6936272`; `/rds/user/dca36/hpc-work/logs/v2-pa_esm_check_20261122_235357/`, `problems.csv` 0
  rows, `.pass` beside it): of the 9,044 genomes of `genomes/bakta_annotation.csv` (md5 `d4df2c0f…`), every one has
  a protein parquet whose count is CDS − pseudogenes, contigs unchanged since annotation, and a `.pt` in
  `$AMR/processed/esm/` (bf16, 960-dim) — the 3,677 reused genomes' being §3.6's own.

**Numbers of record — duplicate assemblies, PsA v2.** `engine/genomes/find_duplicate_assemblies` (§3.6's screen,
`c94df7f`: same total length and contig count, then contig sequences hashed) over the 9,044 QC-pass genomes of
`genome_qc.csv` md5 `ae5365c2…` → `genomes/duplicate_assemblies.csv`: **12 groups of 2 confirmed identical** (2
refuted) — the same 12 as before the DNBSEQ genomes joined.
- 11 pairs are genomes built here, all NCBI GenBank assemblies — 7 from study 42168836 (consecutive BioSamples, two
  GCAs each), 2 more same-study pairs, and one pair that is PAO1's own reference sequence (SAMN02603714, whose record
  names GCA_000006765.1, and SAMN31936973). 1 pair is §3.6's (SAMN03105472/SAMN03105661).
- **Labels:** 8 pairs agree on every drug both carry; 4 disagree on one or two (§3.6's pair: meropenem and
  levofloxacin; others: ceftazidime, imipenem ×2). Two readings, untested and implying different actions: a noisy
  AST record, or an assembly attached to the wrong BioSample upstream.
- **The rate follows the assembly source, not the species:** §3.6's whole Enterobacterales cohort (67,326, no NCBI
  assembly) has 5 confirmed pairs (§3.6 *Caveats*); 11 of v2 PsA's 12 are NCBI GenBank deposits. A policy cannot be
  calibrated on §3.6 alone.
- **PAO1's 6,264,404 bp sequence is a v2 genome twice:** SAMN02603714 (PAO1's own BioSample, labelled on one drug)
  and SAMN31936973, whose GenBank assembly is PAO1's sequence exactly — so not, on its face, that isolate's own.
- **Decided (§6, 2026-11-22): one genome of each confirmed pair is deleted**, the lexicographically later Sample kept
  (the §3.6 agent's rule, `apps/entero/excluded_genomes.tsv`). v2's own lists — PsA's 12 now, Enterobacterales' after
  its collate — go to its split builder as `--exclude`. ⚠ An `--exclude` re-draws a whole split (*Next*, 4: what
  "contains §3.6's holdout" needs from the builder).
- **Enterobacterales v2** (`enterobacterales_cabbage_v2/genomes/duplicate_assemblies.csv` md5 `515228bf…`, the same screen over the
  117,277 of `embedding_input_unfiltered.csv`, inline, 39 s): 122 fingerprint groups (246 genomes) → **41 pairs
  confirmed identical** (81 refuted). 34 pairs are genomes built here; 5 are both §3.6's — exactly §3.6's 5, whose
  earlier Sample §3.6 already dropped; 2 pair a genome built here with a §3.6 one, and in both the v2 genome is the
  earlier Sample, so §3.6's is kept. Sources: ATB/ATB 24, NCBI/NCBI 16, portal 1 — unlike PsA, not mostly NCBI.
- **19 of the 41 pairs disagree on at least one drug** (ciprofloxacin 7, aztreonam 5, imipenem 4, meropenem 3,
  levofloxacin 2, four others 1 each; `genomes/duplicate_assemblies_pairs.csv` md5 `c82fb80c…`, one row per pair with
  origin, source, ST, study and the drugs that differ, against `binary_ast.csv`). Two runs of pairs carry
  most of it: 6 *Salmonella* ST34 pairs SAMN13074842–49 / SAMN35680651–58 (studies 32411120 / 39033143; 5 differ on
  ciprofloxacin) and 6 *Escherichia* sp. pairs SAMN11246194–202 / SAMN17183805–19 (31652858 / 35236849; 4 differ on
  aztreonam). **Hypothesis, untested:** a later study re-registered earlier isolates under new BioSamples, deposited
  the same assemblies and re-tested them.
- **The lists:** `apps/pa/excluded_genomes_cabbage_v2.tsv` (12) and `apps/entero/excluded_genomes_cabbage_v2.tsv`
  (41), `Sample, reason` as §3.6's: the earlier Sample of each confirmed pair, with any label disagreement named.
  The Enterobacterales list holds all 5 of §3.6's. PsA's holds SAMN03105472, of the pair §3.6 kept both of: both
  were in §3.6's train, so dropping it moves no §3.6 holdout genome.

**In flight.**
- **PsA:** nothing — genomes, Bakta and ESM-C are done (above). Splits wait on the exclude list and David's
  ANI-thinning question (*Next*, 4).
- **Enterobacterales ESM-C, pre-QC cohort.** Array `41659165` (`FLOTO-SL2-GPU`, 73 tasks of ≤ 300, 59–72 min
  each, 2026-11-22 22:40 → 2026-11-23 04:27) embedded 21,852 genomes; **`check` PASS 31,888/31,888** (job
  `41701657`; `/rds/user/dca36/hpc-work/logs/v2-entero_esm_check_20261123_042902/`, `problems.csv` 0 rows; 10,036
  of them Kp links, fp32). **Third round** (at `6936272`, a local chain of the driver's stages): `annotated` →
  `preqc_annotated.csv` 37,687 (27,547 ingested, 10,140 our run; 45 running, 12,841 not yet) → collate (job
  `41702229`) 37,687 of 37,687 → proteins (job `41702770`, 5,799 new) → reextract (job `41703431`, 37,687
  unchanged) → **check-proteins PASS 37,687/37,687** (job `41704298`) → link-kp (job `41704789`; 668 *Klebsiella*
  without a `.pt`, none in the Kp store) → **array `41705160`** (5,799 genomes, `genomes/esm_samples_to_embed.txt`
  md5 `23c1f8b1…`, 20 tasks of ≤ 290); then `check`. Once more on the post-QC cohort when Bakta ends.
- **Enterobacterales genomes: built, QC'd and collated** (*Numbers of record* above); the duplicate screen and its
  exclude list done. Left in stage 2: Bakta.
  **Bakta started before QC on SL3** (§6): `41627214` (13,096 genomes built so far and not in BakRep —
  6,920 NCBI, 5,704 ATB, 438 shovill, 34 portal; cohort `genomes/bakta/built_genomes_preqc_sl3.csv`, i.e.
  `built_genomes.csv` without the 31 NCBI genomes the re-plan rebuilds) and `41627455` (the 528 refused), 56 tasks at a
  time (SL3's 448-core cap) — 590 tasks done in the first ~5.3 h, 0 failed (~445 genomes an hour); 2,036 of 3,274
  done by 2026-11-23 00:04. **25 tasks (~100 genomes) failed in ~5 s each, every one on `cpu-q-391`**, whose `/tmp`
  is full (the array runs from `0dd9f51`, before the `TMPDIR=/local` fix): a black-hole node, taking a new task every
  few minutes. Excluded from the pending tasks of both arrays (`scontrol update … ExcNodeList=cpu-q-391`); its genomes
  go to the post-QC `submit`. CheckM2 list 01 (35,032 genomes, 18 tasks of 2,000): 80–105 min per task, MaxRSS
  ≤ 69.5 GB at 100G. **Against the QC-pass cohort** (`annotate_entero_genomes_cabbage_v2.sh status`, 2026-11-23,
  after `plan`: the BakRep roster unchanged at 28,075): of the 50,083 built here, 27,374 ingested, 8,829 with our
  run, **13,880 with neither** — the rest of the two SL3 rosters and ~8,300 genomes built after them; of the 67,194
  reused, 63,196 + 3,998, none missing. **Next: `submit all` on SL3** (§6's three days run to 2026-11-25) as soon as
  both SL3 arrays end — submit refuses while this cohort's own run arrays are live — then `collate` on
  `embedding_input_unfiltered.csv`.
- **`/tmp` is full on some icelake nodes:** bash 4.4 writes every here-string and here-document to `$TMPDIR`, so a job
  script fails at its first `<<<` (a CheckM2 task, in 3 s). v2's CheckM2, skani and Bakta jobs take `TMPDIR=/local`
  (`94938a3`). §3.6's own drivers do not.
- Two adversarial reviews of the build code ran first (16 and 6 findings survived, all fixed at `c66a737` /
  `d6c3e05`); a third, of the Bakta step (4 lenses, 5 findings), confirmed none.

**Next.** Agreed with David 2026-11-21, in this order, each stage §3.6's procedure:
1. **Labels — done** (above). `cabbage_v2_phenotype.py` replaced the planned `cabbage_v2_tests.py` and the optional
   `--phenotype`: writing v2 in the portal schema needed no change to any existing engine file.
2. **Genomes — PsA done; Enterobacterales built and collated, its Bakta running.** Only BioSamples absent from every §3.6 genome manifest: the per-sample stores are keyed by
   BioSample and shared, so v2 reuses every §3.6 genome with its shard, CheckM2, skani and `genome_qc.csv` rows, and the
   two agents never build the same one. ATB → portal GFF → shovill → **NCBI GenBank assembly** (§6, 2026-11-21);
   CheckM2 + skani; MLST; then Bakta (BakRep ingest or our run). Machinery: `engine/genomes/reuse_built_genomes.py`,
   `engine/genomes/scripts/build_genomes_reusing_built.sh` (+ `build_genomes_array_disjoint.sh`), configured by
   `apps/{pa,entero}/scripts/build_{pa,entero}_genomes_cabbage_v2.sh`. Agreed with the §3.6 agent 2026-11-21:
   - **The disjointness is an assertion, not a convention:** checked at plan, at every submission, by every array task
     before it builds, and at collate; §3.6's two manifests are now read-only. `build_genomes.build_one` deletes and
     rebuilds a genome's contigs when the stored `src=` header differs from the manifest row's source, so an
     overlapping row would destroy a §3.6 genome rather than skip it.
   - **MLST pinned at submission, one array per scheme:** per genus, the scheme each genus ended on in §3.6's re-typed
     `genome_qc.csv` (*Escherichia*, *Shigella* → `ecoli_achtman_4`; `salmonella`, `klebsiella`, `ecloacae`, `serratia`,
     `proteus`, `providencia`; *Morganella* and other scheme-less genera auto), plus genera §3.6 did not have
     (*Citrobacter* `cfreundii`, *Yersinia* `yersinia`, *Cronobacter* `cronobacter`). Per species, over the genus:
     aerogenes on `kaerogenes` under both names (§3.6 *Enterobacter aerogenes*, v2 *Klebsiella aerogenes*; the §3.6
     agent's correction) and the *K. oxytoca* complex on `koxytoca`. After collate, a genome built here must be on its
     pin and a reused genome must carry exactly its owner's scheme and ST — the check that catches §3.6's re-typed
     shards losing their mtime order.
   - **Aerogenes is sized as §3.6 sized it** (4.9 Mbp) under its v2 name, so its read depth is computed the same way.
   - **Bakta, §3.6's procedure** (BakRep's GFF3 where BakRep holds the genome on our ATB contigs, our run of the same
     release, flags, env and database otherwise — submit refuses unless the env's fingerprint is §3.6's) through
     `engine/genomes/scripts/annotate_genomes_reusing_built.sh`, configured by
     `apps/{pa,entero}/scripts/annotate_*_genomes_cabbage_v2.sh`. **A reused genome keeps its owner's annotation:** it
     is never on v2's ingest roster, and goes to v2's run only if no annotation record of it exists at all; v2's
     collate then checks every cohort genome has exactly one current record. Submit refuses while an owner's Bakta job
     is live. **Cost:** ~1.1 core-h per genome of our run (§3.6's sacct) — PsA ~2,800 core-h; Enterobacterales ~24,000,
     36 % of the 66,585 h left on `FLOTO-PROJECT-K-SL2-CPU` (`mybalance`, 2026-11-22). **The Enterobacterales
     array runs on `FLOTO-SL3-CPU` first** (§6, 2026-11-22: three days, then SL2 if it gets nowhere).
3. **ESM-C** (bf16) into the shared `$AMR/processed/esm/` through a sample list, **on `FLOTO-SL2-GPU`** (§6,
   2026-11-22), starting now on every genome whose Bakta annotation exists. Reused genomes keep their owner's `.pt`
   (§3.6): only genomes built here are ever on v2's array or link list, and `check` fails on a reused genome without
   one. Machinery: `engine/embedding/scripts/build_embeddings_reusing_built.sh` (stages proteins, check-proteins,
   link-kp, submit, check — §3.6's extractor, array, settings and checks; no smoke, the code being §3.6's; every job
   from a pinned clean worktree with `TMPDIR=/local`), configured by `apps/{pa,entero}/scripts/build_*_embeddings_cabbage_v2.sh`.
   - **PsA — done** (array `41656695`, `check` PASS 9,044/9,044; *Numbers of record*): 18 tasks × ~1.5 h ≈ 27 A100-h.
   - **Enterobacterales, before QC:** `annotate_entero_genomes_cabbage_v2.sh annotated` (re-plans, then
     `reuse_built_genomes bakta-annotated`: genomes built here with exactly one finished record — an ingest, or our run
     of §3.6's fingerprint — newer than their contigs, so a genome the DNBSEQ re-plan rebuilds stays out until
     re-annotated) → `genomes/preqc_annotated.csv` → `COHORT=… LIVE_RUNS_OK=1 … collate` (safe while the SL3 runs go
     on: a run task skips a genome whose run is current) → `bakta_annotation_preqc_annotated.csv` → proteins →
     check-proteins → link-kp (the Kp ESM-C store, as §3.6) → submit. Re-run as Bakta completes, then once more on the
     post-QC cohort (`ANNOTATION=bakta_annotation_embedding_input_unfiltered.csv`); submit embeds only genomes with no `.pt`.
   - **Ambiguous codons → `X`** (§6, 2026-11-22): the driver's `proteins` passes `--ambiguous-codons-as-x`, and
     `reextract` rewrites any v2 parquet written before it whose proteins change (reused genomes excluded). Then
     check-proteins, link-kp (only genomes still without a `.pt`) and submit again, for both cohorts; the 27 rejoin
     the Enterobacterales cohort.
4. **Splits and training** — new split tables, retrained models, a second PsA baseline, every holdout read through
   `engine.splits.load_splits` only. **The lineage holdout is §3.6's, under the same ≤ 5 % cap:** PsA **ST155**
   (replacing ST235 — David, 2026-11-21, relayed by the §3.6 agent; §3.6 records it); Enterobacterales *E. coli* ST10,
   *Klebsiella* ST37, *Salmonella* ST32, *Enterobacter* ST171, none for *S. sonnei* or *S. flexneri* (§6). **v2's
   held-out set contains §3.6's** — the same held-out ST plus §3.6's random holdout (David, 2026-11-21, §6): v2's PsA arm
   builds with `--assert-same-as paeruginosa=$AMR/processed/paeruginosa/splits` against the ST155 tables
   (`split_manifest.json` md5 `908fc0af…`), and the Enterobacterales arm against §3.6's once they exist. ⚠ **As
   written that cannot pass:** `assert_same_as` (`engine/splits/build_amr_holdout_splits.py`) requires a species'
   rows to *equal* the other tables' — the same genomes, the same splits — and v2's PsA has 9,044 QC-pass genomes to
   §3.6's 3,662. **`--pin-from NAME=DIR` does it** (additive; reviewed and approved by the §3.6 agent, 2026-11-23):
   §3.6's split kept per genome, read from the union of its tables; only v2's new genomes drawn, topping the holdout
   up to floor(0.2 × pool); refuses a two-split genome, another fold, or a pinned `holdout_st` whose ST is not held
   out. **Dry run** (local copies of v2 PsA's inputs, `--holdout-st paeruginosa:155`, both exclude lists and §3.6's
   groups, pinned from `$AMR/processed/paeruginosa/splits`; nothing written on CSD3): 3,661 of §3.6's 3,662 here
   (SAMN03105472 excluded, from train), none absent, no ST changed; train 5,625 · validate 1,409 · holdout 1,758 ·
   holdout_st 225, of which pinned 2,289 · 574 · 716 · 82. Each stratum's holdout rate is 0.20 (pinned 716/3,579,
   new 1,042/5,213): a proportional stratified sample, not a skewed mixture. **But v2's holdout and §3.6's share 716
   genomes (and 82 ST155):** their headline numbers are paired, not independent — right for a v1-vs-v2 comparison,
   wrong to read as a replication. Whether to accept that trade is put to David by the §3.6 agent. **Compare v2
   only with §3.6's ST155 arm:** the ST swap re-drew train, validate and the random holdout too, so an ST235-era number
   (`_superseded_st235/`) is on different genomes. The §3.6 agent is adding a **Klebsiella-only arm** (KpSC unit,
   held-out `klebsiella:37`) at David's request: its Klebsiella rows must be the same in v2 (a third place for
   `--assert-same-as`). **The PsA builder takes §3.6's two filters too:** `--exclude apps/pa/excluded_genomes.tsv`
   (the 15 Ecuador assemblies) and `--groups apps/pa/duplicate_isolate_groups.csv` (3 CDC strains under two BioSamples
   each), as `train_pa_models.sh splits` passes them — they are applied inside the builder, so §3.6's eligible list
   (3,677) is not its split cohort (3,662). Without them the 15 would land in a v2 holdout §3.6's baseline excluded.
   No duplicate-isolate list exists for any Enterobacterales species; whether one is built before v2's splits (and
   §3.6's Kp arm) is asked of David. Species units name the KpSC species explicitly, never a "Klebsiella" prefix (v2's 424
   *K. aerogenes* are not KpSC); E. coli/Shigella skani calls do not decide any unit (the panel cannot separate them).
   **Before the v2 splits, put to David** (with §3.6's evidence, not a recommendation): whether the holdout is thinned
   by ANI and which metric is read — §3.6's clonal nearest-neighbour lookup, no model, reaches mean AUROC 0.684 on the
   PsA random holdout and 0.516 on ST155 against the fine-tunes' 0.802 and 0.708
   (`$AMR/processed/paeruginosa/ani/psa_nn_lookup_baseline.csv`, `models/holdout_metrics.csv`; a lower bound on
   memorisation, not a ceiling — §3.6). **After the v2 models train:** §3.6's breakpoint label-noise measurement on
   v2 (`apps/pa/{mic_distance_bands,holdout_error_by_mic_distance}.py`, `docs/holdout_clonality_and_label_noise.md`;
   no engine code) — v2 PsA has 19.8–42.3 % of labels within one dilution of the breakpoint, Enterobacterales 3.9–10.2 %.

**Caveats.**
- **puffin** — above. A puffin-free label set is a filter on `database`, not a rebuild.
- v2's species labels include non-bacteria (*Homo*, metagenome, *Candida*) and genus-only names; the genus filter and
  the skani species check handle them. About 95 Enterobacterales genomes sit in genera outside the breakpoint
  taxonomy's 21 (*Lelliottia* 12, *Plesiomonas* 10, …) and are excluded, as in §3.6.
- **CABBAGE's `Updated_phenotype_EUCAST` mixes ECOFF calls with clinical ones** — where `used_ECOFF` = yes it calls
  15,481 of 17,838 Enterobacterales tests R that clinical breakpoints call S (`cabbage_eucast_vs_our_eucast_v15.csv`).
  Never a label; the rule recomputes from the MIC. Not carried for disk tests either, where v2's `phenotype` is
  sometimes empty (the 74 *Salmonella* above): those tests give no label (David, 2026-11-21, §6).
- **Labels within one dilution of the breakpoint** (PsA 19.8–42.3 % by drug) — David: "important… maybe a reason the
  model is so hard for Bacformer". An open question (§6), not yet analysed.
- Phenotypes beyond R/S (intermediate, less susceptible, non-susceptible, SDD) give no call to a test with no MIC, as
  in §3.6. 3,334 (BioSample, drug) pairs are duplicated within v2 (314 with differing phenotypes); averaging handles
  them (155 conflicts in the dry run).
- **The v1 directory keeps its EBI name `2026-07`** rather than the dated name proposed: both §3.6 builds and their
  manifests reference it. A rename would need a `2026-07` symlink.
- Shared code, one branch: each build's `manifest.json` records the git SHA, so an engine change between the two
  experiments' runs stays traceable.
- **One file per genome, shared by both experiments:** a reused genome's contigs and Bakta GFF3 are §3.6's own files,
  not copies. Anything that rewrites them for §3.6 changes v2's input too; v2's collates re-check each against its md5.
- **NCBI GenBank assemblies are the submitters' own**, assembled by whatever pipeline they used, unlike ATB's uniform
  shovill/SKESA: they pass QC at about ATB's rate (PsA 99.1 % against 96.3 %), but are a source to stratify by if a v2
  result looks odd.

**Owns.** `engine/ast_labels/cabbage_v2_phenotype.py` (+ its test), `engine/genomes/reuse_built_genomes.py` (+ its
test), `engine/genomes/scripts/{build_genomes_reusing_built,build_genomes_array_disjoint,annotate_genomes_reusing_built}.sh`, the `ncbi_gca` source and
`--genome-size-by-species` in `engine/genomes/{plan_genome_sources,build_genomes}.py` (opt-in; §3.6's path unchanged),
the `*_cabbage_v2` run scripts,
`$AMR/raw/ebi_amr_portal/cabbage_v2_20261121/`, `$AMR/processed/ebi_amr_portal/scoping_cabbage_v2_20261121/`,
`$AMR/processed/{paeruginosa,enterobacterales}_cabbage_v2/`.

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
| `$AMR/processed/paeruginosa/binary_ast.csv` | `engine/ast_labels/build_binary_ast_labels` (via `apps/pa/scripts/build_pa_ast_labels.sh`) | PsA split tables (`engine/splits`) | Every PsA split and model (§3.6) |
| `$AMR/processed/paeruginosa/cohort.csv` | the same build | PsA genome acquisition (`genome_route`, reads/assembly accessions) | The PsA genome manifest |
| `$AMR/processed/paeruginosa/genomes/genome_manifest.csv` (+ `ena_filereport_cache.tsv`) | `engine/genomes/plan_genome_sources` (`build_pa_genomes.sh plan`) | the build arrays, `collate` | Every PsA genome built from it; a re-plan that changes a genome's source rebuilds it in place |
| `$AMR/raw/{contigs,prodigal}/<Sample>.{fna,gff}.gz` (keyed by BioSample, shared across organisms) | `engine/genomes/build_genomes` arrays | QC, `collate`, the protein extractor | Protein parquets and ESM-C embeddings of that genome (the PsA `check` stage flags a parquet older than its GFF) |
| `$AMR/processed/paeruginosa/genomes/{genome_qc,embedding_input,st_counts_by_drug}.csv` | `engine/genomes/collate_genomes` | Bakta annotation (`embedding_input.csv` is the cohort annotated and embedded); the held-out-ST choice | The PsA annotation table and embeddings' cohort, the ST holdout, every PsA split |
| `$AMR/raw/ebi_amr_portal/cabbage_v2_20261121/cabbage_v2.csv` | external (given to the project as `cabbage_v2.csv.bz2`); staged and md5-checked 2026-11-21 | `engine/ast_labels/cabbage_v2_phenotype.py` (§3.7); the scoping in `processed/ebi_amr_portal/scoping_cabbage_v2_20261121/` | Every §3.7 label, genome manifest, split and model |
| `$AMR/processed/ebi_amr_portal/cabbage_v2_20261121/phenotype.parquet` (+ `.json`) | `engine/ast_labels/cabbage_v2_phenotype` (run by either `*_cabbage_v2` label script if absent) | both §3.7 label builds | Every §3.7 label; delete it to rebuild from the CSV |
| `$AMR/processed/{paeruginosa,enterobacterales}_cabbage_v2/{binary_ast,cohort,label_counts}.csv` | `build_binary_ast_labels` via `apps/{pa,entero}/scripts/build_*_ast_labels_cabbage_v2.sh` | §3.7 genome acquisition and splits | Every §3.7 genome manifest, split and model |
| `$AMR/processed/{paeruginosa,enterobacterales}_cabbage_v2/genomes/{genome_manifest,reused_genomes}.csv`, `by_scheme/` | `engine/genomes/reuse_built_genomes` + `plan_genome_sources --ncbi-fallback` via `apps/{pa,entero}/scripts/build_*_genomes_cabbage_v2.sh plan` (read-only once written) | the §3.7 build arrays (by row position), QC and collate | Every §3.7 build shard (a re-plan moves `by_scheme/` aside; `missing-tasks` reruns any task whose rows moved) |
| `$AMR/processed/{paeruginosa,enterobacterales}_cabbage_v2/genomes/{genome_qc,embedding_input,st_counts_by_drug}.csv` (+ `embedding_input_unfiltered.csv` for Enterobacterales) | `collate_genomes` over `collate_inputs/` (this cohort's builds + the reused genomes' §3.6 rows) via `build_*_genomes_cabbage_v2.sh collate`, published only after `check-schemes` passes | §3.7 Bakta (the cohort annotated), ESM-C, splits | The §3.7 annotation table, embeddings list and every §3.7 split |
| `apps/{pa,entero}/excluded_genomes_cabbage_v2.tsv` | the earlier Sample of each confirmed pair in `{paeruginosa,enterobacterales}_cabbage_v2/genomes/duplicate_assemblies.csv` (`engine/genomes/find_duplicate_assemblies` over the QC-pass set; §6 2026-11-22) | the §3.7 split builder (`--exclude`) | every §3.7 split |
| `$AMR/processed/{paeruginosa,enterobacterales}_cabbage_v2/genomes/bakta/{built_genomes.csv,bakrep_roster.tsv,ingest_status.tsv,run_roster_*.tsv}` → `genomes/bakta_annotation*.csv` | `reuse_built_genomes bakta-plan` / `bakta-roster` and `bakta.py ingest` / `run` / `collate` via `annotate_*_genomes_cabbage_v2.sh` (run rosters dated, read-only) | the §3.7 protein stage and ESM-C (`bakta_annotation*.csv`) | The proteins and ESM-C embeddings of every genome built for v2 |
| `$AMR/processed/enterobacterales/{binary_ast,cohort,label_counts}.csv` | `engine/ast_labels/build_binary_ast_labels` (via `apps/entero/scripts/build_entero_ast_labels.sh`) | Enterobacterales genome acquisition and splits | Every Enterobacterales genome manifest, split and model |
| `$AMR/raw/bakrep/v2_export_20250612/bakrep-export.tsv.gz` + `$AMR/processed/bakrep_coverage_20261119/` | BakRep's published export; `join.py` | Which genomes take BakRep's Bakta annotation and which get our own Bakta run | The annotation source of every PsA and Enterobacterales genome |
| `$AMR/raw/bakta_db/v5.1_full/` (+ `v5.1_full.md5`, `README.txt`) | copied from nuna's `refs/bakta/db` (job `41494699`); AMRFinderPlus DB `latest` → 2024-07-22.1 | every Bakta run of ours (`engine/genomes/bakta.py`) | Every our-run annotation (a run's fingerprint includes this DB's `version.json` and AMRFinderPlus DB) |
| `$AMR/raw/bakta/<Sample>.bakta.gff3.gz` + `.ingest.json` (BakRep's) or `.run.json` (ours) | `engine/genomes/bakta.py` `ingest` / `run` (`annotate_pa_genomes.sh`) | `bakta.py collate` | Protein parquets and ESM-C embeddings of that genome |
| `$AMR/processed/paeruginosa/genomes/bakta_annotation.csv` | `bakta.py collate` (`annotate_pa_genomes.sh collate`) | `build_pa_embeddings.sh` (its cohort) | PsA protein parquets and ESM-C embeddings |
| `$AMR/processed/enterobacterales/genomes/bakta_annotation_embedding_input_unfiltered.csv` | `bakta.py collate` (`annotate_entero_genomes.sh collate`, `COHORT=embedding_input_unfiltered.csv`) | `build_entero_embeddings.sh` (its cohort, via `ANNOTATION`) | Enterobacterales protein parquets, the Kp link and ESM-C embeddings |
| `$AMR/processed/enterobacterales/genomes/esm_linked_kp.tsv` + the symlinks in `$AMR/processed/esm/` | `build_entero_embeddings.sh link-kp` (`engine/embedding/link_existing_embeddings.py`) over `david/bacformer_processed/esm/klebsiella/` | `build_entero_embeddings.sh submit` (skips genomes with a `.pt`), `check_cohort_store` | `esm_samples_to_embed.txt`; a Kp source `.pt` that changes silently changes our store |
| ESM-C sample lists: `$AMR/processed/paeruginosa/esm_samples{,_smoke}.txt`, `$AMR/processed/enterobacterales/genomes/esm_samples_{to_embed,smoke}.txt` | `build_pa_embeddings.sh` `submit` writes `esm_samples.txt` (sorted Samples of `bakta_annotation.csv`); `build_entero_embeddings.sh` `submit` writes `esm_samples_to_embed.txt` (sorted cohort Samples with no `.pt` in the store yet); `smoke` its own `_smoke` list | `esm_embeddings_array.sh` (`SAMPLES`, md5-pinned) | Pending tasks of an array submitted with the old list: they refuse to run (exit 3) |
| `$AMR/processed/bakta_vs_bakrep_validation/` | `engine/genomes/scripts/validate_bakta_against_bakrep.sh` | the decision to annotate BakRep-less genomes with our own run | — (evidence only) |
| `$AMR/processed/paeruginosa/ani/{psa_heldout_max_ani_to_train,psa_nn_lookup_baseline}.csv` (+ `psa_heldout_split_ids.txt`) | `skani dist` (job `41644484`) → `engine/splits/holdout_lookup_baseline` | the ANI-thinning threshold decision; how much of §3.6's random-holdout AUROC is clonal lookup | — (evidence only). ⚠ Tied to the ST155 split tables: rebuild the splits and these describe a holdout that no longer exists. The 599 MB pairwise TSV behind them is **not** kept (`hpc-work` scratch, swept) — it regenerates in ~10 min on 32 cores |
| `$AMR/processed/paeruginosa/mic_bands/<drug>_mic_bands.csv` + `mic_band_summary.csv` | `apps/pa/mic_distance_bands` over `ast_tests_called.parquet` + `eucast_v15_breakpoints.csv` | `apps/pa/holdout_error_by_mic_distance`; the exclude-vs-multi-class-head decision | — (evidence only). Rebuilt if the breakpoint table or the labels change |
| `$AMR/processed/paeruginosa/mic_bands/error_by_distance/{error_by_band,near_vs_far,error_concentration,conditional_logit,per_drug_vs_auroc,per_drug_correlations,leave_out_ceiling}.csv` | `apps/pa/holdout_error_by_mic_distance` over `models/<drug>/eval_scores*.npz` + `results.json` | [`docs/holdout_clonality_and_label_noise.md`](docs/holdout_clonality_and_label_noise.md) | — (evidence only). ⚠ Tied to **these** checkpoints and **these** split tables: retrain or re-split and every row is stale |

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
| CSD3, multi-species AMR (§3.6) | `$R/amr_prediction` (referred to as `$AMR`; `$BACPREDICT_AMR_ROOT` overrides it; its scripts do not read `$BACPREDICT_DATA_ROOT`) |
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
| 2026-11-18 | **Gene attribution is run on the LOGIT, with each genome's base logit stored beside it.** Attributing `sigmoid(logit)` directly is what the published Bacformer code does, and it **saturates**: 1,895 of 4,242 Salmonella genomes came back with zero attribution variance, contributing nothing. Our two models are more confident still. Storing the logit shift loses nothing because `delta_probability()` reconstructs the bounded −1..1 probability score exactly, so one run yields both scales. |
| 2026-11-18 | **Attribution's primary statistic is per-genome-normalised \|attribution\|; resistant-vs-susceptible per gene is secondary.** David's choice. Per-genome normalisation also makes the logit and probability scales interchangeable for ranking. Zero-variance genomes are **counted and reported**, never silently zeroed — that count is the saturation diagnostic. |
| 2026-11-18 | **A within-lineage repeat of the R-vs-S contrast is what licenses any causal reading of attribution.** A determinant signal survives stratification by TB lineage / Kp sublineage; a lineage marker collapses. Until it is read, attribution is descriptive only. This is why the cohort is stratified on lineage at selection time rather than sampled at random. |

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
| 2026-11-18 | **captum must be pinned `>=0.9.0`.** captum **0.8.0 pins `numpy<2.0`** and silently drags the whole shared environment back to numpy 1.26 (taking zarr, scanpy and rich down with it) — measured here: a `uv sync` downgraded the shared `.venv` before it was caught and reverted. 0.9.0 drops the pin and locks as a purely additive 20-line change. The venv is shared by several concurrent agents, so this is not a local inconvenience. |

### Multi-species AMR

| Date | Decision |
|---|---|
| 2026-11-18 | **Label rule, every species:** EUCAST v15.0 from the MIC — resistant iff MIC > the R breakpoint, "I" counts as S. A censored MIC that straddles R is dropped, not rescued by the source's call. A test with no MIC keeps the source's own R/S. Portal MICs with no unit recorded (not disk tests) are called as mg/L ("MIC first"). |
| 2026-11-18 | **One ten-drug panel for PsA and Enterobacterales:** ciprofloxacin, ceftazidime, meropenem, tobramycin, amikacin, imipenem, piperacillin-tazobactam, aztreonam, levofloxacin, cefepime. Gentamicin (no EUCAST PsA breakpoint) and colistin dropped; ceftazidime-avibactam and ceftolozane-tazobactam deferred. |
| 2026-11-18 | **Holdout:** one MLST ST with ≥50 genomes, held out zero-shot, alongside other holdouts. Built by `engine/splits` like every other split. |
| 2026-11-18 | **An even genome surface, for publication.** Shovill for every genome assembled from reads, including the 195 Khaledi et al. genomes that NCBI already holds assemblies for. AllTheBacteria assemblies (themselves shovill) where they exist, including the undocumented 2026-09 batch. One Prodigal 2.6.3 run with Prokka's flags on every genome, then `mlst`. Genomes that fail the species check are excluded. RNA-seq-only and ONT-only BioSamples are dropped (shovill is short-read only). Reads under 10× depth are not assembled. New pixi env (bioconda shovill / mlst / prodigal 2.6.3) approved. |
| 2026-11-18 | **ESM-C embeddings are stored in bf16** in a new project_k store. They are computed in bf16, so this is lossless and halves the store. It does not bear on the bf16 attribution finding in §3.1, which concerns the model's logits, not its stored inputs. |
| 2026-11-18 | **Later experiments, not the primary test:** benchmark on AMR-GNN's own cuts, which needs a restricted training set; and thin the training set by ANI so that a dominant species such as *Salmonella* cannot dominate. |
| 2026-11-19 | **Scope: PsA and Enterobacterales only for now**; the all-species cohort stays planned. Its labels need a decision first: our breakpoint table holds only Enterobacterales and *Pseudomonas* rows, and many species × drug pairs (all of *M. tuberculosis*) have no EUCAST breakpoint. |
| 2026-11-19 | **Bakta is the annotation standard, not Prodigal** (amends the 2026-11-18 surface decision's "one Prodigal 2.6.3 run"). Bakta 1.9.4 with the full DB v5.1: BakRep's annotation where BakRep holds the genome, our own run of the same version and flags otherwise, gated by a reproduction check. Even annotation is sorted before anything is embedded. `mlst` stays ours (2.35.0) on every genome. |
| 2026-11-19 | **CheckM2: our own run on every genome** (one QC surface), not AllTheBacteria's published values. scikit-learn stays unpinned: 1.6.1 and 1.9.1 give byte-identical reports on 50 genomes. |
| 2026-11-19 | **PsA: embed every QC-passing genome; decide the held-out ST and the two cohort drops at the splits stage.** Embeddings are per genome and label-free. **ESM-C runs on FLOTO-SL2-GPU.** |
| 2026-11-20 | **The Bakta gate passed, so a genome BakRep lacks gets our Bakta run** (as agreed 2026-11-19, conditional on the gate): 15/15 identical to BakRep in every feature, attribute and AMR call. BakRep's GFF3 is used only when its embedded contigs equal ours name for name; any genome it refuses is also annotated by us. **AMRFinderPlus DB 2024-07-22.1**, not the 2023-11-15.1 that DB v5.1 ships: AMRFinderPlus 3.12.8 (what Bakta 1.9.4 resolves to) refuses it ("requires database version at least 2023-12-15.2"); with 2024-07-22.1 every AMR call matched BakRep's. |
| 2026-11-20 | **Enterobacterales pipeline choices** (David): (i) **annotate and embed every genome that passes the non-species QC**; the species rule is set from the label × call crosstab and applied at splits. (ii) **A BakRep file ingest refuses (contig names, release, content) is annotated by our Bakta run** ("use Bakta, that is the point"). (iii) **Existing *Klebsiella* ESM-C embeddings are reused where done**: symlinked into `$AMR/processed/esm/` and skipped by the ESM array. |
| 2026-11-20 | **Cross-species design** (David): **one model per drug**. **PsA-only is the baseline; PsA + all Enterobacterales is the comparison experiment.** H0: PsA alone is the best we can do; H1: cross-species training does better on PsA and/or makes attribution more accurate. **Holdouts:** one large, scoreable ST per species (PsA: ST235) held out zero-shot beside a random 20 % holdout; ST membership **exact ST only** (not urgent). **PsA drops:** the 15 Ecuador long-read portal assemblies (SAMN338421xx) are left out ("inconsequential, not worth time"); the low-depth Cortes-Lara genomes that pass QC stay. **No per-class minimum:** the ten drugs were chosen with > 1,000 genomes each. |
| 2026-11-20 | **All multi-species AMR data lives under `project_k/david/amr_prediction/{raw,processed,final}`**, never in `~/rds/hpc-work` (David's scratch: logs and temporary files only). Per-sample files keyed by BioSample are flat and shared across species (`raw/{contigs,bakta}`, `processed/{protein_sequences,esm}`); per-species batch tables go in a species folder (`processed/paeruginosa/`, `processed/enterobacterales/`). Raw inputs and annotation in `raw/`; QC, proteins, embeddings, splits and models in `processed/`. |
| 2026-11-21 | **CABBAGE v2 is a second experiment, not an update of §3.6** (David: "second test, same procedures"). v2 labels are neither merged with nor substituted for the portal 2026-07 labels; every stage repeats §3.6's procedure (labels with the AMR-GNN supplement for PsA, genomes, QC, MLST, Bakta, ESM-C), then **new splits, retrained models and a second PsA baseline**, compared with §3.6's models. §3.6 continues on 2026-07. |
| 2026-11-21 | **A v2 MIC with no sign is read as "="** (David). Where v1 holds the same test, its sign was "=" for 94,642 of 94,657 Enterobacterales and 4,529 of 4,529 PsA pairs. |
| 2026-11-21 | **puffin is labelled under the same rule as every source**, with each test keeping its `database` so that a puffin-free label set is a filter (agreed with the §3.7 label stage). What puffin is remains to be asked of the CABBAGE side. |
| 2026-11-21 | **One branch for both experiments** (David: "if either of you … change shared code then the two experiments differ"). v2 work stays on `refactor/consolidate-engine`; changes to shared code are additive and communicated between the two agents. This overrides root `CLAUDE.md` §0.5 "one branch per agent" for this pair. |
| 2026-11-21 | **The lineage holdout is capped at 5 % of a drug's labelled set, and is additional to the random 20 %** (David). The random holdout is taken *after* the ST is removed, so the two are disjoint and both are scored — that part of the 2026-11-20 design was already right. What was wrong was the size: **PsA's ST235 is 7.1–9.9 % (mean 8.6 %)**, so 26–28 % of genomes were held out of training — "close to 30 % of genomes. Worth nothing." **PsA switches to ST155** (2.1–3.9 %, 8 of 10 drugs scoreable, as ST235 was) and **all ten models are retrained now** — "Switch now. Better to do it now." ⛔ Because the builder permutes the whole non-held-out pool, this re-draws train, validate and the random holdout too, so the entire ST235 baseline is superseded rather than just its ST column; the old artifacts are preserved under `_superseded_st235/`. **Enterobacterales under the same cap:** ST131 is out (22.3 % of a drug's labels, and 30.4 % at worst, after the MLST re-type doubled it to 3,029 genomes); **ST37 for *Klebsiella*** (2.2 %); ***E. coli* ST10** (756 genomes, 2.4 % mean / 3.7 % max, 7 of 10 drugs scoreable; David chose it over ST38 2026-11-21 — he had written "ST37" for *E. coli*, and there is no *E. coli* ST37 under any scheme). *Salmonella* ST32 and *Enterobacter* ST171 stand (each scores only 2 drugs either way). **No ST holdout for *S. sonnei* or *S. flexneri*** — one clone is 97 % and 84 % of them, so holding it out is a species holdout. |
| 2026-11-21 | **Klebsiella is a single-species arm too, built exactly like PsA** (David: "Can we run Klebsiella as a single species with hold out, like PsA… We have run klebsiella already, but didn't have any holdouts. A major drawback of what we did. And we need this consistent so we can check if Kleb improves with cross species too."). Unit `kpsc` (the five KpSC species, named — *K. aerogenes* excluded), held-out ST `klebsiella:37`, a random 20 % of the rest, and the PsA driver's stages, guards and hyperparameters unchanged, so the two baselines differ in organism alone. **This makes three single-species baselines to keep aligned, not two**, and gives `--assert-same-as` a second job: the Klebsiella rows must be identical across the Kp-only arm, the cross-species arm and v2. **Structural consequence:** in the cross-species arm KpSC and PsA are each their own `--species` unit and the other Enterobacterales unit excludes them, because the builder refuses a genome eligible in two units and a shared unit would shuffle the Kp rows differently. |
| 2026-11-21 | **MLST is typed with the scheme pinned per species** (David: "Just do it per species. I don't want to spend more time on this."). Auto-detection mis-typed 5,164 of 67,326 QC-pass Enterobacterales genomes (7.7 %), which would have put genomes of a held-out clone into train under a bogus `st_key`. Fixed by re-typing those genomes, not by changing `engine/genomes` — the CABBAGE v2 experiment (§3.7) runs the same code, so a behaviour change there would make the two experiments differ. v2 pins the scheme at submission the same way. |
| 2026-11-21 | **CABBAGE v2 labels accepted as built** (David, on the §3.7 checks). Where v2 and §3.6 disagree on a label, v2's EUCAST v15 re-call of its own MIC stands — "a consistent MIC call to label any with EUCAST", as §3.6 did. The 273 §3.6-labelled genomes v2 does not label are ignored ("tiny … not going to change any results"). v2 disk tests with no R/S call give no label: CABBAGE's `Updated_phenotype_*` is not used ("I'd just drop them"). |
| 2026-11-21 | **v2 genomes with no reads take the submitter's NCBI GenBank assembly** (David: "An NCBI assembly is adequate … use NCBI when no reads exist … better than throwing out that much data"). Source `ncbi_gca`, last in the order ATB → portal assembly → shovill → NCBI, so the even surface stays first wherever it is attainable; like `portal_gff` it is a submitter assembly. §3.6's planner had no such route because every §3.6 genome had reads or a portal assembly; ~17,500 v2 genomes (15,135 Enterobacterales, 1,392 PsA) have neither — no runs in ENA, and none in SRA for 12 of 12 checked. |
| 2026-11-21 | **v2's held-out set contains §3.6's** (David: "v2 held out should include v1 and same sublineage. Yes."): the same held-out ST per species plus §3.6's random holdout, enforced with `--assert-same-as`, so v1 and v2 models are scored on common genomes. |
| 2026-11-22 | **v2 assembles DNBSEQ/BGISEQ paired reads** (David: "DNBSEQ / BGSEQ - yes"), with shovill as for Illumina, but only for a genome with no usable Illumina run (`plan_genome_sources --fallback-read-platforms`, opt-in; §3.6's plans are unchanged). By the priority order this moves the genomes that dropped as `other_strategy` with such reads (PsA 414, Enterobacterales 510, all ≥ 10×) and 31 Enterobacterales genomes that had taken an NCBI assembly to `shovill`. Applied by `replan`, which publishes only if no other row moved. |
| 2026-11-22 | **The Enterobacterales v2 Bakta run goes on the free `FLOTO-SL3-CPU` first** (David: "lets try the free SL3. See how much we get from it in the next 3 days and if none, will go SL2"). SL3 is QOS `cpu2`: 12 h wall, 448 cores per user at once, priority 1,000 against SL2's 5,000 (`sacctmgr show qos`). |
| 2026-11-22 | **An ST is held out only where a species unit has ≥ 1,000 genomes, and the ST ≥ 50** (David: "we only hold out an ST if there will be at least 50 samples. We need at least 1000 samples from the species to make that even a concern"). Grouping a small genus into a unit matters only for that holdout, so no v2 minor genus needs one: the largest, *Serratia*, has 553 labelled genomes (v2 `cohort.csv`). |
| 2026-11-22 | **One genome of each byte-identical duplicate pair is deleted** (David: "just delete one of each of duplicates. keep it simple. You are talking about tiny numbers."). The pairs are `engine/genomes/find_duplicate_assemblies`' confirmed groups; the lexicographically later Sample is kept (§3.6: 5 Enterobacterales genomes, `apps/entero/excluded_genomes.tsv`; §3.6 PsA's one pair, both in train, left as it is). v2 applies the same rule to its own screens (PsA: 12 pairs). |
| 2026-11-22 | **All v2 GPU work runs on `FLOTO-SL2-GPU`** (David: "The work is too important for the lab. Just run SL2 for all GPU work." — in his experience SL3 never picks up GPU jobs). SL3 is tried only for the Enterobacterales Bakta CPU array, which would cost ~£240 on SL2 (~24,000 core-h at ~£0.01), because SL3 does sometimes pick up CPU work. ESM-C starts on every v2 genome that already has its Bakta annotation, without waiting for the rest. |
| 2026-11-22 | **v2 translates every codon holding a base other than A, C, G or T as `X`**, as Bakta's own `.faa` does (David: "b) is fine. so few that it doesn't matter."). NCBI GenBank contigs carry IUPAC ambiguity codes; Biopython resolves an ambiguous codon where it can (`ACY` → T, `RAY` → B) and reads `TRA` as a stop, so the extractor's internal-stop rule dropped that CDS (27 Enterobacterales genomes, one protein each). `extract_proteins_from_gff_fna(ambiguous_codons_as_x=True)` / `--ambiguous-codons-as-x`, opt-in: §3.6's stores and drivers are unchanged. v2's parquets written before it are brought up to it by `reextract_proteins`, which rewrites only those that change. |

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

- ~~**Near-breakpoint labels and the PsA ceiling**~~ — **MEASURED 2026-11-22, the hypothesis holds** (§3.6
  *Numbers of record — PsA label noise at the breakpoint*). David, 2026-11-21: "Labels within 1 dilution of
  breakpoint is important. That is a huge number in PsA and maybe a reason the model is so hard for Bacformer."
  It is: class-conditional holdout error is **3.9× (susceptible) and 2.9× (resistant)** higher at the cut than
  far from it, and conditional on genome the odds of error are **2.52× (95 % CI 2.01–3.16)** for a drug where
  that genome sits within one dilution. Removing the near band is worth a mean **+0.068 AUROC** on the 7 of 10
  drugs that clear both deletion nulls. Still open, and David's call, now **three** options: **exclude** those
  genomes; a **multi-class head** (⚠ 7 drugs have no S category, so its classes would have to be defined on
  measurement distance, not the EUCAST category); or **regress log2 MIC** directly and threshold afterwards
  (David, 2026-11-22: "We might need to consider predicting log MIC instead of the binary call"). ⚠ The third
  changes the task, the loss and every downstream comparison, and ~⅓ of tests are censored (`≤`/`>`), which a
  plain regression cannot consume honestly — it needs interval-censored handling. The MIC is already retained
  per test in `ast_tests_called.parquet` and reaches nothing today. Also still open: the same measurement for
  Enterobacterales (3.9–10.2 % near-breakpoint, so a much smaller expected effect), which has no trained arm yet.
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

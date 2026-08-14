# `visualisations/` — what is in this tree and what it is safe to quote

**This tree is a publication mirror, not a source of truth.** Figures and tables are rendered here so
they can be reviewed and cited without cluster access. Every number in it was read from an artifact
on the cluster, and **the artifact is the authority** — if the two disagree, the artifact wins and the
file here is stale.

The state of every layer, and the numbers that are of record, live in
[`PROJECT_STATE.md`](../../../PROJECT_STATE.md) at the repo root.

## The one file you may quote a catalogue ceiling from

| File | Contents |
|---|---|
| `kp/catalogue_ceiling_panel.csv` | All 22 Kp drugs, CARD, **current** |
| `tb/catalogue_ceiling_panel.csv` | 9 of 10 TB drugs, WHO/TB-Profiler, **PROVISIONAL** |

Each row carries its own provenance, so the caveat travels with the number rather than living in a
doc someone may not read:

- `ceiling_catalogue` — `CARD` (Kp) or `WHO_tbprofiler` (TB).
- `ceiling_grain` — `allele` for Kp (a `family` grain also exists on the cluster and differs slightly);
  `one_hot` for TB.
- `ceiling_estimator` — **this is the one that matters.** `deployment_holdout` means the determinant LR
  was fit and scored through the same `<drug>_split.csv` holdout the fine-tune was evaluated on, so it
  is directly comparable to an FT number. `kfold_probe` is the retired whole-cohort k-fold probe: a
  *different estimator on a different evaluation set*, so a TB ceiling-vs-FT gap is not yet a
  like-for-like comparison.
- `ceiling_status` — `current` or `provisional`.

**Rebuilt from** (CSD3, `$R = /home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david`):

- Kp — `$R/bac_ast_prediction/processed/train_kleb_ast/card_ceiling/<drug>/card_determinant_lr_<drug>_allele.csv`,
  row `__ALL_CARD__`.
- TB — `$R/processed/train_tb_ast/snp_embeddings/tbprofiler_gene_lr/tbprofiler_gene_lr_<drug>.csv`,
  row `__ALL_WHO_one_hot__`.

### Telling the two estimators apart

Both schemas carry a `mut_auroc_sd` column, so its *presence* distinguishes nothing. The marker is its
**value**: the deployment-holdout scorer fits once and reports `mut_auroc_sd == 0.0` exactly, while the
k-fold probe reports the spread across folds (non-zero — TB ranges 0.0002 to 0.0084). The column
vocabulary differs too: the Kp/CARD schema has `category`, `n_determinants` and `is_causal`; the
retired TB schema has `region`, `n_variants` and no `is_causal`.

### Why TB is provisional

Four independent problems, all of which need the ceiling rebuilt rather than patched:

1. Built **2026-06-19**, against a cohort superseded on 2026-07-08.
2. Read from `$R/processed/…`, which is the **deprecated** May tree, not the canonical
   `$R/bac_ast_prediction/…`.
3. **Missing rifabutin** — 9 of 10 drugs.
4. **Different estimator** (above), which is the defect that makes the number non-comparable rather
   than merely old.

Rebuilding it — all 10 drugs, through `load_splits` + `score_onehot_frame`, into a Kp-mirroring
`who_ceiling/` layout — is the first task of any TB work. **Do not copy the June CSVs into the
canonical path; that launders them.**

## `_superseded/`

Artifacts kept only so the history of a wrong number is traceable. Nothing in that directory may be
quoted or consumed — see its own README for what each file is and which numbers came out of it.

## Everything else in this tree

Per-drug figures and tables rendered by the plotting modules. They carry no provenance columns, so
treat any number in a filename or a figure as **undated** unless `PROJECT_STATE.md` names it.

**Figures are the easiest thing here to quote by accident**, because a reader sees a number without
ever opening a CSV, and no test covers image content. Known stale:

- `*_card_cause_histogram_*.png` — predate the presentation fixes.
- `*_amr_ladder_table.csv` (untracked) — predate the read-out leak fix; the *cluster* ladder tables
  are current, it is the checked-in mirror that is not.
- `amr_panel_auroc.png` (both organisms) — **quarantined**; they plotted the superseded per-drug
  fine-tune numbers with those numbers in the captions.

Regenerate rather than cite any of them.

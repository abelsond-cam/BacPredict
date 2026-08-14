# Handover — the state and memory audit, 2026-08-14

> **✅ The independent review asked for below has been done, and its findings are fixed.** This file
> is kept as the record of what was asked and what the original agent was unsure of. **§4's items 1
> and 3 were both found to be wrong and have since been corrected in the repo** — do not act on them
> as written. What the review changed:
>
> - The `mut_auroc_sd == 0.0` rule in §4.1 was a **weaker check than claimed**: it answers "fit
>   once?", not "fit on the deployment holdout?", and the k-fold probe *can* report exactly 0.0. The
>   test now reads the explicit `ceiling_estimator` field instead.
> - The `Bacformer_FT_DEFICITS.md` closure claim in §4.3 **was** overstated, exactly as suspected —
>   the two numbers are not commensurable. The banner now says so.
> - Two genuinely wrong numbers were found: the lineage-cluster coverage (91.2% is *label* coverage;
>   the clusters hold **54.9%**) and a "re-checked in full" stamp on `LR_SCORING_AUDIT.md` whose table
>   is stale.
> - Three documents the audit missed entirely: `README.md`, root `CLAUDE.md`'s "Training data
>   architecture" section, and a surviving `dtype="auto"` to-do.
>
> §4.4 (leave the 44 PNGs uncommitted), §4.5 (`docs/api.md` as a pointer) and the two data-safety
> guards were all reviewed and upheld.

**For an agent with no prior context. Your job is to try to break this, not to confirm it.**

I did the cleanup described below, which makes me the worst available judge of whether it is right.
What I want from you is a list of things that are wrong, unsupported, or that read as settled when
they are not. **Report; do not silently fix.** A finding I disagree with is still worth having.

If you find nothing wrong, say so plainly — but check the specific claims in §4 first, because those
are the ones I am least sure of.

---

## 1. What happened, and why

An agent quoted four Bacformer fine-tune AUROCs in a comparison table. All four were wrong — colistin
by 0.10, rifampin by 0.06 — because they were read from a checked-in summary panel that predated the
July 2026 re-runs. Two of the four drug-selection rationales inverted as a result: colistin was
billed as the worst Kp drug (it is not; azithromycin is) and rifampin as underperforming its
catalogue (it does not; it matches it).

The table was fixed. Three read-only audits then found the same failure everywhere:

- **The two documents presenting themselves as authoritative were the least reliable.** Root
  `CLAUDE.md` described a package layout that stopped existing on 2026-07-11. `ToDo.md` said Kp
  models were "not formally evaluated" when 22 checkpoints existed, and had no entry at all for the
  work at HEAD.
- **The wrong numbers had a physical origin still tracked in git** — the panel CSV itself.
- **51 memory files carried 18 mutual contradictions**, including one recommending `dtype="auto"`,
  the documented root cause of an fp32 bug costing ~5 pp AUROC, still filed as durable advice.

**Nothing was wrong with any run.** The record had drifted, and there was no single place to check.

## 2. What changed

| Change | Where |
|---|---|
| **`PROJECT_STATE.md`** — new, the single authority on state: layers, then an artifact dependency table | repo root |
| Cross-project rule: every project has exactly one, and it wins over any memory | `~/.claude/CLAUDE.md` |
| `ToDo.md` **retired** with a banner listing what it was wrong about; parked milestones carried out first | `docs/_retired/`, `docs/_parked/` |
| Root `CLAUDE.md`: layout rewritten, `splits/` added to the stage list, dead paths fixed, the leak that *actually* happened documented, all results removed | `CLAUDE.md` |
| **Catalogue ceiling rebuilt** from source artifacts, with per-row provenance | `visualisations/{kp,tb}/catalogue_ceiling_panel.csv` |
| Stale panels **quarantined** with a README naming which wrong number came from which file | `visualisations/_superseded/` |
| **`tests/docs/`** — five invariants, enforced | `tests/docs/test_docs_stay_true.py` |
| **Memories 51 → 19**, `MEMORY.md` rebuilt as an index (17 KB → ~3 KB); admissibility rule added | memory dir, `CLAUDE.md` §0.6 |
| 7 plot tests fixed that had been red since 2026-08-05 | `tests/engine/plots/` |

The pre-consolidation memories are preserved verbatim at
`docs/_archive/memory_snapshot_2026-08-14/` — nothing was lost, and every deletion is reversible.

## 3. How to verify each claim independently

Do not take my word for any of this.

**The suite.** `uv run pytest tests/` → expect 624 passing. `uv run pytest tests/docs/` is the new
part; read the test file, it explains what defect each check corresponds to.

**The fine-tune numbers** (`PROJECT_STATE.md` §3.1). Every one came from a checkpoint's own
`results.json` on CSD3. Spot-check a few:

```
R=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david
ssh dca36@login.hpc.cam.ac.uk "python -c \"
import json;print(json.load(open('$R/bac_ast_prediction/processed/train_kleb_ast/models/finetune/klebsiella_pneumoniae_colistin_lr_0.00015_finetuned_fold00_seed1/results.json'))['metrics']['auroc'])\""
```

**The ceilings.** Kp: `__ALL_CARD__` row of
`$R/bac_ast_prediction/processed/train_kleb_ast/card_ceiling/<drug>/card_determinant_lr_<drug>_allele.csv`.
TB: `__ALL_WHO_one_hot__` row of
`$R/processed/train_tb_ast/snp_embeddings/tbprofiler_gene_lr/tbprofiler_gene_lr_<drug>.csv`. Compare
against the two `catalogue_ceiling_panel.csv` files.

**The split-table ↔ deployed-holdout equivalence.** This is the claim that makes the whole
unitig-vs-fine-tune comparison legitimate, so it is worth re-checking rather than trusting: for each
drug, the FT `results.json` `n_samples` must equal the row count of the `<drug>_split.csv` holdout.
I found 0 mismatches across all 32. If you find one, the comparison for that drug is not valid.

## 4. Traps, and the things I am least sure of

**Report on these specifically.**

1. **I corrected my own earlier claim about the ceiling estimator marker, and I may have
   over-corrected.** I had recorded that `mut_auroc_sd` as a *column* distinguishes the retired
   k-fold probe from the deployment-holdout scorer. It does not — **both** schemas carry it. The
   marker is its **value** (holdout scorer fits once → exactly `0.0`; k-fold probe → the spread
   across folds). The test I wrote encodes the value rule. **Check that the value rule actually
   holds** rather than being a second plausible-sounding story.

2. **The TB ceiling is called "provisional" on four grounds** — built 2026-06-19 against a
   superseded cohort, read from the deprecated `processed/` tree, missing rifabutin, and a different
   estimator. I am confident about rifabutin and the tree. **I am least confident that the estimator
   difference is the *decisive* one**, versus the cohort staleness.

3. **I put a strong banner on `Bacformer_FT_DEFICITS.md`** saying its headline anomaly has largely
   closed (TB rifampicin 0.9642 against a 0.9666 ceiling, a gap of ~0.002 rather than ~0.06). I
   deliberately did **not** decide what that means for the programme hypothesis, and flagged it for
   David. **Check I have not overstated the closure** — note the ceiling it is being compared
   against is itself the provisional one, so the comparison is doubtful in both directions.

4. **The 44 `card_cause_histogram` PNGs are modified in the working tree and I did not commit them.**
   Their mtime puts them after the split fix but before the presentation fix, and the copies in HEAD
   are older still. I judged the provenance genuinely ambiguous and left them uncommitted rather than
   promoting them to of-record. **This may be the wrong call** — say so if you think it is. I also
   did not `git checkout` them, because that destroys and was not authorised.

5. **`docs/api.md` I reduced to a pointer** rather than regenerating it, because every `automodule`
   target was unimportable. That is a judgement about value, not correctness.

6. **The 18 memories I wrote (a sibling has since added a 19th) are my compression of 51.** Read them against the snapshot in
   `docs/_archive/memory_snapshot_2026-08-14/` and tell me what I dropped that mattered. I am
   most exposed on the **decisions of record** (`PROJECT_STATE.md` §6): nine memories held the sole
   record of a decision, and if I mis-transcribed one it is now the only copy that anyone will read.

**Two data-safety guards must survive any further cleanup.** If they are missing from §6, that is a
serious finding:

- `min_size` stays 100 for the invasion lineage clusters; **do not regenerate the min50 variant.**
- Untracked `*_amr_ladder_table.csv` regenerate freely, but the **curated** publication tables
  (`*_card_ladder_table_family.csv`, `rif_ladder_table.md`, and the `esm_vs_ft_*` chain) need
  David's confirmation before deletion.

## 5. What is genuinely open — do not treat these as settled

- **Does the chromosomal-blindness hypothesis survive?** It was motivated by a TB rifampicin gap
  that has largely closed, while the chromosomal / rRNA / promoter drugs stay weakest in both
  organisms. For David, not for an agent.
- **Head vs mean across all drugs.** One point measured; a full column and scatter were wanted.
- **Invasion mechanism.** The unitig mapping is a measurement. Whether the plasmid share is causal or
  a co-inherited lineage marker is unresolved, and geNomad cannot resolve it (virus and prophage
  only — not plasmids, integrons or IS).
- **Pooled vs `all_samples_2`** for the lab-collection ranking. The rule and its asymmetry are in
  §6: a collapse is clean evidence, **a win is not conclusive** without a near-duplicate audit.

## 6. What is next

The Kp unitig fan-out — the remaining 20 drugs, in batches of ~5, with an LD-deduped control and an
FT re-score on all 20 (without `eval_scores.npz` there is no paired CI, and a gap cannot be
distinguished from a tie). Costs and the checkpoint after batch 1 are in `PROJECT_STATE.md` §3.3.

**TB is not next**, and its first task when it comes is rebuilding the ceiling — all 10 drugs
including rifabutin, through `load_splits` + `score_onehot_frame`. **Do not copy the June CSVs into
the canonical path; that launders them.**

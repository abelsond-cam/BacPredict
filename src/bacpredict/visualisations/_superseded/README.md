# Superseded artifacts — kept so wrong numbers stay traceable

**Nothing in this directory may be quoted or consumed.** These files are here because numbers out of
them were repeated in documents, memories and a comparison table long after the runs that produced
them had been redone. Deleting them would hide where those numbers came from; keeping them anywhere
else would let a `grep` find them again.

The current catalogue ceilings are `../kp/catalogue_ceiling_panel.csv` and
`../tb/catalogue_ceiling_panel.csv`. Fine-tune numbers come from each checkpoint's `results.json`, or
from an `engine.finetune.evaluate` re-score — **never** from a summary panel. See
[`../PROVENANCE.md`](../PROVENANCE.md) and `PROJECT_STATE.md`.

## What each file is

| File | Was | Why it is here |
|---|---|---|
| `kp_amr_summary_panel.csv` | 7-drug Kp panel: ceiling / ft / concat | **The physical origin of the wrong fine-tune numbers.** Its `ft_auroc` column reads colistin **0.8072** and azithromycin **0.8268**; the current values are **0.9094** and **0.7993**. Its ceiling column is stale too (cefotaxime 0.9802 against CARD's 0.9841) |
| `tb_amr_summary_panel.csv` | 5-drug TB panel, same schema | `ft_auroc` rifampicin **0.9046**, against a current **0.9642** — the source of the "rifampin underperforms its catalogue" reading, which is false |
| `kp_card_vs_best_bacformer.csv` | CARD ceiling vs best Bacformer, with a `gap` column | Same vintage. Its azithromycin row (ceiling 0.5552, bacformer 0.8564) predates both the re-runs and the read-out leak fix |
| `kp_ceiling_concat_panel_2026-07-21.csv` | ceiling + concat only, no `ft_` column | Was untracked, misfiled under `tb/` as `kp_amr_summary_panel.csv`. Its `concat_*` values predate the leak fix |
| `tb_ceiling_concat_panel_2026-07-21.csv` | ceiling + concat only | Was untracked as `tb/tb_amr_summary_panel.csv` and cited by `ast_gwas/CLAUDE.md` as the TB ceiling source. Ceiling values are the provisional June k-fold probe; `concat_*` predates the leak fix |
| `*.png` | the rendered forms of the above | Same numbers, harder to grep |

## The two defects behind all of them

**1. The re-runs.** Every fine-tune was redone between 2026-07-15 and 07-21 on
`macwiatrak/bacformer-large-masked-complete-genomes`. These panels were written before that and were
never regenerated, so every `ft_*` value in this directory is from a superseded model.

**2. The read-out leak.** Until `9060617` (2026-07-23), downstream scoring resolved "evaluate" from
the CSV's single-split `train_val_eval` column while the models are k-fold trained. For Kp
azithromycin, **81%** of the genomes scored as "held out" were in the model's own train/val split.
Any `concat_*` column here inherits that. The structural fix is `eb39ce5` (materialised
`<drug>_split.csv`) and `25e48cc` (the ladder reads it).

The fine-tunes themselves were **never** affected — the leak was in read-out scoring only, which is
why the July checkpoints remain authoritative.

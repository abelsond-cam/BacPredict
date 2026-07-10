# `_archive/tb_snp_diagnostic/` — the concluded TB SNP read-out diagnostic

This directory is a **frozen record**, not live code. It is excluded from the wheel, from `ruff`,
and from `pytest`. The modules' `from pangena_predict.…` / `from tl.…` imports reflect the
pre-consolidation package layout and are **not maintained** after the engine move — treat these as a
snapshot of what was run, not runnable code.

## What this strand established

*M. tuberculosis* rifampicin AST underperformed (deployed Bacformer eval AUROC ~0.905, WHO-catalogue
ceiling ≥0.96) while *Klebsiella* AST was strong. This diagnostic localised **where** the single
rpoB/RRDR residue signal is lost. Headline: the signal survives in Bacformer's contextualised rpoB
token (AUROC 0.953) and is destroyed by the protein→genome **mean-pool** (0.788); fine-tuning the
mean partly recovers it (0.905), a naive learned attention pool does worse (0.868). The failure is
in the read-out pooling, not the embedding.

The full write-up is the source of truth — this code only reproduces its figures:
- `docs/findings/ft_deficits.md` §1–3 (the localization ladder + attention-head negative result)
- `docs/_archive/PROGRESS_REPORT.md` (the original progress report)

## Contents → what each reported

| module | role |
|---|---|
| `rpob_genotype.py` | RRDR allele from the assembled CDS + rpoB-copy QC + H37Rv reference (`reference_gene/rpoB_H37Rv.faa`) |
| `snp_vs_esm_ladder.py` | the rpoB/rifampicin localization-ladder driver (Steps 1/2/3a/2b/2c) — extracted from the former `snp_vs_esm_prediction.py`; its generic probe primitives stayed live in the engine |
| `llr_distribution_probe.py` | per-residue + per-protein surprisal (the masked-vs-unmasked proxy proof) |
| `unmasked_surprisal_scan.py` | genome-wide unmasked-surprisal scan (manifest + array) |
| `surprisal_analysis.py` | read-only figures over the surprisal sidecars |
| `build_surprisal_store.py` | re-key raw surprisal dumps → per-sample panel npz (the label-blind panel; the live per-gene-LR store is its supervised replacement) |
| `geometry_probe.py` | per-residue WT→mutant ESM-C geometry (d_site/d_window/d_pool) |
| `intrinsic_attention_probe.py` | Bacformer's internal self-attention on rpoB |
| `head_pool_attention_probe.py` | the prediction head's learned pool weights on a trained checkpoint |
| `eval_attn_pool_on_full_split.py` | scored a trained attention-pool checkpoint on the full AST holdout (the 0.977 manifest confound test) |

`scripts/` holds the sbatch wrappers; `tests/` holds the three unit tests for these modules (not
collected by `pytest`, whose `testpaths` is `tests/`).

## Reviving anything

If the read-out work restarts, a module here is one `git mv` back into the engine (and an import
refresh) away from live. The generic gene-LR primitives it builds on
(`resolve_clean_splits`, `load_pooled_gene_vectors`, `load_bacformer_vectors`, `fit_score_step`) are
still maintained in the engine's `gene_lr` stage.

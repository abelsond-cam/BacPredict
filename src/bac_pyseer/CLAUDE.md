# bac_pyseer — Pyseer / GWAS

New package under `src/` compartmentalising **pyseer + GWAS** analyses, with **one
subfolder per task**. See the root [CLAUDE.md](../../CLAUDE.md) for §0 global conventions;
cross-task status is in [ToDo.md](../../ToDo.md) under "Pyseer GWAS".

| Task folder | Scope |
|---|---|
| [kleb_iso_source](kleb_iso_source/CLAUDE.md) | *Klebsiella* isolation source (blood/faeces, faeces/respiratory) — variants, unitigs, MGE placement |
| [ast_gwas](ast_gwas/CLAUDE.md) | **AMR**, both organisms, all antibiotics — unitig GWAS → LR baseline vs Bacformer fine-tuning |

`ast_gwas` is deliberately organism-agnostic (`--organism {kp,tb}`) rather than a folder per
organism: one pipeline serves 2 organisms × 32 drugs, and it reuses `kleb_iso_source`'s GGCAT build,
sharded LMM and postprocessing via env overrides rather than forking them.

This work runs on **variant calls + unitigs + Panaroo GPA** — it is *not* Bacformer
fine-tuning but will be used to compare the results from Bacformer and potentially to
create extra features Bacformer can use.

Per-task plans and running notes live in each subfolder's own `CLAUDE.md`.

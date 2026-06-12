# bac_pyseer — Pyseer / GWAS

New package under `src/` compartmentalising **pyseer + GWAS** analyses, with **one
subfolder per task** (`kleb_iso_source` first; `tb_ast` and others to follow). See
the root [CLAUDE.md](../../CLAUDE.md) for §0 global conventions; cross-task status is
in [ToDo.md](../../ToDo.md) under "Pyseer GWAS".

This work runs on **variant calls + unitigs + Panaroo GPA** — it is *not* Bacformer
fine-tuning. It was moved here from the BacHGT tracker so all pyseer work for every
task lives in one place.

Per-task plans and running notes live in each subfolder's own `CLAUDE.md`.

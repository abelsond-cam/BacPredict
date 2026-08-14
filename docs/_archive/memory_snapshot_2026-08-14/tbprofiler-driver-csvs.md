---
name: tbprofiler-driver-csvs
description: Location + schema of the per-drug tbprofiler driver CSVs that seed the TB all-drivers panel (Stage 2 step 2)
metadata:
  node_type: memory
  type: reference
  originSessionId: f3c1d41f-8ba4-45e4-98f4-8a0db0e1b64a
---

Per-drug TB **driver lists** (each mutation/gene that drives resistance, with its one-hot ceiling) live at
`src/pangena_predict/docs/visualisations/tb_<drug>/tbprofiler_gene_lr_<drug>.csv`. These seed the
"all driving mutations" panel (see [[pangena-predict-stage2-state]] step 2): the final table adds
**baclm / ESM / Bacformer** AUROC+AUPRC columns beside the CSV's existing one-hot, per driver, + a grouped
column chart (AUROC & AUPRC) per drug.

**Drug folders (10):** ethambutol, ethionamide, isoniazid, kanamycin, levofloxacin, moxifloxacin,
pyrazinamide, rifabutin, rifampicin, streptomycin. **Name mapping to the AST column** in
`binary_ast_with_split.csv`: `rifampicin`→**`rifampin`** (US); most others match; verify each drug has an
AST column before running (AST has amikacin/ethambutol/ethionamide/isoniazid/kanamycin/moxifloxacin/
pyrazinamide/rifampin/streptomycin; levofloxacin & rifabutin may be absent → skip or map to a fluoroquinolone).

**CSV columns:** `gene_name`, `region` (`coding`/`non-coding`/`all`), `site` (e.g. `inhA (promoter)`,
`ethA`), `mut_auroc`, `mut_auroc_sd`, `mut_auprc`, `mut_auprc_sd` (= the **one-hot WHO** ceiling for that
driver), `n_variants`, `n_genomes_with_variant`, `embeddable` (bool — coding, ESM/Bacformer-able),
`is_rrna` (bool), `is_noncoding` (bool — promoter/IGR). Row `__ALL_WHO_one_hot__` (region `all`) = the full
WHO one-hot model AUROC/AUPRC (e.g. ethionamide 0.871 / 0.719).

**Routing per driver:** coding (`embeddable`) → baclm coding vec (`coding_amr_lr`) + ESM + Bacformer
(`bacformer_genome_vectors`, GPU); non-coding promoter (`is_noncoding` & not `is_rrna`) → baclm IGR
(`igr_amr_lr`, anchor on flank gene); `is_rrna` → baclm named-RNA vector, only AFTER the 2d re-embed.
ESM/Bacformer are protein models → N/A for non-coding rows.

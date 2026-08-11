"""Unitig GWAS → logistic-regression baseline for AMR, scored against Bacformer fine-tuning.

One organism-agnostic pipeline (``--organism {kp,tb}``) serving both *K. pneumoniae* and
*M. tuberculosis* across every antibiotic in their AST panels. It reuses the proven
:mod:`bac_pyseer.kleb_iso_source` GWAS machinery — GGCAT unitig build, sharded ``pyseer --lmm``,
Bonferroni-on-patterns, the cached hit-submatrix extraction — and pins the read-out to the *same*
``<drug>_split.csv`` tables the fine-tuned checkpoints were evaluated on, so the two arms are
directly comparable.

The design point that makes the comparison honest: the GWAS phenotype file carries **train +
validate only**, so unitig selection never sees a holdout label. The holdout genomes appear only
in the design matrix (feature *values*, built unsupervised over the whole cohort) and in the final
scoring pass. See :mod:`bac_pyseer.ast_gwas.build_ast_phenotype`.

Stage order (unitigs and kinship are built **once per organism** and reused for every drug)::

    resolve_ast_assemblies  ->  Sample<TAB>assembly path
    run_ggcat_unitigs.sh    ->  unitigs.pyseer.gz          (once per organism)
    mash_kinship            ->  similarity.tsv             (once per organism)
    lineage_from_distances  ->  lineage_clusters.tsv       (once per organism)
    build_ast_phenotype     ->  phenotype.tsv              (per drug, train+validate)
    pyseer --lmm (sharded)  ->  <drug>.assoc + patterns.txt
    pyseer_postprocess      ->  <drug>_hits_annotated.tsv
    unitig_design_matrix    ->  presence.npz               (per drug, ALL samples)
    unitig_lr               ->  results.json               (per drug, schema v1.2)
"""

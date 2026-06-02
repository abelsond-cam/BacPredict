# API

The `bacpredict` distribution is split into five top-level packages: one shared toolbox umbrella (`tl`, containing `tl.embed`, `tl.genome_download`, `tl.train`) plus four task-scoped packages (`tb_ast`, `kleb_ast`, `kleb_iso_source`, `predict_hgt`). Task packages depend on `tl.*`; `tl.*` does not depend on task packages.

Tasks 4 (mixed-assembly detection) and 5 (DefensePredictor on short reads) are deferred — see `ToDo.md`; their task packages will be added when work resumes.

## Embed (`tl.embed`)

```{eval-rst}
.. module:: tl.embed
.. currentmodule:: tl.embed

.. autosummary::
    :toctree: generated

    extract_proteins_from_gff_fna
    preprocess_assemblies_to_protein_sequences
    generate_embeddings
    genome_assemblies_from_bacformer_embeddings
```

## Genome download (`tl.genome_download`)

```{eval-rst}
.. module:: tl.genome_download
.. currentmodule:: tl.genome_download

.. autosummary::
    :toctree: generated

    download_bakrep_gbff_files
```

## Train (`tl.train`) — shared k-fold + lazy-dataset helpers

```{eval-rst}
.. module:: tl.train
.. currentmodule:: tl.train

.. autosummary::
    :toctree: generated

    split_utils
    datasets
```

## Task: TB AST (`tb_ast`)

```{eval-rst}
.. module:: tb_ast
.. currentmodule:: tb_ast

.. autosummary::
    :toctree: generated

    build_tb_input_csv
```

## Task: Klebsiella AST (`kleb_ast`)

```{eval-rst}
.. module:: kleb_ast
.. currentmodule:: kleb_ast

.. autosummary::
    :toctree: generated

    train_amr
    prepare_esmc_embeddings_and_labels_to_finetune_amr
    preprocess_ebi_amr_records
    convert_ast_data
    add_paths_gff_fna_to_metadata
    add_bakta_gbff_downloaded_flag
    find_missing_embeddings
    filter_esmc_embeddings_by_klebsiella
    extract_anndata_with_bacformer_protein_embeddings
```

## Task: Klebsiella isolation source (`kleb_iso_source`)

```{eval-rst}
.. module:: kleb_iso_source
.. currentmodule:: kleb_iso_source

.. autosummary::
    :toctree: generated

    train_isolation_source
    prepare_esmc_embeddings_and_labels_to_finetune_isolation_source
    stratified_isolation_source_sampling
    isolation_source_cli_parsing
```

## Task: `predictHGT` (`predict_hgt`)

Empty Python package — diagnostic, no entrypoints yet. See [src/predict_hgt/CLAUDE.md](../src/predict_hgt/CLAUDE.md) for the plan.

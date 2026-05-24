# API

The `bacpredict` package is organised into four task-scoped subpackages.

## Genome download (`bacpredict.genome_download`)

```{eval-rst}
.. module:: bacpredict.genome_download
.. currentmodule:: bacpredict

.. autosummary::
    :toctree: generated

    genome_download.add_bakta_gbff_downloaded_flag
    genome_download.add_paths_gff_fna_to_metadata
    genome_download.build_tb_input_csv
    genome_download.download_bakrep_gbff_files
```

## Embed (`bacpredict.embed`)

```{eval-rst}
.. module:: bacpredict.embed
.. currentmodule:: bacpredict

.. autosummary::
    :toctree: generated

    embed.extract_proteins_from_gff_fna
    embed.preprocess_assemblies_to_protein_sequences
    embed.generate_embeddings
    embed.find_missing_embeddings
    embed.genome_assemblies_from_bacformer_embeddings
    embed.filter_esmc_embeddings_by_klebsiella
    embed.extract_anndata_with_bacformer_protein_embeddings
```

## Sample and label (`bacpredict.sample_and_label`)

```{eval-rst}
.. module:: bacpredict.sample_and_label
.. currentmodule:: bacpredict

.. autosummary::
    :toctree: generated

    sample_and_label.preprocess_ebi_amr_records
    sample_and_label.convert_ast_data
    sample_and_label.stratified_isolation_source_sampling
    sample_and_label.isolation_source_cli_parsing
    sample_and_label.prepare_esmc_embeddings_and_labels_to_finetune_amr
    sample_and_label.prepare_esmc_embeddings_and_labels_to_finetune_isolation_source
```

## Train (`bacpredict.train`)

```{eval-rst}
.. module:: bacpredict.train
.. currentmodule:: bacpredict

.. autosummary::
    :toctree: generated

    train.split_utils
    train.datasets
    train.train_amr
    train.train_isolation_source
```

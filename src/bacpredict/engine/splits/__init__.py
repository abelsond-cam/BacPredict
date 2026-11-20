"""The single source of truth for sampling + labels.

One materialized table per drug — ``<drug>_split.csv`` — is written once by a builder and read by the ONE reader
:func:`bacpredict.engine.splits.load_splits.load_splits` (plus ``load_split_ids`` beside it, the same read for
one named split). Every computation (the fine-tuned genome-mean, every
ESM/baclm per-segment LR, the catalogue one-hot, and the trainer) reads this table: fit on ``train``, select
by train-OOF, evaluate on ``holdout``. Because the trainer and the read-out share the same table, a
fine-tuned feature is provably scored on the model's own holdout — the train/test leak is impossible by
construction, not guarded at runtime.

Two builders, one reader:

- :mod:`~bacpredict.engine.splits.generate_kfold_splits` — TB / Kp: ``Sample, ast_label, split`` with
  ``split ∈ {train, validate, holdout}``, the deployed models' random k-fold partition, drawn per drug.
- :mod:`~bacpredict.engine.splits.build_amr_holdout_splits` — the multi-species AMR arms (PsA, Enterobacterales):
  ``Sample, species, st_key, ast_label, split``, adding ``holdout_st`` (an exact-ST zero-shot holdout). Assignment
  is per genome and per species with a per-species RNG, so a species' rows are the same in every arm. Mechanics
  are in that module's docstring.

``load_splits`` returns ``train``/``validate``/``holdout`` from either kind; ``holdout_st`` is read only by
``load_split_ids``.
"""

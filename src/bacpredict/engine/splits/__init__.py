"""The single source of truth for sampling + labels.

One materialized table per drug — ``<drug>_split.csv`` (``Sample, ast_label, split∈{train,validate,holdout}``)
— is written once by :mod:`bacpredict.engine.splits.generate_kfold_splits` and read by the ONE reader
:func:`bacpredict.engine.splits.load_splits.load_splits`. Every computation (the fine-tuned genome-mean, every
ESM/baclm per-segment LR, the catalogue one-hot, and the trainer) reads this table: fit on ``train``, select
by train-OOF, evaluate on ``holdout``. Because the trainer and the read-out share the same table, a
fine-tuned feature is provably scored on the model's own holdout — the train/test leak is impossible by
construction, not guarded at runtime.
"""

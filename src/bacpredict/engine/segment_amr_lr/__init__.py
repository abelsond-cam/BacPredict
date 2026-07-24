"""Segment-based AMR logistic regression — the LR analysis layer.

A *segment* is a genomic region whose per-genome pooled embedding predicts resistance: a coding
**protein**, an intergenic region (**igr**), an rRNA/named body (**unit**), or a promoter/**upstream**
region. ``per_segment_lr`` screens every segment of a type through the same fit/score primitive
(:mod:`bacpredict.engine.segment_amr_lr.fit_lr`), and ``concat/`` builds the concatenated read-out on top.
Everything fits on the deployed model's ``train`` split, selects by train-OOF AUROC, and evaluates on the
``holdout`` — via the one materialized split table (:mod:`bacpredict.engine.splits`).
"""

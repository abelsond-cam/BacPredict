"""How much of a holdout's score is available from clonal lookup alone.

A holdout drawn at random from a clonal population is not a test of generalisation if the model can
find a near-identical training genome and copy its label. This module measures that directly: for
each held-out genome take its highest-ANI **labelled** training genome and predict that genome's
label. No model is involved. The resulting AUROC is the share of the holdout's headline number that
needs no biology at all — a floor under "how well would pure memorisation do here".

Read it against the fine-tune's own ``results.json`` for the same drug and split. The metrics come
from :func:`bacpredict.engine.finetune.metrics.compute_full_metrics`, the same function that wrote
those numbers, so the two are computed identically and are directly comparable.

⚠ **It is a LOWER bound on memorisation, not a ceiling.** Whole-genome ANI is dominated by the core
genome, while most acquired AMR determinants are accessory. A lookup keyed on accessory gene content,
or a k-NN in the embedding space itself, would score higher. A low number here does not license the
claim that a model generalises; a high one does convict the holdout.

⚠ **ANI saturates.** skani prints two decimals and scores a genome against itself at 99.99, so
``>= 99.99`` is the resolution limit, not a measurement: it cannot tell a clone from the same genome.

Ties are not broken arbitrarily. Every labelled training genome at the maximum ANI votes, and the
score is the fraction of them that are resistant — so the output is a continuous score in [0, 1] and
its AUROC is well defined, rather than an arbitrary pick among genomes the metric cannot separate.

Input is a ``skani dist`` TSV of held-out genomes against training genomes (``Ref_file``,
``Query_file``, ``ANI``, …), where the queries are the held-out genomes. Only the top ``k`` neighbours
of each query are kept, so memory is O(k × held-out) rather than O(pairs): a within-cohort triangle
for a cohort of ~10,000 genomes runs to tens of millions of rows.

Example
-------
::

    python -m bacpredict.engine.splits.holdout_lookup_baseline \\
        --ani-tsv heldout_vs_train.tsv --split-dir $AMR/processed/paeruginosa/splits \\
        --out psa_nn_lookup_baseline.csv --per-genome-out psa_heldout_max_ani_to_train.csv
"""

from __future__ import annotations

import argparse
import csv
import heapq
import logging
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd

from bacpredict.engine.finetune.metrics import compute_full_metrics
from bacpredict.engine.splits.load_splits import load_split_ids, load_splits

logger = logging.getLogger(__name__)

#: Splits scored against the training set. ``validate`` is excluded: it is part of model selection.
HELDOUT_SPLITS = ("holdout", "holdout_st")

#: Neighbours kept per held-out genome. A drug labels only part of the training set, so the nearest
#: *labelled* neighbour may not be the nearest neighbour; 100 makes running out vanishingly unlikely
#: while keeping the table small. ``n_without_labelled_neighbour`` reports any genome that did.
DEFAULT_TOP_K = 100


def sample_of(path: str) -> str:
    """``/…/SAMN03105472.fna.gz`` → ``SAMN03105472`` (the store is keyed by BioSample)."""
    return os.path.basename(path).split(".")[0]


def nearest_neighbours(ani_tsv: str | Path, top_k: int = DEFAULT_TOP_K) -> dict[str, list[tuple[float, str]]]:
    """Stream a ``skani dist`` TSV → each query's ``top_k`` references, highest ANI first.

    Parameters
    ----------
    ani_tsv : str or Path
        ``skani dist`` output: a header, then ``Ref_file``, ``Query_file``, ``ANI`` in the first three
        columns. References are the training genomes, queries the held-out ones.
    top_k : int
        Neighbours kept per query (see :data:`DEFAULT_TOP_K`).

    Returns
    -------
    dict[str, list[tuple[float, str]]]
        ``query Sample -> [(ani, ref Sample), …]`` sorted by descending ANI.
    """
    heaps: dict[str, list[tuple[float, str]]] = defaultdict(list)
    with open(ani_tsv) as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader, None)  # header
        for row in reader:
            if len(row) < 3:
                continue
            try:
                ani = float(row[2])
            except ValueError:  # a malformed line is skipped, not fatal: skani can emit a warning row
                continue
            heap = heaps[sample_of(row[1])]
            entry = (ani, sample_of(row[0]))
            if len(heap) < top_k:
                heapq.heappush(heap, entry)
            elif ani > heap[0][0]:
                heapq.heapreplace(heap, entry)
    return {query: sorted(heap, reverse=True) for query, heap in heaps.items()}


def per_genome_similarity(
    neighbours: dict[str, list[tuple[float, str]]], heldout_split: dict[str, str]
) -> pd.DataFrame:
    """Each held-out genome's maximum ANI to any training genome, whether that genome is labelled or not.

    Parameters
    ----------
    neighbours : dict
        :func:`nearest_neighbours` output.
    heldout_split : dict[str, str]
        ``Sample -> split`` over the held-out genomes to report.

    Returns
    -------
    pandas.DataFrame
        ``Sample, split, max_ani_to_train, n_train_at_max_ani, nearest_train_sample``. A genome skani
        reported no match for carries NaN and 0 — it is distant, not missing.
    """
    rows = []
    for sample in sorted(heldout_split):
        hits = neighbours.get(sample, [])
        if not hits:
            rows.append((sample, heldout_split[sample], float("nan"), 0, ""))
            continue
        top = hits[0][0]
        tied = [ref for ani, ref in hits if ani == top]
        rows.append((sample, heldout_split[sample], top, len(tied), tied[0]))
    return pd.DataFrame(
        rows, columns=["Sample", "split", "max_ani_to_train", "n_train_at_max_ani", "nearest_train_sample"]
    )


def lookup_baseline(
    split_table: str | Path, neighbours: dict[str, list[tuple[float, str]]], *, species: str | None = None
) -> list[dict[str, object]]:
    """Score the nearest-labelled-neighbour lookup on one drug's holdouts.

    Splits are resolved through :mod:`bacpredict.engine.splits.load_splits` — never by reading the
    table directly — so the held-out sets are exactly the ones the fine-tune was scored on.

    Parameters
    ----------
    split_table : str or Path
        The drug's ``<drug>_split.csv``.
    neighbours : dict
        :func:`nearest_neighbours` output.
    species : str, optional
        Restrict to one species' rows, as in :func:`~bacpredict.engine.splits.load_splits.load_splits`.

    Returns
    -------
    list[dict]
        One row per split in :data:`HELDOUT_SPLITS` that has at least one scoreable genome: the
        drug, the split, ``n``, the §0.4 metrics of the lookup, and the ANI summary it leaned on.
    """
    label_map, train_ids, _validate, holdout_ids = load_splits(split_table, species=species)
    train_labels = {s: label_map[s] for s in train_ids}
    drug = Path(split_table).name.replace("_split.csv", "")
    ids_by_split = {"holdout": holdout_ids, "holdout_st": load_split_ids(split_table, "holdout_st", species=species)}

    rows: list[dict[str, object]] = []
    for split in HELDOUT_SPLITS:
        y_true, y_prob, top_anis, unreachable = [], [], [], 0
        for sample in ids_by_split[split]:
            labelled = [(ani, ref) for ani, ref in neighbours.get(sample, []) if ref in train_labels]
            if not labelled:
                unreachable += 1
                continue
            top = labelled[0][0]
            votes = [train_labels[ref] for ani, ref in labelled if ani == top]
            y_true.append(label_map[sample])
            y_prob.append(sum(votes) / len(votes))
            top_anis.append(top)
        if not y_true:
            continue
        metrics = compute_full_metrics(y_true, y_prob)
        anis = pd.Series(top_anis)
        rows.append({
            "drug": drug,
            "split": split,
            "n": len(y_true),
            "nn_auroc": metrics["auroc"],
            "nn_auprc": metrics["auprc"],
            "nn_balanced_accuracy": metrics["balanced_accuracy"],
            "prevalence": metrics["prevalence"],
            "median_nearest_ani": anis.median(),
            "pct_nearest_ge_99_99": 100.0 * float((anis >= 99.99).mean()),
            "n_without_labelled_neighbour": unreachable,
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    """CLI: ``--ani-tsv`` + ``--split-dir``/``--split-table`` → the per-drug baseline CSV."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ani-tsv", required=True, help="skani dist TSV: held-out genomes (query) vs train (ref)")
    parser.add_argument("--split-dir", help="directory of <drug>_split.csv; all of them are scored")
    parser.add_argument("--split-table", action="append", default=[], help="one split table (repeatable)")
    parser.add_argument("--species", help="restrict to one species' rows of a multi-species table")
    parser.add_argument("--out", required=True, help="per-drug baseline CSV")
    parser.add_argument("--per-genome-out", help="optional: each held-out genome's max ANI to train")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help=f"neighbours kept (default {DEFAULT_TOP_K})")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tables = [Path(t) for t in args.split_table]
    if args.split_dir:
        tables += sorted(Path(args.split_dir).glob("*_split.csv"))
    if not tables:
        parser.error("give --split-dir or at least one --split-table")

    logger.info("reading %s (top %d neighbours per held-out genome)", args.ani_tsv, args.top_k)
    neighbours = nearest_neighbours(args.ani_tsv, args.top_k)
    logger.info("%d held-out genomes have at least one reported neighbour", len(neighbours))

    rows: list[dict[str, object]] = []
    for table in tables:
        rows += lookup_baseline(table, neighbours, species=args.species)
    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    logger.info("wrote %s (%d rows)", args.out, len(out))

    if args.per_genome_out:
        heldout_split: dict[str, str] = {}
        for table in tables:  # a genome labelled for several drugs sits in the same split in each
            for split in HELDOUT_SPLITS:
                for sample in load_split_ids(table, split, species=args.species):
                    heldout_split[sample] = split
        per_genome = per_genome_similarity(neighbours, heldout_split)
        Path(args.per_genome_out).parent.mkdir(parents=True, exist_ok=True)
        per_genome.to_csv(args.per_genome_out, index=False)
        logger.info("wrote %s (%d genomes)", args.per_genome_out, len(per_genome))

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

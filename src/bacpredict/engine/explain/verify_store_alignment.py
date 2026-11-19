r"""CPU — check that every embedding row is the protein the annotation parquet says it is.

The flat index is a **positional** join: embedding row *i* is taken to be protein *i* of the
genome's `*_protein_sequences.parquet`, walked in contig order. Everything gene-level rests on that
— attribution, per-gene LR, the CARD sidecar, the nuna assignments. If the two disagree anywhere,
every protein **after** the first disagreement is attributed to the wrong gene, silently.

Why a total-count check is not enough
-------------------------------------
``verify_protein_count`` compares the two totals, and that catches the gross case. It cannot catch a
**compensating** change: a re-annotation that adds one CDS on one contig and drops one on another
leaves the total identical while shifting every index between them. That is not hypothetical here —
the Kp parquets were regenerated on 2026-06-20 against embeddings written 2026-02-07, and of the
genomes that *do* fail the total check, 53 of 57 differ by exactly +3, which is one extra CDS on each
of three different contigs, scattered through the genome rather than appended at the end. A genome
with one addition and one deletion would have passed.

This module compares the **per-contig** protein counts, which is the strongest invariant the store
actually carries: the `.pt` holds only ``protein_embeddings``, ``contig_ids`` and ``attention_mask``
— no protein ids, no sequences, no lengths — so there is nothing finer to align against.

Reported per genome
-------------------
``status``, both totals, both contig counts, the first contig where they diverge, the flat index at
which the shift begins, and ``n_proteins_after_break`` — the number of proteins whose gene identity
is wrong, which is the number that actually matters and is typically most of the genome.

Statuses: ``aligned`` · ``aligned_contig_ids_clamped`` (>1,000 contigs; flat index still
valid) · ``contig_mismatch`` (totals agree, contigs do not — the silent case) ·
``count_mismatch`` · ``truncated`` (embedding capped at the embedding-time limit; the rows that
exist still align) · ``missing_pt`` / ``missing_parquet``.

Examples
--------
Sweep the Kp AST cohort::

    python -m bacpredict.engine.explain.verify_store_alignment \\
        --organism kp --samples-from "$B/train_kleb_ast/binary_ast_with_split.csv" \\
        --out "$B/train_kleb_ast/store_alignment_kp.csv"
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from bacpredict.engine.config import store_paths

logger = logging.getLogger(__name__)

TRUNCATION_LIMIT = 6000
# The embedding pipeline clamps contig_ids at this many distinct ids: every protein on contig 1000
# and beyond is labelled 999. Measured on SAMEA7546160, where id 999 holds 93 proteins against a
# median contig of 3. The protein TOTAL is preserved, so the flat index is still valid and only the
# contig label is wrong past the clamp -- a different finding from a misalignment, and one that
# would otherwise be reported as the alarming "totals agree but contigs do not" case.
CONTIG_ID_LIMIT = 1000


def contig_sizes_from_store(esm_path: Path) -> np.ndarray | None:
    """Per-contig protein counts from a genome's ``.pt``, without reading its embeddings.

    Uses ``mmap=True`` so only ``contig_ids`` is faulted in — measured 0.01 s against 0.43 s for a
    full load, a 50x saving that is what makes a whole-cohort sweep affordable.

    Parameters
    ----------
    esm_path : pathlib.Path
        The genome's ``{Sample}_esm_embeddings.pt``.

    Returns
    -------
    numpy.ndarray or None
        Protein count per contig, in contig order, or ``None`` if the store carries no contig ids.
    """
    try:
        store = torch.load(esm_path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:  # noqa: BLE001 - a store we cannot open is reported, never fatal
        store = torch.load(esm_path, map_location="cpu", weights_only=False)
    ids = store.get("contig_ids", store.get("contig_idx"))
    if ids is None:
        return None
    ids = ids.squeeze(0) if ids.dim() > 1 else ids
    values = ids.to(torch.int64).numpy()
    # contig_ids is run-length structured (all of contig 0, then contig 1, ...). Count per id in
    # ascending id order, which is the order flatten_proteins walks.
    unique, counts = np.unique(values, return_counts=True)
    return counts[np.argsort(unique)]


def contig_sizes_from_parquet(parquet_path: Path) -> np.ndarray:
    """Per-contig protein counts from the annotation parquet, matching ``flatten_proteins``' walk.

    Parameters
    ----------
    parquet_path : pathlib.Path
        The genome's ``{Sample}_protein_sequences.parquet``.

    Returns
    -------
    numpy.ndarray
        Protein count per contig, in the stored contig order.
    """
    nested = pq.read_table(parquet_path, columns=["protein_sequence"]).column("protein_sequence")[0]
    return np.array([len(contig) for contig in nested.as_py()], dtype=np.int64)


def compare_one(sample: str, esm_path: Path, parquet_path: Path) -> dict:
    """Compare one genome's embedding store against its annotation parquet.

    Parameters
    ----------
    sample : str
        Sample ID.
    esm_path : pathlib.Path
        The genome's embedding store.
    parquet_path : pathlib.Path
        The genome's annotation parquet.

    Returns
    -------
    dict
        One row: status, totals, contig counts, and where the first divergence begins.
    """
    row: dict = {"Sample": sample, "status": "aligned", "n_pt": -1, "n_parquet": -1,
                 "n_contigs_pt": -1, "n_contigs_parquet": -1,
                 "first_divergent_contig": -1, "first_divergent_flat_index": -1,
                 "n_proteins_after_break": 0}
    if not esm_path.exists():
        row["status"] = "missing_pt"
        return row
    if not parquet_path.exists():
        row["status"] = "missing_parquet"
        return row

    pt_sizes = contig_sizes_from_store(esm_path)
    pq_sizes = contig_sizes_from_parquet(parquet_path)
    if pt_sizes is None:
        row["status"] = "no_contig_ids"
        return row

    row.update(n_pt=int(pt_sizes.sum()), n_parquet=int(pq_sizes.sum()),
               n_contigs_pt=len(pt_sizes), n_contigs_parquet=len(pq_sizes))

    if len(pt_sizes) != len(pq_sizes):
        clamped = (len(pt_sizes) == CONTIG_ID_LIMIT
                   and len(pq_sizes) > CONTIG_ID_LIMIT
                   and row["n_pt"] == row["n_parquet"]
                   and np.array_equal(pt_sizes[: CONTIG_ID_LIMIT - 1], pq_sizes[: CONTIG_ID_LIMIT - 1]))
        if clamped:
            # Every protein is present and in flat order; only contig labels past 999 are wrong.
            row["status"] = "aligned_contig_ids_clamped"
            return row
        row["status"] = "contig_count_mismatch"
        row["n_proteins_after_break"] = row["n_pt"]
        return row

    divergent = np.flatnonzero(pt_sizes != pq_sizes)
    if divergent.size == 0:
        # Totals and every contig agree. This is the strongest statement the store supports.
        return row

    first = int(divergent[0])
    row["first_divergent_contig"] = first
    row["first_divergent_flat_index"] = int(pt_sizes[:first].sum())
    row["n_proteins_after_break"] = int(row["n_pt"] - pt_sizes[:first].sum())
    # Totals agreeing while contigs do not is the compensating case a total-count check misses.
    if row["n_pt"] == row["n_parquet"]:
        row["status"] = "contig_mismatch"
    elif row["n_pt"] == TRUNCATION_LIMIT and row["n_parquet"] > TRUNCATION_LIMIT:
        row["status"] = "truncated"
    else:
        row["status"] = "count_mismatch"
    return row


def main() -> None:
    """CLI entry point — sweep a cohort and write one alignment row per genome."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--organism", required=True, choices=("tb", "kp", "pa"), help="Organism key.")
    parser.add_argument("--samples-from", type=Path, required=True,
                        help="Any CSV with a Sample column — a cohort table, the AST sheet, or embedding_input.csv.")
    parser.add_argument("--out", type=Path, required=True, help="Where the per-genome CSV is written.")
    parser.add_argument("--esm-dir", type=Path, default=None, help="Override the ESM store directory.")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N genomes.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

    paths = store_paths(args.organism)
    esm_dir = Path(args.esm_dir or paths.esm_dir)
    samples = pd.read_csv(args.samples_from)["Sample"].astype(str).drop_duplicates().tolist()
    if args.limit:
        samples = samples[: args.limit]
    logger.info("checking %d genomes: %s against %s", len(samples), esm_dir, paths.parquet_dir)

    rows = []
    for index, sample in enumerate(samples):
        rows.append(compare_one(
            sample,
            esm_dir / f"{sample}{paths.esm_suffix}",
            Path(paths.parquet_dir) / f"{sample}{paths.parquet_suffix}",
        ))
        if index and index % 500 == 0:
            logger.info("  %d/%d", index, len(samples))

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    tally = frame["status"].value_counts()
    logger.info("wrote %s\n%s", args.out, tally.to_string())
    broken = frame[frame["status"].isin(["contig_mismatch", "count_mismatch", "contig_count_mismatch"])]
    if len(broken):
        logger.warning(
            "%d of %d genomes are MISALIGNED; median %d proteins per genome carry the wrong gene",
            len(broken), len(frame), int(broken["n_proteins_after_break"].median()))
        silent = frame[frame["status"] == "contig_mismatch"]
        if len(silent):
            logger.warning(
                "%d of those pass a TOTAL-count check and would not have been caught: %s",
                len(silent), ", ".join(silent["Sample"].head(5)))


if __name__ == "__main__":
    main()

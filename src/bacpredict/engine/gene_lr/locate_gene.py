"""Recover a target gene's flat protein index + sequence from a sample's parquet.

The per-sample ``{sample_id}_protein_sequences.parquet`` (written by
:mod:`bacpredict.engine.embedding.preprocess_assemblies_to_protein_sequences`) stores the protein
annotation as *lists-of-lists*: one inner list per contig, in ``contig_idx``
order. The ESM-C embedding store drops the per-protein labels, so to pull a
single gene's pooled vector back out of ``{sample_id}_esm_embeddings.pt`` we have
to reconstruct the *flat* protein index that matches the embedding tensor.

The flat order is the concatenation of every contig's ``protein_sequence`` list
in ``contig_idx`` order — exactly the order ``compute_genome_protein_embeddings``
feeds ESM-C and that ``protein_embeddings_to_inputs`` preserves (real proteins
carry the ``PROT_EMB`` token; see :mod:`bacpredict.engine.embedding.protein_pooling` for
how the matching rows are selected back out of the stored tensor).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Columns we flatten in lock-step (every per-contig list has the same length).
_NESTED_COLUMNS = ("gene_name", "protein_name", "protein_sequence", "start", "end", "protein_id")


@dataclass(frozen=True)
class GeneHit:
    """One protein matched by :func:`locate_gene`.

    Attributes
    ----------
    flat_index : int
        Position of this protein in the flattened protein list — the index into
        the ``PROT_EMB`` rows of ``{sample_id}_esm_embeddings.pt``.
    gene_name : str | None
        The matched ``gene_name`` value (the GFF ``gene`` attribute, or the
        locus-tag fallback).
    sequence : str
        Amino-acid sequence as embedded by ESM-C (the exact string the model saw).
    contig_idx : int
        Index of the contig this protein sits on.
    start : int | None
        1-based start coordinate on the contig.
    end : int | None
        End coordinate on the contig.
    """

    flat_index: int
    gene_name: str | None
    sequence: str
    contig_idx: int
    start: int | None
    end: int | None


def _as_list(value) -> list:
    """Coerce a parquet cell (numpy array / list / scalar) to a plain list."""
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, list):
        return value
    return [value]


def flatten_proteins(row: pd.Series | pd.DataFrame) -> list[dict]:
    """Flatten one sample's nested protein lists into flat per-protein records.

    Parameters
    ----------
    row : pandas.Series or single-row pandas.DataFrame
        One row of a ``*_protein_sequences.parquet`` file.

    Returns
    -------
    list of dict
        One dict per protein, in flat (embedding) order, each carrying
        ``flat_index``, ``contig_idx`` and the per-protein annotation fields.
    """
    if isinstance(row, pd.DataFrame):
        if len(row) != 1:
            raise ValueError(f"Expected a single-row frame, got {len(row)} rows")
        row = row.iloc[0]

    contig_idx_list = _as_list(row.get("contig_idx"))
    nested = {col: _as_list(row.get(col)) for col in _NESTED_COLUMNS}

    # `contig_idx` is range(n_contigs); each nested column is a list with one
    # inner list per contig. Walk contigs in order so the flat index lines up
    # with the embedding tensor.
    n_contigs = len(nested["protein_sequence"])
    if not contig_idx_list:
        contig_idx_list = list(range(n_contigs))

    records: list[dict] = []
    flat_index = 0
    for c in range(n_contigs):
        seqs = _as_list(nested["protein_sequence"][c])
        per_col = {col: _as_list(nested[col][c]) if c < len(nested[col]) else [] for col in _NESTED_COLUMNS}
        contig_label = contig_idx_list[c] if c < len(contig_idx_list) else c
        for j in range(len(seqs)):
            records.append(
                {
                    "flat_index": flat_index,
                    "contig_idx": int(contig_label),
                    "gene_name": per_col["gene_name"][j] if j < len(per_col["gene_name"]) else None,
                    "protein_name": per_col["protein_name"][j] if j < len(per_col["protein_name"]) else None,
                    "protein_sequence": seqs[j],
                    "start": per_col["start"][j] if j < len(per_col["start"]) else None,
                    "end": per_col["end"][j] if j < len(per_col["end"]) else None,
                    "protein_id": per_col["protein_id"][j] if j < len(per_col["protein_id"]) else None,
                }
            )
            flat_index += 1
    return records


def locate_gene(
    parquet: str | Path | pd.DataFrame,
    gene: str,
    *,
    aliases: tuple[str, ...] = (),
) -> list[GeneHit]:
    """Find every protein whose ``gene_name`` matches ``gene`` (case-insensitive).

    Parameters
    ----------
    parquet : str or Path or pandas.DataFrame
        Path to a ``*_protein_sequences.parquet`` file, or an already-loaded
        single-row frame.
    gene : str
        Target gene symbol (e.g. ``"rpoB"``). Matched case-insensitively.
    aliases : tuple of str, optional
        Additional accepted gene symbols (e.g. alternative locus names).

    Returns
    -------
    list of GeneHit
        All matching proteins, in flat order. Usually length 1 for single-copy
        core genes such as ``rpoB``; an empty list means the gene was not
        annotated in this sample.
    """
    if isinstance(parquet, (str, Path)):
        df = pd.read_parquet(parquet)
    else:
        df = parquet
    records = flatten_proteins(df)

    wanted = {gene.lower(), *(a.lower() for a in aliases)}
    hits = [
        GeneHit(
            flat_index=r["flat_index"],
            gene_name=r["gene_name"],
            sequence=r["protein_sequence"],
            contig_idx=r["contig_idx"],
            start=r["start"],
            end=r["end"],
        )
        for r in records
        if r["gene_name"] is not None and str(r["gene_name"]).lower() in wanted
    ]
    return hits


def build_gene_presence_table(
    sample_ids: list[str],
    parquet_dir: str | Path,
    gene: str,
    *,
    aliases: tuple[str, ...] = (),
    parquet_suffix: str = "_protein_sequences.parquet",
    qc_log_path: str | Path | None = None,
) -> pd.DataFrame:
    """Per-genome single-copy flat index of ``gene`` — the generic, any-gene presence table.

    The gene-agnostic analogue of the archived rpoB genotyper's ``build_genotype_table``, without
    the rpoB-specific RRDR allele calling: for each sample, flatten its proteins and keep the genomes
    where ``gene`` (case-insensitive, plus ``aliases``) appears **exactly once** (single-copy). Genomes
    where the gene is absent or multi-copy are skipped and counted (optionally logged to
    ``qc_log_path``). This is the substrate for any "ESM gene vector ⊕ Bacformer genome mean" concat.

    Parameters
    ----------
    sample_ids
        Sample IDs to scan (``{sample}{parquet_suffix}`` under ``parquet_dir``).
    parquet_dir, parquet_suffix
        Location + suffix of the ``*_protein_sequences.parquet`` files.
    gene, aliases
        Target gene symbol (matched case-insensitively) and any accepted alternative symbols.
    qc_log_path
        If given, append a one-line skip summary (missing / absent / multi-copy counts).

    Returns
    -------
    pandas.DataFrame
        Indexed by ``Sample`` with ``protein_index`` (index into the genome's ESM-C ``PROT_EMB``
        rows), ``n_proteins`` (flattened protein count — the alignment guard), ``gene_name`` (matched
        symbol), and ``annotation`` (the matched protein's ``protein_name`` / product). Empty if no
        single-copy genome is found.
    """
    parquet_dir = Path(parquet_dir)
    wanted = {gene.lower(), *(a.lower() for a in aliases)}
    rows: list[dict] = []
    skips = {"missing_parquet": 0, "absent": 0, "multi_copy": 0}
    for sid in sample_ids:
        pq = parquet_dir / f"{sid}{parquet_suffix}"
        if not pq.exists():
            skips["missing_parquet"] += 1
            continue
        records = flatten_proteins(pd.read_parquet(pq))
        hits = [r for r in records if r["gene_name"] is not None and str(r["gene_name"]).lower() in wanted]
        if not hits:
            skips["absent"] += 1
            continue
        if len(hits) > 1:
            skips["multi_copy"] += 1
            continue
        hit = hits[0]
        rows.append({
            "Sample": str(sid),
            "protein_index": int(hit["flat_index"]),
            "n_proteins": len(records),
            "gene_name": hit["gene_name"],
            "annotation": hit["protein_name"],
        })

    n_kept = len(rows)
    logger.info(
        "gene presence (%s): %d single-copy of %d samples (skipped %s)", gene, n_kept, len(sample_ids), skips
    )
    if qc_log_path is not None:
        with Path(qc_log_path).open("a") as fh:
            fh.write(f"gene={gene} single_copy={n_kept} of={len(sample_ids)} skips={skips}\n")
    if not rows:
        return pd.DataFrame(columns=["protein_index", "n_proteins", "gene_name", "annotation"]).rename_axis("Sample")
    return pd.DataFrame(rows).set_index("Sample")

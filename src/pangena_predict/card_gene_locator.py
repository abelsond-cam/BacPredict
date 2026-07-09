"""Locate a CARD AMR gene family's protein → flat embedding index per genome (for acquired Kp genes).

Bakta under-annotates acquired AMR genes (AAC(6'), ArmA, bla_KPC, …), so the driver panel cannot find
them by Bakta ``gene_name`` the way it finds core genes like *rpoB*/*gyrA*. Kleborate/CARD *can*: the
sidecar ``{Sample}_amr.parquet`` written by :mod:`kleb_ast.annotate_amr_sidecar` re-identifies every AMR
gene by ``minimap2`` of the vendored CARD refs against the assembly and records, **per hit**, the
**flat protein index** (the row into ``{Sample}_esm_embeddings.pt`` / baclm ``protein_embeddings``) plus
its ``amr_gene_family`` — built against the *same* assembly the protein parquet came from, with a
flat-order validation guard.

This module turns those sidecars into the **same presence-table schema** that
:func:`pangena_predict.coding_amr_lr.build_multi_gene_presence` produces (a ``DataFrame`` per gene,
indexed by ``Sample``, with ``gene_flat_index`` / ``n_proteins`` / ``gene_name`` / ``annotation``) — so
``load_baclm_gene_vectors`` / ``load_pooled_gene_vectors`` and the Bacformer sweep consume it unchanged.
The only difference from the Bakta path is *how the flat index is found* (CARD family match, not gene
name). Single-copy only, mirroring the Bakta locator; ``flat_index < 0`` (Bakta-missed, no CDS) rows
carry no embedding and are ignored.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from pangena_predict.locate_gene import flatten_proteins

logger = logging.getLogger(__name__)


def _card_presence_one(sid: str, amr_pq: Path, protein_pq: Path, wanted_by_family: dict[str, frozenset]):
    """Locate every target CARD family in one genome's sidecar (worker-safe).

    Returns ``(sid, per_family)`` where ``per_family[family]`` is the single-copy hit dict or ``None``
    (absent / multi-copy); ``per_family`` is ``None`` if the sidecar is missing (uncovered genome).
    """
    if not amr_pq.exists():
        return sid, None
    df = pd.read_parquet(amr_pq)
    if "flat_index" not in df.columns or "amr_gene_family" not in df.columns:
        return sid, None
    df = df[df["flat_index"].astype(int) >= 0]  # drop Bakta-missed (no CDS -> no embedding) rows
    fam_lower = df["amr_gene_family"].astype(str).str.lower()
    # n_proteins from the protein parquet's flat count — the same guard build_multi_gene_presence uses.
    n_prot = len(flatten_proteins(pd.read_parquet(protein_pq))) if protein_pq.exists() else None

    per_family: dict = {}
    for family, wanted in wanted_by_family.items():
        rows = df[fam_lower.isin(wanted)]
        flat_ids = sorted({int(x) for x in rows["flat_index"]})
        if len(flat_ids) == 1:
            best = rows.iloc[0]
            per_family[family] = {
                "gene_flat_index": flat_ids[0], "n_proteins": n_prot,
                "gene_name": family, "annotation": best.get("amr_allele"),
            }
        else:
            per_family[family] = None  # absent (0) or multi-copy (>1) — ambiguous which vector to pull
    return sid, per_family


def build_card_presence(
    sample_ids: list[str],
    amr_sidecar_dir: Path,
    parquet_dir: Path,
    family_specs: list[tuple[str, tuple[str, ...]]],
    *,
    amr_suffix: str = "_amr.parquet",
    parquet_suffix: str = "_protein_sequences.parquet",
    pool_workers: int = 1,
) -> dict[str, pd.DataFrame]:
    """One sidecar sweep → per-CARD-family single-copy presence tables (drop-in for build_multi_gene_presence).

    ``family_specs`` is a list of ``(amr_gene_family, aliases_tuple)`` (aliases let a CSV label map onto
    the sidecar's family string). Returns ``dict[family] → DataFrame`` with the same columns as
    :func:`pangena_predict.coding_amr_lr.build_multi_gene_presence`, indexed by ``Sample`` (single-copy).
    """
    wanted_by_family = {f: frozenset([f.lower(), *(a.lower() for a in aliases)]) for f, aliases in family_specs}
    amr_sidecar_dir, parquet_dir = Path(amr_sidecar_dir), Path(parquet_dir)
    tasks = [
        (str(s), amr_sidecar_dir / f"{s}{amr_suffix}", parquet_dir / f"{s}{parquet_suffix}", wanted_by_family)
        for s in sample_ids
    ]
    if pool_workers > 1:
        import multiprocessing as mp

        with mp.Pool(pool_workers) as pool:
            results = pool.starmap(_card_presence_one, tasks)
    else:
        results = [_card_presence_one(*t) for t in tasks]

    rows_by_family: dict = {f: [] for f, _ in family_specs}
    n_missing = 0
    for sid, per_family in results:
        if per_family is None:
            n_missing += 1
            continue
        for family, hit in per_family.items():
            if hit is not None:
                rows_by_family[family].append({"Sample": sid, **hit})
    empty = pd.DataFrame(columns=["gene_flat_index", "n_proteins", "gene_name", "annotation"]).rename_axis("Sample")
    tables = {f: (pd.DataFrame(rows).set_index("Sample") if rows else empty.copy()) for f, rows in rows_by_family.items()}
    for f, t in tables.items():
        logger.info("CARD presence: %s single-copy in %d/%d genomes (missing sidecar=%d)",
                    f, len(t), len(sample_ids), n_missing)
    return tables


def sidecar_dir_available(amr_sidecar_dir: Path, sample_ids: list[str], *, amr_suffix: str = "_amr.parquet") -> bool:
    """True if the sidecar dir exists and at least one sample's ``{Sample}_amr.parquet`` is present."""
    amr_sidecar_dir = Path(amr_sidecar_dir)
    if not amr_sidecar_dir.is_dir():
        return False
    return any((amr_sidecar_dir / f"{s}{amr_suffix}").exists() for s in sample_ids[:200])

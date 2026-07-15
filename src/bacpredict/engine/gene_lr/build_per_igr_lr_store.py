"""Build the per-IGR logistic-regression ranking over the baclm non-coding channel (Phase 3).

The non-coding sibling of :mod:`bacpredict.engine.gene_lr.build_per_gene_lr_store`. Where that ranks
core **genes** by "does this gene's own embedding predict resistance?", this ranks core **intergenic
regions** by "does this region's baclm non-coding embedding predict resistance?" — and the top region
is the best-IGR block for the 3-way concat (``bacformerFT-mean ⊕ baclm-best-gene ⊕ baclm-best-IGR``).

**IGR identity = ordered 5′→3′ flanking-gene pair** ``left_gene→right_gene`` (ascending genome
coordinate, i.e. the forward strand — *oblivious to coding direction*; a finer strand/operon-aware
naming comes from the sister Nuna project). A region is named only when **both** directly-abutting
flanks are consistently-named ``gene=`` symbols (mirrors the per-gene "recurring symbol" rule — a CDS
with only a per-genome locus-tag never recurs, so it can't seed a core pair). Regions abutting an
unnamed CDS, an RNA, or a contig end are left unnamed and dropped.

For every core IGR pair (single-copy in > ``--min-prevalence`` of the *train* genomes) we fit a
stand-alone out-of-fold ``LogisticRegression`` on that region's 960-d baclm embedding predicting the
binary resistance label, and rank pairs by out-of-fold train AUROC. The leakage discipline and the LR
itself are **reused verbatim** from the per-gene store (:func:`fit_per_gene`): train genomes get an
out-of-fold probability (K-fold within train), so the AUROC is a held-out estimate, never in-sample.

The baclm intergenic store lives inside each ``{sample}_baclm_embeddings.pt`` as ``noncoding_*`` (the
2d re-embed's maximal non-CDS runs) or the legacy ``intergenic_*`` keys, one row per region with its
``seqid/start/end`` — the same layout :mod:`bacpredict.engine.gene_lr.igr_amr_lr` reads. Flanking-gene
coordinates come from the sample's Bakta GFF (``input_csv`` → ``sr_gff_file``), because the region's
string ``seqid`` joins to the GFF, not to the parquet's integer ``contig_idx``.

Output (mirrors the per-gene ranking contract so later drugs merge onto ``igr_pair``):

    <out-dir>/per_igr_lr_<drug>.csv        # igr_pair,left_gene,right_gene,prevalence,lr_auroc_<drug>,n_train,n_pos,kept
    <out-dir>/igr_lr_auroc.csv             # per-pair out-of-fold train AUROC + provenance
    <out-dir>/igr_prevalence.csv
    <out-dir>/per_igr_lr_build_summary.json

CPU-only (GFF parse + sklearn LRs over precomputed baclm vectors). Stage-A smoke (``--max-train-genomes
30``) runs on a login node; the full cohort is a CPU sbatch (tens of thousands of ``.pt``/GFF reads).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bacpredict.engine.config import store_paths
from bacpredict.engine.gene_lr.build_per_gene_lr_store import (
    fit_per_gene,
    load_splits,
    subsample_balanced,
)
from bacpredict.engine.gene_lr.igr_amr_lr import _parse_gff

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------------------------------------------------------------------
# Per-genome IGR identity: read the baclm intergenic rows, name each by its
# ordered 5′→3′ flanking-gene pair from the GFF.
# ---------------------------------------------------------------------------


def _genes_by_seqid(gff_path: Path) -> dict[str, list[tuple[int, int, str]]]:
    """``seqid → sorted [(start, end, gene_name_lower)]`` for every ``gene=`` CDS in the GFF.

    Reuses :func:`igr_amr_lr._parse_gff` (which keys named genes by symbol) and inverts it to a
    per-contig coordinate-sorted list — the substrate for locating each IGR's abutting flanks.
    """
    _feats, genes = _parse_gff(gff_path)
    by_seqid: dict[str, list[tuple[int, int, str]]] = {}
    for gname, hits in genes.items():
        for seqid, start, end, _strand in hits:
            by_seqid.setdefault(seqid, []).append((int(start), int(end), gname))
    for seqid in by_seqid:
        by_seqid[seqid].sort()
    return by_seqid


def _read_intergenic(pt_path: Path) -> tuple[np.ndarray, list[str], list[int], list[int]] | None:
    """Load one baclm store's intergenic rows: ``(emb[n, dim], seqids, starts, ends)`` or ``None``.

    Prefers the 2d-re-embed ``noncoding_*`` keys, falls back to the legacy ``intergenic_*`` keys, so
    this reads both the old and re-embedded stores (same fallback as :mod:`igr_amr_lr`).
    """
    if not pt_path.exists():
        return None
    store = torch.load(pt_path, map_location="cpu", mmap=True, weights_only=True)
    if "noncoding_embeddings" in store:
        emb, seqid, start, end = (
            store["noncoding_embeddings"], store["noncoding_seqid"], store["noncoding_start"], store["noncoding_end"],
        )
    elif "intergenic_embeddings" in store:
        emb, seqid, start, end = (
            store["intergenic_embeddings"], store["intergenic_seqid"], store["intergenic_start"], store["intergenic_end"],
        )
    else:
        return None
    return emb.float().numpy(), list(seqid), [int(s) for s in start], [int(e) for e in end]


def _flank_pair(
    genes_here: list[tuple[int, int, str]], igr_start: int, igr_end: int, *, boundary_tol: int
) -> tuple[str, str] | None:
    """Ordered 5′→3′ flanking-gene pair for one IGR, or ``None`` if either flank is unnamed/absent.

    ``left`` = the named gene directly abutting the region's low-coordinate boundary (``end`` closest
    below ``igr_start``, within ``boundary_tol``); ``right`` = the named gene abutting the high boundary
    (``start`` closest above ``igr_end``). Requiring abutment within a few bp is what enforces the
    "consistently-named flank" rule: if the immediately-adjacent CDS is unnamed (locus-tag only) the
    nearest *named* gene lies beyond it, its gap exceeds the tolerance, and the region is dropped.
    """
    left, left_gap = None, boundary_tol + 1
    right, right_gap = None, boundary_tol + 1
    for start, end, gname in genes_here:
        if end < igr_start:
            gap = igr_start - 1 - end
            if gap <= boundary_tol and gap < left_gap:
                left, left_gap = gname, gap
        if start > igr_end:
            gap = start - (igr_end + 1)
            if gap <= boundary_tol and gap < right_gap:
                right, right_gap = gname, gap
    if left is None or right is None:
        return None
    return left, right


def _genome_igr_records(
    sid: str, gff_path: str, pt_path: str, boundary_tol: int = 3
) -> tuple[str, list[tuple[str, np.ndarray]]] | None:
    """One genome's ``[(igr_pair, embedding)]`` — the named, CDS-flanked baclm intergenic regions.

    ``igr_pair`` is ``"left→right"`` (ascending genome coord). Returns ``None`` when the GFF or ``.pt``
    is missing/unreadable (the genome is skipped, not imputed).
    """
    gpath, ppath = Path(gff_path), Path(pt_path)
    if not gpath.exists() or not ppath.exists():
        return None
    try:
        by_seqid = _genes_by_seqid(gpath)
    except (OSError, ValueError):
        return None
    read = _read_intergenic(ppath)
    if read is None:
        return None
    emb, seqids, starts, ends = read
    records: list[tuple[str, np.ndarray]] = []
    for i, (sq, s, e) in enumerate(zip(seqids, starts, ends, strict=True)):
        pair = _flank_pair(by_seqid.get(sq, []), s, e, boundary_tol=boundary_tol)
        if pair is None:
            continue
        records.append((f"{pair[0]}→{pair[1]}", emb[i]))
    return sid, records


def collect_igr_matrices(
    train_ids: list[str],
    sample_gff: dict[str, str],
    baclm_dir: Path,
    *,
    baclm_suffix: str = "_baclm_embeddings.pt",
    boundary_tol: int = 3,
    pool_workers: int = 1,
) -> tuple[dict[str, tuple[list[str], np.ndarray]], pd.DataFrame, list[str]]:
    """One GFF+``.pt`` sweep → per-IGR-pair single-copy design matrices + prevalence + read universe.

    For each train genome, name every CDS-flanked baclm intergenic region by its ordered flanking pair
    and keep the **single-copy** pairs (a pair occurring exactly once in the genome — mirrors the gene
    store's single-copy rule, excluding recurrent tandem/paralog ambiguity). Returns
    ``({pair: (sample_ids, X[m, dim])}, prevalence_table, read_ids)``: the per-pair matrices over the
    genomes carrying the pair single-copy, a ``pair/n_single_copy/prevalence`` table, and the list of
    genomes successfully read (the universe for a zero-impute fit — a *read* genome lacking a pair is
    genuinely pair-absent; a *skipped* genome has no data).
    """
    tasks = [
        (str(s), sample_gff.get(str(s), ""), str(Path(baclm_dir) / f"{s}{baclm_suffix}"))
        for s in train_ids if str(s) in sample_gff
    ]
    if pool_workers > 1:
        # joblib (not multiprocessing.Pool) for the sweep: this module then runs a joblib Parallel fit
        # (fit_per_gene), and mixing an mp.Pool with a following loky Parallel corrupts the resource
        # tracker on aarch64 (worker crash). One consistent backend for sweep + fit avoids it.
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=pool_workers)(
            delayed(_genome_igr_records)(sid, gff, pt, boundary_tol) for sid, gff, pt in tasks
        )
    else:
        results = [_genome_igr_records(sid, gff, pt, boundary_tol) for sid, gff, pt in tasks]

    ids_by_pair: dict[str, list[str]] = {}
    vecs_by_pair: dict[str, list[np.ndarray]] = {}
    single_copy_genomes: Counter[str] = Counter()
    read_ids: list[str] = []
    n_skipped = 0
    for res in results:
        if res is None:
            n_skipped += 1
            continue
        sid, records = res
        read_ids.append(sid)
        counts = Counter(pair for pair, _ in records)
        for pair, vec in records:
            if counts[pair] == 1:  # single-copy occurrence in this genome
                single_copy_genomes[pair] += 1
                ids_by_pair.setdefault(pair, []).append(sid)
                vecs_by_pair.setdefault(pair, []).append(vec)
    if n_skipped:
        logger.warning("IGR sweep: skipped %d train genomes (missing/unreadable GFF or .pt)", n_skipped)

    n = len(read_ids)
    prevalence = pd.DataFrame(
        [{"igr_pair": p, "n_single_copy": c, "prevalence": c / max(n, 1)} for p, c in single_copy_genomes.items()]
    ).sort_values("prevalence", ascending=False).reset_index(drop=True)
    matrices = {p: (ids_by_pair[p], np.vstack(vecs_by_pair[p])) for p in ids_by_pair}
    logger.info("IGR sweep: %d named single-copy pairs over %d read genomes", len(matrices), n)
    return matrices, prevalence, read_ids


# ---------------------------------------------------------------------------
# Ranking table
# ---------------------------------------------------------------------------


def write_igr_drug_table(
    fitted: dict[str, dict],
    prevalence: pd.DataFrame,
    *,
    drug: str,
    filtered_pairs: set[str],
    out_path: Path,
) -> None:
    """Write the wide IGR×drug ranking: ``igr_pair,left_gene,right_gene,prevalence,lr_auroc_<drug>,…``.

    One row per fitted pair, ranked by AUROC. The per-drug AUROC column is ``lr_auroc_<drug>`` so later
    drugs merge onto ``igr_pair`` into one wide table. ``left_gene``/``right_gene`` split the ordered
    pair back out for readability.
    """
    prev_by_pair = dict(zip(prevalence["igr_pair"], prevalence["prevalence"], strict=False))
    rows = []
    for pair, f in sorted(fitted.items(), key=lambda kv: kv[1]["auroc"], reverse=True):
        left, _, right = pair.partition("→")
        rows.append({
            "igr_pair": pair,
            "left_gene": left,
            "right_gene": right,
            "prevalence": prev_by_pair.get(pair, float("nan")),
            f"lr_auroc_{drug}": f["auroc"],
            "n_train": f["n_train"],
            "n_pos": f["n_pos"],
            "kept_filtered": pair in filtered_pairs,
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.info("Wrote wide IGR×drug table (%d pairs) to %s", len(rows), out_path)


def run(
    *,
    split_csv: Path,
    drug: str,
    input_csv: Path,
    baclm_dir: Path,
    out_dir: Path,
    min_prevalence: float,
    auroc_filter: float,
    n_folds: int,
    seed: int,
    n_jobs: int = 1,
    max_train_genomes: int | None = None,
    sample_seed: int = 1,
    boundary_tol: int = 3,
    baclm_suffix: str = "_baclm_embeddings.pt",
    pool_workers: int = 1,
    impute_absent_zero: bool = False,
) -> dict:
    """Name IGRs, fit per-IGR LRs on a (sub)sample of train, write the wide IGR×drug ranking table."""
    label_map, train_ids, validate_ids, evaluate_ids = load_splits(split_csv, drug)
    fit_train_ids = subsample_balanced(train_ids, label_map, max_n=max_train_genomes, seed=sample_seed)

    inp = pd.read_csv(input_csv, usecols=["Sample", "sr_gff_file"])
    sample_gff = dict(zip(inp["Sample"].astype(str), inp["sr_gff_file"].astype(str), strict=True))

    matrices, prevalence, read_ids = collect_igr_matrices(
        fit_train_ids, sample_gff, baclm_dir,
        baclm_suffix=baclm_suffix, boundary_tol=boundary_tol, pool_workers=pool_workers,
    )
    core_pairs = set(prevalence.loc[prevalence["prevalence"] > min_prevalence, "igr_pair"])
    core_matrices = {p: m for p, m in matrices.items() if p in core_pairs}
    logger.info("Core IGR pairs (single-copy in >%.0f%% of %d train): %d of %d named pairs",
                100 * min_prevalence, len(read_ids), len(core_matrices), len(matrices))

    fitted = fit_per_gene(
        core_matrices, label_map, n_folds=n_folds, seed=seed, n_jobs=n_jobs,
        all_ids=read_ids, impute_absent_zero=impute_absent_zero,
    )
    filtered_pairs = {p for p, f in fitted.items() if f["auroc"] > auroc_filter}
    logger.info("Filter (AUROC > %.2f): %d of %d fitted pairs kept", auroc_filter, len(filtered_pairs), len(fitted))

    out_dir.mkdir(parents=True, exist_ok=True)
    auroc_rows = [
        {"igr_pair": p, "auroc": f["auroc"], "n_train": f["n_train"], "n_pos": f["n_pos"],
         "kept_filtered": p in filtered_pairs}
        for p, f in sorted(fitted.items(), key=lambda kv: kv[1]["auroc"], reverse=True)
    ]
    pd.DataFrame(auroc_rows).to_csv(out_dir / "igr_lr_auroc.csv", index=False)
    write_igr_drug_table(fitted, prevalence, drug=drug, filtered_pairs=filtered_pairs,
                         out_path=out_dir / f"per_igr_lr_{drug}.csv")
    prevalence.to_csv(out_dir / "igr_prevalence.csv", index=False)

    best = max(fitted.items(), key=lambda kv: kv[1]["auroc"], default=(None, None))
    summary = {
        "task": "pangena_predict",
        "analysis": "build_per_igr_lr_store",
        "drug": drug,
        "embedding_store": "baclm_intergenic",
        "split_csv": str(split_csv),
        "n_train": len(train_ids),
        "n_train_fit": len(fit_train_ids),
        "max_train_genomes": max_train_genomes,
        "sample_seed": sample_seed,
        "n_validate": len(validate_ids),
        "n_evaluate": len(evaluate_ids),
        "n_read_genomes": len(read_ids),
        "boundary_tol": boundary_tol,
        "min_prevalence": min_prevalence,
        "n_named_pairs": len(matrices),
        "n_core_pairs": len(core_matrices),
        "n_fitted_pairs": len(fitted),
        "auroc_filter": auroc_filter,
        "n_filtered_pairs": len(filtered_pairs),
        "n_folds": n_folds,
        "seed": seed,
        "impute_absent_zero": impute_absent_zero,
        "best_igr_pair": best[0],
        "best_igr_auroc": best[1]["auroc"] if best[1] else None,
    }
    (out_dir / "per_igr_lr_build_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("build_summary: %s", json.dumps({k: summary[k] for k in (
        "n_named_pairs", "n_core_pairs", "n_fitted_pairs", "n_filtered_pairs", "best_igr_pair", "best_igr_auroc")}))
    return summary


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--species", choices=["tb", "kp"], default="tb", help="Resolve default store paths.")
    parser.add_argument("--split-csv", type=Path, help="CSV with Sample, <drug>, train_val_eval (default: species sheet).")
    parser.add_argument("--drug", type=str, default="rifampin", help="Binary label column (default rifampin, US).")
    parser.add_argument("--input-csv", type=Path, help="Sample→GFF map (embedding_input.csv; default: species input_csv).")
    parser.add_argument("--baclm-dir", type=Path, help="Dir of *_baclm_embeddings.pt (default: species baclm_dir).")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output base dir for the ranking tables.")
    parser.add_argument("--min-prevalence", type=float, default=0.95,
                        help="Single-copy prevalence threshold over train (0.95 = core; lower to include accessory pairs).")
    parser.add_argument("--auroc-filter", type=float, default=0.8,
                        help="Mark pairs with out-of-fold train AUROC above this as kept_filtered (default 0.8).")
    parser.add_argument("--n-folds", type=int, default=5, help="Out-of-fold cross-fitting folds within train.")
    parser.add_argument("--seed", type=int, default=1, help="Fold-assignment seed.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Worker processes for the per-IGR fits (-1 = all cores).")
    parser.add_argument("--pool-workers", type=int, default=1, help="Worker processes for the GFF+.pt read sweep.")
    parser.add_argument("--max-train-genomes", type=int, default=None,
                        help="Fit on a random, class-balanced subsample of this many train genomes (default: all).")
    parser.add_argument("--sample-seed", type=int, default=1, help="Seed for the train subsample (default 1).")
    parser.add_argument("--boundary-tol", type=int, default=3,
                        help="Max bp gap between an IGR boundary and its abutting named flank gene (default 3).")
    parser.add_argument("--impute-absent-zero", action="store_true",
                        help="Fit each pair over ALL read genomes, zero-imputing the ones lacking it (presence/absence "
                             "signal), instead of dropping absent genomes.")
    args = parser.parse_args()

    paths = store_paths(args.species)
    split_csv = args.split_csv or paths.ast_sheet
    input_csv = args.input_csv or paths.input_csv
    baclm_dir = args.baclm_dir or paths.baclm_dir

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run(
        split_csv=split_csv,
        drug=args.drug,
        input_csv=input_csv,
        baclm_dir=baclm_dir,
        out_dir=args.out_dir,
        min_prevalence=args.min_prevalence,
        auroc_filter=args.auroc_filter,
        n_folds=args.n_folds,
        seed=args.seed,
        n_jobs=args.n_jobs,
        max_train_genomes=args.max_train_genomes,
        sample_seed=args.sample_seed,
        boundary_tol=args.boundary_tol,
        baclm_suffix=paths.baclm_suffix,
        pool_workers=args.pool_workers,
        impute_absent_zero=args.impute_absent_zero,
    )


if __name__ == "__main__":
    main()

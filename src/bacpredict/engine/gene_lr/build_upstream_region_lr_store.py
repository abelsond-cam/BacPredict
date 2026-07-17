"""Per-gene UPSTREAM-REGION LR ranking — the synteny-anchored non-coding screen.

The sibling of :mod:`build_per_igr_lr_store`, but it names each baclm non-coding region by the **gene it
sits immediately 5′ of** (``upstream:<gene>``) instead of by its two flanking genes. This is the fix for
a real capture bug: the flanking-pair scheme requires *both* neighbours to be consistently-named core
genes, so a regulatory region whose far neighbour is an unnamed CDS (locus-tag only) is silently dropped —
including the canonical mabA-inhA operon promoter (the ethionamide/isoniazid −15), which sits 5′ of
``fabG1`` next to an unnamed AbiEi antitoxin CDS and therefore *never appears* in the per-IGR ranking even
though it is embedded in the store. Anchoring on the single downstream gene keeps it (and every other
promoter next to a hypothetical) and names it by the gene it regulates, which is also what the WHO/CARD
catalogues call it ("inhA promoter").

For each named gene in a genome we take the region **abutting its 5′ end** (on ``-`` strand the region
just above the gene end; on ``+`` strand just below the gene start, within ``--boundary-tol`` bp), key it
``upstream:<gene>``, keep the **single-copy** anchors, and fit the same out-of-fold LR harness
(``fit_per_gene``) → per-anchor AUROC. Writes ``per_upstream_lr_<drug>.csv``
(``upstream_gene, gene, prevalence, lr_auroc_<drug>, n_train, n_pos, kept_filtered``).

Reuses the store readers/LR from :mod:`build_per_igr_lr_store` verbatim (both read the legacy
``intergenic_*`` and the re-embed ``noncoding_*`` keys), so it runs on the current store today and on the
re-embedded store unchanged. Longer term the synteny anchoring is the job of the sibling *syntology*
project (bakta-independent homology+synteny); this is the lightweight gene-upstream version.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from bacpredict.engine.config import store_paths
from bacpredict.engine.gene_lr.build_per_gene_lr_store import fit_per_gene, load_splits, subsample_balanced
from bacpredict.engine.gene_lr.build_per_igr_lr_store import _read_intergenic
from bacpredict.engine.gene_lr.igr_amr_lr import _parse_gff

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _upstream_region_index(
    gstart: int, gend: int, strand: str, rows: list[tuple[int, int, int]], *, boundary_tol: int
) -> int | None:
    """Index of the non-coding region abutting a gene's 5′ end, or ``None`` if none within tolerance.

    ``rows`` is the genome's ``[(start, end, row_idx)]`` on the gene's contig. On ``-`` strand the 5′ end
    is the high coordinate (``gend``) and the upstream region's ``start`` abuts ``gend+1``; on ``+`` strand
    the 5′ end is ``gstart`` and the region's ``end`` abuts ``gstart-1``. Ties break to the nearest region.
    """
    best_idx, best_gap = None, boundary_tol + 1
    if strand == "-":
        for s, _e, i in rows:
            gap = s - (gend + 1)  # region sits above the gene
            if -boundary_tol <= gap <= boundary_tol and abs(gap) < best_gap:
                best_idx, best_gap = i, abs(gap)
    else:  # '+' (and default) — upstream is below the gene start
        for _s, e, i in rows:
            gap = (gstart - 1) - e
            if -boundary_tol <= gap <= boundary_tol and abs(gap) < best_gap:
                best_idx, best_gap = i, abs(gap)
    return best_idx


def _genome_upstream_records(
    sid: str, gff_path: str, pt_path: str, boundary_tol: int = 3
) -> tuple[str, list[tuple[str, np.ndarray]]] | None:
    """One genome's ``[(upstream:<gene>, embedding)]`` — each named gene's 5′-abutting baclm region."""
    gpath, ppath = Path(gff_path), Path(pt_path)
    if not gpath.exists() or not ppath.exists():
        return None
    try:
        _feats, genes = _parse_gff(gpath)
    except (OSError, ValueError):
        return None
    read = _read_intergenic(ppath)
    if read is None:
        return None
    emb, seqids, starts, ends = read
    rows_by_seqid: dict[str, list[tuple[int, int, int]]] = {}
    for i, (sq, s, e) in enumerate(zip(seqids, starts, ends, strict=True)):
        rows_by_seqid.setdefault(sq, []).append((s, e, i))

    records: list[tuple[str, np.ndarray]] = []
    for gname, hits in genes.items():
        for seqid, gstart, gend, strand in hits:
            rows = rows_by_seqid.get(seqid)
            if not rows:
                continue
            idx = _upstream_region_index(gstart, gend, strand, rows, boundary_tol=boundary_tol)
            if idx is not None:
                records.append((f"upstream:{gname}", emb[idx]))
    return sid, records


def collect_upstream_matrices(
    train_ids: list[str], sample_gff: dict[str, str], baclm_dir: Path, *,
    baclm_suffix: str = "_baclm_embeddings.pt", boundary_tol: int = 3, eval_ids: set[str] | None = None,
    store_dtype: str = "float32", min_prevalence: float = 0.0,
) -> tuple[dict[str, tuple[list[str], np.ndarray]], pd.DataFrame, list[str]]:
    """GFF+``.pt`` sweep → per-anchor single-copy design matrices + prevalence + read universe.

    **Two-pass, streamed** (one genome at a time — never materialise every genome's records at once): pass
    1 tallies each anchor's single-copy prevalence over the fit genomes and keeps only anchors above
    ``min_prevalence`` (the *core* set); pass 2 re-reads and stores the 960-d vectors for **core anchors
    only**. This mirrors the coding side (:func:`assemble_gene_matrices` collects vectors only for
    pre-discovered core genes) and is essential at the full cohort — holding *every* named gene's region
    (mostly rare accessory anchors) needs >440 GB, while the ~1–2 k core anchors fit easily. The serial
    read (torch.load mmap can't be forked before the process-parallel fit on aarch64) costs two passes over
    the ``.pt`` files; the memory saving is worth the extra I/O.

    ``eval_ids`` (the held-out evaluate split) are swept in pass 2 and added to each core anchor's design
    matrix so the fitted LR can be scored on them, but they are excluded from the pass-1 prevalence /
    core-anchor selection and from the returned ``read_ids`` universe — those stay on the fit
    (train+validate) genomes only, so the evaluate split never influences which anchors are screened.
    """
    eval_ids = eval_ids or set()
    tasks = [
        (str(s), sample_gff.get(str(s), ""), str(Path(baclm_dir) / f"{s}{baclm_suffix}"))
        for s in train_ids if str(s) in sample_gff
    ]
    fit_tasks = [t for t in tasks if t[0] not in eval_ids]

    # Pass 1 (fit genomes only) — prevalence → core anchors; discard vectors as we go.
    single_copy: Counter[str] = Counter()
    read_ids: list[str] = []
    n_skipped = 0
    for sid, gff, pt in fit_tasks:
        res = _genome_upstream_records(sid, gff, pt, boundary_tol)
        if res is None:
            n_skipped += 1
            continue
        _sid, records = res
        read_ids.append(sid)
        counts = Counter(k for k, _ in records)
        single_copy.update(k for k, c in counts.items() if c == 1)
    if n_skipped:
        logger.warning("upstream pass 1: skipped %d fit genomes (missing/unreadable GFF or .pt)", n_skipped)

    n = len(read_ids)
    prevalence = pd.DataFrame(
        [{"upstream_gene": k, "n_single_copy": c, "prevalence": c / max(n, 1)} for k, c in single_copy.items()]
    ).sort_values("prevalence", ascending=False).reset_index(drop=True)
    core = {k for k, c in single_copy.items() if (c / max(n, 1)) > min_prevalence}
    logger.info("upstream pass 1: %d anchors > %.2f prevalence (of %d) over %d fit genomes",
                len(core), min_prevalence, len(single_copy), n)

    # Pass 2 (fit + eval) — store vectors for CORE anchors only (eval rows scored, never selected).
    ids_by_key: dict[str, list[str]] = {}
    vecs_by_key: dict[str, list[np.ndarray]] = {}
    for sid, gff, pt in tasks:
        res = _genome_upstream_records(sid, gff, pt, boundary_tol)
        if res is None:
            continue
        _sid, records = res
        counts = Counter(k for k, _ in records)
        for key, vec in records:
            if counts[key] == 1 and key in core:  # single-copy AND a core anchor
                ids_by_key.setdefault(key, []).append(sid)
                vecs_by_key.setdefault(key, []).append(vec.astype(store_dtype, copy=False))

    matrices = {k: (ids_by_key[k], np.vstack(vecs_by_key[k])) for k in ids_by_key}
    logger.info("upstream pass 2: materialised %d core anchor matrices", len(matrices))
    return matrices, prevalence, read_ids


def run(
    *, split_csv: Path, drug: str, input_csv: Path, baclm_dir: Path, out_dir: Path,
    min_prevalence: float = 0.10, auroc_filter: float = 0.8, n_folds: int = 5, seed: int = 1,
    n_jobs: int = 1, max_train_genomes: int | None = None, sample_seed: int = 1, boundary_tol: int = 3,
    baclm_suffix: str = "_baclm_embeddings.pt", eval_holdout: bool = False, store_dtype: str = "float32",
) -> dict:
    """Anchor regions on the downstream gene, fit per-anchor LRs, write the wide ranking table.

    ``eval_holdout`` gives **real held-out-test** numbers: each anchor's LR is fit on train+validate and
    additionally reports ``eval_auroc_<drug>`` on the untouched evaluate split (vs the OOF-only default).
    """
    label_map, train_ids, validate_ids, evaluate_ids = load_splits(split_csv, drug)
    fit_pool = [*train_ids, *validate_ids] if eval_holdout else list(train_ids)
    eval_set: set[str] | None = set(evaluate_ids) if eval_holdout else None
    fit_train_ids = subsample_balanced(fit_pool, label_map, max_n=max_train_genomes, seed=sample_seed)
    sweep_ids = [*fit_train_ids, *(evaluate_ids if eval_holdout else [])]

    inp = pd.read_csv(input_csv, usecols=["Sample", "sr_gff_file"])
    sample_gff = dict(zip(inp["Sample"].astype(str), inp["sr_gff_file"].astype(str), strict=True))

    matrices, prevalence, read_ids = collect_upstream_matrices(
        sweep_ids, sample_gff, baclm_dir, baclm_suffix=baclm_suffix, boundary_tol=boundary_tol,
        eval_ids=eval_set, store_dtype=store_dtype, min_prevalence=min_prevalence,
    )
    # The collector already returns only core (> min_prevalence) anchor matrices; this is a belt-and-braces
    # filter and the log line for the summary.
    prev = prevalence["prevalence"]
    core = set(prevalence.loc[prev > min_prevalence, "upstream_gene"])
    core_matrices = {k: m for k, m in matrices.items() if k in core}
    logger.info("upstream core anchors > %.2f prevalence over %d fit genomes: %d (of %d total anchors)",
                min_prevalence, len(read_ids), len(core_matrices), len(prevalence))

    fitted = fit_per_gene(core_matrices, label_map, n_folds=n_folds, seed=seed, n_jobs=n_jobs,
                          all_ids=read_ids, impute_absent_zero=False, eval_ids=eval_set)
    filtered = {k for k, f in fitted.items() if f["auroc"] > auroc_filter}

    out_dir.mkdir(parents=True, exist_ok=True)
    prev_map = dict(zip(prevalence["upstream_gene"], prevalence["prevalence"], strict=True))
    rows = [
        {"upstream_gene": k, "gene": k.split("upstream:", 1)[-1], "prevalence": prev_map.get(k, float("nan")),
         f"lr_auroc_{drug}": f["auroc"], f"eval_auroc_{drug}": f.get("eval_auroc", float("nan")),
         "n_train": f["n_train"], "n_pos": f["n_pos"], "n_eval": f.get("n_eval", 0),
         "n_eval_pos": f.get("n_eval_pos", 0), "kept_filtered": k in filtered}
        for k, f in sorted(fitted.items(), key=lambda kv: kv[1]["auroc"], reverse=True)
    ]
    table = out_dir / f"per_upstream_lr_{drug}.csv"
    pd.DataFrame(rows).to_csv(table, index=False)
    prevalence.to_csv(out_dir / "upstream_prevalence.csv", index=False)

    best = max(fitted.items(), key=lambda kv: kv[1]["auroc"], default=(None, None))
    summary = {
        "analysis": "build_upstream_region_lr_store", "drug": drug, "split_csv": str(split_csv),
        "eval_holdout": eval_holdout, "n_train_fit": len(fit_train_ids), "n_read_genomes": len(read_ids),
        "n_evaluate": len(evaluate_ids) if eval_holdout else 0, "boundary_tol": boundary_tol,
        "min_prevalence": min_prevalence, "n_anchors": len(matrices), "n_core": len(core_matrices),
        "n_fitted": len(fitted), "auroc_filter": auroc_filter, "n_filtered": len(filtered),
        "best_anchor": best[0], "best_auroc": best[1]["auroc"] if best[1] else None,
    }
    (out_dir / "per_upstream_lr_build_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("build_summary: %s", json.dumps({k: summary[k] for k in (
        "n_anchors", "n_core", "n_fitted", "n_filtered", "best_anchor", "best_auroc")}))
    return summary


def main() -> None:
    """CLI entry point (mirrors build_per_igr_lr_store's args)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--species", choices=["tb", "kp"], default="tb")
    p.add_argument("--split-csv", type=Path)
    p.add_argument("--drug", type=str, default="ethionamide")
    p.add_argument("--input-csv", type=Path)
    p.add_argument("--baclm-dir", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--min-prevalence", type=float, default=0.10)
    p.add_argument("--auroc-filter", type=float, default=0.8)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--max-train-genomes", type=int, default=None)
    p.add_argument("--sample-seed", type=int, default=1)
    p.add_argument("--boundary-tol", type=int, default=3)
    p.add_argument("--eval-holdout", action="store_true",
                   help="Real held-out-test numbers: fit each anchor's LR on train+validate and additionally "
                        "report eval_auroc_<drug> on the untouched evaluate split (vs the OOF-only default).")
    p.add_argument("--store-dtype", choices=["float32", "float16"], default="float32",
                   help="Design-matrix storage precision. float16 halves the full-cohort footprint (the LR "
                        "still fits in float32); use for whole-cohort --eval-holdout runs.")
    args = p.parse_args()

    sp = store_paths(args.species)
    split_csv = args.split_csv or sp.ast_sheet
    input_csv = args.input_csv or sp.input_csv
    baclm_dir = args.baclm_dir or sp.baclm_dir
    run(split_csv=split_csv, drug=args.drug, input_csv=input_csv, baclm_dir=baclm_dir, out_dir=args.out_dir,
        min_prevalence=args.min_prevalence, auroc_filter=args.auroc_filter, n_folds=args.n_folds, seed=args.seed,
        n_jobs=args.n_jobs, max_train_genomes=args.max_train_genomes, sample_seed=args.sample_seed,
        boundary_tol=args.boundary_tol, eval_holdout=args.eval_holdout, store_dtype=args.store_dtype)


if __name__ == "__main__":
    main()

"""Per-unit (named non-CDS body) LR ranking — the RNA/element screen over the baclm re-embed store.

The third non-coding sibling of :mod:`build_per_igr_lr_store` (flank-pair key) and
:mod:`build_upstream_region_lr_store` (``upstream:<gene>`` key). Those two key the **whole-IGR**
channel (``noncoding_*``) by synteny; this one keys the **named-body** channel (``feature_*``) by the
element's own identity — ``<feature_type>:<feature_name>`` (e.g. ``rrna:rrs``, ``rrna:rrl``,
``regulatory_region:<name>``, ``crispr:<name>``). It is the screen that can finally ask "does the
16S/23S rRNA body itself predict resistance?" — ``rrs``→streptomycin/kanamycin, ``rrl``→azithromycin —
which the synteny keys structurally cannot, because an rRNA gene is *carved out* of the whole-IGR run
and never named as a flank.

The named bodies live only in the **2d re-embed** store (``baclm_reembed/``) as ``feature_embeddings``
[n_feat, 960] + parallel ``feature_type`` / ``feature_name`` / ``feature_seqid`` lists (written by
:mod:`bacpredict.engine.embedding.extract_intergenic_from_gff_fna` →
:mod:`bacpredict.engine.embedding.baclm_embed`). The legacy ``baclm/`` store has no ``feature_*`` keys,
so ``--baclm-dir`` **must** point at the re-embed dir; a genome read from the stale store yields no
units and the run logs a prominent "0 units — wrong store?" warning.

**Relaxed single-copy gate (the key difference from the synteny siblings).** rRNA is multi-copy — a
genome carries several *rrn* operons, so ``rrna:rrs`` occurs many times. The flank/upstream screens keep
only single-copy anchors and would drop every rRNA; here we instead **mean-pool all copies of a unit
within a genome** into one per-genome vector (copies of ``rrs`` are near-identical, so the mean is a
stable representation, and one-row-per-genome avoids the pseudo-replication that per-copy rows would
inject into the LR). Prevalence is then simply the fraction of read genomes carrying the unit at all
(≥1 copy).

Everything else is reused verbatim from the per-gene harness (:func:`fit_per_segment` /
:func:`fit_one_segment` / :func:`fit_one_segment_imputed`) and mirrors the upstream sibling's three fit modes:
carrier-only (default; drop-absent, present-conditioned), ``--impute-absent-zero`` (fit over the full
read universe, zero-imputing absent genomes — the selection = usage AUROC the concat's zero-imputed
block consumes), and ``--feature presence`` (the presence/absence one-hot lineage control). Imputed and
presence rankings route to their own ``--out-dir`` so the carrier-only file is never overwritten;
presence writes ``per_unit_presence_lr_<drug>.csv``.

Output (mirrors the sibling ranking contract; ``feature_type``/``feature_name`` columns so the shared
:mod:`bacpredict.engine.plots.plot_igr_lr_ranking` labels rows out of the box):

    <out-dir>/per_unit_lr_<drug>.csv     # unit,feature_type,feature_name,prevalence,lr_auroc_<drug>,…
    <out-dir>/unit_prevalence.csv
    <out-dir>/per_unit_lr_build_summary.json

Fragments (``fragment_*``, the anonymous inter-feature spacers) are **not** screened here — they carry
no type/name, so they are not body-keyable; the fragmented-spacer *promoter* view is the job of the
synteny-keyed modules pointed at that channel, not of this per-unit body screen.

CPU-only (``.pt`` reads + sklearn LRs over precomputed baclm vectors). No GFF parse — units self-identify
from the store — so the read sweep is lighter than the flank/upstream siblings.
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
from bacpredict.engine.finetune.holdout import load_splits
from bacpredict.engine.gene_lr.build_per_gene_lr_store import fit_per_segment, subsample_balanced

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _read_features(pt_path: Path) -> tuple[np.ndarray, list[str], list[str]] | None:
    """Load one re-embed store's named-body rows: ``(emb[n, dim], feature_types, feature_names)`` or ``None``.

    ``None`` means the genome is unreadable *as a feature source* — the ``.pt`` is missing, has no
    ``feature_embeddings`` key (the legacy ``baclm/`` store), or its parallel type/name lists are
    length-mismatched (a schema break). A readable store with **zero** feature rows returns empty arrays,
    not ``None``: the genome is a valid, feature-less member of the read universe (a genuine absence for
    the zero-impute fit), distinct from a genome we could not read at all.
    """
    if not pt_path.exists():
        return None
    store = torch.load(pt_path, map_location="cpu", mmap=True, weights_only=True)
    if "feature_embeddings" not in store:
        return None
    emb = store["feature_embeddings"]
    n = int(emb.shape[0]) if emb is not None else 0
    ftypes = [str(t) for t in store.get("feature_type", [])]
    fnames = [str(nm) for nm in store.get("feature_name", [])]
    if len(ftypes) != n or len(fnames) != n:
        return None
    if n == 0:
        return np.zeros((0, 1), dtype=np.float32), [], []
    return emb.float().numpy(), ftypes, fnames


def _unit_key(ftype: str, fname: str) -> str:
    """``<feature_type>:<feature_name>`` — type lower-cased, name stripped (``unnamed`` if blank)."""
    return f"{ftype.strip().lower()}:{fname.strip() or 'unnamed'}"


def _genome_unit_records(
    sid: str, pt_path: str, type_filter: set[str] | None = None
) -> tuple[str, list[tuple[str, np.ndarray]]] | None:
    """One genome's ``[(unit_key, mean_pooled_embedding)]`` — one row per unit, copies mean-pooled.

    Multi-copy bodies (the several *rrn* copies of ``rrna:rrs``) are averaged into a single per-genome
    vector, so a genome contributes at most one row per unit. ``type_filter`` (lower-cased feature types)
    restricts to a subset of the vocabulary (e.g. ``{"rrna"}``); ``None`` keeps every named body.
    """
    read = _read_features(Path(pt_path))
    if read is None:
        return None
    emb, ftypes, fnames = read
    by_key: dict[str, list[np.ndarray]] = {}
    for i, (ftype, fname) in enumerate(zip(ftypes, fnames, strict=True)):
        if type_filter is not None and ftype.strip().lower() not in type_filter:
            continue
        by_key.setdefault(_unit_key(ftype, fname), []).append(emb[i])
    records = [(k, np.mean(np.vstack(v), axis=0).astype(np.float32)) for k, v in by_key.items()]
    return sid, records


def collect_unit_matrices(
    train_ids: list[str], baclm_dir: Path, *, baclm_suffix: str = "_baclm_embeddings.pt",
    eval_ids: set[str] | None = None, store_dtype: str = "float32",
    min_prevalence: float = 0.0, max_prevalence: float = 1.0, unit_types: set[str] | None = None,
) -> tuple[dict[str, tuple[list[str], np.ndarray]], pd.DataFrame, list[str]]:
    """``.pt`` sweep → per-unit (mean-pooled, one-row-per-genome) design matrices + prevalence + universe.

    **Two-pass, streamed** (mirrors the upstream collector): pass 1 tallies each unit's prevalence over
    the fit genomes (one count per carrying genome, copies already pooled) and keeps only units in the
    ``(min_prevalence, max_prevalence]`` band; pass 2 re-reads and stores the mean-pooled vectors for
    those in-band units. ``eval_ids`` genomes are swept in pass 2 and appended to each unit's matrix (so
    the fitted LR can be scored on them) but excluded from the pass-1 prevalence / band selection and from
    the returned ``read_ids`` universe — the evaluate split never influences which units are screened.
    """
    eval_ids = eval_ids or set()
    tasks = [(str(s), str(Path(baclm_dir) / f"{s}{baclm_suffix}")) for s in train_ids]
    fit_tasks = [t for t in tasks if t[0] not in eval_ids]

    # Pass 1 (fit genomes only) — prevalence → in-band units; vectors discarded as we go.
    present: Counter[str] = Counter()
    unit_meta: dict[str, tuple[str, str]] = {}
    read_ids: list[str] = []
    n_skipped = 0
    for sid, pt in fit_tasks:
        res = _genome_unit_records(sid, pt, type_filter=unit_types)
        if res is None:
            n_skipped += 1
            continue
        _sid, records = res
        read_ids.append(sid)
        present.update(k for k, _ in records)
        for key, _vec in records:
            unit_meta.setdefault(key, tuple(key.split(":", 1)))  # (feature_type, feature_name)
    if n_skipped:
        logger.warning("per-unit pass 1: skipped %d fit genomes (missing/unreadable/no feature_* keys)", n_skipped)

    n = len(read_ids)
    prevalence = pd.DataFrame(
        [{"unit": k, "feature_type": unit_meta[k][0], "feature_name": unit_meta[k][1],
          "n_present": c, "prevalence": c / max(n, 1)} for k, c in present.items()],
        columns=["unit", "feature_type", "feature_name", "n_present", "prevalence"],  # survive an empty sweep
    ).sort_values("prevalence", ascending=False).reset_index(drop=True)
    core = {k for k, c in present.items() if min_prevalence < (c / max(n, 1)) <= max_prevalence}
    logger.info("per-unit pass 1: %d units in prevalence band (%.2f, %.2f] (of %d) over %d fit genomes",
                len(core), min_prevalence, max_prevalence, len(present), n)

    # Pass 2 (fit + eval) — store mean-pooled vectors for in-band units only (eval rows scored, not selected).
    ids_by_key: dict[str, list[str]] = {}
    vecs_by_key: dict[str, list[np.ndarray]] = {}
    for sid, pt in tasks:
        res = _genome_unit_records(sid, pt, type_filter=unit_types)
        if res is None:
            continue
        _sid, records = res
        for key, vec in records:
            if key in core:
                ids_by_key.setdefault(key, []).append(sid)
                vecs_by_key.setdefault(key, []).append(vec.astype(store_dtype, copy=False))

    matrices = {k: (ids_by_key[k], np.vstack(vecs_by_key[k])) for k in ids_by_key}
    logger.info("per-unit pass 2: materialised %d in-band unit matrices", len(matrices))
    return matrices, prevalence, read_ids


def run(
    *, split_csv: Path, drug: str, baclm_dir: Path, out_dir: Path,
    min_prevalence: float = 0.0, auroc_filter: float = 0.8, n_folds: int = 5, seed: int = 1,
    n_jobs: int = 1, max_train_genomes: int | None = None, sample_seed: int = 1,
    baclm_suffix: str = "_baclm_embeddings.pt", eval_holdout: bool = False, store_dtype: str = "float32",
    impute_absent_zero: bool = False, feature: str = "embedding", max_prevalence: float = 1.0,
    unit_types: set[str] | None = None,
) -> dict:
    """Screen named non-CDS bodies, fit one LR per unit, write the wide per-unit ranking table.

    ``feature="embedding"`` fits each unit's 960-d baclm body embedding; ``feature="presence"`` fits a
    single presence/absence one-hot (the lineage control). ``eval_holdout`` additionally reports
    ``eval_auroc_<drug>`` on the untouched evaluate split (real held-out-test numbers).
    """
    label_map, train_ids, validate_ids, evaluate_ids = load_splits(split_csv, drug)
    fit_pool = [*train_ids, *validate_ids] if eval_holdout else list(train_ids)
    eval_set: set[str] | None = set(evaluate_ids) if eval_holdout else None
    fit_train_ids = subsample_balanced(fit_pool, label_map, max_n=max_train_genomes, seed=sample_seed)
    sweep_ids = [*fit_train_ids, *(evaluate_ids if eval_holdout else [])]

    matrices, prevalence, read_ids = collect_unit_matrices(
        sweep_ids, baclm_dir, baclm_suffix=baclm_suffix, eval_ids=eval_set, store_dtype=store_dtype,
        min_prevalence=min_prevalence, max_prevalence=max_prevalence, unit_types=unit_types,
    )

    presence = feature == "presence"
    impute_mode = "presence" if presence else ("imputed_zero" if impute_absent_zero else "carrier_only")
    if presence:
        # Replace each unit's embedding block with a ones-column; zero-imputing over the read universe
        # then makes the full design a 1/0 presence indicator (carrier=1, absent=0) — the pure one-hot LR.
        matrices = {k: (ids, np.ones((len(ids), 1), dtype=np.float32)) for k, (ids, _v) in matrices.items()}
        impute_absent_zero = True
    auroc_col = f"presence_lr_auroc_{drug}" if presence else f"lr_auroc_{drug}"
    table_name = f"per_unit_presence_lr_{drug}.csv" if presence else f"per_unit_lr_{drug}.csv"

    out_dir.mkdir(parents=True, exist_ok=True)
    prevalence.to_csv(out_dir / "unit_prevalence.csv", index=False)
    meta_by_unit = dict(zip(prevalence["unit"], zip(prevalence["feature_type"], prevalence["feature_name"],
                                                    strict=True), strict=True))
    prev_map = dict(zip(prevalence["unit"], prevalence["prevalence"], strict=True))

    if not matrices:
        logger.warning("per-unit: 0 units in band — is --baclm-dir the re-embed store (feature_* keys)? "
                       "The legacy baclm/ store has no named bodies.")
        pd.DataFrame(columns=["unit", "feature_type", "feature_name", "prevalence", auroc_col,
                              f"eval_auroc_{drug}", "n_train", "n_pos", "kept_filtered", "impute_mode"]).to_csv(
            out_dir / table_name, index=False)
        summary = {"analysis": "build_per_unit_lr_store", "drug": drug, "feature": feature,
                   "n_units": 0, "n_core": 0, "n_fitted": 0, "n_read_genomes": len(read_ids),
                   "best_unit": None, "best_auroc": None}
        (out_dir / "per_unit_lr_build_summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    fitted = fit_per_segment(matrices, label_map, n_folds=n_folds, seed=seed, n_jobs=n_jobs,
                          all_ids=read_ids, impute_absent_zero=impute_absent_zero, eval_ids=eval_set)
    filtered = {k for k, f in fitted.items() if f["auroc"] > auroc_filter}

    rows = [
        {"unit": k, "feature_type": meta_by_unit.get(k, ("", ""))[0],
         "feature_name": meta_by_unit.get(k, ("", ""))[1], "prevalence": prev_map.get(k, float("nan")),
         auroc_col: f["auroc"], f"eval_auroc_{drug}": f.get("eval_auroc", float("nan")),
         "n_train": f["n_train"], "n_pos": f["n_pos"], "n_eval": f.get("n_eval", 0),
         "n_eval_pos": f.get("n_eval_pos", 0), "kept_filtered": k in filtered, "impute_mode": impute_mode}
        for k, f in sorted(fitted.items(), key=lambda kv: kv[1]["auroc"], reverse=True)
    ]
    pd.DataFrame(rows).to_csv(out_dir / table_name, index=False)

    best = max(fitted.items(), key=lambda kv: kv[1]["auroc"], default=(None, None))
    summary = {
        "analysis": "build_per_unit_lr_store", "drug": drug, "split_csv": str(split_csv),
        "eval_holdout": eval_holdout, "feature": feature, "impute_absent_zero": impute_absent_zero,
        "n_train_fit": len(fit_train_ids), "n_read_genomes": len(read_ids),
        "n_evaluate": len(evaluate_ids) if eval_holdout else 0,
        "min_prevalence": min_prevalence, "max_prevalence": max_prevalence,
        "unit_types": sorted(unit_types) if unit_types else None,
        "n_units": len(prevalence), "n_core": len(matrices), "n_fitted": len(fitted),
        "auroc_filter": auroc_filter, "n_filtered": len(filtered),
        "best_unit": best[0], "best_auroc": best[1]["auroc"] if best[1] else None,
    }
    (out_dir / "per_unit_lr_build_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("build_summary: %s", json.dumps({k: summary[k] for k in (
        "n_units", "n_core", "n_fitted", "n_filtered", "best_unit", "best_auroc")}))
    return summary


def main() -> None:
    """CLI entry point (mirrors the sibling non-coding LR stores; no --input-csv — units self-identify)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--species", choices=["tb", "kp"], default="tb")
    p.add_argument("--split-csv", type=Path)
    p.add_argument("--drug", type=str, default="streptomycin")
    p.add_argument("--baclm-dir", type=Path,
                   help="Dir of *_baclm_embeddings.pt — MUST be the re-embed store (feature_* keys); the "
                        "legacy baclm/ store has no named bodies. Default: species baclm_dir (usually stale).")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--unit-types", type=str, default=None,
                   help="Comma-separated feature_type filter (e.g. 'rrna,ncrna'); default: all named bodies.")
    p.add_argument("--min-prevalence", type=float, default=0.0,
                   help="Prevalence floor over fit genomes (default 0.0 — keep every carried unit, incl. "
                        "near-universal rRNA; the synteny screens' core floor does not apply to bodies).")
    p.add_argument("--max-prevalence", type=float, default=1.0,
                   help="Prevalence ceiling (default 1.0; set e.g. 0.99 for the presence one-hot to drop "
                        "near-ubiquitous, uninformative units).")
    p.add_argument("--auroc-filter", type=float, default=0.8)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--max-train-genomes", type=int, default=None)
    p.add_argument("--sample-seed", type=int, default=1)
    p.add_argument("--eval-holdout", action="store_true",
                   help="Real held-out-test numbers: fit each unit's LR on train+validate and additionally "
                        "report eval_auroc_<drug> on the untouched evaluate split (vs the OOF-only default).")
    p.add_argument("--store-dtype", choices=["float32", "float16"], default="float32",
                   help="Design-matrix storage precision. float16 halves the full-cohort footprint (the LR "
                        "still fits in float32); use for whole-cohort --eval-holdout runs.")
    p.add_argument("--feature", choices=["embedding", "presence"], default="embedding",
                   help="embedding = fit the 960-d baclm body embedding (default); presence = fit a single "
                        "presence/absence one-hot (the lineage/synteny control).")
    p.add_argument("--impute-absent-zero", action="store_true",
                   help="Fit each unit over ALL read genomes, zero-imputing the ones lacking it, instead of "
                        "dropping absent genomes. Selection = usage: the concat feeds a zero-imputed block, so "
                        "select the non-coding rung on this AUROC (route to a distinct --out-dir).")
    args = p.parse_args()

    sp = store_paths(args.species)
    split_csv = args.split_csv or sp.ast_sheet
    baclm_dir = args.baclm_dir or sp.baclm_dir
    unit_types = {t.strip().lower() for t in args.unit_types.split(",") if t.strip()} if args.unit_types else None
    run(split_csv=split_csv, drug=args.drug, baclm_dir=baclm_dir, out_dir=args.out_dir,
        min_prevalence=args.min_prevalence, auroc_filter=args.auroc_filter, n_folds=args.n_folds, seed=args.seed,
        n_jobs=args.n_jobs, max_train_genomes=args.max_train_genomes, sample_seed=args.sample_seed,
        eval_holdout=args.eval_holdout, store_dtype=args.store_dtype, impute_absent_zero=args.impute_absent_zero,
        feature=args.feature, max_prevalence=args.max_prevalence, unit_types=unit_types)


if __name__ == "__main__":
    main()

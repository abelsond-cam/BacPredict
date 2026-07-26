"""Does baclm embed CODING regions with the information ESM holds?

A like-for-like, per-gene logistic-regression probe. For a target gene + drug it pulls each
sample's pooled gene 960-vector from **both** the ESM-C store and the baclm store and scores them
head-to-head through the canonical k=5 × s=3 harness
(:func:`bacpredict.engine.gene_lr.kfold_probe.run_kfold_probe`): within-fold CV AUROC mean ± sd on a fixed
evaluate holdout, plus the **paired** ``baclm − ESM`` delta run-for-run.

Why first: baclm's coding channel reusing ESM's information is the precondition for trusting its
**non-coding** (IGR / RNA) channel. If baclm-coding ≈ ESM-coding here, the non-coding
work is on solid ground; if not, the embedding — not the read-out — is the problem.

Both stores are 960-dim and share the same ``*_protein_sequences.parquet`` flat order, so a gene's
``protein_index`` (from :func:`bacpredict.engine.gene_lr.locate_gene.build_gene_presence_table`) indexes the
same protein row in each. The only new code here is the baclm reader: baclm's ``.pt`` holds
``protein_embeddings`` as a plain ``[n_cds, 960]`` matrix (one row per CDS in flat order, **no** batch
dim / attention mask / special tokens), so its row selection is a direct ``[flat_index]`` — unlike the
ESM store, which interleaves/pads and needs
:func:`bacpredict.engine.embedding.protein_pooling.real_protein_indices`.

CPU-only (sklearn LR over precomputed embeddings) — Stage-A smoke (``--n 10``) runs on a login node;
the full cohort is a CPU sbatch (tens of thousands of single-row mmap reads; use ``--pool-workers``).
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bacpredict.engine.config import StorePaths, store_paths
from bacpredict.engine.finetune.holdout import resolve_clean_splits
from bacpredict.engine.finetune.split_utils import generate_kfold_splits
from bacpredict.engine.gene_lr.kfold_probe import FeatureSpec, run_kfold_probe, summarise_kfold
from bacpredict.engine.gene_lr.linear_probe import fit_score_step
from bacpredict.engine.gene_lr.locate_gene import build_gene_presence_table, flatten_proteins
from bacpredict.engine.gene_lr.pooled_cds_vectors import load_pooled_gene_vectors

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Target panel — primary AMR determinant genes whose coding embedding should carry the signal.
# TB is fully specified (all drug columns exist in binary_ast_with_split.csv). Kp entries are
# chromosomal single-copy mutation targets (the true rpoB analogue — acquired plasmid genes are
# present-only in carriers, a different setup); confirm the Kp AST column names before running Kp.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProteinTarget:
    """One (gene, drug) probe. ``aliases`` are alternative gene symbols matched case-insensitively."""

    gene: str
    drug: str
    aliases: tuple[str, ...] = ()
    note: str = ""


PANEL: dict[str, list[ProteinTarget]] = {
    "tb": [
        ProteinTarget("rpoB", "rifampin", note="RRDR anchor — the whole TB story"),
        ProteinTarget("katG", "isoniazid", note="catalase-peroxidase LoF"),
        ProteinTarget("gyrA", "moxifloxacin", note="fluoroquinolone QRDR"),
        ProteinTarget("pncA", "pyrazinamide", note="pncA gene-body LoF (promoter is the 2c IGR target)"),
        ProteinTarget("embB", "ethambutol", note="arabinosyltransferase"),
        ProteinTarget("rpoC", "rifampin", note="compensatory — weak signal expected"),
    ],
    # Kp: chromosomal single-copy mutation targets. Paths + drug columns need Isambard confirmation.
    "kp": [
        ProteinTarget("gyrA", "ciprofloxacin", note="QRDR"),
        ProteinTarget("parC", "ciprofloxacin", note="QRDR"),
    ],
}


# Store paths + the species→task resolver now live in the shared organism config
# (``StorePaths`` / ``store_paths``). Kept as aliases here for call-site compatibility.
SpeciesPaths = StorePaths
default_paths = store_paths


# ---------------------------------------------------------------------------
# baclm coding-vector reader (the only new I/O — ESM reuses load_pooled_gene_vectors)
# ---------------------------------------------------------------------------

def _read_baclm_gene_one(
    sample_id: str,
    flat_index: int,
    n_expected: int | None,
    pt_path: Path,
) -> tuple[str, np.ndarray | None, str | None]:
    """Read one sample's baclm gene 960-vector from its mmap'd ``.pt`` (worker-safe).

    baclm's ``protein_embeddings`` is ``[n_cds, 960]`` in flat CDS order — no batch dim, no mask —
    so the gene is a direct ``[flat_index]`` read. The ``n_proteins`` guard mirrors the ESM path:
    the stored row count must equal the parquet's flat protein count or the index is meaningless.
    Returns ``(sample_id, vector_or_None, skip_reason)``.
    """
    if not pt_path.exists():
        return sample_id, None, "missing_pt"
    store = torch.load(pt_path, map_location="cpu", mmap=True, weights_only=True)
    prot = store["protein_embeddings"]  # [n_cds, 960]
    n_rows = int(prot.shape[0])
    if n_expected is not None and n_rows != n_expected:
        return sample_id, None, "count_mismatch"
    if flat_index >= n_rows:
        return sample_id, None, "out_of_range"
    return sample_id, prot[flat_index].float().clone().numpy(), None


def _scan_one_parquet_multi(sid, pq_path, wanted_by_gene):
    """Flatten one genome's parquet **once** and locate every target gene (worker-safe).

    Returns ``(sid, per_gene)`` where ``per_gene[gene]`` is the single-copy hit dict or ``None``
    (absent / multi-copy); ``per_gene`` itself is ``None`` if the parquet is missing.
    """
    if not pq_path.exists():
        return sid, None
    records = flatten_proteins(pd.read_parquet(pq_path))
    n_prot = len(records)
    per_gene: dict = {}
    for gene, wanted in wanted_by_gene.items():
        hits = [r for r in records if r["gene_name"] is not None and str(r["gene_name"]).lower() in wanted]
        per_gene[gene] = (
            {"protein_index": int(hits[0]["flat_index"]), "n_proteins": n_prot,
             "gene_name": hits[0]["gene_name"], "annotation": hits[0].get("protein_name")}
            if len(hits) == 1 else None
        )
    return sid, per_gene


def build_multi_gene_presence(sample_ids, parquet_dir, gene_specs, *,
                              parquet_suffix="_protein_sequences.parquet", pool_workers=1):
    """One parquet sweep → per-gene single-copy presence tables for every gene in ``gene_specs``.

    ``gene_specs`` is a list of ``(gene, aliases_tuple)``. Reads each genome's parquet **once** (the
    dominant cost) and locates all target genes, instead of :func:`build_gene_presence_table` re-reading
    the whole cohort per gene — a ~len(genes)× I/O saving for the panel. Returns ``dict[gene] → DataFrame``
    with the same schema build_gene_presence_table produces (indexed by Sample, single-copy only).
    """
    wanted_by_gene = {g: frozenset([g.lower(), *(a.lower() for a in aliases)]) for g, aliases in gene_specs}
    parquet_dir = Path(parquet_dir)
    tasks = [(str(s), parquet_dir / f"{s}{parquet_suffix}", wanted_by_gene) for s in sample_ids]
    if pool_workers > 1:
        import multiprocessing as mp

        with mp.Pool(pool_workers) as pool:
            results = pool.starmap(_scan_one_parquet_multi, tasks)
    else:
        results = [_scan_one_parquet_multi(*t) for t in tasks]

    rows_by_gene: dict = {g: [] for g, _ in gene_specs}
    n_missing = 0
    for sid, per_gene in results:
        if per_gene is None:
            n_missing += 1
            continue
        for gene, hit in per_gene.items():
            if hit is not None:
                rows_by_gene[gene].append({"Sample": sid, **hit})
    empty = pd.DataFrame(columns=["protein_index", "n_proteins", "gene_name", "annotation"]).rename_axis("Sample")
    tables = {g: (pd.DataFrame(rows).set_index("Sample") if rows else empty.copy()) for g, rows in rows_by_gene.items()}
    for g, t in tables.items():
        logger.info("multi-gene presence: %s single-copy in %d/%d genomes (missing parquet=%d)",
                    g, len(t), len(sample_ids), n_missing)
    return tables


def load_baclm_gene_vectors(
    gene_table: pd.DataFrame,
    baclm_dir: Path,
    *,
    flat_index_col: str = "protein_index",
    pt_suffix: str = "_baclm_embeddings.pt",
    pool_workers: int = 1,
) -> pd.DataFrame:
    """Baclm analogue of :func:`pooled_cds_vectors.load_pooled_gene_vectors`.

    Same signature/semantics — pulls each sample's pooled gene vector from the baclm store, mmap'd,
    one row materialised, dropping samples whose ``.pt`` is missing or whose index fails the guards.
    Returns a ``[N, 960]`` DataFrame indexed by Sample.
    """
    tasks = [
        (
            str(sample_id),
            int(row[flat_index_col]),
            int(row["n_proteins"]) if "n_proteins" in row and not pd.isna(row["n_proteins"]) else None,
            baclm_dir / f"{sample_id}{pt_suffix}",
        )
        for sample_id, row in gene_table.iterrows()
    ]
    if pool_workers > 1:
        import multiprocessing as mp

        with mp.Pool(pool_workers) as pool:
            results = pool.starmap(_read_baclm_gene_one, tasks)
    else:
        results = [_read_baclm_gene_one(*t) for t in tasks]

    skips: dict[str, int] = {}
    vectors: list[np.ndarray] = []
    kept: list[str] = []
    for sample_id, vec, reason in results:
        if reason is not None:
            skips[reason] = skips.get(reason, 0) + 1
            continue
        vectors.append(vec)
        kept.append(sample_id)
    if skips:
        logger.warning("baclm gene vectors: skipped %s", skips)
    if not vectors:
        return pd.DataFrame()
    return pd.DataFrame(np.vstack(vectors), index=pd.Index(kept, name="Sample"))


# ---------------------------------------------------------------------------
# One (gene, drug) ESM-vs-baclm comparison
# ---------------------------------------------------------------------------

@dataclass
class ComparisonResult:
    """One (gene, drug) ESM-vs-baclm comparison result (carrying the per-fold k-fold metrics)."""

    gene: str
    drug: str
    n_esm: int
    n_baclm: int
    kfold: dict = field(default_factory=dict)
    error: str | None = None


def _build_frames(
    target: ProteinTarget,
    paths: SpeciesPaths,
    *,
    pool_workers: int = 1,
    sample_limit: int | None = None,
    gene_table: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int], str | None]:
    """Resolve clean labels and build the aligned ESM + baclm gene frames for one (gene, drug).

    Shared by the k-fold panel (:func:`run_gene_comparison`) and the learning-curve ladder
    (:func:`run_gene_ladder`): both need the same pair of 960-vector frames over the same samples.
    ``gene_table`` (optional) is a pre-built presence table from :func:`build_multi_gene_presence` —
    pass it in panel mode to avoid re-reading the parquet cohort per gene; it is subset to this drug's
    labelled samples. When ``None`` the single-gene :func:`build_gene_presence_table` sweep is used.
    Returns ``(esm, baclm, label_map, error)`` — ``error`` is ``None`` on success.
    """
    label_map, *_split = resolve_clean_splits(paths.ast_sheet, target.drug)
    sample_ids = sorted(label_map)
    if sample_limit is not None:
        sample_ids = sample_ids[:sample_limit]
        label_map = {s: label_map[s] for s in sample_ids}

    if gene_table is None:
        gene_table = build_gene_presence_table(
            sample_ids, paths.parquet_dir, target.gene, aliases=target.aliases,
            parquet_suffix=paths.parquet_suffix,
        )
    else:
        gene_table = gene_table.loc[gene_table.index.intersection(sample_ids)]
    empty = pd.DataFrame()
    if gene_table.empty:
        return empty, empty, label_map, "no single-copy genomes for gene"

    esm = load_pooled_gene_vectors(
        gene_table, paths.esm_dir, pt_suffix=paths.esm_suffix, pool_workers=pool_workers,
    )
    baclm = load_baclm_gene_vectors(
        gene_table, paths.baclm_dir, pt_suffix=paths.baclm_suffix, pool_workers=pool_workers,
    )
    if esm.empty or baclm.empty:
        return esm, baclm, label_map, f"empty frame (esm={len(esm)} baclm={len(baclm)})"
    return esm, baclm, label_map, None


def run_gene_comparison(
    target: ProteinTarget,
    paths: SpeciesPaths,
    *,
    n_folds: int = 5,
    seeds: tuple[int, ...] = (1, 2, 3),
    pool_workers: int = 1,
    sample_limit: int | None = None,
    gene_table: pd.DataFrame | None = None,
) -> ComparisonResult:
    """Build ESM + baclm gene frames for one (gene, drug) and score them through the k-fold harness."""
    esm, baclm, label_map, error = _build_frames(
        target, paths, pool_workers=pool_workers, sample_limit=sample_limit, gene_table=gene_table,
    )
    if error is not None:
        return ComparisonResult(target.gene, target.drug, len(esm), len(baclm), error=error)

    specs = {
        "esm": FeatureSpec(frame=esm, kind="numeric", standardise=True),
        "baclm": FeatureSpec(frame=baclm, kind="numeric", standardise=True),
    }
    kfold = run_kfold_probe(specs, label_map, n_folds=n_folds, seeds=seeds)
    logger.info("[%s / %s]\n%s", target.gene, target.drug, summarise_kfold(kfold))
    return ComparisonResult(target.gene, target.drug, len(esm), len(baclm), kfold=kfold)


def _result_to_dict(r: ComparisonResult) -> dict:
    """Flatten a ComparisonResult to a JSON-serialisable summary (drops per-run arrays for brevity)."""
    out: dict = {"gene": r.gene, "drug": r.drug, "n_esm": r.n_esm, "n_baclm": r.n_baclm, "error": r.error}
    if r.kfold:
        frames = r.kfold["frames"]
        out["auroc"] = {name: frames[name]["aggregate"]["auroc"] for name in frames}
        out["auprc"] = {name: frames[name]["aggregate"]["auprc"] for name in frames}
        out["paired_delta_baclm_minus_esm"] = _baclm_minus_esm(r.kfold["paired_auroc_deltas"])
        out["n_evaluate"] = r.kfold["config"]["n_evaluate"]
        out["n_universe"] = r.kfold["config"]["n_universe"]
    return out


def _baclm_minus_esm(paired: dict) -> dict | None:
    """Normalise the harness's unordered pair delta to a ``baclm − ESM`` orientation.

    ``run_kfold_probe`` keys the pair by frame-insertion order (``esm__minus__baclm`` here); flip its
    sign so a positive ``mean_delta``/``win_fraction`` always means baclm beats ESM, regardless of order.
    """
    if "baclm__minus__esm" in paired:
        return paired["baclm__minus__esm"]
    d = paired.get("esm__minus__baclm")
    if d is None:
        return None
    return {
        **d,
        "mean_delta": -d["mean_delta"],
        "n_first_wins": d["n_runs"] - d["n_first_wins"],
        "win_fraction": 1.0 - d["win_fraction"],
    }


# ---------------------------------------------------------------------------
# Learning-curve ladder — AUROC vs training-set size, ESM vs baclm, fixed evaluate holdout
# ---------------------------------------------------------------------------

def _agg(values: list[float]) -> dict | None:
    """Mean / sd (ddof=1) / min / max / n over a rung's per-seed values (``None`` if empty)."""
    vals = np.asarray([v for v in values if v is not None], dtype=float)
    if vals.size == 0:
        return None
    return {
        "mean": float(vals.mean()),
        "sd": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
        "min": float(vals.min()),
        "max": float(vals.max()),
        "n": int(vals.size),
    }


def _stratified_order(ids: list[str], label_map: dict[str, int], seed: int) -> list[str]:
    """Seeded, label-stratified, **nested** ordering of ``ids``.

    Shuffles the positives and negatives separately, then interleaves them by fractional rank so that
    *any prefix* preserves the class ratio — a prefix of length ``n`` is a stratified subsample, and the
    length-``n+step`` subsample contains it (nested), which cuts rung-to-rung noise in the curve.
    """
    rng = np.random.default_rng(seed)
    pos = [s for s in ids if label_map[s] == 1]
    neg = [s for s in ids if label_map[s] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    def ranked(lst: list[str]) -> list[tuple[float, str]]:
        n = len(lst)
        return [((i + 0.5) / n, s) for i, s in enumerate(lst)]

    merged = sorted(ranked(pos) + ranked(neg), key=lambda t: t[0])
    return [s for _, s in merged]


def _ladder_grid(pool_size: int, step: int, fine_until: int) -> list[int]:
    """Training sizes to evaluate: fine ``step`` up to ``fine_until``, then ``4×step``, always full.

    The learning curve is steep at low n (where the ESM/baclm gap either closes or persists) and flat at
    high n, so we sample densely below ``fine_until`` and coarsely above it — the endpoint (full pool) is
    always included. Pass ``fine_until >= pool_size`` to force literal ``step``-increments throughout.
    """
    grid: list[int] = []
    n = step
    while n < pool_size:
        grid.append(n)
        n += step if n < fine_until else max(step, 4 * step)
    grid.append(pool_size)
    seen: set[int] = set()
    out: list[int] = []
    for x in grid:
        if 0 < x <= pool_size and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def ladder_over_frames(
    frames: dict[str, pd.DataFrame],
    label_map: dict[str, int],
    *,
    seeds: tuple[int, ...] = (1, 2, 3),
    step: int = 500,
    fine_until: int = 6000,
    evaluate_seed: int = 1,
    evaluate_fraction: float = 0.20,
    pair: tuple[str, str] | None = None,
) -> dict:
    """Generic learning curve for any set of pre-aligned feature frames on a fixed evaluate holdout.

    The evaluate holdout is pinned by ``evaluate_seed`` (identical to the k-fold harness's). For each
    rung ``n`` and seed, a label-stratified subsample of ``n`` training genomes is drawn from the shared
    pool and the *same* rows feed every frame's LR — so per-rung frame AUROCs are paired. ``pair``
    (``(a, b)``) additionally records the paired ``a − b`` AUROC per rung. Reused by the coding
    ESM-vs-baclm ladder and the IGR baclm ladder. Returns rung records + universe/pool sizes.
    """
    universe = sorted(set.intersection(*[set(f.index) for f in frames.values()]) & set(label_map))
    uni_df = pd.DataFrame({"Sample": universe})
    evaluate_ids_set, _folds = generate_kfold_splits(
        uni_df, n_folds=5, seed=evaluate_seed,
        evaluate_fraction=evaluate_fraction, evaluate_seed=evaluate_seed,
    )
    evaluate_ids = sorted(evaluate_ids_set)
    pool = [s for s in universe if s not in evaluate_ids_set]
    if len(pool) < step:
        return {"error": f"train pool ({len(pool)}) smaller than one rung ({step})",
                "n_universe": len(universe), "n_train_pool": len(pool)}

    grid = _ladder_grid(len(pool), step, fine_until)
    orders = {seed: _stratified_order(pool, label_map, seed) for seed in seeds}
    rungs: list[dict] = []
    for n_train in grid:
        scores: dict[str, list[float]] = {name: [] for name in frames}
        deltas: list[float] = []
        for seed in seeds:
            train_ids = orders[seed][:n_train]
            per_seed: dict[str, float | None] = {}
            for name, frame in frames.items():
                res = fit_score_step(
                    frame, kind="numeric", standardise=True, label_map=label_map,
                    train_ids=train_ids, validate_ids=[], evaluate_ids=evaluate_ids,
                )
                auroc = res["metrics"]["auroc"] if "metrics" in res else None
                per_seed[name] = auroc
                if auroc is not None:
                    scores[name].append(auroc)
            if pair is not None and per_seed.get(pair[0]) is not None and per_seed.get(pair[1]) is not None:
                deltas.append(per_seed[pair[0]] - per_seed[pair[1]])
        rung = {"n_train": n_train, **{name: _agg(scores[name]) for name in frames}}
        if pair is not None:
            rung[f"delta_{pair[0]}_minus_{pair[1]}"] = _agg(deltas)
        rungs.append(rung)

    n_pos = sum(1 for s in pool if label_map[s] == 1)
    return {
        "n_universe": len(universe),
        "n_evaluate": len(evaluate_ids),
        "n_train_pool": len(pool),
        "pool_pos": n_pos,
        "pool_neg": len(pool) - n_pos,
        "seeds": list(seeds),
        "step": step,
        "fine_until": fine_until,
        "rungs": rungs,
    }


def run_gene_ladder(
    target: ProteinTarget,
    paths: SpeciesPaths,
    *,
    seeds: tuple[int, ...] = (1, 2, 3),
    step: int = 500,
    fine_until: int = 6000,
    evaluate_seed: int = 1,
    evaluate_fraction: float = 0.20,
    pool_workers: int = 1,
    sample_limit: int | None = None,
    gene_table: pd.DataFrame | None = None,
) -> dict:
    """AUROC vs training-set size for ESM and baclm on one (gene, drug), on a **fixed** evaluate holdout.

    Both frames are scored on the identical holdout (pinned by ``evaluate_seed`` — the same one the
    k-fold panel uses, so the top rung is comparable to the panel's full-N point). Answers whether
    baclm's coding gap to ESM **closes with data** (data-hungry embedding) or **persists** (lower
    ceiling). Thin wrapper over :func:`ladder_over_frames` with the ESM/baclm coding frames.
    """
    esm, baclm, label_map, error = _build_frames(
        target, paths, pool_workers=pool_workers, sample_limit=sample_limit, gene_table=gene_table,
    )
    if error is not None:
        return {"gene": target.gene, "drug": target.drug, "error": error,
                "n_esm": len(esm), "n_baclm": len(baclm)}

    lad = ladder_over_frames(
        {"esm": esm, "baclm": baclm}, label_map, seeds=seeds, step=step, fine_until=fine_until,
        evaluate_seed=evaluate_seed, evaluate_fraction=evaluate_fraction, pair=("baclm", "esm"),
    )
    out = {"gene": target.gene, "drug": target.drug, "n_esm": len(esm), "n_baclm": len(baclm), **lad}
    if lad.get("error"):
        return out
    top = lad["rungs"][-1]
    if top.get("esm") and top.get("baclm"):
        logger.info("[%s / %s] full pool (n=%d): ESM %.4f  baclm %.4f  Δ=%+.4f",
                    target.gene, target.drug, top["n_train"],
                    top["esm"]["mean"], top["baclm"]["mean"], top["delta_baclm_minus_esm"]["mean"])
    return out


def plot_ladder(payload: dict, png_path: Path) -> None:
    """Render the ESM-vs-baclm learning curves (one panel per gene) with ±sd bands to ``png_path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ladders = [lad for lad in payload["ladders"] if lad.get("rungs")]
    if not ladders:
        logger.warning("no ladders with rungs to plot")
        return
    ncol = min(3, len(ladders))
    nrow = (len(ladders) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.0 * nrow), squeeze=False)
    esm_c, bac_c = "#7e3f9e", "#1e8449"
    for ax, lad in zip(axes.flat, ladders, strict=False):
        ns = [r["n_train"] for r in lad["rungs"]]
        for name, colour in (("esm", esm_c), ("baclm", bac_c)):
            m = np.array([r[name]["mean"] if r[name] else np.nan for r in lad["rungs"]])
            sd = np.array([r[name]["sd"] if r[name] else 0.0 for r in lad["rungs"]])
            ax.plot(ns, m, "-o", ms=3, color=colour, label=name)
            ax.fill_between(ns, m - sd, m + sd, color=colour, alpha=0.15)
        ax.set_title(f"{lad['gene']} / {lad['drug']}  (n={lad['n_train_pool']} train, eval={lad['n_evaluate']})",
                     fontsize=10)
        ax.set_xlabel("training genomes")
        ax.set_ylabel("evaluate AUROC")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    for ax in axes.flat[len(ladders):]:
        ax.set_visible(False)
    fig.suptitle(f"Coding baclm vs ESM — learning curves ({payload['species']})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    logger.info("wrote %s", png_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point: run one (or the whole panel of) gene ESM-vs-baclm comparison(s)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", choices=["tb", "kp"], default="tb")
    ap.add_argument("--panel", action="store_true", help="run every gene in the species panel")
    ap.add_argument("--gene", help="single gene (overrides --panel)")
    ap.add_argument("--drug", help="drug column for --gene")
    ap.add_argument("--aliases", default="", help="comma-separated alternative gene symbols")
    ap.add_argument("--ast-sheet", type=Path, help="override AST label CSV")
    ap.add_argument("--esm-dir", type=Path, help="override ESM store dir")
    ap.add_argument("--baclm-dir", type=Path, help="override baclm store dir")
    ap.add_argument("--parquet-dir", type=Path, help="override protein_sequences parquet dir")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seeds", default="1,2,3", help="comma-separated seeds")
    ap.add_argument("--pool-workers", type=int, default=1)
    ap.add_argument("--n", type=int, default=None, help="Stage-A smoke: cap #samples")
    ap.add_argument("--ladder", action="store_true",
                    help="learning-curve mode: AUROC vs training-set size (ESM vs baclm) instead of k-fold")
    ap.add_argument("--ladder-step", type=int, default=500, help="fine training-size increment (ladder)")
    ap.add_argument("--ladder-fine-until", type=int, default=6000,
                    help="use the fine step up to this n, then 4× coarser (ladder); ≥ pool forces step throughout")
    ap.add_argument("--plot", action="store_true", help="also render a PNG next to --output (ladder mode)")
    ap.add_argument("--output", type=Path, required=True, help="results JSON path")
    args = ap.parse_args()

    paths = default_paths(args.species)
    if args.ast_sheet:
        paths.ast_sheet = args.ast_sheet
    if args.esm_dir:
        paths.esm_dir = args.esm_dir
    if args.baclm_dir:
        paths.baclm_dir = args.baclm_dir
    if args.parquet_dir:
        paths.parquet_dir = args.parquet_dir

    if args.gene:
        if not args.drug:
            ap.error("--gene requires --drug")
        targets = [ProteinTarget(args.gene, args.drug, tuple(a for a in args.aliases.split(",") if a))]
    else:
        targets = PANEL[args.species]

    seeds = tuple(int(s) for s in args.seeds.split(","))

    # Panel: sweep the parquet cohort ONCE for all genes, then reuse the presence tables per (gene, drug).
    presence: dict[str, pd.DataFrame] | None = None
    if len(targets) > 1:
        all_ids = sorted(pd.read_csv(paths.ast_sheet, usecols=["Sample"])["Sample"].astype(str).unique())
        if args.n is not None:
            all_ids = all_ids[: args.n]
        gene_specs = list({(t.gene, t.aliases) for t in targets})
        logger.info("panel: one parquet sweep over %d genomes for %d genes", len(all_ids), len(gene_specs))
        presence = build_multi_gene_presence(
            all_ids, paths.parquet_dir, gene_specs,
            parquet_suffix=paths.parquet_suffix, pool_workers=args.pool_workers,
        )

    if args.ladder:
        ladders = [
            run_gene_ladder(
                t, paths, seeds=seeds, step=args.ladder_step, fine_until=args.ladder_fine_until,
                pool_workers=args.pool_workers, sample_limit=args.n,
                gene_table=(presence[t.gene] if presence is not None else None),
            )
            for t in targets
        ]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "task": "pangena_predict",
            "analysis": "coding_amr_lr_baclm_vs_esm_ladder",
            "species": args.species,
            "seeds": list(seeds),
            "ladder_step": args.ladder_step,
            "ladder_fine_until": args.ladder_fine_until,
            "sample_limit": args.n,
            "paths": {k: str(v) for k, v in vars(paths).items()},
            "ladders": ladders,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
        logger.info("wrote %s (%d gene ladders)", args.output, len(ladders))
        if args.plot:
            plot_ladder(payload, args.output.with_suffix(".png"))

        # Console headline: full-pool endpoint + does the gap close from the first rung?
        print("\n=== baclm vs ESM coding learning curve (AUROC; low-n → full pool) ===")
        for lad in ladders:
            if lad.get("error"):
                print(f"  {lad['gene']:<8} {lad['drug']:<14} ERROR: {lad['error']}")
                continue
            r0, rN = lad["rungs"][0], lad["rungs"][-1]
            d0 = r0["delta_baclm_minus_esm"]["mean"] if r0["delta_baclm_minus_esm"] else float("nan")
            dN = rN["delta_baclm_minus_esm"]["mean"] if rN["delta_baclm_minus_esm"] else float("nan")
            print(
                f"  {lad['gene']:<8} {lad['drug']:<14} "
                f"n={r0['n_train']:<6}Δ={d0:+.4f}  →  n={rN['n_train']:<6}Δ={dN:+.4f}  "
                f"(ESM {rN['esm']['mean']:.4f} / baclm {rN['baclm']['mean']:.4f}; pool={lad['n_train_pool']})"
            )
        return

    results = [
        run_gene_comparison(
            t, paths, n_folds=args.n_folds, seeds=seeds,
            pool_workers=args.pool_workers, sample_limit=args.n,
            gene_table=(presence[t.gene] if presence is not None else None),
        )
        for t in targets
    ]

    payload = {
        "schema_version": SCHEMA_VERSION,
        "task": "pangena_predict",
        "analysis": "coding_amr_lr_baclm_vs_esm",
        "species": args.species,
        "n_folds": args.n_folds,
        "seeds": list(seeds),
        "sample_limit": args.n,
        "paths": {k: str(v) for k, v in vars(paths).items()},
        "results": [_result_to_dict(r) for r in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    logger.info("wrote %s (%d gene comparisons)", args.output, len(results))

    # Console headline: baclm − ESM AUROC delta per gene (the go/no-go).
    print("\n=== baclm vs ESM coding AUROC (mean ± sd, k-fold) ===")
    for r in results:
        d = _result_to_dict(r)
        if r.error:
            print(f"  {r.gene:<8} {r.drug:<14} ERROR: {r.error}")
            continue
        e, b = d["auroc"]["esm"], d["auroc"]["baclm"]
        delta = d.get("paired_delta_baclm_minus_esm") or {}
        print(
            f"  {r.gene:<8} {r.drug:<14} ESM {e['mean']:.4f}±{e['sd']:.4f} | "
            f"baclm {b['mean']:.4f}±{b['sd']:.4f} | Δ={delta.get('mean_delta', float('nan')):+.4f} "
            f"win={delta.get('win_fraction', float('nan')):.2f} (n_univ={d['n_universe']})"
        )


if __name__ == "__main__":
    main()

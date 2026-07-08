"""Stage 2a — does baclm embed CODING regions with the information ESM holds?

A like-for-like, per-gene logistic-regression probe. For a target gene + drug it pulls each
sample's pooled gene 960-vector from **both** the ESM-C store and the baclm store and scores them
head-to-head through the canonical k=5 × s=3 harness
(:func:`pangena_predict.kfold_probe.run_kfold_probe`): within-fold CV AUROC mean ± sd on a fixed
evaluate holdout, plus the **paired** ``baclm − ESM`` delta run-for-run.

Why first: baclm's coding channel reusing ESM's information is the precondition for trusting its
**non-coding** (IGR / RNA) channel (Stages 2b–2e). If baclm-coding ≈ ESM-coding here, the non-coding
work is on solid ground; if not, the embedding — not the read-out — is the problem.

Both stores are 960-dim and share the same ``*_protein_sequences.parquet`` flat order, so a gene's
``gene_flat_index`` (from :func:`pangena_predict.locate_gene.build_gene_presence_table`) indexes the
same protein row in each. The only new code here is the baclm reader: baclm's ``.pt`` holds
``protein_embeddings`` as a plain ``[n_cds, 960]`` matrix (one row per CDS in flat order, **no** batch
dim / attention mask / special tokens), so its row selection is a direct ``[flat_index]`` — unlike the
ESM store, which interleaves/pads and needs
:func:`pangena_predict.snp_vs_esm_prediction._real_protein_indices`.

CPU-only (sklearn LR over precomputed embeddings) — Stage-A smoke (``--n 10``) runs on a login node;
the full cohort is a CPU sbatch (tens of thousands of single-row mmap reads; use ``--pool-workers``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pangena_predict.kfold_probe import FeatureSpec, run_kfold_probe, summarise_kfold
from pangena_predict.locate_gene import build_gene_presence_table, flatten_proteins
from pangena_predict.snp_vs_esm_prediction import load_pooled_gene_vectors, resolve_clean_splits

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
class GeneTarget:
    """One (gene, drug) probe. ``aliases`` are alternative gene symbols matched case-insensitively."""

    gene: str
    drug: str
    aliases: tuple[str, ...] = ()
    note: str = ""


PANEL: dict[str, list[GeneTarget]] = {
    "tb": [
        GeneTarget("rpoB", "rifampin", note="RRDR anchor — the whole TB story"),
        GeneTarget("katG", "isoniazid", note="catalase-peroxidase LoF"),
        GeneTarget("gyrA", "moxifloxacin", note="fluoroquinolone QRDR"),
        GeneTarget("pncA", "pyrazinamide", note="pncA gene-body LoF (promoter is the 2c IGR target)"),
        GeneTarget("embB", "ethambutol", note="arabinosyltransferase"),
        GeneTarget("rpoC", "rifampin", note="compensatory — weak signal expected"),
    ],
    # Kp: chromosomal single-copy mutation targets. Paths + drug columns need Isambard confirmation.
    "kp": [
        GeneTarget("gyrA", "ciprofloxacin", note="QRDR"),
        GeneTarget("parC", "ciprofloxacin", note="QRDR"),
    ],
}


@dataclass
class SpeciesPaths:
    """Per-species store locations (Isambard ``$SCRATCHDIR`` defaults; override on the CLI)."""

    ast_sheet: Path
    esm_dir: Path
    baclm_dir: Path
    parquet_dir: Path
    esm_suffix: str = "_esm_embeddings.pt"
    baclm_suffix: str = "_baclm_embeddings.pt"
    parquet_suffix: str = "_protein_sequences.parquet"


def default_paths(species: str) -> SpeciesPaths:
    """Isambard ``$SCRATCHDIR`` defaults for a species (``tb`` → ``train_tb_ast`` etc.)."""
    scratch = os.environ.get("SCRATCHDIR", "")
    task = {"tb": "train_tb_ast", "kp": "train_kleb_ast"}[species]
    root = Path(scratch) / "processed" / task
    return SpeciesPaths(
        ast_sheet=root / "binary_ast_with_split.csv",
        esm_dir=root / "esm",
        baclm_dir=root / "baclm",
        parquet_dir=root / "protein_sequences",
    )


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
            {"gene_flat_index": int(hits[0]["flat_index"]), "n_proteins": n_prot,
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
    empty = pd.DataFrame(columns=["gene_flat_index", "n_proteins", "gene_name", "annotation"]).rename_axis("Sample")
    tables = {g: (pd.DataFrame(rows).set_index("Sample") if rows else empty.copy()) for g, rows in rows_by_gene.items()}
    for g, t in tables.items():
        logger.info("multi-gene presence: %s single-copy in %d/%d genomes (missing parquet=%d)",
                    g, len(t), len(sample_ids), n_missing)
    return tables


def load_baclm_gene_vectors(
    gene_table: pd.DataFrame,
    baclm_dir: Path,
    *,
    flat_index_col: str = "gene_flat_index",
    pt_suffix: str = "_baclm_embeddings.pt",
    pool_workers: int = 1,
) -> pd.DataFrame:
    """baclm analogue of :func:`snp_vs_esm_prediction.load_pooled_gene_vectors`.

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
    gene: str
    drug: str
    n_esm: int
    n_baclm: int
    kfold: dict = field(default_factory=dict)
    error: str | None = None


def run_gene_comparison(
    target: GeneTarget,
    paths: SpeciesPaths,
    *,
    n_folds: int = 5,
    seeds: tuple[int, ...] = (1, 2, 3),
    pool_workers: int = 1,
    sample_limit: int | None = None,
    gene_table: pd.DataFrame | None = None,
) -> ComparisonResult:
    """Build ESM + baclm gene frames for one (gene, drug) and score them through the k-fold harness.

    ``gene_table`` (optional) is a pre-built presence table from :func:`build_multi_gene_presence` —
    pass it in panel mode to avoid re-reading the parquet cohort per gene; it is subset to this drug's
    labelled samples. When ``None`` the single-gene :func:`build_gene_presence_table` sweep is used.
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
    if gene_table.empty:
        return ComparisonResult(target.gene, target.drug, 0, 0, error="no single-copy genomes for gene")

    esm = load_pooled_gene_vectors(
        gene_table, paths.esm_dir, pt_suffix=paths.esm_suffix, pool_workers=pool_workers,
    )
    baclm = load_baclm_gene_vectors(
        gene_table, paths.baclm_dir, pt_suffix=paths.baclm_suffix, pool_workers=pool_workers,
    )
    if esm.empty or baclm.empty:
        return ComparisonResult(
            target.gene, target.drug, len(esm), len(baclm),
            error=f"empty frame (esm={len(esm)} baclm={len(baclm)})",
        )

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
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
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
        targets = [GeneTarget(args.gene, args.drug, tuple(a for a in args.aliases.split(",") if a))]
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

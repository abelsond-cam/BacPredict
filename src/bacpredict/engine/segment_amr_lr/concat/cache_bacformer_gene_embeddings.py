"""GPU — cache a drug's fine-tuned Bacformer genome-mean AND per-gene contextualised embeddings.

One FT forward per genome yields, in the same pass:

- **FT genome-mean** (mask-normalised mean of ``last_hidden_state`` over the real proteins) — the pool
  the deployed classification head averages over. Cached to ``ft_genome_mean_<drug>.npz`` so the
  FT-mean ⊕ ESM ladder rung can be scored CPU-only (concat probe ``--bacformer-vectors``).
- **Per-gene FT Bacformer tokens** for the **top-N genes** (by out-of-fold ESM-LR AUROC, floored at a
  threshold) from the per-gene ESM screen — the contextualised FT embedding of each of those genes,
  saved per gene (carriers only) to ``gene_emb/<gene>.npz``. The substrate for a future *multi-gene
  Bacformer* concat (inject the top-k Bacformer gene tokens instead of, or with, the ESM gene tokens).

Why top-N and not "all > 0.6": the pervasive lineage signal means 600–1700 genes clear 0.6 per drug, so
"all" would be ~26 GB/drug. Top-N (default 50, AUROC > 0.6 floor) is generous over the ~20 a multi-gene
concat would use and keeps the store ~1 GB/drug.

Reuses the engine forward helpers (``load_model`` / ``forward_inputs`` finetuned backbone,
``real_protein_indices``, ``flatten_proteins``, ``bacformer_last_hidden_state``). GPU; one drug per run.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bacpredict.engine.embedding.generate_embeddings import bacformer_last_hidden_state
from bacpredict.engine.embedding.protein_pooling import genome_mean_pool, real_protein_indices, real_protein_rows
from bacpredict.engine.gene_lr.locate_gene import flatten_proteins
from bacpredict.engine.segment_amr_lr.concat.bacformer_genome_vectors import forward_inputs, load_model
from bacpredict.engine.splits.load_splits import load_splits
from bacpredict.engine.splits.subsample import subsample_balanced

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def select_top_genes(ranking_csv: Path, drug: str, *, threshold: float, top_n: int) -> pd.DataFrame:
    """Top-``top_n`` genes with out-of-fold ESM-LR AUROC > ``threshold`` (gene_name, auroc, prevalence)."""
    df = pd.read_csv(ranking_csv)
    col = f"lr_auroc_{drug}" if f"lr_auroc_{drug}" in df.columns else next(
        c for c in df.columns if c.startswith("lr_auroc_"))
    keep = df[df[col] > threshold].sort_values(col, ascending=False).head(top_n).copy()
    keep = keep.rename(columns={col: "auroc"})[["gene_name", "auroc", "prevalence"]]
    logger.info("Top genes for %s: %d with AUROC>%.2f (capped at %d)", drug, len(keep), threshold, top_n)
    return keep.reset_index(drop=True)


def _sanitize(gene: str) -> str:
    """Filesystem-safe gene token (tet(A) -> tet_A_, blaKPC-2 -> blaKPC_2)."""
    return re.sub(r"[^A-Za-z0-9]", "_", str(gene))


def _corrected_cache_exists(out_dir: Path, drug: str, scope: str) -> bool:
    """True iff a VALID corrected cache is already on disk (for ``--skip-existing`` idempotent fan-out).

    Requires the scope-tagged ``ft_genome_mean_<drug>_<scope>.npz`` AND a ``cache_summary_<drug>.json`` whose
    ``scope`` matches and whose holdout coverage is complete (``n_holdout >= 0.95 * n_evaluate_expected``).

    Deliberately does NOT match the pre-fix **leaky** caches: those wrote the un-scoped
    ``ft_genome_mean_<drug>.npz`` and a summary with no ``scope`` field, so this returns ``False`` for them
    and ``--skip-existing`` re-forwards them (they are the bug). A partial/aborted corrected forward (short
    holdout) is also rebuilt.
    """
    npz = out_dir / f"ft_genome_mean_{drug}_{scope}.npz"
    summ = out_dir / f"cache_summary_{drug}.json"
    if not npz.exists() or not summ.exists():
        return False
    try:
        s = json.loads(summ.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if s.get("scope") != scope:
        return False
    n_exp, n_hold = s.get("n_evaluate_expected"), s.get("n_holdout")
    if n_exp and n_hold is not None and n_hold < 0.95 * n_exp:
        return False  # partial / aborted forward — rebuild it
    return True


def run(
    *,
    split_table: Path,
    drug: str,
    parquet_dir: Path,
    esm_store_dir: Path,
    checkpoint: Path | None,
    ranking_csv: Path,
    out_dir: Path,
    threshold: float,
    top_n: int,
    device: str,
    mode: str = "finetuned",
    pt_suffix: str = "_esm_embeddings.pt",
    max_samples: int | None = None,
    scope: str = "trainholdout",
    max_train_sample: int | None = 4000,
    skip_existing: bool = False,
) -> None:
    """Forward the deployed model's genomes through the FT backbone; cache the scope-tagged genome-mean + top-gene tokens.

    Split scope comes from the deployed ``<drug>_split.csv`` table (:func:`load_splits`) — the ONE source of
    truth for who is train vs the FT-unseen holdout, so the cache can never be built on a leaky CSV
    single-split. ``checkpoint`` now only selects the **FT backbone to forward** (the model), never the
    split. ``scope``:

    - ``trainholdout`` (default) — a class-balanced sample of the deployed **train** (capped at
      ``max_train_sample``) PLUS the **full** holdout. What the corrected ladder needs: fit the
      read-out LR on FT-train genome-means, test on the FT-unseen holdout.
    - ``eval`` — the holdout only (honest, ~5x cheaper, but no train side for a fit-on-train probe).

    Writes ``ft_genome_mean_<drug>_<scope>.npz`` + ``cache_summary_<drug>.json`` (scope + checkpoint +
    holdout provenance the ladder's coverage guard reads). ``skip_existing`` returns early (no GPU forward)
    when a VALID corrected cache is already present — for idempotent fan-out re-runs; it never skips the
    pre-fix leaky caches.
    """
    if skip_existing and _corrected_cache_exists(out_dir, drug, scope):
        logger.info("%s: a valid corrected cache (scope=%s) already exists in %s — skipping the GPU forward "
                    "(--skip-existing). Delete it to force a rebuild.", drug, scope, out_dir)
        return
    label_map, train_ids, validate_ids, holdout_ids = load_splits(split_table)
    holdout_set = set(holdout_ids)
    if scope == "trainholdout":
        train_sample = subsample_balanced([*train_ids, *validate_ids], label_map, max_n=max_train_sample, seed=1)
        all_ids = [*holdout_ids, *train_sample]  # holdout first so a --max-samples smoke keeps the guarded set
    elif scope == "eval":
        all_ids = list(holdout_ids)
    else:
        raise ValueError(f"scope must be 'trainholdout' or 'eval', got {scope!r}")
    if max_samples is not None:
        all_ids = all_ids[:max_samples]
    n_holdout_planned = sum(1 for s in all_ids if s in holdout_set)
    logger.info("Forwarding %d genomes for %s (scope=%s: holdout=%d, train=%d; split-table holdout=%d)",
                len(all_ids), drug, scope, n_holdout_planned,
                len(all_ids) - n_holdout_planned, len(holdout_ids))

    top = select_top_genes(ranking_csv, drug, threshold=threshold, top_n=top_n)
    top_set = set(top["gene_name"].astype(str))

    # mode="frozen" forwards the base backbone (no checkpoint) — to cache frozen tokens for the same top
    # genes, so the per-gene plot can show ESM → frozen → fine-tuned for non-AMR lineage genes too.
    model = load_model(device, mode=mode, checkpoint=checkpoint if mode == "finetuned" else None)
    model_dtype = next(model.parameters()).dtype

    mean_ids: list[str] = []
    mean_vecs: list[np.ndarray] = []
    gene_ids: dict[str, list[str]] = {g: [] for g in top_set}
    gene_vecs: dict[str, list[np.ndarray]] = {g: [] for g in top_set}
    skips: dict[str, int] = {}

    for k, sid in enumerate(all_ids, 1):
        pq = parquet_dir / f"{sid}_protein_sequences.parquet"
        pt = esm_store_dir / f"{sid}{pt_suffix}"
        if not pq.exists() or not pt.exists():
            skips["missing"] = skips.get("missing", 0) + 1
            continue
        gene_names = [r["gene_name"] for r in flatten_proteins(pd.read_parquet(pq))]
        store = torch.load(pt, map_location="cpu")
        real_idx = real_protein_indices(store, store["protein_embeddings"].shape[1])
        n_real = int(real_idx.numel())
        if n_real > len(gene_names):
            skips["misaligned"] = skips.get("misaligned", 0) + 1
            continue
        gene_names = gene_names[:n_real]
        counts = Counter(g for g in gene_names if g in top_set)

        inputs = forward_inputs(store, device, model_dtype)
        lhs = bacformer_last_hidden_state(model, inputs)
        # [n_real, dim] numpy, flat-aligned with gene_names — materialised for per-gene token indexing below.
        real_rows = real_protein_rows(lhs, real_idx, input_len=store["protein_embeddings"].shape[1]).cpu().numpy()

        mean_ids.append(str(sid))
        mean_vecs.append(genome_mean_pool(real_rows))
        for i, g in enumerate(gene_names):
            if g in top_set and counts[g] == 1:  # single-copy occurrence only
                gene_vecs[g].append(real_rows[i])
                gene_ids[g].append(str(sid))
        if k % 200 == 0:
            logger.info("  forward: %d/%d genomes (kept mean=%d)", k, len(all_ids), len(mean_ids))

    if skips:
        logger.warning("skipped genomes: %s", skips)
    if not mean_ids:
        raise RuntimeError("No genomes forwarded — check paths / .pt suffix.")

    out_dir.mkdir(parents=True, exist_ok=True)
    mean_npz = out_dir / f"ft_genome_mean_{drug}_{scope}.npz"
    np.savez(mean_npz, sample_ids=np.array(mean_ids), mean_vectors=np.vstack(mean_vecs).astype(np.float32))
    n_holdout_cached = sum(1 for s in mean_ids if s in holdout_set)
    logger.info("Wrote FT genome-mean (%d genomes: holdout=%d, train=%d) -> %s",
                len(mean_ids), n_holdout_cached, len(mean_ids) - n_holdout_cached, mean_npz)

    gene_dir = out_dir / "gene_emb"
    gene_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for _, r in top.iterrows():
        g = str(r["gene_name"])
        if not gene_ids[g]:
            continue
        san = _sanitize(g)
        np.savez(gene_dir / f"{san}.npz", sample_ids=np.array(gene_ids[g]),
                 vectors=np.vstack(gene_vecs[g]).astype(np.float32))
        manifest.append({"gene_name": g, "sanitized": san, "esm_lr_auroc": float(r["auroc"]),
                         "prevalence": float(r["prevalence"]), "n_carriers": len(gene_ids[g])})
    pd.DataFrame(manifest).to_csv(out_dir / f"top_gene_manifest_{drug}.csv", index=False)
    (out_dir / f"cache_summary_{drug}.json").write_text(json.dumps({
        "drug": drug, "mode": mode, "checkpoint": str(checkpoint) if checkpoint else None,
        "scope": scope, "split_table": str(split_table),
        "n_evaluate_expected": len(holdout_ids),
        "n_genomes": len(mean_ids), "n_holdout": n_holdout_cached,
        "holdout_ids": [s for s in mean_ids if s in holdout_set],
        "n_top_genes": len(manifest), "threshold": threshold, "top_n": top_n,
        "ranking_csv": str(ranking_csv),
    }, indent=2))
    logger.info("Wrote %d per-gene FT Bacformer stores -> %s", len(manifest), gene_dir)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-table", type=Path, required=True,
                        help="<drug>_split.csv (Sample, ast_label, split) — the deployed split table; the cache "
                             "forwards its train+holdout genomes. --bacformer-checkpoint selects only the model.")
    parser.add_argument("--drug", type=str, required=True)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--esm-store-dir", type=Path, required=True)
    parser.add_argument("--mode", type=str, default="finetuned", choices=["finetuned", "frozen"],
                        help="finetuned = the drug's FT checkpoint; frozen = base backbone (no checkpoint).")
    parser.add_argument("--bacformer-checkpoint", type=Path, default=None,
                        help="The drug's deployed FT AMR checkpoint dir (required for --mode finetuned).")
    parser.add_argument("--ranking-csv", type=Path, required=True,
                        help="per_gene_lr_<drug>.csv (imputed) — selects the top genes to cache.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--auroc-threshold", type=float, default=0.6, help="Floor for top-gene selection.")
    parser.add_argument("--top-n", type=int, default=50, help="Cap on the number of top genes cached.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--scope", choices=["trainholdout", "eval"], default="trainholdout",
                        help="trainholdout (default): deployed train-sample + full k-fold holdout (for the "
                             "fit-on-train/test-on-holdout ladder). eval: the k-fold holdout only.")
    parser.add_argument("--max-train-sample", type=int, default=4000,
                        help="Cap the class-balanced deployed-train sample in scope=trainholdout (default 4000; "
                             "plenty for a 960-d LR — keeps the TB ~30k train side cheap). The holdout is full.")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap total forwarded genomes (smoke).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip the GPU forward when a VALID corrected cache (scope-tagged npz + summary "
                             "with full holdout coverage) already exists. Never skips the pre-fix leaky "
                             "un-scoped caches — those are re-forwarded. For idempotent fan-out re-runs.")
    args = parser.parse_args()
    if args.mode == "finetuned" and args.bacformer_checkpoint is None:
        parser.error("--bacformer-checkpoint is required for --mode finetuned")
    run(
        split_table=args.split_table, drug=args.drug, parquet_dir=args.parquet_dir,
        esm_store_dir=args.esm_store_dir, checkpoint=args.bacformer_checkpoint, ranking_csv=args.ranking_csv,
        out_dir=args.out_dir, threshold=args.auroc_threshold, top_n=args.top_n, device=args.device,
        mode=args.mode, max_samples=args.max_samples, scope=args.scope, max_train_sample=args.max_train_sample,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()

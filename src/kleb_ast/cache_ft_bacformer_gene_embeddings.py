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

Reuses the snp_embeddings forward helpers (``_load_model`` / ``_forward_inputs`` finetuned backbone,
``_real_protein_indices``, ``flatten_proteins``, ``bacformer_last_hidden_state``). GPU; one drug per run.
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

from snp_embeddings.bacformer_genome_vectors import _forward_inputs, _load_model
from snp_embeddings.locate_gene import flatten_proteins
from snp_embeddings.snp_vs_esm_prediction import _real_protein_indices, resolve_clean_splits
from tl.embed.generate_embeddings import bacformer_last_hidden_state

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


def run(
    *,
    ast_sheet: Path,
    drug: str,
    parquet_dir: Path,
    esm_store_dir: Path,
    checkpoint: Path,
    ranking_csv: Path,
    out_dir: Path,
    threshold: float,
    top_n: int,
    device: str,
    pt_suffix: str = "_esm_embeddings.pt",
    max_samples: int | None = None,
) -> None:
    """Forward each labelled genome through the FT backbone; save the genome-mean + top-gene tokens."""
    _lm, train_ids, validate_ids, evaluate_ids, _info = resolve_clean_splits(ast_sheet, drug)
    all_ids = [*train_ids, *validate_ids, *evaluate_ids]
    if max_samples is not None:
        all_ids = all_ids[:max_samples]

    top = select_top_genes(ranking_csv, drug, threshold=threshold, top_n=top_n)
    top_set = set(top["gene_name"].astype(str))

    model = _load_model(device, mode="finetuned", checkpoint=checkpoint)
    model_dtype = next(model.parameters()).dtype

    mean_ids: list[str] = []
    mean_vecs: list[np.ndarray] = []
    gene_ids: dict[str, list[str]] = {g: [] for g in top_set}
    gene_vecs: dict[str, list[np.ndarray]] = {g: [] for g in top_set}
    skips: dict[str, int] = {}
    length_checked = False

    for k, sid in enumerate(all_ids, 1):
        pq = parquet_dir / f"{sid}_protein_sequences.parquet"
        pt = esm_store_dir / f"{sid}{pt_suffix}"
        if not pq.exists() or not pt.exists():
            skips["missing"] = skips.get("missing", 0) + 1
            continue
        gene_names = [r["gene_name"] for r in flatten_proteins(pd.read_parquet(pq))]
        store = torch.load(pt, map_location="cpu")
        real_idx = _real_protein_indices(store, store["protein_embeddings"].shape[1])
        n_real = int(real_idx.numel())
        if n_real > len(gene_names):
            skips["misaligned"] = skips.get("misaligned", 0) + 1
            continue
        gene_names = gene_names[:n_real]
        counts = Counter(g for g in gene_names if g in top_set)

        inputs = _forward_inputs(store, device, model_dtype)
        lhs = bacformer_last_hidden_state(model, inputs)
        lhs = lhs[0] if lhs.dim() == 3 else lhs
        if not length_checked:
            if lhs.shape[0] != store["protein_embeddings"].shape[1]:
                raise RuntimeError(
                    f"last_hidden_state length {lhs.shape[0]} != input {store['protein_embeddings'].shape[1]} "
                    f"for {sid}: gene-token indexing would be misaligned. Aborting.")
            length_checked = True
        real_rows = lhs[real_idx].float().cpu().numpy()  # [n_real, dim], flat-aligned with gene_names

        mean_ids.append(str(sid))
        mean_vecs.append(real_rows.mean(axis=0))
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
    mean_npz = out_dir / f"ft_genome_mean_{drug}.npz"
    np.savez(mean_npz, sample_ids=np.array(mean_ids), mean_vectors=np.vstack(mean_vecs).astype(np.float32))
    logger.info("Wrote FT genome-mean (%d genomes) -> %s", len(mean_ids), mean_npz)

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
        "drug": drug, "checkpoint": str(checkpoint), "n_genomes": len(mean_ids),
        "n_top_genes": len(manifest), "threshold": threshold, "top_n": top_n,
        "ranking_csv": str(ranking_csv),
    }, indent=2))
    logger.info("Wrote %d per-gene FT Bacformer stores -> %s", len(manifest), gene_dir)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ast-sheet-path", type=Path, required=True)
    parser.add_argument("--drug", type=str, required=True)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--esm-store-dir", type=Path, required=True)
    parser.add_argument("--bacformer-checkpoint", type=Path, required=True,
                        help="The drug's deployed FT AMR checkpoint dir (the FT backbone forward).")
    parser.add_argument("--ranking-csv", type=Path, required=True,
                        help="per_gene_lr_<drug>.csv (imputed) — selects the top genes to cache.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--auroc-threshold", type=float, default=0.6, help="Floor for top-gene selection.")
    parser.add_argument("--top-n", type=int, default=50, help="Cap on the number of top genes cached.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap genomes (smoke).")
    args = parser.parse_args()
    run(
        ast_sheet=args.ast_sheet_path, drug=args.drug, parquet_dir=args.parquet_dir,
        esm_store_dir=args.esm_store_dir, checkpoint=args.bacformer_checkpoint, ranking_csv=args.ranking_csv,
        out_dir=args.out_dir, threshold=args.auroc_threshold, top_n=args.top_n, device=args.device,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()

"""GPU — cache the **frozen** Bacformer genome-mean + per-AMR-gene contextualised tokens (reliable carriers).

The ``mode="frozen"`` counterpart of :mod:`kleb_ast.cache_ft_amr_proteins`. Identical extraction — one forward
per eval genome, keyed off the CARD/Kleborate ``{Sample}_amr.parquet`` sidecars — but through the **base**
``macwiatrak/bacformer-large-masked-complete-genomes`` backbone (no fine-tuned checkpoint), so the per-gene
token is the *frozen* Bacformer representation. This is the missing ingredient for Plot #5
(:mod:`kleb_ast.gene_ingredient_concat`): is the **frozen-Bacformer gene**, the **frozen-ESM gene**, or the
**fine-tuned-Bacformer gene** the better block to concat onto a genome mean?

Saved per drug to ``<out>/<drug>/``:

- ``frozen_genome_mean_<drug>.npz`` — {sample_ids, mean_vectors}: the frozen mask-mean over real proteins.
- ``frozen_amr_emb/<gene>.npz`` — {sample_ids, vectors, bakta_match}: per AMR gene (family or allele,
  per ``--grain``), its frozen token for each single-copy carrier.
- ``frozen_amr_gene_manifest_<drug>.csv`` — gene, sanitized, amr_source, n_carriers, n_bakta.

Eval-holdout only (matches the FT cache's scope so the two are directly comparable). GPU; one drug per run.
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

from bacpredict.engine.concat.bacformer_genome_vectors import forward_inputs, load_model
from bacpredict.engine.embedding.generate_embeddings import bacformer_last_hidden_state
from bacpredict.engine.gene_lr.locate_gene import flatten_proteins
from bacpredict.engine.gene_lr.snp_vs_esm_prediction import real_protein_indices, resolve_clean_splits
from kleb_ast.cache_ft_amr_proteins import _bakta_matches, _sanitize

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def run(
    *,
    ast_sheet: Path,
    drug: str,
    parquet_dir: Path,
    esm_store_dir: Path,
    sidecar_dir: Path,
    out_dir: Path,
    grain: str,
    device: str,
    pt_suffix: str = "_esm_embeddings.pt",
    max_samples: int | None = None,
) -> None:
    """Forward each eval genome through the **frozen** backbone; save the genome-mean + per-AMR-gene tokens."""
    label_col = "amr_gene_family" if grain == "family" else "amr_allele"
    _lm, _train, _val, evaluate_ids, _info = resolve_clean_splits(ast_sheet, drug)
    all_ids = list(evaluate_ids)
    if max_samples is not None:
        all_ids = all_ids[:max_samples]
    logger.info("Forwarding %d eval genomes for %s (grain=%s, FROZEN backbone)", len(all_ids), drug, grain)

    model = load_model(device, mode="frozen", checkpoint=None)
    model_dtype = next(model.parameters()).dtype

    mean_ids: list[str] = []
    mean_vecs: list[np.ndarray] = []
    gene_ids: dict[str, list[str]] = {}
    gene_vecs: dict[str, list[np.ndarray]] = {}
    gene_bakta: dict[str, list[bool]] = {}
    gene_source: dict[str, str] = {}
    skips: dict[str, int] = {}
    length_checked = False

    for k, sid in enumerate(all_ids, 1):
        pq = parquet_dir / f"{sid}_protein_sequences.parquet"
        pt = esm_store_dir / f"{sid}{pt_suffix}"
        side = sidecar_dir / f"{sid}_amr.parquet"
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

        inputs = forward_inputs(store, device, model_dtype)
        lhs = bacformer_last_hidden_state(model, inputs)
        lhs = lhs[0] if lhs.dim() == 3 else lhs
        if not length_checked:
            if lhs.shape[0] != store["protein_embeddings"].shape[1]:
                raise RuntimeError(
                    f"last_hidden_state length {lhs.shape[0]} != input {store['protein_embeddings'].shape[1]} "
                    f"for {sid}: gene-token indexing would be misaligned. Aborting.")
            length_checked = True
        real_rows = lhs[real_idx].float().cpu().numpy()  # [n_real, dim], flat-aligned

        mean_ids.append(str(sid))
        mean_vecs.append(real_rows.mean(axis=0))

        if not side.exists():
            continue
        calls = pd.read_parquet(side)
        calls = calls[calls["amr_source"].isin(["acquired", "chromosomal"])]
        calls = calls[(calls["flat_index"] >= 0) & (calls["flat_index"] < n_real)]
        counts = Counter(str(v) for v in calls[label_col].dropna())
        for _, r in calls.iterrows():
            label = str(r[label_col]) if not pd.isna(r[label_col]) else None
            if label is None or counts[label] != 1:
                continue
            gene_ids.setdefault(label, []).append(str(sid))
            gene_vecs.setdefault(label, []).append(real_rows[int(r["flat_index"])])
            gene_bakta.setdefault(label, []).append(
                _bakta_matches(r.get("bakta_gene_name"), str(r.get("amr_gene_family")), str(r.get("amr_allele"))))
            gene_source.setdefault(label, str(r["amr_source"]))
        if k % 200 == 0:
            logger.info("  forward: %d/%d genomes (kept mean=%d, genes=%d)",
                        k, len(all_ids), len(mean_ids), len(gene_ids))

    if skips:
        logger.warning("skipped genomes: %s", skips)
    if not mean_ids:
        raise RuntimeError("No genomes forwarded — check paths / .pt suffix.")

    out_dir = out_dir / drug
    out_dir.mkdir(parents=True, exist_ok=True)
    mean_npz = out_dir / f"frozen_genome_mean_{drug}.npz"
    np.savez(mean_npz, sample_ids=np.array(mean_ids), mean_vectors=np.vstack(mean_vecs).astype(np.float32))
    logger.info("Wrote frozen genome-mean (%d genomes) -> %s", len(mean_ids), mean_npz)

    gene_dir = out_dir / "frozen_amr_emb"
    gene_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for label in gene_ids:
        san = _sanitize(label)
        np.savez(gene_dir / f"{san}.npz", sample_ids=np.array(gene_ids[label]),
                 vectors=np.vstack(gene_vecs[label]).astype(np.float32),
                 bakta_match=np.array(gene_bakta[label], dtype=bool))
        manifest.append({"gene_family": label, "sanitized": san, "amr_source": gene_source[label],
                         "n_carriers": len(gene_ids[label]), "n_bakta": int(sum(gene_bakta[label]))})
    pd.DataFrame(manifest).sort_values("n_carriers", ascending=False).to_csv(
        out_dir / f"frozen_amr_gene_manifest_{drug}.csv", index=False)
    (out_dir / f"frozen_cache_summary_{drug}.json").write_text(json.dumps({
        "drug": drug, "mode": "frozen", "n_genomes": len(mean_ids),
        "n_genes": len(manifest), "grain": grain,
    }, indent=2))
    logger.info("Wrote %d per-gene frozen token stores -> %s", len(manifest), gene_dir)


def main() -> None:
    """CLI entry point."""
    rds = Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ast-sheet-path", type=Path,
                   default=rds / "processed" / "train_kleb_ast" / "binary_ast_with_split.csv")
    p.add_argument("--drug", type=str, required=True)
    p.add_argument("--parquet-dir", type=Path, default=rds / "processed" / "klebsiella_protein_sequences")
    p.add_argument("--esm-store-dir", type=Path, default=rds / "processed" / "klebsiella_esm_embeddings")
    p.add_argument("--sidecar-dir", type=Path, default=rds / "processed" / "train_kleb_ast" / "amr_annotation")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--grain", choices=["family", "allele"], default="family")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--max-samples", type=int, default=None, help="Cap genomes (smoke).")
    args = p.parse_args()
    run(
        ast_sheet=args.ast_sheet_path, drug=args.drug, parquet_dir=args.parquet_dir,
        esm_store_dir=args.esm_store_dir, sidecar_dir=args.sidecar_dir, out_dir=args.out_dir,
        grain=args.grain, device=args.device, max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()

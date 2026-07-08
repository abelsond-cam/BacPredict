"""GPU — cache a drug's fine-tuned Bacformer genome-mean + per-AMR-gene contextualised tokens (reliable).

The reliable-label counterpart of :mod:`kleb_ast.cache_ft_bacformer_gene_embeddings`. Where that selected
genes by Bakta ``gene_name`` and the per-gene ESM ranking, this keys off the **CARD/Kleborate AMR
sidecars** (:mod:`kleb_ast.annotate_amr_sidecar`): one FT forward per genome, and for every AMR protein the
sidecar identifies (by ``flat_index``) we save the **fine-tuned Bacformer contextualised token**
``last_hidden_state[flat_index]`` — grouped by AMR gene-family. So the FT side of the per-gene ESM-vs-FT
head-to-head (and the concat) is computed on the *reliable* carrier sets, not Bakta's.

Saved per drug to ``<out>/<drug>/``:

- ``ft_genome_mean_<drug>.npz`` — {sample_ids, mean_vectors}: the mask-mean over real proteins (the
  deployed pool), reused as the concat's genome context.
- ``ft_amr_emb/<family>.npz`` — {sample_ids, vectors, bakta_match}: per AMR gene-family, its FT token for
  each single-copy carrier, plus whether Bakta also named that protein (for the reliable-vs-Bakta split).
- ``amr_gene_manifest_<drug>.csv`` — family, sanitized, amr_source, n_carriers, n_bakta.

Eval-holdout only (the FT-unseen, honest scope). Reuses the pangena_predict forward helpers. GPU; one drug
per run. The CPU consumer (:mod:`kleb_ast.reliable_ft_concat`) reads these to compute reliable FT-LR + concat.
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

from pangena_predict.bacformer_genome_vectors import _forward_inputs, _load_model
from pangena_predict.locate_gene import flatten_proteins
from pangena_predict.snp_vs_esm_prediction import _real_protein_indices, resolve_clean_splits
from tl.embed.generate_embeddings import bacformer_last_hidden_state

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_NORM = re.compile(r"[^a-z0-9]")


def _sanitize(gene: str) -> str:
    """Filesystem-safe gene token (tet(A) -> tet_A_, CTX-M -> CTX_M)."""
    return re.sub(r"[^A-Za-z0-9]", "_", str(gene))


def _bakta_matches(bakta_gene_name, amr_gene_family: str, amr_allele: str) -> bool:
    """Mirror of the Phase-2a Bakta-name match (does Bakta plausibly name this CARD family?)."""
    g = _NORM.sub("", str(bakta_gene_name).lower()) if bakta_gene_name is not None else ""
    if not g or g == "nan":
        return False
    fam = _NORM.sub("", str(amr_gene_family).lower())
    allele = _NORM.sub("", str(amr_allele).lower())
    return bool(fam) and (fam in g or g in fam or g in allele)


def run(
    *,
    ast_sheet: Path,
    drug: str,
    parquet_dir: Path,
    esm_store_dir: Path,
    sidecar_dir: Path,
    checkpoint: Path,
    out_dir: Path,
    grain: str,
    device: str,
    pt_suffix: str = "_esm_embeddings.pt",
    max_samples: int | None = None,
) -> None:
    """Forward each eval genome through the FT backbone; save the genome-mean + per-AMR-family FT tokens."""
    label_col = "amr_gene_family" if grain == "family" else "amr_allele"
    _lm, _train, _val, evaluate_ids, _info = resolve_clean_splits(ast_sheet, drug)
    all_ids = list(evaluate_ids)
    if max_samples is not None:
        all_ids = all_ids[:max_samples]
    logger.info("Forwarding %d eval genomes for %s (grain=%s)", len(all_ids), drug, grain)

    model = _load_model(device, mode="finetuned", checkpoint=checkpoint)
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
        real_idx = _real_protein_indices(store, store["protein_embeddings"].shape[1])
        n_real = int(real_idx.numel())
        if n_real > len(gene_names):
            skips["misaligned"] = skips.get("misaligned", 0) + 1
            continue

        inputs = _forward_inputs(store, device, model_dtype)
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
            logger.info("  forward: %d/%d genomes (kept mean=%d, families=%d)",
                        k, len(all_ids), len(mean_ids), len(gene_ids))

    if skips:
        logger.warning("skipped genomes: %s", skips)
    if not mean_ids:
        raise RuntimeError("No genomes forwarded — check paths / .pt suffix.")

    out_dir = out_dir / drug
    out_dir.mkdir(parents=True, exist_ok=True)
    mean_npz = out_dir / f"ft_genome_mean_{drug}.npz"
    np.savez(mean_npz, sample_ids=np.array(mean_ids), mean_vectors=np.vstack(mean_vecs).astype(np.float32))
    logger.info("Wrote FT genome-mean (%d genomes) -> %s", len(mean_ids), mean_npz)

    gene_dir = out_dir / "ft_amr_emb"
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
        out_dir / f"amr_gene_manifest_{drug}.csv", index=False)
    (out_dir / f"cache_summary_{drug}.json").write_text(json.dumps({
        "drug": drug, "checkpoint": str(checkpoint), "n_genomes": len(mean_ids),
        "n_families": len(manifest), "grain": grain,
    }, indent=2))
    logger.info("Wrote %d per-family FT token stores -> %s", len(manifest), gene_dir)


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
    p.add_argument("--bacformer-checkpoint", type=Path, required=True,
                   help="The drug's deployed FT AMR checkpoint dir (the FT backbone forward).")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--grain", choices=["family", "allele"], default="family")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--max-samples", type=int, default=None, help="Cap genomes (smoke).")
    args = p.parse_args()
    run(
        ast_sheet=args.ast_sheet_path, drug=args.drug, parquet_dir=args.parquet_dir,
        esm_store_dir=args.esm_store_dir, sidecar_dir=args.sidecar_dir,
        checkpoint=args.bacformer_checkpoint, out_dir=args.out_dir, grain=args.grain,
        device=args.device, max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()

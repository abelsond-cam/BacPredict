"""GPU — Bacformer contextualised gene-token vectors for a PANEL of genes, one forward pass per genome.

The driver panel needs Bacformer's contextualised token for many driver genes. A full Bacformer forward
already produces every gene's token, so we sweep each genome **once** and extract the token of every
panel gene present (single-copy) in it — not one full sweep per gene. Reuses the frozen-model loader,
forward-input builder, and flat-index guards from :mod:`bacpredict.engine.segment_amr_lr.concat.bacformer_genome_vectors`.

Output NPZ carries, per gene, its sample list + token matrix: keys ``<gene>__ids`` (str array) and
``<gene>__tok`` (``[N, dim]`` float32), consumed by :mod:`bacpredict.engine.plots.driver_panel`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bacpredict.engine.embedding.generate_embeddings import bacformer_last_hidden_state
from bacpredict.engine.embedding.protein_pooling import real_protein_indices, real_protein_rows
from bacpredict.engine.gene_lr.card_gene_locator import build_card_presence, sidecar_dir_available
from bacpredict.engine.gene_lr.coding_amr_lr import build_multi_gene_presence
from bacpredict.engine.segment_amr_lr.concat.bacformer_genome_vectors import forward_inputs, load_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _per_sample_genes(presence: dict[str, pd.DataFrame]) -> dict[str, dict[str, tuple[int, int]]]:
    """Invert per-gene presence tables → ``{sample: {gene: (flat_index, n_proteins)}}`` for one-pass extraction."""
    per_sample: dict[str, dict[str, tuple[int, int]]] = {}
    for gene, table in presence.items():
        for sample_id, row in table.iterrows():
            per_sample.setdefault(str(sample_id), {})[gene] = (int(row["gene_flat_index"]), int(row["n_proteins"]))
    return per_sample


@torch.no_grad()
def sweep_gene_tokens(
    per_sample: dict[str, dict[str, tuple[int, int]]],
    esm_store_dir: Path,
    *,
    device: str,
    pt_suffix: str = "_esm_embeddings.pt",
) -> dict[str, tuple[list[str], np.ndarray]]:
    """One Bacformer forward per genome → every panel gene's contextualised token.

    Returns ``{gene: (sample_ids, token_matrix [N, dim])}``. Genomes whose ``.pt`` is missing or whose
    real-protein count fails the flat-order guard are skipped (per gene). A day-one length assertion
    confirms ``last_hidden_state`` aligns 1:1 with the input rows before any token is trusted.
    """
    model = load_model(device, mode="frozen", checkpoint=None)
    model_dtype = next(model.parameters()).dtype
    tokens: dict[str, list[np.ndarray]] = {}
    ids: dict[str, list[str]] = {}
    skips: dict[str, int] = {}

    for i, (sample_id, genes) in enumerate(per_sample.items(), 1):
        pt_path = esm_store_dir / f"{sample_id}{pt_suffix}"
        if not pt_path.exists():
            skips["missing_pt"] = skips.get("missing_pt", 0) + 1
            continue
        store = torch.load(pt_path, map_location="cpu")
        input_len = store["protein_embeddings"].shape[1]
        real_idx = real_protein_indices(store, input_len)
        # Any gene whose recorded n_proteins disagrees with this store is unsafe to index.
        usable = {g: fi for g, (fi, n_exp) in genes.items() if real_idx.numel() == n_exp and fi < real_idx.numel()}
        if not usable:
            skips["count_mismatch_or_oor"] = skips.get("count_mismatch_or_oor", 0) + 1
            continue
        inputs = forward_inputs(store, device, model_dtype)
        lhs = bacformer_last_hidden_state(model, inputs)
        real_rows = real_protein_rows(lhs, real_idx, input_len=input_len)
        for gene, flat_index in usable.items():
            tokens.setdefault(gene, []).append(real_rows[flat_index].cpu().numpy())
            ids.setdefault(gene, []).append(str(sample_id))
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        if i % 500 == 0:
            logger.info("swept %d/%d genomes", i, len(per_sample))

    if skips:
        logger.warning("Bacformer panel sweep: skipped %s", skips)
    return {g: (ids[g], np.vstack(tokens[g])) for g in tokens}


def _derive_panel_genes(csv_dir: Path, csv_prefix: str, csv_suffix: str = "") -> list[str]:
    """Union of embeddable (coding) driver genes across every per-drug driver CSV in ``csv_dir``.

    Discovers each ``<csv-dir>/<drug>/`` folder by the presence of its driver CSV (robust to the
    org-parented ``visualisations/<organism>/<drug>/`` layout, no folder prefix). Schema-agnostic:
    gates on ``embeddable`` and excludes ``is_noncoding``/``is_rrna`` (works for both the TB-Profiler
    and Kp-CARD CSVs), rather than a ``region == coding`` string.
    """
    genes: set[str] = set()
    for folder in sorted(p for p in csv_dir.iterdir() if p.is_dir()):
        drug = folder.name
        csv = folder / f"{csv_prefix}_{drug}{csv_suffix}.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        if "embeddable" not in df.columns:
            continue
        keep = df["embeddable"].astype(bool) & ~df.get("is_noncoding", False).astype(bool) & ~df.get("is_rrna", False).astype(bool)
        genes.update(str(g) for g in df.loc[keep, "gene_name"] if not str(g).startswith("__ALL"))
    return sorted(genes)


def main() -> None:
    """CLI: sweep Bacformer gene tokens for a panel of genes over the labelled cohort → NPZ."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ast-sheet-path", type=Path, required=True,
                    help="binary_ast_with_split.csv — its full Sample set is the (drug-agnostic) sweep universe.")
    ap.add_argument("--parquet-dir", type=Path, required=True, help="dir of *_protein_sequences.parquet.")
    ap.add_argument("--esm-store-dir", type=Path, required=True, help="dir of *_esm_embeddings.pt (Bacformer inputs).")
    ap.add_argument("--output-npz", type=Path, required=True, help="NPZ of per-gene {<gene>__ids, <gene>__tok}.")
    ap.add_argument("--genes", nargs="*", default=None, help="explicit gene list (else derive from --csv-dir).")
    ap.add_argument("--csv-dir", type=Path, default=None, help="driver-CSV dir to derive the coding-gene union from.")
    ap.add_argument("--csv-prefix", default="tbprofiler_gene_lr")
    ap.add_argument("--csv-suffix", default="")
    ap.add_argument("--amr-sidecar-dir", type=Path, default=None,
                    help="dir of {Sample}_amr.parquet CARD sidecars — locate genes by CARD family (Kp).")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--pool-workers", type=int, default=8)
    args = ap.parse_args()

    genes = args.genes or (
        _derive_panel_genes(args.csv_dir, args.csv_prefix, args.csv_suffix)
        if args.csv_dir else None)
    if not genes:
        raise SystemExit("no genes: pass --genes or --csv-dir")
    logger.info("panel genes (%d): %s", len(genes), ", ".join(genes))

    # Drug-agnostic: gene tokens are the same regardless of drug, so cover every sample in the split.
    all_ids = sorted(pd.read_csv(args.ast_sheet_path)["Sample"].astype(str).unique())
    specs = [(g, ()) for g in genes]
    if args.amr_sidecar_dir is not None and sidecar_dir_available(args.amr_sidecar_dir, all_ids):
        logger.info("locating genes by CARD family from sidecars in %s", args.amr_sidecar_dir)
        presence = build_card_presence(all_ids, args.amr_sidecar_dir, args.parquet_dir, specs, pool_workers=args.pool_workers)
    else:
        presence = build_multi_gene_presence(all_ids, args.parquet_dir, specs, pool_workers=args.pool_workers)
    per_sample = _per_sample_genes(presence)
    logger.info("sweeping Bacformer over %d genomes for %d genes on %s", len(per_sample), len(genes), args.device)

    result = sweep_gene_tokens(per_sample, args.esm_store_dir, device=args.device)
    if not result:
        raise RuntimeError("no gene tokens recovered — check esm_store_dir / .pt suffix.")
    payload: dict[str, np.ndarray] = {}
    for gene, (sids, mat) in result.items():
        payload[f"{gene}__ids"] = np.array(sids)
        payload[f"{gene}__tok"] = mat.astype(np.float32)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, **payload)
    logger.info("wrote %d gene token matrices to %s", len(result), args.output_npz)


if __name__ == "__main__":
    main()

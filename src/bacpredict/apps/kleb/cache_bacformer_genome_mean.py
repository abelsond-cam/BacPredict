"""GPU — cache the Kp frozen Bacformer genome-mean per sample (drug-agnostic), for the CPU concat sweep.

The **genome mean** is the mask-normalised mean of Bacformer's ``last_hidden_state`` over the real
protein tokens — exactly the pool the genome-classification head averages, and **gene/drug-agnostic**.
Computing it once over the whole Kp cohort lets every per-drug concat probe + ladder
(:mod:`bacpredict.engine.concat.concatenate_bacformer_genome_esm_protein_emb`) run **CPU-only** via
``--bacformer-vectors``, the lesson from the TB sweep.

Parquet-free: unlike :mod:`bacpredict.engine.concat.bacformer_genome_vectors` (which also extracts a gene token at
a gene's flat index, needing the protein parquet and a single-copy filter), the genome mean needs only
the ESM-C store's real-protein rows — so this covers *every* Kp genome, with no gene-presence filter.
Reuses the shared forward helpers (``load_model`` / ``forward_inputs`` / ``real_protein_indices`` /
``bacformer_last_hidden_state``) so the model load, dtype casts, and flat-order guard match the deployed
evaluator exactly. Output NPZ ``{sample_ids, mean_vectors}`` is the ``--bacformer-vectors`` contract.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bacpredict.engine.concat.bacformer_genome_vectors import forward_inputs, load_model
from bacpredict.engine.config import KP, resolve_data_root
from bacpredict.engine.embedding.generate_embeddings import bacformer_last_hidden_state
from bacpredict.engine.gene_lr.snp_vs_esm_prediction import real_protein_indices

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

AST_SAMPLE_ALIASES = ("Sample", "phenotype-BioSample_ID", "sample_accession")


def load_sample_ids(ast_sheet: Path) -> list[str]:
    """Sorted unique Sample IDs from the AST sheet (drug-agnostic — every labelled Kp genome)."""
    df = pd.read_csv(ast_sheet, low_memory=False)
    col = next((c for c in AST_SAMPLE_ALIASES if c in df.columns), None)
    if col is None:
        raise ValueError(f"{ast_sheet} has no sample-id column (looked for {AST_SAMPLE_ALIASES}).")
    return sorted(df[col].astype(str).unique())


def compute_genome_means(
    sample_ids: list[str], esm_store_dir: Path, *, device: str, pt_suffix: str = "_esm_embeddings.pt"
) -> tuple[np.ndarray, list[str]]:
    """Frozen Bacformer genome-mean per genome → ``(mean_matrix [N, dim], kept_sample_ids)``.

    One forward pass per genome; the mean is over the real-protein rows (``real_protein_indices``).
    A day-one guard asserts ``last_hidden_state`` aligns 1:1 with the input rows. Genomes whose ``.pt``
    is missing or has no real proteins are skipped (counted in the warning).
    """
    model = load_model(device, mode="frozen", checkpoint=None)
    model_dtype = next(model.parameters()).dtype

    means: list[np.ndarray] = []
    kept: list[str] = []
    skips: dict[str, int] = {}
    length_checked = False
    for k, sid in enumerate(sample_ids, 1):
        pt_path = esm_store_dir / f"{sid}{pt_suffix}"
        if not pt_path.exists():
            skips["missing_pt"] = skips.get("missing_pt", 0) + 1
            continue
        store = torch.load(pt_path, map_location="cpu")
        input_len = store["protein_embeddings"].shape[1]
        real_idx = real_protein_indices(store, input_len)
        if real_idx.numel() == 0:
            skips["no_real_proteins"] = skips.get("no_real_proteins", 0) + 1
            continue

        inputs = forward_inputs(store, device, model_dtype)
        lhs = bacformer_last_hidden_state(model, inputs)
        lhs = lhs[0] if lhs.dim() == 3 else lhs
        if not length_checked:
            if lhs.shape[0] != input_len:
                raise RuntimeError(
                    f"Bacformer last_hidden_state length {lhs.shape[0]} != input length {input_len} "
                    f"for {sid}: the real-protein mask would be misaligned. Aborting."
                )
            length_checked = True
        means.append(lhs[real_idx].float().mean(dim=0).cpu().numpy())
        kept.append(str(sid))
        if k % 500 == 0:
            logger.info("  genome-mean: %d/%d genomes (kept %d)", k, len(sample_ids), len(kept))

    if skips:
        logger.warning("genome-mean: skipped %s", skips)
    mat = np.vstack(means) if means else np.empty((0, 0))
    return mat, kept


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ast-sheet-path", type=Path, default=None,
                        help="binary_ast_with_split.csv — defines the cohort (all unique Samples); "
                        "default: <data-root>/processed/train_kleb_ast/binary_ast_with_split.csv.")
    parser.add_argument("--esm-store-dir", type=Path, default=None,
                        help="Dir of {sample}_esm_embeddings.pt "
                        "(default: <data-root>/processed/klebsiella_esm_embeddings).")
    parser.add_argument("--output-npz", type=Path, default=None,
                        help="NPZ to write {sample_ids, mean_vectors} (the --bacformer-vectors contract); "
                        "default: <data-root>/processed/train_kleb_ast/bacformer_frozen_genome_mean.npz.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device (default cuda:0; cpu for smoke).")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap the cohort (smoke; default: all).")
    args = parser.parse_args()
    ast_sheet_path = args.ast_sheet_path or KP.data_root() / "binary_ast_with_split.csv"
    esm_store_dir = args.esm_store_dir or resolve_data_root() / "processed" / "klebsiella_esm_embeddings"
    output_npz = args.output_npz or KP.data_root() / "bacformer_frozen_genome_mean.npz"

    sample_ids = load_sample_ids(ast_sheet_path)
    if args.max_samples is not None:
        sample_ids = sample_ids[: args.max_samples]
    logger.info("Caching frozen Bacformer genome-mean over %d Kp genomes on %s", len(sample_ids), args.device)

    mean_mat, kept = compute_genome_means(sample_ids, esm_store_dir, device=args.device)
    if not kept:
        raise RuntimeError("No genome-mean vectors recovered — check esm-store-dir / .pt suffix.")

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_npz, sample_ids=np.array(kept), mean_vectors=mean_mat)
    logger.info("Wrote %d frozen Bacformer genome-mean vectors (dim=%d) to %s",
                len(kept), mean_mat.shape[1] if mean_mat.size else 0, output_npz)


if __name__ == "__main__":
    main()

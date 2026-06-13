"""Step 2b (bonus, GPU) — frozen Bacformer contextualised rpoB protein token.

The Step-2 probe shows the *frozen ESM-C mean-pooled* rpoB vector loses the
rifampicin signal. Step 2b asks whether Bacformer's cross-protein attention adds
anything back: it runs the frozen Bacformer complete-genomes model forward on
each genome's stored ESM inputs and pulls out the **contextualised rpoB protein
token** from ``last_hidden_state`` — the representation Bacformer's own mean-pool
classification head would average over. A linear probe on this token that is no
better than Step 2 means the loss was already sealed at the ESM-C pool (Bacformer
recovers nothing); that probe is fit downstream by
:mod:`snp_embeddings.snp_vs_esm_prediction` (``--steps bacformer_rpob_token``)
on the NPZ this script writes.

It is a *bonus*: it needs a GPU forward, so it rides with the Step-3 GPU pass.

Indexing the rpoB token
-----------------------
The rpoB token sits at the same flat-protein position used in Step 2: the real
proteins of the stored ``.pt`` in flat order (``special_tokens_mask == 4`` for the
Bacformer-input bundle, or ``attention_mask == 1`` for the plain per-protein TB
store), indexed by the rpoB flat index from the genotype table. A **day-one
assertion** confirms ``last_hidden_state`` has the same length as the input (no
silently-injected CLS shifting the index) before any vector is trusted.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from snp_embeddings.rpob_genotype import build_genotype_table, load_reference
from snp_embeddings.snp_vs_esm_prediction import _real_protein_indices, resolve_clean_splits
from tl.embed.generate_embeddings import bacformer_last_hidden_state, load_bacformer_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _forward_inputs(store: dict, device: str, model_dtype: torch.dtype) -> dict:
    """Build the Bacformer forward kwargs from a stored ``.pt`` (bundle or plain).

    Mirrors the casts the deployed evaluator uses: float embeddings/masks → the
    model dtype, integer index tensors left as-is.
    """
    if "special_tokens_mask" in store:
        # Bacformer-input bundle (from protein_embeddings_to_inputs) — its tensors
        # already are the forward kwargs; cast only the float ones.
        inputs = {}
        for k, v in store.items():
            if not torch.is_tensor(v):
                continue
            inputs[k] = v.to(device=device, dtype=model_dtype) if v.is_floating_point() else v.to(device)
        return inputs
    # Plain per-protein store — the kwargs the deployed sequence model consumes.
    contig = store.get("contig_ids", store.get("contig_idx", store.get("token_type_ids")))
    return {
        "protein_embeddings": store["protein_embeddings"].to(device=device, dtype=model_dtype),
        "attention_mask": store["attention_mask"].to(device=device, dtype=model_dtype),
        "contig_ids": contig.to(device),
    }


def compute_rpob_tokens(
    genotype,
    esm_store_dir: Path,
    *,
    device: str,
    pt_suffix: str = "_esm_embeddings.pt",
) -> tuple[np.ndarray, list[str]]:
    """Frozen-Bacformer contextualised rpoB token per single-copy genome.

    Returns ``(matrix [N, dim], sample_ids)``. Samples whose ``.pt`` is missing or
    whose real-protein count fails the flat-order guard are skipped.
    """
    model = load_bacformer_model(device, dtype="auto")
    model_dtype = next(model.parameters()).dtype

    vectors: list[np.ndarray] = []
    kept: list[str] = []
    skips: dict[str, int] = {}
    length_checked = False
    for sample_id, row in genotype.iterrows():
        pt_path = esm_store_dir / f"{sample_id}{pt_suffix}"
        if not pt_path.exists():
            skips["missing_pt"] = skips.get("missing_pt", 0) + 1
            continue
        store = torch.load(pt_path, map_location="cpu")
        input_len = store["protein_embeddings"].shape[1]
        real_idx = _real_protein_indices(store, input_len)
        n_expected = int(row["n_proteins"])
        if real_idx.numel() != n_expected:
            skips["count_mismatch"] = skips.get("count_mismatch", 0) + 1
            continue
        flat_index = int(row["rpob_flat_index"])
        if flat_index >= real_idx.numel():
            skips["out_of_range"] = skips.get("out_of_range", 0) + 1
            continue

        inputs = _forward_inputs(store, device, model_dtype)
        lhs = bacformer_last_hidden_state(model, inputs)
        lhs = lhs[0] if lhs.dim() == 3 else lhs
        if not length_checked:
            # Day-one guard: the output must align 1:1 with the input rows, or the
            # rpoB flat index points at the wrong token (e.g. an injected CLS).
            if lhs.shape[0] != input_len:
                raise RuntimeError(
                    f"Bacformer last_hidden_state length {lhs.shape[0]} != input length {input_len} "
                    f"for {sample_id}: the rpoB token index would be misaligned. Aborting Step 2b."
                )
            length_checked = True
        raw = int(real_idx[flat_index])
        vectors.append(lhs[raw].float().cpu().numpy())
        kept.append(str(sample_id))

    if skips:
        logger.warning("frozen Bacformer rpoB token: skipped %s", skips)
    matrix = np.vstack(vectors) if vectors else np.empty((0, 0))
    return matrix, kept


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ast-sheet-path", type=Path, required=True,
                        help="binary_ast_with_split.csv (defines the labelled cohort to cover).")
    parser.add_argument("--parquet-dir", type=Path, required=True, help="Dir of *_protein_sequences.parquet.")
    parser.add_argument("--esm-store-dir", type=Path, required=True, help="Dir of *_esm_embeddings.pt.")
    parser.add_argument("--output-npz", type=Path, required=True,
                        help="Where to write the {sample_ids, vectors} NPZ (consumed by --steps bacformer_rpob_token).")
    parser.add_argument("--drug", type=str, default="rifampin", help="Phenotype column (default rifampin).")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device (default cuda:0).")
    parser.add_argument("--qc-log", type=Path, default=Path("rpob_copy_qc.log"),
                        help="Where to write the rpoB-copy QC log (default: ./rpob_copy_qc.log).")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap the number of samples (smoke; default: all).")
    args = parser.parse_args()

    reference = load_reference()
    _label_map, train_ids, validate_ids, evaluate_ids, _info = resolve_clean_splits(args.ast_sheet_path, args.drug)
    all_ids = [*train_ids, *validate_ids, *evaluate_ids]
    if args.max_samples is not None:
        all_ids = all_ids[: args.max_samples]

    logger.info("Genotyping %d labelled samples (single-copy rpoB only) for the rpoB flat index", len(all_ids))
    genotype = build_genotype_table(all_ids, args.parquet_dir, reference, qc_log_path=args.qc_log)
    logger.info("Running frozen Bacformer over %d genomes on %s", len(genotype), args.device)

    matrix, kept = compute_rpob_tokens(genotype, args.esm_store_dir, device=args.device)
    if not kept:
        raise RuntimeError("No rpoB tokens recovered — check esm_store_dir / .pt suffix.")

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, sample_ids=np.array(kept), vectors=matrix)
    logger.info("Wrote %d Bacformer rpoB-token vectors to %s", len(kept), args.output_npz)


if __name__ == "__main__":
    main()

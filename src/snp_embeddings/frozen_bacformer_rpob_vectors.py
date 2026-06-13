"""GPU — frozen Bacformer contextualised vectors: the rpoB token AND the genome mean.

Runs the frozen Bacformer complete-genomes model forward on each genome's
**stored ESM-C inputs** (no re-embedding) and extracts two vectors from the same
forward pass:

- **rpoB token** — ``last_hidden_state`` at the rpoB protein index (the gene's
  contextualised representation).
- **genome mean** — the mask-normalised mean of ``last_hidden_state`` over the
  real protein tokens: exactly the pool ``BacformerGenomeClassificationHead``
  averages over, but on the **frozen** base model (no fine-tuning).

Both feed linear probes in :mod:`snp_embeddings.snp_vs_esm_prediction`
(``--steps bacformer_rpob_token bacformer_mean``), giving the comparison:

- frozen rpoB token vs the ESM-C pool (Step 2) — does Bacformer's cross-protein
  attention enrich rpoB beyond ESM-C?
- frozen genome mean vs the **fine-tuned** deployed model (~0.905) — how much did
  fine-tuning the Bacformer weights through the mean-pool head actually buy, over
  a linear probe on the *frozen* mean? (We know the mean can't be fine-tuned well;
  this quantifies the gap.)

Indexing
--------
The rpoB token sits at the flat-protein position from the genotype table: the real
proteins of the stored ``.pt`` in flat order (``special_tokens_mask == 4`` for the
Bacformer-input bundle, or ``attention_mask == 1`` for the plain per-protein TB
store). The genome mean averages over those same real-protein rows. A **day-one
assertion** confirms ``last_hidden_state`` aligns 1:1 with the input rows (no
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


def compute_bacformer_vectors(
    genotype,
    esm_store_dir: Path,
    *,
    device: str,
    pt_suffix: str = "_esm_embeddings.pt",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Frozen-Bacformer rpoB token + genome mean per single-copy genome.

    Returns ``(rpob_matrix [N, dim], mean_matrix [N, dim], sample_ids)`` — both
    pulled from the same forward pass. Samples whose ``.pt`` is missing or whose
    real-protein count fails the flat-order guard are skipped.
    """
    model = load_bacformer_model(device, dtype="auto")
    model_dtype = next(model.parameters()).dtype

    rpob_vectors: list[np.ndarray] = []
    mean_vectors: list[np.ndarray] = []
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
                    f"for {sample_id}: the rpoB token index would be misaligned. Aborting."
                )
            length_checked = True
        real_rows = lhs[real_idx].float()  # contextualised real-protein tokens
        raw = int(real_idx[flat_index])
        rpob_vectors.append(lhs[raw].float().cpu().numpy())
        # Genome mean = the mask-normalised mean the classification head pools over.
        mean_vectors.append(real_rows.mean(dim=0).cpu().numpy())
        kept.append(str(sample_id))

    if skips:
        logger.warning("frozen Bacformer vectors: skipped %s", skips)
    rpob = np.vstack(rpob_vectors) if rpob_vectors else np.empty((0, 0))
    mean = np.vstack(mean_vectors) if mean_vectors else np.empty((0, 0))
    return rpob, mean, kept


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ast-sheet-path", type=Path, required=True,
                        help="binary_ast_with_split.csv (defines the labelled cohort to cover).")
    parser.add_argument("--parquet-dir", type=Path, required=True, help="Dir of *_protein_sequences.parquet.")
    parser.add_argument("--esm-store-dir", type=Path, required=True, help="Dir of *_esm_embeddings.pt.")
    parser.add_argument("--output-npz", type=Path, required=True,
                        help="NPZ to write {sample_ids, rpob_vectors, mean_vectors} "
                             "(consumed by --steps bacformer_rpob_token / bacformer_mean).")
    parser.add_argument("--drug", type=str, default="rifampin", help="Phenotype column (default rifampin).")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device (default cuda:0).")
    parser.add_argument("--qc-log", type=Path, default=Path("rpob_copy_qc.log"),
                        help="Where to write the rpoB-copy QC log (default: ./rpob_copy_qc.log).")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap the number of samples (smoke; default: all).")
    args = parser.parse_args()

    reference = load_reference()
    _label_map, train_ids, validate_ids, evaluate_ids, _info = resolve_clean_splits(args.ast_sheet_path, args.drug)
    if args.max_samples is not None:
        # Proportional slice so all three splits stay represented downstream.
        total = len(train_ids) + len(validate_ids) + len(evaluate_ids)
        frac = min(1.0, args.max_samples / max(1, total))
        train_ids = train_ids[: max(1, round(len(train_ids) * frac))]
        validate_ids = validate_ids[: max(1, round(len(validate_ids) * frac))]
        evaluate_ids = evaluate_ids[: max(1, round(len(evaluate_ids) * frac))]
    all_ids = [*train_ids, *validate_ids, *evaluate_ids]

    logger.info("Genotyping %d labelled samples (single-copy rpoB only) for the rpoB flat index", len(all_ids))
    genotype = build_genotype_table(all_ids, args.parquet_dir, reference, qc_log_path=args.qc_log)
    logger.info("Running frozen Bacformer over %d genomes on %s", len(genotype), args.device)

    rpob, mean, kept = compute_bacformer_vectors(genotype, args.esm_store_dir, device=args.device)
    if not kept:
        raise RuntimeError("No Bacformer vectors recovered — check esm_store_dir / .pt suffix.")

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, sample_ids=np.array(kept), rpob_vectors=rpob, mean_vectors=mean)
    logger.info("Wrote %d Bacformer rpoB-token + genome-mean vectors to %s", len(kept), args.output_npz)


if __name__ == "__main__":
    main()

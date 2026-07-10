"""GPU — Bacformer contextualised genome vectors: a gene token AND the genome mean, frozen or fine-tuned.

Runs Bacformer forward on each genome's **stored ESM-C inputs** (no re-embedding) and extracts two
vectors from the same forward pass:

- **gene token** — ``last_hidden_state`` at one gene's flat protein index (default *rpoB*) — the
  gene's contextualised representation.
- **genome mean** — the mask-normalised mean of ``last_hidden_state`` over the real protein tokens:
  exactly the pool the genome-classification head averages over. **Gene-agnostic.**

Two **modes** (``--bacformer-checkpoint`` selects):

- **frozen** — the base complete-genomes model (``load_bacformer_model``).
- **fine-tuned** — the backbone of a deployed AMR checkpoint (the 0.905 mean-pool model), via
  ``load_finetuned_bacformer_backbone``.

Both feed linear probes in :mod:`pangena_predict.snp_vs_esm_prediction` (``bacformer_gene_token`` /
``bacformer_mean``) and the concat driver
:mod:`pangena_predict.concatenate_bacformer_genome_esm_protein_emb` (genome mean ⊕ ESM gene vector).

Indexing
--------
The gene token sits at the flat-protein index from the gene-presence table
(:func:`pangena_predict.locate_gene.build_gene_presence_table`): the real proteins of the stored
``.pt`` in flat order (``special_tokens_mask == 4`` for the Bacformer-input bundle, or
``attention_mask == 1`` for the plain per-protein TB store). The genome mean averages over those same
real-protein rows. A **day-one assertion** confirms ``last_hidden_state`` aligns 1:1 with the input
rows (no silently-injected CLS shifting the index) before any vector is trusted.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from pangena_predict.locate_gene import build_gene_presence_table
from pangena_predict.snp_vs_esm_prediction import real_protein_indices, resolve_clean_splits
from tl.embed.generate_embeddings import bacformer_last_hidden_state, load_bacformer_model
from tl.train.evaluate import resolve_checkpoint_dir

# Default cohort drug (TB rpoB/rifampicin). Superseded by OrganismConfig in a later refactor step.
RIFAMPIN_COLUMN = "rifampin"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def forward_inputs(store: dict, device: str, model_dtype: torch.dtype) -> dict:
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


def load_finetuned_bacformer_backbone(checkpoint: Path, device: str) -> torch.nn.Module:
    """Load the **Bacformer backbone** of a fine-tuned AMR classification checkpoint, in eval mode.

    The deployed AST model is a ``BacformerForGenomeClassification`` (backbone ``.bacformer`` — a
    ``BacformerModel`` — plus a classification head). For the fine-tuned genome mean we want the
    backbone's ``last_hidden_state`` (the same pool the head averages), with the *fine-tuned* weights.
    Loads exactly as :mod:`tl.train.evaluate` does (``trust_remote_code``, ``torch_dtype="auto"``,
    ``.float()`` on CPU so Stage-A smokes work), resolves the best ``checkpoint-*`` subdir, and returns
    ``model.bacformer``.
    """
    from transformers import AutoModelForSequenceClassification

    model_dir = resolve_checkpoint_dir(Path(checkpoint))
    clf = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), num_labels=1, problem_type="binary_classification",
        return_dict=True, trust_remote_code=True, torch_dtype="auto",
    )
    if device == "cpu":
        clf = clf.float()
    backbone = clf.bacformer
    logger.info("Loaded fine-tuned Bacformer backbone from %s", model_dir)
    return backbone.to(device).eval()


def load_model(device: str, *, mode: str, checkpoint: Path | None) -> torch.nn.Module:
    """The frozen base model (``mode="frozen"``) or a fine-tuned checkpoint backbone (``"finetuned"``)."""
    if mode == "finetuned":
        if checkpoint is None:
            raise ValueError("mode='finetuned' requires a checkpoint.")
        return load_finetuned_bacformer_backbone(checkpoint, device)
    if mode != "frozen":
        raise ValueError(f"Unknown mode {mode!r}; expected 'frozen' or 'finetuned'.")
    return load_bacformer_model(device, dtype="auto")


def _extract_gene_token_and_mean(
    model: torch.nn.Module,
    gene_table,
    esm_store_dir: Path,
    *,
    device: str,
    pt_suffix: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run ``model`` forward per genome → contextualised gene token + genome mean.

    ``gene_table`` is indexed by Sample and carries ``gene_flat_index`` + ``n_proteins`` (from
    :func:`pangena_predict.locate_gene.build_gene_presence_table`). Returns ``(token [N, dim],
    mean [N, dim], sample_ids)`` from one forward pass each, with the day-one length guard and the
    missing/misaligned-genome skips. The genome mean is gene-agnostic; only the token uses the index.
    """
    model_dtype = next(model.parameters()).dtype

    token_vectors: list[np.ndarray] = []
    mean_vectors: list[np.ndarray] = []
    kept: list[str] = []
    skips: dict[str, int] = {}
    length_checked = False
    for sample_id, row in gene_table.iterrows():
        pt_path = esm_store_dir / f"{sample_id}{pt_suffix}"
        if not pt_path.exists():
            skips["missing_pt"] = skips.get("missing_pt", 0) + 1
            continue
        store = torch.load(pt_path, map_location="cpu")
        input_len = store["protein_embeddings"].shape[1]
        real_idx = real_protein_indices(store, input_len)
        n_expected = int(row["n_proteins"])
        if real_idx.numel() != n_expected:
            skips["count_mismatch"] = skips.get("count_mismatch", 0) + 1
            continue
        flat_index = int(row["gene_flat_index"])
        if flat_index >= real_idx.numel():
            skips["out_of_range"] = skips.get("out_of_range", 0) + 1
            continue

        inputs = forward_inputs(store, device, model_dtype)
        lhs = bacformer_last_hidden_state(model, inputs)
        lhs = lhs[0] if lhs.dim() == 3 else lhs
        if not length_checked:
            # Day-one guard: the output must align 1:1 with the input rows, or the
            # gene flat index points at the wrong token (e.g. an injected CLS).
            if lhs.shape[0] != input_len:
                raise RuntimeError(
                    f"Bacformer last_hidden_state length {lhs.shape[0]} != input length {input_len} "
                    f"for {sample_id}: the gene token index would be misaligned. Aborting."
                )
            length_checked = True
        real_rows = lhs[real_idx].float()  # contextualised real-protein tokens
        raw = int(real_idx[flat_index])
        token_vectors.append(lhs[raw].float().cpu().numpy())
        # Genome mean = the mask-normalised mean the classification head pools over.
        mean_vectors.append(real_rows.mean(dim=0).cpu().numpy())
        kept.append(str(sample_id))

    if skips:
        logger.warning("Bacformer vectors: skipped %s", skips)
    token = np.vstack(token_vectors) if token_vectors else np.empty((0, 0))
    mean = np.vstack(mean_vectors) if mean_vectors else np.empty((0, 0))
    return token, mean, kept


def compute_bacformer_vectors(
    gene_table,
    esm_store_dir: Path,
    *,
    device: str,
    mode: str = "frozen",
    checkpoint: Path | None = None,
    pt_suffix: str = "_esm_embeddings.pt",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Bacformer gene token + genome mean per single-copy genome, ``mode`` in {frozen, finetuned}.

    Returns ``(token_matrix [N, dim], mean_matrix [N, dim], sample_ids)`` — both from the same forward
    pass. ``mode="frozen"`` uses the base complete-genomes model; ``mode="finetuned"`` uses the
    ``checkpoint`` backbone (the deployed ~0.905 mean-pool model — A.1.i). Samples whose ``.pt`` is
    missing or whose real-protein count fails the flat-order guard are skipped.
    """
    model = load_model(device, mode=mode, checkpoint=checkpoint)
    return _extract_gene_token_and_mean(model, gene_table, esm_store_dir, device=device, pt_suffix=pt_suffix)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ast-sheet-path", type=Path, required=True,
                        help="binary_ast_with_split.csv (defines the labelled cohort to cover).")
    parser.add_argument("--parquet-dir", type=Path, required=True, help="Dir of *_protein_sequences.parquet.")
    parser.add_argument("--esm-store-dir", type=Path, required=True, help="Dir of *_esm_embeddings.pt.")
    parser.add_argument("--output-npz", type=Path, required=True,
                        help="NPZ to write {sample_ids, gene_token_vectors, mean_vectors} "
                             "(consumed by --steps bacformer_gene_token / bacformer_mean).")
    parser.add_argument("--gene", type=str, default="rpoB", help="Gene whose contextualised token to extract (default rpoB).")
    parser.add_argument("--gene-aliases", type=str, nargs="*", default=[], help="Alternative accepted gene symbols.")
    parser.add_argument("--drug", type=str, default=RIFAMPIN_COLUMN, help="Phenotype column defining the cohort (default rifampin).")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device (default cuda:0).")
    parser.add_argument("--bacformer-checkpoint", type=Path, default=None,
                        help="Fine-tuned AMR checkpoint dir: extract the *fine-tuned* backbone's gene token + "
                             "genome mean instead of the frozen base model.")
    parser.add_argument("--qc-log", type=Path, default=Path("gene_presence_qc.log"),
                        help="Where to write the gene-presence QC log (default: ./gene_presence_qc.log).")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap the number of samples (smoke; default: all).")
    args = parser.parse_args()

    _label_map, train_ids, validate_ids, evaluate_ids, _info = resolve_clean_splits(args.ast_sheet_path, args.drug)
    if args.max_samples is not None:
        # Proportional slice so all three splits stay represented downstream.
        total = len(train_ids) + len(validate_ids) + len(evaluate_ids)
        frac = min(1.0, args.max_samples / max(1, total))
        train_ids = train_ids[: max(1, round(len(train_ids) * frac))]
        validate_ids = validate_ids[: max(1, round(len(validate_ids) * frac))]
        evaluate_ids = evaluate_ids[: max(1, round(len(evaluate_ids) * frac))]
    all_ids = [*train_ids, *validate_ids, *evaluate_ids]

    mode = "finetuned" if args.bacformer_checkpoint else "frozen"
    logger.info("Locating single-copy %s in %d labelled samples for the flat index", args.gene, len(all_ids))
    gene_table = build_gene_presence_table(
        all_ids, args.parquet_dir, args.gene, aliases=tuple(args.gene_aliases), qc_log_path=args.qc_log
    )
    logger.info("Running %s Bacformer over %d single-copy genomes on %s", mode, len(gene_table), args.device)

    token, mean, kept = compute_bacformer_vectors(
        gene_table, args.esm_store_dir, device=args.device, mode=mode, checkpoint=args.bacformer_checkpoint
    )
    if not kept:
        raise RuntimeError("No Bacformer vectors recovered — check esm_store_dir / .pt suffix.")

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, sample_ids=np.array(kept), gene_token_vectors=token, mean_vectors=mean)
    logger.info("Wrote %d Bacformer %s-token + genome-mean vectors to %s", len(kept), args.gene, args.output_npz)


if __name__ == "__main__":
    main()

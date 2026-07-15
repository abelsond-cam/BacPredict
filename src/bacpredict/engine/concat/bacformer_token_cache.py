"""GPU — cache a drug's Bacformer genome-mean + per-gene contextualised tokens for reliable carrier calls.

The unified primitive behind the fine-tuned and frozen per-AMR-gene token caches: one forward per eval
genome through either the drug's FT checkpoint (``mode="finetuned"``) or the base backbone
(``mode="frozen"``), and for every carrier call a ``calls_fn`` yields — one per genome, each a
``(label, flat_index, source, tag_match)`` :class:`bacpredict.engine.gene_lr.reliable_gene_vectors.GeneCall`
(the sidecar-agnostic seam, CARD/Kleborate for Kp) — save the contextualised token
``last_hidden_state[flat_index]`` grouped by label. So the reliable per-gene ESM-vs-frozen-vs-FT
head-to-head and the concat are computed on the *reliable* carrier sets, not Bakta's.

Saved per drug to ``<out>/<drug>/`` (``prefix`` = ``ft`` or ``frozen``):

- ``<prefix>_genome_mean_<drug>.npz`` — {sample_ids, mean_vectors}: the mask-mean over real proteins.
- ``<prefix>_amr_emb/<label>.npz`` — {sample_ids, vectors, bakta_match}: per label, its token for each
  single-copy carrier, plus ``tag_match`` (whether Bakta also named it — the reliable-vs-Bakta split).
- ``amr_gene_manifest_<drug>.csv`` (FT) / ``frozen_amr_gene_manifest_<drug>.csv`` (frozen) — label,
  sanitized, amr_source, n_carriers, n_bakta.

Eval-holdout only (the FT-unseen, honest scope). Reuses the engine forward helpers. GPU; one drug per run.
The Kp CARD ``calls_fn`` + data-root defaults live in the thin ``apps/kleb`` CLIs
(``cache_ft_amr_proteins`` / ``cache_frozen_amr_proteins``).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bacpredict.engine.concat.bacformer_genome_vectors import forward_inputs, load_model
from bacpredict.engine.embedding.generate_embeddings import bacformer_last_hidden_state
from bacpredict.engine.gene_lr.locate_gene import flatten_proteins
from bacpredict.engine.gene_lr.reliable_gene_vectors import CallsFn
from bacpredict.engine.gene_lr.snp_vs_esm_prediction import real_protein_indices, resolve_clean_splits

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _sanitize(gene: str) -> str:
    """Filesystem-safe gene token (tet(A) -> tet_A_, CTX-M -> CTX_M, blaKPC-2 -> blaKPC_2)."""
    return re.sub(r"[^A-Za-z0-9]", "_", str(gene))


def run(
    *,
    ast_sheet: Path,
    drug: str,
    parquet_dir: Path,
    esm_store_dir: Path,
    calls_fn: CallsFn,
    out_dir: Path,
    mode: str,
    checkpoint: Path | None,
    prefix: str,
    device: str,
    grain: str = "family",
    pt_suffix: str = "_esm_embeddings.pt",
    max_samples: int | None = None,
) -> None:
    """Forward each eval genome through the ``mode`` backbone; save the genome-mean + per-label tokens.

    ``prefix`` (``ft``/``frozen``) names the output stores; ``calls_fn(sid, n_real)`` yields the reliable
    carrier calls (label + flat index + source + ``tag_match``) — identical selection to the CPU collector.
    """
    _lm, _train, _val, evaluate_ids, _info = resolve_clean_splits(ast_sheet, drug)
    all_ids = list(evaluate_ids)
    if max_samples is not None:
        all_ids = all_ids[:max_samples]
    logger.info("Forwarding %d eval genomes for %s (mode=%s, grain=%s)", len(all_ids), drug, mode, grain)

    model = load_model(device, mode=mode, checkpoint=checkpoint if mode == "finetuned" else None)
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

        for call in calls_fn(str(sid), n_real):
            gene_ids.setdefault(call.label, []).append(str(sid))
            gene_vecs.setdefault(call.label, []).append(real_rows[call.flat_index])
            gene_bakta.setdefault(call.label, []).append(call.tag_match)
            gene_source.setdefault(call.label, call.source)
        if k % 200 == 0:
            logger.info("  forward: %d/%d genomes (kept mean=%d, families=%d)",
                        k, len(all_ids), len(mean_ids), len(gene_ids))

    if skips:
        logger.warning("skipped genomes: %s", skips)
    if not mean_ids:
        raise RuntimeError("No genomes forwarded — check paths / .pt suffix.")

    out_dir = out_dir / drug
    out_dir.mkdir(parents=True, exist_ok=True)
    mean_npz = out_dir / f"{prefix}_genome_mean_{drug}.npz"
    np.savez(mean_npz, sample_ids=np.array(mean_ids), mean_vectors=np.vstack(mean_vecs).astype(np.float32))
    logger.info("Wrote %s genome-mean (%d genomes) -> %s", prefix, len(mean_ids), mean_npz)

    gene_dir = out_dir / f"{prefix}_amr_emb"
    gene_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for label in gene_ids:
        san = _sanitize(label)
        np.savez(gene_dir / f"{san}.npz", sample_ids=np.array(gene_ids[label]),
                 vectors=np.vstack(gene_vecs[label]).astype(np.float32),
                 bakta_match=np.array(gene_bakta[label], dtype=bool))
        manifest.append({"gene_family": label, "sanitized": san, "amr_source": gene_source[label],
                         "n_carriers": len(gene_ids[label]), "n_bakta": int(sum(gene_bakta[label]))})
    # FT manifest/summary keep their historical un-prefixed names (the CPU consumers read amr_gene_manifest_*).
    manifest_name = f"amr_gene_manifest_{drug}.csv" if prefix == "ft" else f"{prefix}_amr_gene_manifest_{drug}.csv"
    summary_name = f"cache_summary_{drug}.json" if prefix == "ft" else f"{prefix}_cache_summary_{drug}.json"
    pd.DataFrame(manifest).sort_values("n_carriers", ascending=False).to_csv(out_dir / manifest_name, index=False)
    (out_dir / summary_name).write_text(json.dumps({
        "drug": drug, "mode": mode, "checkpoint": str(checkpoint) if checkpoint else None,
        "n_genomes": len(mean_ids), "n_families": len(manifest), "grain": grain,
    }, indent=2))
    logger.info("Wrote %d per-label %s token stores -> %s", len(manifest), prefix, gene_dir)

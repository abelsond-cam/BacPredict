"""Per-protein LR-probability panel store — the supervised attention-head channel.

The *supervised* cousin of the label-blind surprisal panel: where that channel says "this protein is
anomalous", this one says "this protein's own embedding predicts resistance" — an explicit, per-protein
pointer the gated-attention head can route to. It rests on the coding per-segment LRs
(:func:`bacpredict.engine.segment_amr_lr.per_segment_lr.run` with ``segment_type="coding"``,
``write_panels=True``): each protein row carries its gene's predicted resistance probability, non-core
proteins carry 0.

**Leakage discipline.** The per-gene probability is label-derived, so a fit *train* genome gets its
**out-of-fold** value (stored on the fitted LR) and every other genome the **full-fit** probability —
:func:`_prob_for` routes the two. The filtered store additionally zeroes proteins whose gene's out-of-fold
train AUROC did not clear the ranking filter; standardisation accumulates over the fit-train genomes only.
Two stores are written — ``filtered/`` (denoised) and ``unfiltered/`` (all core genes; let the head choose)
— sharing the same fitted LRs.

Output (mirrors the surprisal-panel contract so it is a drop-in ``--panel-store`` for the trainer's
``PanelDataset``)::

    <dir>/{filtered,unfiltered}/{sample}_panel.npz             # panel [n,1], flat_index, n_proteins, columns
    <dir>/{filtered,unfiltered}/panel_standardization.json     # fit-train-only mean/std (1 column)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from bacpredict.engine.embedding.segment_locator import read_genome

logger = logging.getLogger(__name__)

PANEL_COLUMNS = ["lr_resistance_prob"]


def _prob_for(gene: str, sample_id: str, emb_row: np.ndarray, fitted: dict[str, dict]) -> float:
    """Leakage-safe probability: stored out-of-fold value for fit train genomes, else full-fit."""
    f = fitted[gene]
    oof = f["oof_prob"].get(sample_id)
    if oof is not None:
        return oof
    return float(f["clf"].predict_proba(f["scaler"].transform(emb_row[None, :]))[0, 1])


class _Standardizer1D:
    """Streaming mean/std over the single panel column of the standardisation id set."""

    def __init__(self) -> None:
        self.sum = 0.0
        self.sumsq = 0.0
        self.count = 0

    def update(self, col: np.ndarray) -> None:
        """Fold one genome's panel column into the accumulators."""
        self.sum += float(col.sum())
        self.sumsq += float(np.square(col.astype(np.float64)).sum())
        self.count += col.shape[0]

    def finalize(self) -> tuple[float, float]:
        """Return ``(mean, std)``; a zero-variance column gets ``std=1`` (no-op scaling)."""
        mean = self.sum / max(self.count, 1)
        var = max(self.sumsq / max(self.count, 1) - mean * mean, 0.0)
        std = float(np.sqrt(var)) or 1.0
        return mean, std


def _write_sample(out_dir: Path, sample: str, panel: np.ndarray) -> None:
    """Write one ``{sample}_panel.npz`` (panel [n, 1] in flat protein order)."""
    np.savez(
        out_dir / f"{sample}_panel.npz",
        panel=panel.astype(np.float32),
        flat_index=np.arange(panel.shape[0], dtype=np.int64),
        n_proteins=np.array(panel.shape[0], dtype=np.int64),
        columns=np.array(PANEL_COLUMNS),
    )


def _write_standardization(out_dir: Path, std: _Standardizer1D) -> None:
    """Write ``panel_standardization.json`` (fit-train-only mean/std, single column)."""
    mean, scale = std.finalize()
    payload = {
        "columns": list(PANEL_COLUMNS),
        "mean": [mean],
        "std": [scale],
        "n_proteins_used": int(std.count),
        "standardize_ids_restricted": True,
    }
    (out_dir / "panel_standardization.json").write_text(json.dumps(payload, indent=2))


def build_panels(
    all_ids: list[str],
    fitted: dict[str, dict],
    filtered_genes: set[str],
    embed_dir: Path,
    parquet_dir: Path,
    *,
    train_set: set[str],
    filtered_dir: Path,
    unfiltered_dir: Path,
    store_kind: str = "esm",
) -> int:
    """Write the filtered + unfiltered panel stores in one pass over ``all_ids``.

    Each protein row carries its gene's resistance probability (out-of-fold for fit train
    genomes, full-fit for the rest); non-core proteins carry 0. The filtered store additionally
    zeroes proteins whose gene's out-of-fold train AUROC did not clear the filter. Standardisation
    accumulates over the fit-train genomes (``train_set``) only.
    """
    core_genes = set(fitted)
    std_filtered, std_unfiltered = _Standardizer1D(), _Standardizer1D()
    n_written = 0
    for k, sid in enumerate(all_ids, 1):
        read = read_genome(sid, embed_dir, parquet_dir, store_kind=store_kind)
        if read is None:
            continue
        gene_names, emb = read
        unfiltered = np.zeros(len(gene_names), dtype=np.float32)
        flt = np.zeros(len(gene_names), dtype=np.float32)
        for i, g in enumerate(gene_names):
            if g in core_genes:
                p = _prob_for(g, sid, emb[i], fitted)
                unfiltered[i] = p
                if g in filtered_genes:
                    flt[i] = p
        _write_sample(unfiltered_dir, sid, unfiltered[:, None])
        _write_sample(filtered_dir, sid, flt[:, None])
        if sid in train_set:
            std_unfiltered.update(unfiltered)
            std_filtered.update(flt)
        n_written += 1
        if k % 200 == 0:
            logger.info("  panel write: %d/%d genomes", k, len(all_ids))

    _write_standardization(unfiltered_dir, std_unfiltered)
    _write_standardization(filtered_dir, std_filtered)
    return n_written

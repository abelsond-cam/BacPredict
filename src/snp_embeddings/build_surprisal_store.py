"""Build the per-protein **surprisal panel** store consumed by the attention head.

The Task-7 diagnostic showed the genome mean-pool is what dilutes a single causal
protein (e.g. an *rpoB* RRDR point mutation) out of the genome vector, and that a
learned gated-attention pool cannot, on the resistance label alone, find that
protein among ~4,000 others (frozen ≈ 0.78, e2e ≈ 0.82 vs the 0.905 mean-pool
baseline). The remedy is to hand the attention an explicit, label-blind
"this protein is anomalous" signal: the ESM-C **surprisal panel**.

This module re-keys the genome-wide unmasked-surprisal scan
(:mod:`snp_embeddings.unmasked_surprisal_scan`) into one small per-sample
``{sample}_panel.npz`` that lines up row-for-row with the sample's
``protein_embeddings`` (flat protein order), plus a train-only
``panel_standardization.json``. **No GPU and no ESM forward** — the per-residue
surprisal is already on disk.

The 9-panel (one row per protein, concatenated onto each 960-d backbone token
*after* the backbone — never its input):

    [s1=max, s2=2nd, s3=3rd, s10=10th, p95, p90, p50=median,
     participation_ratio, kurtosis]

Two sources:

- ``--source raw`` — read the per-residue dumps ``scan_raw_shard*.npz``
  (``values``/``offsets``/``samples``/``flat_index``) and compute every panel
  member with :func:`~snp_embeddings.llr_distribution_probe.protein_surprisal_stats`
  (so the panel is *by construction* the same statistic the figures use). This is
  the path for the Phase-P prototype (the 1000-genome scan already on disk).
- ``--source parquet`` — read the per-protein stats parquets ``scan_stats_shard*.parquet``
  and select the 9 columns directly (all present once the scan is re-run with the
  s3/s10/p90 stats). The Phase-F path for the remaining ~37k genomes, where keeping
  raw per-residue dumps would cost ~200 GB.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from snp_embeddings.llr_distribution_probe import protein_surprisal_stats

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Canonical panel order — the stat keys produced by ``protein_surprisal_stats``.
PANEL_KEYS = (
    "max_surprisal",      # s1
    "top2_surprisal",     # s2
    "top3_surprisal",     # s3
    "top10_surprisal",    # s10
    "p95_surprisal",      # p95
    "p90_surprisal",      # p90
    "median_surprisal",   # p50
    "participation_ratio",
    "kurtosis_surprisal",
)
PANEL_DIM = len(PANEL_KEYS)


def panel_from_stats(stats: dict) -> list[float]:
    """Map one ``protein_surprisal_stats`` dict to the 9-panel row, imputing undefined entries.

    Short proteins leave ``top3``/``top10`` undefined → fall back to the max
    (the protein's whole top tail is its max); ``kurtosis`` → 0.0 (no excess);
    ``participation_ratio`` → the protein length. The store is therefore NaN-free.
    """
    s1 = float(stats["max_surprisal"])

    def _or(key: str, default: float) -> float:
        v = stats.get(key)
        if v is None:
            return default
        v = float(v)
        return v if np.isfinite(v) else default

    return [
        s1,
        _or("top2_surprisal", s1),
        _or("top3_surprisal", s1),
        _or("top10_surprisal", s1),
        _or("p95_surprisal", s1),
        _or("p90_surprisal", s1),
        _or("median_surprisal", s1),
        _or("participation_ratio", float(stats["length"])),
        _or("kurtosis_surprisal", 0.0),
    ]


class _Standardizer:
    """Streaming per-column mean/std over the proteins of the standardisation id set."""

    def __init__(self) -> None:
        self.sum = np.zeros(PANEL_DIM, dtype=np.float64)
        self.sumsq = np.zeros(PANEL_DIM, dtype=np.float64)
        self.count = 0

    def update(self, panel: np.ndarray) -> None:
        """Fold one sample's ``[M, 9]`` panel into the accumulators."""
        self.sum += panel.sum(axis=0)
        self.sumsq += np.square(panel.astype(np.float64)).sum(axis=0)
        self.count += panel.shape[0]

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(mean, std)``; zero-variance columns get ``std=1`` (no-op scaling)."""
        mean = self.sum / max(self.count, 1)
        var = np.maximum(self.sumsq / max(self.count, 1) - mean * mean, 0.0)
        std = np.sqrt(var)
        std[std == 0.0] = 1.0
        return mean.astype(np.float32), std.astype(np.float32)


def _write_sample(out_dir: Path, sample: str, flat_index: np.ndarray, panel: np.ndarray) -> None:
    """Write one ``{sample}_panel.npz`` (panel rows in flat protein order)."""
    np.savez(
        out_dir / f"{sample}_panel.npz",
        panel=panel.astype(np.float32),
        flat_index=flat_index.astype(np.int64),
        n_proteins=np.array(panel.shape[0], dtype=np.int64),
        columns=np.array(PANEL_KEYS),
    )


def build_from_raw(
    *,
    raw_glob: str,
    out_dir: Path,
    standardize_ids: set[str] | None,
) -> tuple[int, int]:
    """Build the panel store from the per-residue raw dumps. Returns ``(n_samples, n_proteins)``."""
    raw_files = sorted(glob.glob(raw_glob))
    if not raw_files:
        raise FileNotFoundError(f"No raw dumps matched {raw_glob!r}")
    logger.info("Building panel store from %d raw shards", len(raw_files))

    std = _Standardizer()
    n_samples = n_proteins = 0
    for rf in raw_files:
        z = np.load(rf)
        values = z["values"]
        offsets = z["offsets"]
        samples = np.asarray(z["samples"]).astype(str)
        flat = np.asarray(z["flat_index"]).astype(np.int64)
        # Proteins of one genome are contiguous within a shard (scan appends per genome).
        order = np.argsort(samples, kind="stable")
        s_sorted = samples[order]
        boundaries = np.flatnonzero(np.r_[True, s_sorted[1:] != s_sorted[:-1], True])
        for b0, b1 in zip(boundaries[:-1], boundaries[1:], strict=False):
            idx = order[b0:b1]
            sample = s_sorted[b0]
            rows = []
            for i in idx:
                surprisal = values[offsets[i]:offsets[i + 1]]
                # protein_surprisal_stats expects log P; the dump stores surprisal = -log P.
                rows.append(panel_from_stats(protein_surprisal_stats(-surprisal)))
            panel = np.asarray(rows, dtype=np.float32)
            fi = flat[idx]
            _write_sample(out_dir, sample, fi, panel)
            if standardize_ids is None or sample in standardize_ids:
                std.update(panel)
            n_samples += 1
            n_proteins += panel.shape[0]
        z.close()
        logger.info("  ...%s done (%d samples, %d proteins so far)", Path(rf).name, n_samples, n_proteins)

    _write_standardization(out_dir, std, n_used_filter=standardize_ids)
    return n_samples, n_proteins


def build_from_parquet(
    *,
    stats_glob: str,
    out_dir: Path,
    standardize_ids: set[str] | None,
) -> tuple[int, int]:
    """Build the panel store from the per-protein stats parquets. Returns ``(n_samples, n_proteins)``."""
    stats_files = sorted(glob.glob(stats_glob))
    if not stats_files:
        raise FileNotFoundError(f"No stats parquets matched {stats_glob!r}")
    missing = [k for k in PANEL_KEYS if k not in pd.read_parquet(stats_files[0]).columns]
    if missing:
        raise ValueError(f"Stats parquet {stats_files[0]} lacks panel columns {missing}; re-run the scan.")
    logger.info("Building panel store from %d stats parquets", len(stats_files))

    cols = ["sample", "flat_index", "length", *PANEL_KEYS]
    std = _Standardizer()
    n_samples = n_proteins = 0
    for sf in stats_files:
        df = pd.read_parquet(sf, columns=cols)
        for sample, g in df.groupby("sample", sort=False):
            g = g.sort_values("flat_index")
            rows = [panel_from_stats({k: (None if pd.isna(r[k]) else r[k]) for k in (*PANEL_KEYS, "length")})
                    for _, r in g.iterrows()]
            panel = np.asarray(rows, dtype=np.float32)
            _write_sample(out_dir, str(sample), g["flat_index"].to_numpy(), panel)
            if standardize_ids is None or str(sample) in standardize_ids:
                std.update(panel)
            n_samples += 1
            n_proteins += panel.shape[0]
        logger.info("  ...%s done (%d samples so far)", Path(sf).name, n_samples)

    _write_standardization(out_dir, std, n_used_filter=standardize_ids)
    return n_samples, n_proteins


def _write_standardization(out_dir: Path, std: _Standardizer, *, n_used_filter: set[str] | None) -> None:
    """Write ``panel_standardization.json`` (train-only mean/std + provenance)."""
    mean, scale = std.finalize()
    payload = {
        "columns": list(PANEL_KEYS),
        "mean": mean.tolist(),
        "std": scale.tolist(),
        "n_proteins_used": int(std.count),
        "standardize_ids_restricted": n_used_filter is not None,
    }
    path = out_dir / "panel_standardization.json"
    path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %s (mean/std over %d proteins, restricted=%s)", path, std.count, n_used_filter is not None)


def _load_standardize_ids(split_csv: Path | None, split_value: str) -> set[str] | None:
    """Read the train-only sample ids from a split CSV (``Sample``/``train_val_eval``), or None."""
    if split_csv is None:
        return None
    df = pd.read_csv(split_csv)
    id_col = "Sample" if "Sample" in df.columns else df.columns[0]
    ids = set(df.loc[df["train_val_eval"] == split_value, id_col].astype(str))
    logger.info("Standardisation restricted to %d %r ids from %s", len(ids), split_value, split_csv)
    return ids


def main() -> None:
    """CLI entry point — re-key the surprisal scan into the per-sample panel store."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["raw", "parquet"], default="raw")
    parser.add_argument("--scan-raw-glob", type=str, default=None, help="Glob for scan_raw_shard*.npz (source=raw).")
    parser.add_argument("--scan-stats-glob", type=str, default=None,
                        help="Glob for scan_stats_shard*.parquet (source=parquet).")
    parser.add_argument("--out-dir", type=Path, required=True, help="Panel store output dir (one npz per sample).")
    parser.add_argument("--split-csv", type=Path, default=None,
                        help="Split CSV; restrict standardisation to its train ids (else use all built samples).")
    parser.add_argument("--split-value", type=str, default="train")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    standardize_ids = _load_standardize_ids(args.split_csv, args.split_value)

    if args.source == "raw":
        if args.scan_raw_glob is None:
            parser.error("--scan-raw-glob is required for --source raw")
        n_samples, n_proteins = build_from_raw(
            raw_glob=args.scan_raw_glob, out_dir=args.out_dir, standardize_ids=standardize_ids
        )
    else:
        if args.scan_stats_glob is None:
            parser.error("--scan-stats-glob is required for --source parquet")
        n_samples, n_proteins = build_from_parquet(
            stats_glob=args.scan_stats_glob, out_dir=args.out_dir, standardize_ids=standardize_ids
        )
    logger.info("Panel store complete: %d samples, %d proteins → %s", n_samples, n_proteins, args.out_dir)


if __name__ == "__main__":
    main()

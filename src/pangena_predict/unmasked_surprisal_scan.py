"""Scalable genome-wide unmasked-surprisal pass (experiment-4 feature selection).

The Phase-0 probe (:mod:`pangena_predict.llr_distribution_probe`) established, on rpoB,
that the resistance SNP is a single-residue masked spike and that the cheap **unmasked**
surprisal tracks the masked ablation. This module scales the unmasked signal to a **solid
sample of whole genomes** to answer the two remaining questions before choosing the
experiment-4 per-protein feature:

1. **Genome-wide spatial autocorrelation** — does a residue's surprisal predict its
   neighbours'? (If not, a single-residue peak statistic suffices; no window needed.) We
   accumulate the pooled autocorrelation function (lag 1..K) and the surprisal profile
   *around each protein's top residue*, streaming, as proteins are scored.
2. **Per-protein concentration statistics** — which summary best isolates the single-SNP
   protein from the long / divergent / multiply-mutated crowd? Every protein's
   :func:`~pangena_predict.llr_distribution_probe.protein_surprisal_stats` row is written to
   a parquet (magnitude *and* concentration/shape stats).

Two modes:

- ``--mode manifest`` (CPU): genotype a pool of the canonical split once and write a
  ``manifest.csv`` of ~N resistant rpoB-mutant + ~N WT samples (sample, role, rpoB flat
  index, genotype). Reuses :func:`resolve_clean_splits`, :func:`build_genotype_table`,
  :func:`select_isolates`.
- ``--mode scan`` (GPU, sharded by ``--shard-index/--n-shards``): forward every protein of
  this shard's genomes once (:func:`unmasked_logprobs`), writing a per-shard stats parquet,
  a per-shard autocorrelation NPZ, and (optionally) a per-shard raw per-residue dump.

This is **not** a throwaway diagnostic: the per-protein stats parquet is the start of the
genome-wide feature precompute (Phase 1).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pangena_predict.llr_distribution_probe import (
    _NEEDED_COLS,
    hotspot_codons,
    hotspot_label,
    protein_surprisal_stats,
    select_isolates,
)
from pangena_predict.locate_gene import flatten_proteins
from pangena_predict.rpob_genotype import build_genotype_table, load_reference
from pangena_predict.snp_vs_esm_prediction import resolve_clean_splits

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------------------------------------------------------------------
# Streaming accumulators
# ---------------------------------------------------------------------------


class AutocorrAccumulator:
    """Streaming pooled autocorrelation of per-residue surprisal, lags 1..``max_lag``.

    Holds the sufficient statistics for a Pearson correlation between ``s_i`` and
    ``s_{i+k}`` pooled over every protein, so the genome-wide ACF can be finalised after a
    sharded run by summing accumulators. Also tracks the surprisal profile *around each
    protein's top residue* (offsets −W..W) and a global background mean.
    """

    def __init__(self, max_lag: int = 20, window: int = 25) -> None:
        self.max_lag = max_lag
        self.window = window
        k = max_lag
        self.sum_x = np.zeros(k)
        self.sum_y = np.zeros(k)
        self.sum_xy = np.zeros(k)
        self.sum_x2 = np.zeros(k)
        self.sum_y2 = np.zeros(k)
        self.count = np.zeros(k)
        w = 2 * window + 1
        self.prof_sum = np.zeros(w)
        self.prof_count = np.zeros(w)
        self.bg_sum = 0.0
        self.bg_count = 0
        self.n_proteins = 0
        self.n_residues = 0

    def update(self, surprisal: np.ndarray) -> None:
        """Fold one protein's per-residue surprisal into the accumulators."""
        s = np.asarray(surprisal, dtype=float)
        n = s.size
        if n == 0:
            return
        self.n_proteins += 1
        self.n_residues += n
        self.bg_sum += float(s.sum())
        self.bg_count += n
        for k in range(1, self.max_lag + 1):
            if n <= k:
                break
            x = s[:-k]
            y = s[k:]
            i = k - 1
            self.sum_x[i] += x.sum()
            self.sum_y[i] += y.sum()
            self.sum_xy[i] += np.dot(x, y)
            self.sum_x2[i] += np.dot(x, x)
            self.sum_y2[i] += np.dot(y, y)
            self.count[i] += x.size
        p = int(np.argmax(s))
        w = self.window
        lo, hi = max(0, p - w), min(n, p + w + 1)
        for q in range(lo, hi):
            self.prof_sum[q - p + w] += s[q]
            self.prof_count[q - p + w] += 1

    def to_dict(self) -> dict:
        """The raw accumulators (for saving a shard NPZ; merge by summing, then finalise)."""
        return {
            "max_lag": np.array(self.max_lag),
            "window": np.array(self.window),
            "sum_x": self.sum_x, "sum_y": self.sum_y, "sum_xy": self.sum_xy,
            "sum_x2": self.sum_x2, "sum_y2": self.sum_y2, "count": self.count,
            "prof_sum": self.prof_sum, "prof_count": self.prof_count,
            "bg_sum": np.array(self.bg_sum), "bg_count": np.array(self.bg_count),
            "n_proteins": np.array(self.n_proteins), "n_residues": np.array(self.n_residues),
        }


# ---------------------------------------------------------------------------
# Mode: manifest (CPU)
# ---------------------------------------------------------------------------


def build_manifest(
    *,
    ast_sheet_path: Path,
    parquet_dir: Path,
    drug: str,
    n_resistant: int,
    n_wt: int,
    pool_size: int,
    out_csv: Path,
    qc_log_path: Path,
) -> pd.DataFrame:
    """Genotype a pool of the canonical split once → a manifest of resistant + WT genomes.

    Resistant = label 1 with ≥1 RRDR substitution (natural distribution, not codon-deduped,
    so it is S450L-heavy and representative); WT = label 0 with zero RRDR substitutions.
    """
    reference = load_reference()
    label_map, train_ids, validate_ids, evaluate_ids, split_info = resolve_clean_splits(ast_sheet_path, drug)
    pool = [*train_ids, *validate_ids, *evaluate_ids][:pool_size]
    logger.info("Genotyping a pool of %d labelled samples to source genomes", len(pool))
    genotype = build_genotype_table(pool, parquet_dir, reference, qc_log_path=qc_log_path)

    resistant_ids, wt_ids = select_isolates(
        genotype, label_map, reference, n_resistant=n_resistant, n_wt=n_wt, diverse=False
    )
    rows: list[dict] = []
    for role, ids in (("resistant", resistant_ids), ("wt", wt_ids)):
        for sid in ids:
            hotspots = hotspot_codons(genotype.loc[sid], reference) if role == "resistant" else []
            rows.append({
                "sample": sid,
                "role": role,
                "rpob_flat_index": int(genotype.loc[sid, "rpob_flat_index"]),
                "genotype": hotspot_label(hotspots),
            })
    manifest = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_csv, index=False)
    logger.info("Wrote manifest %s — %d resistant + %d WT (pool %d, single-copy genotyped %d)",
                out_csv, len(resistant_ids), len(wt_ids), len(pool), len(genotype))
    logger.info("split: %s", split_info)
    return manifest


# ---------------------------------------------------------------------------
# Mode: scan (GPU, sharded)
# ---------------------------------------------------------------------------


def run_scan(
    *,
    manifest_csv: Path,
    parquet_dir: Path,
    shard_index: int,
    n_shards: int,
    device: str,
    out_prefix: Path,
    max_lag: int,
    window: int,
    dump_raw: bool,
) -> None:
    """Forward every protein of this shard's genomes once → stats parquet + ACF NPZ (+ raw)."""
    from tl.embed.esm_residue_level import load_esmc_mlm, unmasked_logprobs

    manifest = pd.read_csv(manifest_csv)
    shard = np.array_split(manifest, n_shards)[shard_index].reset_index(drop=True)
    logger.info("Shard %d/%d: %d of %d genomes", shard_index, n_shards, len(shard), len(manifest))

    model, tokenizer = load_esmc_mlm(device=device)
    acc = AutocorrAccumulator(max_lag=max_lag, window=window)
    rows: list[dict] = []
    raw_values: list[np.ndarray] = []
    raw_offsets: list[int] = [0]
    raw_keys: list[tuple[str, int]] = []

    for g, (_, m) in enumerate(shard.iterrows()):
        sid = str(m["sample"])
        rpob_flat = int(m["rpob_flat_index"])
        role = str(m["role"])
        pq = parquet_dir / f"{sid}_protein_sequences.parquet"
        if not pq.exists():
            # Full-cohort scans may list a genome whose parquet is absent; skip rather than
            # crash a multi-hour shard on one missing file.
            logger.warning("skip %s — no protein parquet at %s", sid, pq)
            continue
        records = flatten_proteins(pd.read_parquet(pq, columns=_NEEDED_COLS))
        for rec in records:
            seq = rec["protein_sequence"]
            if not seq:
                continue
            logp = unmasked_logprobs(model, tokenizer, seq, device=device).numpy()
            surprisal = -logp.astype(float)
            stats = protein_surprisal_stats(logp)
            rows.append({
                "sample": sid, "role": role, "flat_index": rec["flat_index"],
                "gene_name": rec["gene_name"], "is_rpob": rec["flat_index"] == rpob_flat,
                "genotype": str(m["genotype"]), **stats,
            })
            acc.update(surprisal)
            if dump_raw:
                raw_values.append(surprisal.astype(np.float32))
                raw_offsets.append(raw_offsets[-1] + surprisal.size)
                raw_keys.append((sid, int(rec["flat_index"])))
        if (g + 1) % 25 == 0:
            logger.info("  ...%d/%d genomes (%d proteins so far)", g + 1, len(shard), len(rows))

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    pq_path = out_prefix.with_name(out_prefix.name + f"_stats_shard{shard_index:03d}.parquet")
    pd.DataFrame(rows).to_parquet(pq_path, index=False)
    acf_path = out_prefix.with_name(out_prefix.name + f"_acf_shard{shard_index:03d}.npz")
    np.savez(acf_path, **acc.to_dict())
    logger.info("Wrote %s (%d protein rows) + %s", pq_path, len(rows), acf_path)
    if dump_raw:
        raw_path = out_prefix.with_name(out_prefix.name + f"_raw_shard{shard_index:03d}.npz")
        np.savez(
            raw_path,
            values=np.concatenate(raw_values) if raw_values else np.array([], dtype=np.float32),
            offsets=np.array(raw_offsets, dtype=np.int64),
            samples=np.array([k[0] for k in raw_keys]),
            flat_index=np.array([k[1] for k in raw_keys], dtype=np.int64),
        )
        logger.info("Wrote raw per-residue dump %s (%d proteins)", raw_path, len(raw_keys))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point — ``--mode manifest`` (CPU) or ``--mode scan`` (GPU array)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["manifest", "scan"], required=True)
    parser.add_argument("--parquet-dir", type=Path, required=True, help="Dir of *_protein_sequences.parquet.")
    parser.add_argument("--manifest-csv", type=Path, required=True, help="Manifest path (written/read).")
    parser.add_argument("--drug", type=str, default="rifampin")
    # manifest mode
    parser.add_argument("--ast-sheet-path", type=Path, default=None, help="binary_ast_with_split.csv (manifest mode).")
    parser.add_argument("--n-resistant", type=int, default=500)
    parser.add_argument("--n-wt", type=int, default=500)
    parser.add_argument("--pool-size", type=int, default=4000)
    # scan mode
    parser.add_argument("--out-prefix", type=Path, default=None, help="Output prefix for shard files (scan mode).")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-lag", type=int, default=20)
    parser.add_argument("--window", type=int, default=25)
    parser.add_argument("--dump-raw", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.mode == "manifest":
        if args.ast_sheet_path is None:
            parser.error("--ast-sheet-path is required for --mode manifest")
        build_manifest(
            ast_sheet_path=args.ast_sheet_path, parquet_dir=args.parquet_dir, drug=args.drug,
            n_resistant=args.n_resistant, n_wt=args.n_wt, pool_size=args.pool_size,
            out_csv=args.manifest_csv, qc_log_path=args.manifest_csv.with_name("manifest_rpob_copy_qc.log"),
        )
    else:
        if args.out_prefix is None:
            parser.error("--out-prefix is required for --mode scan")
        run_scan(
            manifest_csv=args.manifest_csv, parquet_dir=args.parquet_dir,
            shard_index=args.shard_index, n_shards=args.n_shards, device=args.device,
            out_prefix=args.out_prefix, max_lag=args.max_lag, window=args.window, dump_raw=args.dump_raw,
        )


if __name__ == "__main__":
    main()

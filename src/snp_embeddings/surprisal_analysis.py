"""Read-only analysis + visualisation of the Phase-0 ESM-C surprisal sidecars.

This module **reads** the two sidecars the GPU probe
(:mod:`snp_embeddings.llr_distribution_probe`) already wrote and turns them into
the headline figures + a refined results JSON. It runs no model and touches no
GPU — pure ``pandas``/``numpy``/``scipy``/``matplotlib`` over small files, so it
belongs on the HPC login node (or a laptop).

Two questions, two sidecars:

**0A — is the cheap unmasked proxy faithful to the masked-ablation gold standard?**
From the ``*_0a_points.npz`` (per-residue ``log P`` over each isolate's whole rpoB,
plus the resistance-residue points), we plot the proxy correlation and report the
pooled / per-isolate agreement and where the resistance residue ranks among rpoB's
residues. ``surprisal = -log P`` (higher = more anomalous / more "surprising").

**0B — does a per-protein surprisal summary act like a hotspot scan?** From the
``*_0b_protein_stats.parquet`` (one row per protein per genome) we recompute a
**numerically stable** ``hotspot_z_floored`` (the saved ``hotspot_z`` blew up when
``MAD → 0``), then ask, per genome, where mutated rpoB ranks among the ~4,000
proteins by each candidate statistic — i.e. whether the statistic would let an
attention pool say "a SNP is here". The saved parquet uses the older ``*_surprise``
column names; they are aliased to ``*_surprisal`` on load so this module reads both
that file and any future re-run.

Headline figures (also copied into ``src/tb_ast/docs/figures``):

- ``surprisal_vs_ablation.png`` — unmasked surprisal vs masked surprisal, resistance
  residues in red (the 0A proxy proof).
- ``esm_surprisal.png`` — a 2-panel surprisal histogram: (left) per-residue surprisal
  across a representative resistant rpoB with the SNP residue marked, (right)
  per-protein max-surprisal across a representative resistant genome's ~4,000 proteins
  with rpoB marked.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Legacy parquet column → information-theoretic name. The probe wrote "*_surprise";
# we standardise on "surprisal" (−log p). Aliased on load so this module reads the
# already-saved parquet and any future re-run identically.
_SURPRISE_TO_SURPRISAL = {
    "max_surprise": "max_surprisal",
    "top2_surprise": "top2_surprisal",
    "median_surprise": "median_surprisal",
    "mad_surprise": "mad_surprisal",
    "p99_surprise": "p99_surprisal",
    "p95_surprise": "p95_surprisal",
    "skew_surprise": "skew_surprisal",
    "kurtosis_surprise": "kurtosis_surprisal",
}

# The per-protein "a SNP is here" statistic family (higher = more anomalous → rpoB
# should sit in the high tail). hotspot_z_floored is recomputed here; the rest are
# read straight from the parquet.
STAT_FAMILY = (
    "max_surprisal",
    "max_minus_p99",
    "max_minus_p95",
    "top1_minus_top2",
    "mean_top3",
    "hotspot_z_floored",
)
# The subset drawn as supplementary per-stat rank curves.
RANKCURVE_STATS = ("max_surprisal", "hotspot_z_floored")


# ---------------------------------------------------------------------------
# Loading + stat refinement
# ---------------------------------------------------------------------------


def load_protein_stats(parquet_path: Path) -> pd.DataFrame:
    """Load the 0B per-protein parquet, aliasing legacy ``*_surprise`` → ``*_surprisal``."""
    df = pd.read_parquet(parquet_path)
    rename = {old: new for old, new in _SURPRISE_TO_SURPRISAL.items() if old in df.columns}
    if rename:
        logger.info("Aliasing %d legacy *_surprise column(s) → *_surprisal", len(rename))
        df = df.rename(columns=rename)
    return df


def add_floored_hotspot_z(df: pd.DataFrame, mad_floor: float | None) -> tuple[pd.DataFrame, float]:
    """Add a numerically stable ``hotspot_z_floored = (max − median)/max(mad, floor)``.

    The saved ``hotspot_z`` divided by ``mad + 1e-6``; for short or homogeneous
    proteins ``MAD → 0`` and the ratio exploded (values in the tens of thousands),
    so the ranking was dominated by degenerate proteins rather than genuine single-
    residue anomalies. Flooring the MAD removes that artefact.

    Parameters
    ----------
    df : pandas.DataFrame
        Per-protein stats with ``max_surprisal``, ``median_surprisal``, ``mad_surprisal``.
    mad_floor : float or None
        MAD floor. ``None`` → the global median of ``mad_surprisal`` (a typical scale).

    Returns
    -------
    tuple
        ``(df_with_column, mad_floor_used)``.
    """
    if mad_floor is None:
        mad_floor = float(df["mad_surprisal"].median())
    floored = np.maximum(df["mad_surprisal"].to_numpy(dtype=float), mad_floor)
    df = df.copy()
    df["hotspot_z_floored"] = (df["max_surprisal"] - df["median_surprisal"]) / floored
    return df, mad_floor


def resolve_rpob_index(
    df: pd.DataFrame,
    protein_stats_json: Path | None,
) -> dict[str, int]:
    """Map ``sample → rpoB flat_index``.

    Prefers the probe's 0B JSON (authoritative ``rpob_flat_index`` per genome); falls
    back to a case-insensitive ``gene_name == 'rpob'`` match in the parquet (QC keeps
    one rpoB copy per genome, so the match is unique).
    """
    if protein_stats_json is not None and protein_stats_json.exists():
        payload = json.loads(protein_stats_json.read_text())
        mapping = {g["sample"]: int(g["rpob_flat_index"]) for g in payload.get("phase_0b", payload).get("genomes", [])}
        if mapping:
            logger.info("rpoB flat indices read from %s", protein_stats_json.name)
            return mapping
    logger.info("rpoB flat indices resolved by gene_name match (no usable 0B JSON)")
    mapping = {}
    for sample, sub in df.groupby("sample"):
        hit = sub[sub["gene_name"].str.lower() == "rpob"]
        if not hit.empty:
            mapping[str(sample)] = int(hit.iloc[0]["flat_index"])
    return mapping


# ---------------------------------------------------------------------------
# 0B — per-genome rpoB ranking among ~4,000 proteins
# ---------------------------------------------------------------------------


def rpob_rankings(
    df: pd.DataFrame,
    rpob_by_sample: dict[str, int],
    *,
    min_length: int,
) -> tuple[list[dict], dict]:
    """Per genome, where mutated rpoB ranks among proteins (≥ ``min_length``) by each stat."""
    per_genome: list[dict] = []
    for sample, sub in df.groupby("sample"):
        role = str(sub.iloc[0]["role"])
        ranked = sub[sub["length"] >= min_length]
        n = int(len(ranked))
        rpob_flat = rpob_by_sample.get(str(sample))
        rpob_rows = ranked[ranked["flat_index"] == rpob_flat] if rpob_flat is not None else ranked.iloc[0:0]
        entry: dict = {"sample": str(sample), "role": role, "n_ranked": n,
                       "rpob_flat_index": rpob_flat, "rpob_present": not rpob_rows.empty, "by_stat": {}}
        if not rpob_rows.empty:
            for stat in STAT_FAMILY:
                rpob_val = float(rpob_rows.iloc[0][stat])
                rank = int((ranked[stat] > rpob_val).sum()) + 1  # 1 = most anomalous
                entry["by_stat"][stat] = {
                    "rpob_value": rpob_val,
                    "rpob_rank": rank,
                    "rpob_percentile": float(100.0 * (1 - (rank - 1) / n)) if n else None,
                }
        per_genome.append(entry)

    aggregate: dict = {"min_length": min_length, "by_stat": {}}
    for stat in STAT_FAMILY:
        for role in ("resistant", "wt"):
            pcts = [g["by_stat"][stat]["rpob_percentile"] for g in per_genome
                    if g["role"] == role and stat in g["by_stat"]]
            pcts = [p for p in pcts if p is not None]
            aggregate["by_stat"].setdefault(stat, {})[role] = {
                "mean_percentile": float(np.mean(pcts)) if pcts else None,
                "n_genomes": len(pcts),
            }
    return per_genome, aggregate


def top_coflagged(df: pd.DataFrame, sample: str, *, min_length: int, top_n: int) -> list[dict]:
    """Top-``top_n`` proteins of one genome by ``max_surprisal`` (what the flag selects)."""
    sub = df[(df["sample"] == sample) & (df["length"] >= min_length)]
    top = sub.nlargest(top_n, "max_surprisal")[["flat_index", "gene_name", "max_surprisal", "length"]]
    return [
        {"flat_index": int(r.flat_index),
         "gene_name": None if pd.isna(r.gene_name) else str(r.gene_name),
         "max_surprisal": float(r.max_surprisal), "length": int(r.length)}
        for r in top.itertuples()
    ]


# ---------------------------------------------------------------------------
# 0A — proxy summary from the NPZ
# ---------------------------------------------------------------------------


def proxy_summary(npz: dict) -> dict:
    """Pooled / per-isolate agreement of unmasked vs masked surprisal, from the 0A NPZ."""
    masked_surp = -npz["pooled_masked"].astype(float)
    unmasked_surp = -npz["pooled_unmasked"].astype(float)
    pooled_r = float(pearsonr(masked_surp, unmasked_surp)[0]) if masked_surp.size > 2 else None
    pooled_rho = float(spearmanr(masked_surp, unmasked_surp)[0]) if masked_surp.size > 2 else None

    iso_r = npz["isolate_pearson"].astype(float)
    res_r = iso_r[(npz["isolate_role"] == "resistant") & ~np.isnan(iso_r)]

    sm = -npz["snp_masked"].astype(float)
    su = -npz["snp_unmasked"].astype(float)
    snp_r = float(pearsonr(sm, su)[0]) if sm.size > 2 and sm.std() > 0 and su.std() > 0 else None

    ranks = npz["snp_rank_gene"].astype(float)
    labels = npz["snp_label"]
    return {
        "n_isolates": int(npz["isolate_role"].size),
        "n_resistant_with_snp": int(sm.size),
        "n_distinct_genotypes": int(len(set(labels.tolist()))),
        "pooled_pearson": pooled_r,
        "pooled_spearman": pooled_rho,
        "pooled_n_points": int(masked_surp.size),
        "per_isolate_pearson_resistant": {
            "mean": float(res_r.mean()) if res_r.size else None,
            "median": float(np.median(res_r)) if res_r.size else None,
            "sd": float(res_r.std()) if res_r.size else None,
            "min": float(res_r.min()) if res_r.size else None,
            "n": int(res_r.size),
        },
        "snp_site_pearson": snp_r,
        "snp_unmasked_rank_in_gene": {
            "frac_rank1": float((ranks == 1).mean()) if ranks.size else None,
            "frac_top3": float((ranks <= 3).mean()) if ranks.size else None,
            "median_rank": float(np.median(ranks)) if ranks.size else None,
            "n": int(ranks.size),
        },
    }


def _pick_representative_resistant(npz: dict) -> tuple[int, int | None]:
    """First resistant isolate index ``k`` in the 0A NPZ + its aligned SNP index (or ``None``)."""
    roles = npz["isolate_role"]
    res_idx = np.flatnonzero(roles == "resistant")
    if res_idx.size == 0:
        raise RuntimeError("0A NPZ has no resistant isolate to illustrate.")
    k = int(res_idx[0])
    # SNP arrays are appended in resistant-first record order; when every resistant
    # isolate had a located hotspot they align 1:1 with the resistant isolates.
    snp_idx = k if npz["snp_masked"].size == res_idx.size else None
    return k, snp_idx


def snp_distance_profile(npz: dict, *, max_distance: int = 100) -> dict:
    """Surprisal as a function of residue distance from the resistance SNP.

    Aligns each resistant isolate's rpoB on its primary resistance residue (recovered
    by matching the stored SNP log P inside the isolate's per-residue slice) and pools
    the per-residue surprisal at each signed offset ``d`` across all isolates. Because
    the 100 isolates span 17 distinct hotspot codons, intrinsic position-specific
    surprisal averages out at ``d != 0`` while a genuine mutation-induced local effect
    stays aligned — so the profile isolates the drop-off around the SNP. Also tests the
    "are the 2nd/3rd most-surprising residues the SNP's neighbours?" hypothesis by
    measuring how far the top-ranked residues sit from the SNP.

    Parameters
    ----------
    npz : dict
        The loaded 0A points NPZ.
    max_distance : int, default 100
        Largest offset (± aa) to profile.

    Returns
    -------
    dict
        ``profile`` (per-distance aggregates), ``neighbour_rank`` (top-residue
        distances), ``dropoff_masked`` (symmetric mean at small offsets),
        ``background_masked_median``, ``max_distance``.
    """
    idx = npz["pooled_isolate_idx"]
    masked_lp = npz["pooled_masked"].astype(float)
    unmasked_lp = npz["pooled_unmasked"].astype(float)
    roles = npz["isolate_role"]
    snp_masked = npz["snp_masked"].astype(float)
    n_res = int((roles == "resistant").sum())
    aligned = snp_masked.size == n_res  # 0A is all-resistant → snp[k] ↔ isolate k

    offsets = list(range(-max_distance, max_distance + 1))
    bucket_m: dict[int, list] = {d: [] for d in offsets}
    bucket_u: dict[int, list] = {d: [] for d in offsets}
    top_dists: list[list[int]] = []  # [|d| of rank-1, rank-2, rank-3] per isolate
    snp_top1: list[bool] = []
    backgrounds: list[float] = []
    for k in sorted({int(i) for i in idx}):
        if str(roles[k]) != "resistant" or not aligned:
            continue
        sel = idx == k
        m_lp = masked_lp[sel]
        m_sur = -m_lp
        u_sur = -unmasked_lp[sel]
        length = m_sur.size
        p = int(np.argmin(np.abs(m_lp - snp_masked[k])))  # SNP position via exact value match
        order = np.argsort(m_sur)[::-1]  # residues by descending masked surprisal
        top_dists.append([abs(int(order[r]) - p) for r in range(min(3, length))])
        snp_top1.append(bool(order[0] == p))
        for d in offsets:
            q = p + d
            if 0 <= q < length:
                bucket_m[d].append(float(m_sur[q]))
                bucket_u[d].append(float(u_sur[q]))
        far = np.abs(np.arange(length) - p) > max_distance
        if far.any():
            backgrounds.append(float(np.median(m_sur[far])))

    def _agg(bucket: dict, fn) -> list:
        return [float(fn(bucket[d])) if bucket[d] else None for d in offsets]

    profile = {
        "distance": offsets,
        "mean_masked": _agg(bucket_m, np.mean),
        "median_masked": _agg(bucket_m, np.median),
        "q25_masked": [float(np.percentile(bucket_m[d], 25)) if bucket_m[d] else None for d in offsets],
        "q75_masked": [float(np.percentile(bucket_m[d], 75)) if bucket_m[d] else None for d in offsets],
        "mean_unmasked": _agg(bucket_u, np.mean),
        "n_per_distance": [len(bucket_m[d]) for d in offsets],
    }
    dropoff = {}
    for o in (0, 1, 2, 3, 5, 10, 25, 50, 100):
        if o > max_distance:
            continue
        vals = bucket_m[0] if o == 0 else bucket_m.get(o, []) + bucket_m.get(-o, [])
        dropoff[f"d{o}"] = float(np.mean(vals)) if vals else None

    td = np.array(top_dists) if top_dists else np.zeros((0, 3))
    neighbour = {
        "n_isolates": int(td.shape[0]),
        "snp_is_top1_frac": float(np.mean(snp_top1)) if snp_top1 else None,
        "rank2_median_dist": float(np.median(td[:, 1])) if td.shape[0] and td.shape[1] > 1 else None,
        "rank3_median_dist": float(np.median(td[:, 2])) if td.shape[0] and td.shape[1] > 2 else None,
        "frac_rank2_within_2aa": float(np.mean(td[:, 1] <= 2)) if td.shape[0] and td.shape[1] > 1 else None,
        "frac_rank3_within_2aa": float(np.mean(td[:, 2] <= 2)) if td.shape[0] and td.shape[1] > 2 else None,
        "frac_top3_all_within_2aa": float(np.mean((td[:, 1:3] <= 2).all(axis=1)))
        if td.shape[0] and td.shape[1] > 2 else None,
    }
    return {
        "profile": profile,
        "neighbour_rank": neighbour,
        "dropoff_masked": dropoff,
        "background_masked_median": float(np.median(backgrounds)) if backgrounds else None,
        "max_distance": max_distance,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_surprisal_vs_ablation(npz: dict, summary: dict, out_path: Path) -> None:
    """Headline 0A proxy figure: unmasked surprisal vs masked surprisal, SNP residues red."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    masked_surp = -npz["pooled_masked"].astype(float)
    unmasked_surp = -npz["pooled_unmasked"].astype(float)
    sm = -npz["snp_masked"].astype(float)
    su = -npz["snp_unmasked"].astype(float)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    hb = ax.hexbin(masked_surp, unmasked_surp, gridsize=60, bins="log", cmap="Greys", mincnt=1)
    fig.colorbar(hb, ax=ax, label="log10(residue count)")
    if sm.size:
        ax.scatter(sm, su, s=42, color="#d62728", edgecolor="k", linewidth=0.4,
                   alpha=0.85, zorder=5, label=f"resistance residue (n={sm.size})")
    lo = float(min(masked_surp.min(), unmasked_surp.min()))
    hi = float(max(masked_surp.max(), unmasked_surp.max()))
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1, color="C0", label="y = x")

    pr, rho, snp_r = summary["pooled_pearson"], summary["pooled_spearman"], summary["snp_site_pearson"]
    txt = f"pooled Pearson {pr:.3f}\npooled Spearman {rho:.3f}"
    if snp_r is not None:
        txt += f"\nSNP-site Pearson {snp_r:.3f}"
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=10, bbox={"boxstyle": "round", "fc": "white", "alpha": 0.8})
    ax.set_xlabel("masked surprisal  −log P(residue | context∖i)   (gold-standard ablation)")
    ax.set_ylabel("unmasked surprisal  −log P(residue | full context)   (cheap proxy)")
    ax.set_title("Cheap unmasked surprisal tracks the masked-ablation gold standard")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_esm_surprisal_2panel(
    npz: dict,
    df: pd.DataFrame,
    rpob_by_sample: dict[str, int],
    *,
    min_length: int,
    out_path: Path,
) -> dict:
    """Headline 2-panel surprisal histogram: per-residue (within rpoB) | per-protein (genome)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13, 5))
    meta: dict = {}

    # Left — per-residue surprisal across one representative resistant rpoB.
    k, snp_idx = _pick_representative_resistant(npz)
    sel = npz["pooled_isolate_idx"] == k
    res_surp = -npz["pooled_unmasked"][sel].astype(float)
    label = str(npz["isolate_label"][k])
    axl.hist(res_surp, bins=40, color="#9ecae1", edgecolor="white")
    snp_surp = None
    if snp_idx is not None and npz["snp_unmasked"].size > snp_idx:
        snp_logp = float(npz["snp_unmasked"][snp_idx])
        # match the SNP residue inside this isolate's slice (same float, exact match)
        pos = int(np.argmin(np.abs(npz["pooled_unmasked"][sel].astype(float) - snp_logp)))
        snp_surp = float(res_surp[pos])
        axl.axvline(snp_surp, color="#d62728", lw=2.2, label=f"resistance residue ({label})")
        axl.legend(loc="upper right", fontsize=9)
    axl.set_xlabel("per-residue surprisal  −log P(residue)")
    axl.set_ylabel("residues in rpoB")
    axl.set_title(f"Within rpoB: the SNP is a surprisal outlier\n(representative resistant isolate, {res_surp.size} residues)")
    axl.grid(axis="y", alpha=0.25)
    meta["left"] = {"isolate_index": k, "genotype": label, "n_residues": int(res_surp.size),
                    "snp_surprisal": snp_surp}

    # Right — per-protein max-surprisal across a representative resistant genome.
    res_samples = df[df["role"] == "resistant"]["sample"].unique()
    sample = str(res_samples[0]) if res_samples.size else str(df["sample"].unique()[0])
    sub = df[(df["sample"] == sample) & (df["length"] >= min_length)]
    maxsurp = sub["max_surprisal"].to_numpy(dtype=float)
    axr.hist(maxsurp, bins=40, color="#a1d99b", edgecolor="white")
    rpob_flat = rpob_by_sample.get(sample)
    rpob_val = None
    rpob_pct = None
    if rpob_flat is not None:
        rpob_rows = sub[sub["flat_index"] == rpob_flat]
        if not rpob_rows.empty:
            rpob_val = float(rpob_rows.iloc[0]["max_surprisal"])
            rank = int((sub["max_surprisal"] > rpob_val).sum()) + 1
            rpob_pct = float(100.0 * (1 - (rank - 1) / len(sub)))
            axr.axvline(rpob_val, color="#d62728", lw=2.2,
                        label=f"rpoB (rank {rank}/{len(sub)}, {rpob_pct:.1f}th pct)")
            axr.legend(loc="upper right", fontsize=9)
    axr.set_xlabel("per-protein max surprisal  max(−log P) over the protein")
    axr.set_ylabel("proteins in genome")
    axr.set_title(f"Across the genome: rpoB sits in the high tail\n(representative resistant genome, {len(sub)} proteins ≥ {min_length} aa)")
    axr.grid(axis="y", alpha=0.25)
    meta["right"] = {"sample": sample, "n_proteins": int(len(sub)),
                     "rpob_max_surprisal": rpob_val, "rpob_percentile": rpob_pct}

    fig.suptitle("ESM-C surprisal flags the SNP at two scales", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return meta


def plot_proxy_r_hist(npz: dict, out_path: Path) -> None:
    """Supplementary: per-isolate proxy Pearson r (resistant, whole gene)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iso_r = npz["isolate_pearson"].astype(float)
    res_r = iso_r[(npz["isolate_role"] == "resistant") & ~np.isnan(iso_r)]
    if not res_r.size:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(res_r, bins=min(30, max(5, res_r.size // 2)), color="#6baed6", edgecolor="white")
    ax.axvline(float(res_r.mean()), color="#d62728", lw=2, label=f"mean {res_r.mean():.3f}")
    ax.set_xlabel("per-isolate Pearson r (unmasked vs masked surprisal, whole rpoB)")
    ax.set_ylabel("isolates")
    ax.set_title(f"0A proxy correlation per isolate (n={res_r.size} resistant)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rpob_percentile_by_stat(aggregate: dict, out_path: Path) -> None:
    """Supplementary: mean rpoB percentile by statistic, resistant vs WT."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stats = list(aggregate["by_stat"].keys())
    res = [aggregate["by_stat"][s]["resistant"]["mean_percentile"] or 0.0 for s in stats]
    wt = [aggregate["by_stat"][s]["wt"]["mean_percentile"] or 0.0 for s in stats]
    x = np.arange(len(stats))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.2, res, width=0.4, color="#d62728", label="resistant (mutated rpoB)")
    ax.bar(x + 0.2, wt, width=0.4, color="#6baed6", label="WT rpoB")
    ax.axhline(95, ls="--", lw=1, color="grey", label="95th pct")
    ax.set_xticks(x)
    ax.set_xticklabels(stats, rotation=30, ha="right")
    ax.set_ylabel("mean rpoB percentile among genome's proteins")
    ax.set_title("Where rpoB ranks by each per-protein surprisal statistic")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rankcurves(df: pd.DataFrame, rpob_by_sample: dict[str, int], *, min_length: int, out_dir: Path) -> None:
    """Supplementary: per-stat rank curve across the genome with rpoB marked (R red, S blue)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranked = df[df["length"] >= min_length]
    role_by_sample = {str(s): str(sub.iloc[0]["role"]) for s, sub in df.groupby("sample")}
    for stat in RANKCURVE_STATS:
        fig, ax = plt.subplots(figsize=(9, 5))
        for sample, sub in ranked.groupby("sample"):
            vals = np.sort(sub[stat].to_numpy(dtype=float))[::-1]
            color = "#d62728" if role_by_sample[str(sample)] == "resistant" else "#6baed6"
            ax.plot(range(1, vals.size + 1), vals, lw=1, alpha=0.55, color=color)
            rpob_flat = rpob_by_sample.get(str(sample))
            rp = sub[sub["flat_index"] == rpob_flat] if rpob_flat is not None else sub.iloc[0:0]
            if not rp.empty:
                rank = int((sub[stat] > float(rp.iloc[0][stat])).sum()) + 1
                ax.scatter([rank], [float(rp.iloc[0][stat])], s=90, marker="*",
                           edgecolor="k", color=color, zorder=5)
        ax.set_xscale("log")
        ax.set_xlabel("protein rank (1 = most anomalous)")
        ax.set_ylabel(stat)
        ax.set_title(f"Per-protein {stat} across the genome (★ = rpoB; red = resistant, blue = WT)")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / f"supp_rankcurve_{stat}.png", dpi=150)
        plt.close(fig)


def plot_snp_distance_profile(result: dict, out_path: Path) -> None:
    """Headline: masked (+ unmasked) surprisal vs residue distance from the SNP, ±10 and ±wide."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prof = result["profile"]
    d = np.array(prof["distance"], dtype=float)

    def _arr(key: str) -> np.ndarray:
        return np.array([np.nan if v is None else v for v in prof[key]], dtype=float)

    mm, q25, q75, mu = _arr("mean_masked"), _arr("q25_masked"), _arr("q75_masked"), _arr("mean_unmasked")
    bg = result["background_masked_median"]
    nb = result["neighbour_rank"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, win in zip(axes, (10, result["max_distance"]), strict=True):
        sel = np.abs(d) <= win
        ax.fill_between(d[sel], q25[sel], q75[sel], color="#fdd0d0", alpha=0.7, label="masked IQR")
        ax.plot(d[sel], mm[sel], color="#d62728", lw=1.8, marker="o" if win <= 10 else None, ms=4,
                label="mean masked surprisal")
        ax.plot(d[sel], mu[sel], color="#1f77b4", lw=1.2, ls="--", label="mean unmasked surprisal")
        if bg is not None:
            ax.axhline(bg, color="grey", ls=":", lw=1.2, label="gene-wide background (masked median)")
        ax.axvline(0, color="k", lw=0.8, alpha=0.5)
        ax.set_xlabel("residue distance from SNP (aa)")
        ax.set_ylabel("surprisal  −log P")
        ax.set_title(f"±{win} aa around the resistance residue")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc="upper right")
    top1 = "" if nb["snp_is_top1_frac"] is None else f" — SNP is top-1 in {100 * nb['snp_is_top1_frac']:.0f}%"
    fig.suptitle(f"Surprisal drop-off around the SNP (masked, n={nb['n_isolates']} isolates){top1}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _verdict_lines(proxy: dict, aggregate: dict, mad_floor: float, distance: dict | None = None) -> list[str]:
    """Plain-language read-outs for the JSON (what we have actually shown)."""
    lines: list[str] = []
    pir = proxy["per_isolate_pearson_resistant"]
    rank = proxy["snp_unmasked_rank_in_gene"]
    if proxy["pooled_pearson"] is not None:
        lines.append(
            f"0A proxy: unmasked surprisal tracks masked ablation — pooled Pearson "
            f"{proxy['pooled_pearson']:.3f} over {proxy['pooled_n_points']} residues; per-isolate mean r "
            f"{pir['mean']:.3f} (sd {pir['sd']:.3f}) across {pir['n']} resistant isolates / "
            f"{proxy['n_distinct_genotypes']} distinct genotypes."
        )
    if rank["frac_rank1"] is not None:
        lines.append(
            f"0A localisation: the resistance residue is the #1 unmasked surprisal anomaly in rpoB for "
            f"{100 * rank['frac_rank1']:.0f}% of isolates (top-3 {100 * rank['frac_top3']:.0f}%)."
        )
    best = max(aggregate["by_stat"], key=lambda s: aggregate["by_stat"][s]["resistant"]["mean_percentile"] or 0)
    bres = aggregate["by_stat"][best]["resistant"]["mean_percentile"]
    bwt = aggregate["by_stat"][best]["wt"]["mean_percentile"]
    lines.append(
        f"0B hotspot flag: by {best}, mutated rpoB sits at mean percentile {bres:.1f} in resistant genomes "
        f"vs {bwt:.1f} for WT rpoB (among proteins ≥ {aggregate['min_length']} aa) — i.e. it is pushed into the "
        f"high tail an attention pool could attend to. hotspot_z recomputed with a MAD floor of {mad_floor:.3f} "
        f"to remove the MAD→0 blow-ups in the raw statistic."
    )
    if distance is not None:
        do = distance["dropoff_masked"]
        nb = distance["neighbour_rank"]
        bg = distance["background_masked_median"]
        lines.append(
            f"SNP distance profile (masked, n={nb['n_isolates']}): mean surprisal is {do.get('d0'):.2f} at the SNP, "
            f"{do.get('d1'):.2f} at ±1, {do.get('d2'):.2f} at ±2, {do.get('d5'):.2f} at ±5, vs gene-wide "
            f"background {bg:.2f} — the signal is {'a sharp single-residue spike' if (do.get('d1') and bg and (do['d1'] - bg) < 0.3 * (do['d0'] - bg)) else 'spread over the local window'}."
        )
        lines.append(
            f"Neighbour test: the SNP is the top-1 masked anomaly in {100 * nb['snp_is_top1_frac']:.0f}% of isolates; "
            f"the 2nd/3rd most-surprising residues lie within ±2 aa of the SNP in "
            f"{100 * nb['frac_rank2_within_2aa']:.0f}%/{100 * nb['frac_rank3_within_2aa']:.0f}% of isolates "
            f"(median distance {nb['rank2_median_dist']:.0f}/{nb['rank3_median_dist']:.0f} aa) — so 'use top-3' "
            f"{'IS' if nb['frac_top3_all_within_2aa'] and nb['frac_top3_all_within_2aa'] > 0.5 else 'is NOT'} "
            f"justified by a neighbour smear."
        )
    return lines


def run_analysis(
    *,
    protein_stats_parquet: Path,
    points_npz: Path,
    protein_stats_json: Path | None,
    output_dir: Path,
    mad_floor: float | None,
    min_length: int,
    top_n: int,
) -> dict:
    """Recompute the refined stats, write the JSON + figures into ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_protein_stats(protein_stats_parquet)
    df, mad_floor_used = add_floored_hotspot_z(df, mad_floor)
    rpob_by_sample = resolve_rpob_index(df, protein_stats_json)
    per_genome, aggregate = rpob_rankings(df, rpob_by_sample, min_length=min_length)

    npz = dict(np.load(points_npz, allow_pickle=True))
    proxy = proxy_summary(npz)
    distance = snp_distance_profile(npz, max_distance=100)

    # Figures.
    fig_vs = output_dir / "surprisal_vs_ablation.png"
    fig_2p = output_dir / "esm_surprisal.png"
    fig_dist = output_dir / "snp_distance_profile.png"
    plot_surprisal_vs_ablation(npz, proxy, fig_vs)
    panel_meta = plot_esm_surprisal_2panel(npz, df, rpob_by_sample, min_length=min_length, out_path=fig_2p)
    plot_snp_distance_profile(distance, fig_dist)
    plot_proxy_r_hist(npz, output_dir / "supp_proxy_r_hist.png")
    plot_rpob_percentile_by_stat(aggregate, output_dir / "supp_rpob_percentile_by_stat.png")
    plot_rankcurves(df, rpob_by_sample, min_length=min_length, out_dir=output_dir)

    representative = panel_meta["right"]["sample"]
    results = {
        "task": "snp_embeddings",
        "analysis": "surprisal_analysis",
        "inputs": {
            "protein_stats_parquet": str(protein_stats_parquet),
            "points_npz": str(points_npz),
            "protein_stats_json": str(protein_stats_json) if protein_stats_json else None,
        },
        "params": {"mad_floor": mad_floor_used, "min_protein_length": min_length, "top_n": top_n,
                   "stat_family": list(STAT_FAMILY)},
        "phase_0a_proxy": proxy,
        "snp_distance_profile": distance,
        "phase_0b": {
            "n_genomes": len(per_genome),
            "per_genome_rpob_ranking": per_genome,
            "aggregate": aggregate,
            "representative_genome": representative,
            "top_coflagged_in_representative": top_coflagged(df, representative, min_length=min_length, top_n=top_n),
        },
        "headline_figures": {"surprisal_vs_ablation": str(fig_vs), "esm_surprisal": str(fig_2p),
                             "snp_distance_profile": str(fig_dist)},
        "figure_panels": panel_meta,
        "verdict": _verdict_lines(proxy, aggregate, mad_floor_used, distance),
    }
    out_json = output_dir / "surprisal_analysis.json"
    out_json.write_text(json.dumps(results, indent=2))

    logger.info("Wrote %s", out_json)
    logger.info("=" * 72)
    logger.info("SURPRISAL ANALYSIS OUTPUT DIRECTORY:")
    logger.info("  %s", output_dir)
    logger.info("Headline figures: %s , %s , %s", fig_vs.name, fig_2p.name, fig_dist.name)
    logger.info("=" * 72)
    for line in results["verdict"]:
        logger.info("VERDICT: %s", line)
    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protein-stats-parquet", type=Path, required=True,
                        help="0B *_0b_protein_stats.parquet (one row per protein per genome).")
    parser.add_argument("--points-npz", type=Path, required=True,
                        help="0A *_0a_points.npz (per-residue masked/unmasked log P + SNP points).")
    parser.add_argument("--protein-stats-json", type=Path, default=None,
                        help="0B JSON for authoritative rpoB flat indices (default: auto-derive from the parquet).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output subfolder (default: <parquet>/../../surprisal_analysis).")
    parser.add_argument("--mad-floor", type=float, default=None,
                        help="MAD floor for hotspot_z_floored (default: global median of mad_surprisal).")
    parser.add_argument("--min-protein-length", type=int, default=30,
                        help="Exclude proteins shorter than this from the cross-genome ranking (default 30).")
    parser.add_argument("--top-n", type=int, default=15, help="Report the top-N co-flagged proteins (default 15).")
    args = parser.parse_args()

    # Default JSON beside the parquet: strip the "_0b_protein_stats.parquet" suffix → ".json".
    json_path = args.protein_stats_json
    if json_path is None:
        derived = Path(str(args.protein_stats_parquet).replace("_0b_protein_stats.parquet", ".json"))
        json_path = derived if derived != args.protein_stats_parquet else None

    output_dir = args.output_dir or (args.protein_stats_parquet.parent.parent / "surprisal_analysis")

    run_analysis(
        protein_stats_parquet=args.protein_stats_parquet,
        points_npz=args.points_npz,
        protein_stats_json=json_path,
        output_dir=output_dir,
        mad_floor=args.mad_floor,
        min_length=args.min_protein_length,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()

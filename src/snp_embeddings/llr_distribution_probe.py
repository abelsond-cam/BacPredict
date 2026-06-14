r"""Phase 0 surprisal diagnostic — proxy proof (0A) + per-protein SNP-flag (0B).

Experiment 4 wants to hand each protein an undiluted **per-protein anomaly feature**
(a scalar derived from its residue "surprisal") and let an **attention pool** upweight
the SNP-bearing protein out of the ~4,000 in a genome. Two separable questions:

**0A — is the cheap proxy faithful? (publication-grade, n isolates).** The gold
standard is masked-marginal ablation (``log P(observed | context\i)``, one forward per
residue → ``L`` per protein). The cheap genome-wide signal is the unmasked naturalness
(``log P(observed | full context)``, one forward per protein). For each of N resistant
isolates, score **both over the whole rpoB gene** (``--masked-scope gene``) and:

- per-isolate Pearson/Spearman of masked vs unmasked across all residues;
- the **across-isolate** scatter at the mutated residue (masked vs unmasked, n points);
- the fraction where the resistance residue is the **top unmasked anomaly** in rpoB;
- the **distinct-genotype count**, so an "n=100" claim is honest about how many
  independent mutations it spans.

**0B — does a per-protein summary act as a sparse "a SNP is here" flag? (a handful of
genomes).** This is *not* about resistant-vs-susceptible magnitude within one protein,
nor one neighbour — it is whether, **across all ~4,000 proteins of a genome**, a
per-protein statistic singles the SNP-bearing protein(s) into a short high tail an
attention pool can exploit (long conserved genes carry surprising residues too, so the
tail may not be sparse). We compute a **list** of candidate per-protein statistics
(max-surprisal, hotspot-z, max−p99, top1−top2, …; see :func:`protein_surprisal_stats`),
persist every protein's row to a parquet sidecar so new statistics can be added without
re-running, and report where mutated rpoB ranks among the ~4,000 by each, plus what
*else* gets flagged.

Read-only — no training, no embedding store (runs from the protein parquets + the pinned
ESM-C MLM). Reuses :func:`snp_embeddings.snp_vs_esm_prediction.resolve_clean_splits`,
:func:`snp_embeddings.rpob_genotype.build_genotype_table` /
:func:`~snp_embeddings.rpob_genotype.sample_codon_positions`,
:func:`snp_embeddings.locate_gene.flatten_proteins`, and
:func:`tl.embed.esm_residue_level.unmasked_logprobs` / ``masked_logprobs``.
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, pearsonr, skew, spearmanr

from snp_embeddings.locate_gene import flatten_proteins
from snp_embeddings.rpob_genotype import (
    RRDR_FIRST_CODON,
    RRDR_LAST_CODON,
    build_genotype_table,
    load_reference,
    ref_index_for_codon,
    sample_codon_positions,
)
from snp_embeddings.snp_vs_esm_prediction import resolve_clean_splits
from tl.embed.esm_residue_level import load_esmc_mlm, masked_logprobs, unmasked_logprobs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Columns the all-protein flatten needs (matches build_genotype_table's subset read).
_NEEDED_COLS = ["contig_idx", "gene_name", "protein_sequence"]

# Per-protein statistics ranked to test the "a SNP is here" flag (higher = more
# anomalous → should sit in the high tail). Plotted subset is PLOT_STATS.
RANK_STATS = ("max_surprisal", "hotspot_z", "max_minus_p99", "top1_minus_top2")
PLOT_STATS = ("max_surprisal", "hotspot_z", "max_minus_p99")
_EPS = 1e-6


# ---------------------------------------------------------------------------
# Isolate selection + genotype helpers
# ---------------------------------------------------------------------------


def hotspot_codons(genotype_row: pd.Series, reference: str) -> list[tuple[int, str, str]]:
    """RRDR codons where this sample differs from wild-type → ``(codon, wt, observed)``."""
    hotspots: list[tuple[int, str, str]] = []
    for codon in range(RRDR_FIRST_CODON, RRDR_LAST_CODON + 1):
        observed = genotype_row[f"codon_{codon}"]
        wt = reference[ref_index_for_codon(reference, codon)]
        if observed not in ("-", wt):
            hotspots.append((codon, wt, observed))
    return hotspots


def hotspot_label(hotspots: list[tuple[int, str, str]]) -> str:
    """A compact genotype label, e.g. ``"S450L"`` or ``"D435V+H445Y"`` (``"none"`` if empty)."""
    return "+".join(f"{wt}{codon}{alt}" for codon, wt, alt in hotspots) or "none"


def select_isolates(
    genotype: pd.DataFrame,
    label_map: dict[str, int],
    reference: str,
    *,
    n_resistant: int,
    n_wt: int,
    diverse: bool = True,
) -> tuple[list[str], list[str]]:
    """Pick resistant rpoB-mutant isolates and WT/susceptible controls from the genotype.

    Resistant = label 1 with ≥1 RRDR substitution (a hotspot to centre on); WT = label 0
    with zero RRDR substitutions. With ``diverse`` the resistant picks span **distinct
    hotspot codon-sets** (so clonal S450L alleles — the dominant ~70% RIF-R mutation, with
    a highly-conserved byte-identical rpoB — don't fill every slot); without it the first
    matching isolates in table order are taken (the natural mutation distribution, for the
    n=100 proof). Table order throughout for reproducibility.
    """
    resistant: list[str] = []
    wt: list[str] = []
    seen_signatures: set[frozenset[int]] = set()
    for sample_id, row in genotype.iterrows():
        sid = str(sample_id)
        label = label_map.get(sid)
        n_sub = int(row["n_rrdr_substitutions"])
        if label == 1 and n_sub >= 1 and len(resistant) < n_resistant:
            signature = frozenset(codon for codon, _wt, _alt in hotspot_codons(row, reference))
            if not (diverse and signature in seen_signatures):
                seen_signatures.add(signature)
                resistant.append(sid)
        elif label == 0 and n_sub == 0 and len(wt) < n_wt:
            wt.append(sid)
        if len(resistant) >= n_resistant and len(wt) >= n_wt:
            break
    return resistant, wt


def _rank_ascending(values: np.ndarray, idx: int) -> int:
    """Ascending rank of ``values[idx]`` (1 = lowest). For log P, rank 1 = most surprising."""
    return int((values < values[idx]).sum()) + 1


def _zscore(values: np.ndarray, idx: int) -> float:
    """z-score of ``values[idx]`` (0 if the spread is degenerate)."""
    std = float(values.std())
    return float((values[idx] - float(values.mean())) / std) if std > 0 else 0.0


# ---------------------------------------------------------------------------
# 0A — whole-gene masked-vs-unmasked proxy proof
# ---------------------------------------------------------------------------


def isolate_proxy(
    model,
    tokenizer,
    seq: str,
    reference: str,
    hotspots: list[tuple[int, str, str]],
    *,
    sample_id: str,
    role: str,
    scope: str,
    window: int,
    device: str,
) -> dict:
    """Masked + unmasked surprisal for one isolate's rpoB; correlate, locate the SNP.

    ``scope='gene'`` masks **every** residue (the publication proof); ``scope='window'``
    masks only ``±window`` around the hotspot(s) (cheap smoke). Returns a record plus the
    raw per-position arrays (for the pooled NPZ) under ``_arrays``.
    """
    length = len(seq)
    codon_to_pos = sample_codon_positions(seq, reference, [c for c, _w, _a in hotspots]) if hotspots else {}
    hotspot_pos = {c: p for c, p in codon_to_pos.items() if p is not None}

    if scope == "gene":
        positions = list(range(length))
    else:
        centres = sorted(hotspot_pos.values()) or [length // 2]
        keep: set[int] = set()
        for c in centres:
            keep.update(range(max(0, c - window), min(length, c + window + 1)))
        positions = sorted(keep)

    masked = masked_logprobs(model, tokenizer, seq, positions=positions, device=device).numpy()
    unmasked_full = unmasked_logprobs(model, tokenizer, seq, device=device).numpy()
    unmasked = unmasked_full[positions]
    pos_to_idx = {p: i for i, p in enumerate(positions)}

    spread_ok = masked.size > 2 and masked.std() > 0 and unmasked.std() > 0
    pearson_r = float(pearsonr(masked, unmasked)[0]) if spread_ok else None
    spearman_r = float(spearmanr(masked, unmasked)[0]) if spread_ok else None

    snp_records = []
    for codon, wt, alt in hotspots:
        p = hotspot_pos.get(codon)
        if p is None or p not in pos_to_idx:
            continue
        i = pos_to_idx[p]
        snp_records.append({
            "codon": codon, "wt": wt, "alt": alt, "position": int(p),
            "masked_logp": float(masked[i]), "unmasked_logp": float(unmasked[i]),
            "masked_z_scope": _zscore(masked, i), "unmasked_z_scope": _zscore(unmasked, i),
            "masked_rank_scope": _rank_ascending(masked, i), "unmasked_rank_scope": _rank_ascending(unmasked, i),
            "unmasked_rank_gene": _rank_ascending(unmasked_full, p),
            "masked_scope_n": int(masked.size), "gene_length": length,
        })

    record = {
        "sample": sample_id, "role": role, "genotype": hotspot_label(hotspots),
        "gene_length": length, "scope": scope, "n_scored": int(masked.size),
        "pearson_masked_vs_unmasked": pearson_r, "spearman_masked_vs_unmasked": spearman_r,
        "snp": snp_records,
        "_arrays": {"masked": masked, "unmasked": unmasked, "positions": np.array(positions)},
    }
    return record


def run_0a(
    model,
    tokenizer,
    genotype: pd.DataFrame,
    resistant_ids: list[str],
    wt_ids: list[str],
    reference: str,
    *,
    scope: str,
    window: int,
    device: str,
) -> tuple[dict, dict]:
    """0A over all selected isolates → (JSON payload, NPZ array bundle)."""
    records: list[dict] = []
    for role, ids in (("resistant", resistant_ids), ("wt", wt_ids)):
        for n, sid in enumerate(ids):
            hotspots = hotspot_codons(genotype.loc[sid], reference) if role == "resistant" else []
            logger.info("0A %s %d/%d %s (%s)", role, n + 1, len(ids), sid, hotspot_label(hotspots))
            records.append(isolate_proxy(
                model, tokenizer, genotype.loc[sid, "rpob_sequence"], reference, hotspots,
                sample_id=sid, role=role, scope=scope, window=window, device=device,
            ))

    # Pooled (isolate, position) points + per-isolate / per-SNP aggregates.
    pooled_m, pooled_u, pooled_idx = [], [], []
    iso_pearson, iso_spearman, iso_labels, iso_roles = [], [], [], []
    snp_masked, snp_unmasked, snp_rank_gene, snp_labels = [], [], [], []
    for k, rec in enumerate(records):
        a = rec["_arrays"]
        pooled_m.append(a["masked"])
        pooled_u.append(a["unmasked"])
        pooled_idx.append(np.full(a["masked"].size, k))
        iso_pearson.append(rec["pearson_masked_vs_unmasked"])
        iso_spearman.append(rec["spearman_masked_vs_unmasked"])
        iso_labels.append(rec["genotype"])
        iso_roles.append(rec["role"])
        if rec["snp"]:
            primary = rec["snp"][0]
            snp_masked.append(primary["masked_logp"])
            snp_unmasked.append(primary["unmasked_logp"])
            snp_rank_gene.append(primary["unmasked_rank_gene"])
            snp_labels.append(rec["genotype"])
        del rec["_arrays"]

    pm, pu = np.concatenate(pooled_m), np.concatenate(pooled_u)
    pooled_r = float(pearsonr(pm, pu)[0]) if pm.size > 2 else None
    pooled_rho = float(spearmanr(pm, pu)[0]) if pm.size > 2 else None
    res_pearson = [r for r, role in zip(iso_pearson, iso_roles, strict=True) if role == "resistant" and r is not None]
    sm, su = np.array(snp_masked), np.array(snp_unmasked)
    snp_scatter_r = float(pearsonr(sm, su)[0]) if sm.size > 2 and sm.std() > 0 and su.std() > 0 else None
    snp_scatter_rho = float(spearmanr(sm, su)[0]) if sm.size > 2 and sm.std() > 0 and su.std() > 0 else None
    ranks = np.array(snp_rank_gene) if snp_rank_gene else np.array([])

    summary = {
        "n_isolates": len(records),
        "n_resistant": len(resistant_ids),
        "n_wt": len(wt_ids),
        "distinct_genotypes": dict(Counter(snp_labels)),
        "n_distinct_genotypes": len(set(snp_labels)),
        "scope": scope,
        "pooled_pearson": pooled_r,
        "pooled_spearman": pooled_rho,
        "pooled_n_points": int(pm.size),
        "per_isolate_pearson_resistant": {
            "mean": float(np.mean(res_pearson)) if res_pearson else None,
            "median": float(np.median(res_pearson)) if res_pearson else None,
            "sd": float(np.std(res_pearson)) if res_pearson else None,
            "min": float(np.min(res_pearson)) if res_pearson else None,
            "n": len(res_pearson),
        },
        "snp_site_scatter": {"pearson": snp_scatter_r, "spearman": snp_scatter_rho, "n": int(sm.size)},
        "snp_unmasked_rank_in_gene": {
            "frac_rank1": float((ranks == 1).mean()) if ranks.size else None,
            "frac_top3": float((ranks <= 3).mean()) if ranks.size else None,
            "median_rank": float(np.median(ranks)) if ranks.size else None,
            "n": int(ranks.size),
        },
    }
    payload = {"summary": summary, "isolates": records}
    arrays = {
        "pooled_masked": pm, "pooled_unmasked": pu, "pooled_isolate_idx": np.concatenate(pooled_idx),
        "isolate_pearson": np.array([r if r is not None else np.nan for r in iso_pearson]),
        "isolate_label": np.array(iso_labels), "isolate_role": np.array(iso_roles),
        "snp_masked": sm, "snp_unmasked": su, "snp_rank_gene": ranks, "snp_label": np.array(snp_labels),
    }
    return payload, arrays


# ---------------------------------------------------------------------------
# 0B — per-protein surprisal statistics across the whole genome
# ---------------------------------------------------------------------------


def gini_coefficient(values: np.ndarray) -> float | None:
    """Gini coefficient of non-negative ``values`` (0 = all equal, →1 = one value hogs it).

    Concentration measure for the per-protein surprisal: high when a single residue carries
    most of the surprisal mass. Sorted O(n log n) form; ``None`` for empty, 0.0 for zero-sum.
    """
    x = np.sort(np.asarray(values, dtype=float))
    n = x.size
    if n == 0:
        return None
    total = float(x.sum())
    if total <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * x)) / (n * total) - (n + 1) / n)


def protein_surprisal_stats(logp: np.ndarray) -> dict:
    """Candidate per-protein "a SNP is here" statistics from one protein's residue log P.

    ``surprisal = -log P`` (higher = more anomalous; non-negative). Two complementary groups:
    **magnitude/level** (how high the top residues are — ``max_surprisal``, ``mean_top3``) and
    **concentration/shape** (how *isolated* the peak is — the SNP signature, one residue standing
    out from an otherwise-conserved protein, vs a uniformly hard / multiply-mutated protein).
    Self-normalising shape stats (``self_z``, ``self_z_trimmed``) shrink when the protein's own
    residue spread is large, which is the correction we want against hypervariable proteins. Easy
    to extend: add a key here and it flows to the parquet sidecar + ranking.
    """
    surprisal = -np.asarray(logp, dtype=float)
    n = surprisal.size
    order = np.sort(surprisal)[::-1]  # descending: most surprising first
    top1 = float(order[0])
    top2 = float(order[1]) if n > 1 else None
    median = float(np.median(surprisal))
    mad = float(np.median(np.abs(surprisal - median)))
    p99 = float(np.percentile(surprisal, 99))
    p95 = float(np.percentile(surprisal, 95))
    mean = float(surprisal.mean())
    std = float(surprisal.std())
    total = float(surprisal.sum())
    sumsq = float(np.square(surprisal).sum())
    # self-z with the top residue excluded from the centre/scale (no self-contamination)
    self_z_trimmed = None
    if n > 6:
        rest = order[3:]
        rstd = float(rest.std())
        self_z_trimmed = (top1 - float(rest.mean())) / rstd if rstd > 0 else None
    return {
        "length": int(n),
        # magnitude / level
        "max_surprisal": top1,
        "mean_top3": float(order[: min(3, n)].mean()),
        "top2_surprisal": top2,
        # concentration / shape
        "top1_minus_top2": (top1 - top2) if top2 is not None else None,
        "top_minus_mean_rest": (top1 - (total - top1) / (n - 1)) if n > 1 else None,
        "self_z": (top1 - mean) / std if std > 0 else None,
        "self_z_trimmed": self_z_trimmed,
        "hotspot_z": (top1 - median) / (mad + _EPS),
        "participation_ratio": (total * total) / sumsq if sumsq > 0 else None,
        "gini": gini_coefficient(surprisal),
        "skew_surprisal": float(skew(surprisal)) if n > 2 else None,
        "kurtosis_surprisal": float(kurtosis(surprisal)) if n > 3 else None,
        # extra contrasts / moments (kept for continuity)
        "max_minus_p99": top1 - p99,
        "max_minus_p95": top1 - p95,
        "max_minus_median": top1 - median,
        "median_surprisal": median,
        "mad_surprisal": mad,
        "p99_surprisal": p99,
        "p95_surprisal": p95,
    }


def genome_protein_flags(
    model,
    tokenizer,
    records: list[dict],
    *,
    device: str,
    max_proteins: int | None = None,
) -> list[dict]:
    """Unmasked surprisal stats for every protein in a genome (flat order preserved)."""
    rows: list[dict] = []
    use = records if max_proteins is None else records[:max_proteins]
    for n, rec in enumerate(use):
        seq = rec["protein_sequence"]
        if not seq:
            continue
        logp = unmasked_logprobs(model, tokenizer, seq, device=device).numpy()
        rows.append({"flat_index": rec["flat_index"], "gene_name": rec["gene_name"], **protein_surprisal_stats(logp)})
        if (n + 1) % 1000 == 0:
            logger.info("  ...scored %d/%d proteins", n + 1, len(use))
    return rows


def _rpob_ranking(flags: pd.DataFrame, rpob_flat_index: int, *, min_length: int) -> dict:
    """Where mutated rpoB ranks among the genome's proteins by each candidate statistic."""
    ranked = flags[flags["length"] >= min_length].copy()
    out: dict = {"n_ranked": int(len(ranked)), "min_length": min_length, "by_stat": {}}
    rpob_rows = ranked[ranked["flat_index"] == rpob_flat_index]
    if rpob_rows.empty:
        out["rpob_present_in_ranking"] = False
        return out
    out["rpob_present_in_ranking"] = True
    out["rpob_gene_name"] = str(rpob_rows.iloc[0]["gene_name"])
    n = len(ranked)
    for stat in RANK_STATS:
        if stat not in ranked.columns:
            continue
        s = ranked[stat]
        rpob_val = float(rpob_rows.iloc[0][stat]) if pd.notna(rpob_rows.iloc[0][stat]) else None
        if rpob_val is None:
            continue
        rank = int((s > rpob_val).sum()) + 1  # 1 = most anomalous
        out["by_stat"][stat] = {
            "rpob_value": rpob_val, "rpob_rank": rank, "rpob_percentile": float(100.0 * (1 - (rank - 1) / n)),
        }
    return out


def _top_proteins(flags: pd.DataFrame, *, min_length: int, top_n: int) -> dict:
    """The top-N proteins by each plotted statistic (what the flag actually selects)."""
    ranked = flags[flags["length"] >= min_length]
    out: dict = {}
    for stat in PLOT_STATS:
        top = ranked.nlargest(top_n, stat)[["flat_index", "gene_name", stat, "length"]]
        out[stat] = [
            {"flat_index": int(r.flat_index), "gene_name": (None if pd.isna(r.gene_name) else str(r.gene_name)),
             "value": float(getattr(r, stat)), "length": int(r.length)}
            for r in top.itertuples()
        ]
    return out


def run_0b(
    model,
    tokenizer,
    genotype: pd.DataFrame,
    resistant_ids: list[str],
    wt_ids: list[str],
    parquet_dir: Path,
    *,
    device: str,
    min_length: int,
    top_n: int,
    max_proteins: int | None,
) -> tuple[dict, pd.DataFrame]:
    """0B over a handful of genomes → (JSON payload, per-protein parquet frame)."""
    genomes: list[dict] = []
    all_rows: list[pd.DataFrame] = []
    for role, ids in (("resistant", resistant_ids), ("wt", wt_ids)):
        for sid in ids:
            flat_index = int(genotype.loc[sid, "rpob_flat_index"])
            records = flatten_proteins(pd.read_parquet(parquet_dir / f"{sid}_protein_sequences.parquet",
                                                       columns=_NEEDED_COLS))
            logger.info("0B %s %s: scoring %d proteins (rpoB flat idx %d)", role, sid, len(records), flat_index)
            flags = pd.DataFrame(genome_protein_flags(model, tokenizer, records, device=device,
                                                      max_proteins=max_proteins))
            flags.insert(0, "sample", sid)
            flags.insert(1, "role", role)
            all_rows.append(flags)
            genomes.append({
                "sample": sid, "role": role, "n_proteins": int(len(flags)),
                "rpob_flat_index": flat_index,
                "rpob_ranking": _rpob_ranking(flags, flat_index, min_length=min_length),
                "top_proteins": _top_proteins(flags, min_length=min_length, top_n=top_n),
            })

    parquet_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    # Aggregate: rpoB percentile by stat, resistant vs susceptible.
    agg: dict = {"rpob_percentile_by_stat": {}}
    for stat in RANK_STATS:
        for role in ("resistant", "wt"):
            pcts = [g["rpob_ranking"]["by_stat"].get(stat, {}).get("rpob_percentile")
                    for g in genomes if g["role"] == role and g["rpob_ranking"].get("rpob_present_in_ranking")]
            pcts = [p for p in pcts if p is not None]
            agg["rpob_percentile_by_stat"].setdefault(stat, {})[role] = (
                float(np.mean(pcts)) if pcts else None)
    payload = {"summary": {"n_genomes": len(genomes), "min_length": min_length,
                           "rank_stats": list(RANK_STATS), "aggregate": agg},
               "genomes": genomes}
    return payload, parquet_df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_0a(payload: dict, arrays: dict, out_dir: Path) -> None:
    """0A: across-isolate SNP scatter, per-isolate r histogram, pooled hexbin."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sm, su = arrays["snp_masked"], arrays["snp_unmasked"]
    if sm.size:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(sm, su, s=40, alpha=0.6, color="C3")
        lim = [min(sm.min(), su.min()) - 0.5, max(sm.max(), su.max()) + 0.5]
        ax.plot(lim, lim, ls="--", lw=1, color="grey")
        r = payload["summary"]["snp_site_scatter"]["pearson"]
        ax.set_xlabel("masked log P at resistance residue")
        ax.set_ylabel("unmasked log P at resistance residue")
        ax.set_title(f"0A across-isolate SNP scatter (n={sm.size}, Pearson {r:.3f})" if r else "0A SNP scatter")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "phase0a_snp_scatter.png", dpi=150)
        plt.close(fig)

    rp = arrays["isolate_pearson"][arrays["isolate_role"] == "resistant"]
    rp = rp[~np.isnan(rp)]
    if rp.size:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(rp, bins=min(30, max(5, rp.size)), color="C0", alpha=0.8)
        ax.set_xlabel("per-isolate Pearson r (masked vs unmasked, whole gene)")
        ax.set_ylabel("isolates")
        ax.set_title(f"0A per-isolate proxy correlation (n={rp.size} resistant)")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "phase0a_per_isolate_r_hist.png", dpi=150)
        plt.close(fig)

    pm, pu = arrays["pooled_masked"], arrays["pooled_unmasked"]
    if pm.size:
        fig, ax = plt.subplots(figsize=(6, 6))
        hb = ax.hexbin(pm, pu, gridsize=50, bins="log", cmap="viridis")
        fig.colorbar(hb, ax=ax, label="log10(count)")
        ax.set_xlabel("masked log P(observed)")
        ax.set_ylabel("unmasked log P(observed)")
        ax.set_title(f"0A pooled positions (n={pm.size}, Pearson {payload['summary']['pooled_pearson']:.3f})")
        fig.tight_layout()
        fig.savefig(out_dir / "phase0a_pooled_hexbin.png", dpi=150)
        plt.close(fig)


def plot_0b(payload: dict, parquet_df: pd.DataFrame, *, min_length: int, out_dir: Path) -> None:
    """0B: per-stat rank curve (sorted across proteins) with rpoB marked, R vs S."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rpob_by_sample = {g["sample"]: g["rpob_flat_index"] for g in payload["genomes"]}
    role_by_sample = {g["sample"]: g["role"] for g in payload["genomes"]}
    ranked = parquet_df[parquet_df["length"] >= min_length]
    for stat in PLOT_STATS:
        fig, ax = plt.subplots(figsize=(9, 5))
        for sample, sub in ranked.groupby("sample"):
            vals = np.sort(sub[stat].to_numpy(dtype=float))[::-1]
            role = role_by_sample[sample]
            color = "C3" if role == "resistant" else "C0"
            ax.plot(range(1, vals.size + 1), vals, lw=1, alpha=0.6, color=color)
            rp = sub[sub["flat_index"] == rpob_by_sample[sample]]
            if not rp.empty:
                rank = int((sub[stat] > float(rp.iloc[0][stat])).sum()) + 1
                ax.scatter([rank], [float(rp.iloc[0][stat])], s=80, marker="*",
                           edgecolor="k", color=color, zorder=5)
        ax.set_xscale("log")
        ax.set_xlabel("protein rank (1 = most anomalous)")
        ax.set_ylabel(stat)
        ax.set_title(f"0B per-protein {stat} across the genome (★ = rpoB; red R, blue S)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"phase0b_rankcurve_{stat}.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_probe(
    *,
    ast_sheet_path: Path,
    parquet_dir: Path,
    drug: str,
    phase: str,
    device: str,
    scope: str,
    window: int,
    n_resistant: int,
    n_wt: int,
    pool_size: int,
    diverse: bool,
    min_length: int,
    top_n: int,
    max_proteins: int | None,
    out_json: Path,
    out_dir: Path,
    qc_log_path: Path,
) -> dict:
    """Genotype a pool, select isolates, run the requested phase(s); write JSON + sidecars + plots."""
    reference = load_reference()
    label_map, train_ids, validate_ids, evaluate_ids, split_info = resolve_clean_splits(ast_sheet_path, drug)
    pool = [*train_ids, *validate_ids, *evaluate_ids][:pool_size]
    logger.info("Genotyping a pool of %d labelled samples to source isolates", len(pool))
    genotype = build_genotype_table(pool, parquet_dir, reference, qc_log_path=qc_log_path)

    resistant_ids, wt_ids = select_isolates(
        genotype, label_map, reference, n_resistant=n_resistant, n_wt=n_wt, diverse=diverse
    )
    if not resistant_ids:
        raise RuntimeError("No resistant rpoB-mutant isolate found in the pool — raise --pool-size.")
    logger.info("Selected %d resistant + %d wt isolates", len(resistant_ids), len(wt_ids))

    model, tokenizer = load_esmc_mlm(device=device)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        "schema_version": "2.0",
        "task": "snp_embeddings",
        "analysis": "llr_distribution_probe",
        "drug": drug,
        "device": device,
        "phase": phase,
        "reference": "UniProt P9WGY9 (H37Rv rpoB)",
        "split": split_info,
        "selected": {"resistant": resistant_ids, "wt": wt_ids, "pool_size": len(pool),
                     "n_single_copy_genotyped": int(len(genotype))},
    }

    if phase in ("0a", "both"):
        payload_0a, arrays = run_0a(model, tokenizer, genotype, resistant_ids, wt_ids, reference,
                                    scope=scope, window=window, device=device)
        payload["phase_0a"] = payload_0a
        npz_path = out_json.with_name(out_json.stem + "_0a_points.npz")
        np.savez(npz_path, **arrays)
        logger.info("Wrote %s", npz_path)
        plot_0a(payload_0a, arrays, out_dir)

    if phase in ("0b", "both"):
        payload_0b, parquet_df = run_0b(model, tokenizer, genotype, resistant_ids, wt_ids, parquet_dir,
                                        device=device, min_length=min_length, top_n=top_n, max_proteins=max_proteins)
        payload["phase_0b"] = payload_0b
        if not parquet_df.empty:
            pq_path = out_json.with_name(out_json.stem + "_0b_protein_stats.parquet")
            parquet_df.to_parquet(pq_path, index=False)
            logger.info("Wrote %s (%d protein rows)", pq_path, len(parquet_df))
            plot_0b(payload_0b, parquet_df, min_length=min_length, out_dir=out_dir)

    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    payload["host"] = socket.gethostname()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %s + plots in %s", out_json, out_dir)
    return payload


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ast-sheet-path", type=Path, required=True,
                        help="binary_ast_with_split.csv (Sample/phenotype-BioSample_ID, drug, train_val_eval).")
    parser.add_argument("--parquet-dir", type=Path, required=True, help="Dir of *_protein_sequences.parquet.")
    parser.add_argument("--output-json", type=Path, required=True, help="Where to write the probe JSON.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Dir for plots + sidecars (default: JSON's dir).")
    parser.add_argument("--drug", type=str, default="rifampin", help="Phenotype column (default rifampin).")
    parser.add_argument("--phase", choices=["0a", "0b", "both"], default="both",
                        help="0a = whole-gene proxy proof; 0b = cross-genome per-protein flag (default both).")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device (default cpu — Stage-A smoke).")
    parser.add_argument("--masked-scope", choices=["gene", "window"], default="gene",
                        help="0A masked coverage: 'gene' (every residue, the proof) or 'window' (±W, cheap smoke).")
    parser.add_argument("--window", type=int, default=25,
                        help="±W residues around the hotspot when --masked-scope window (default 25).")
    parser.add_argument("--n-resistant", type=int, default=3, help="Resistant rpoB-mutant isolates (default 3).")
    parser.add_argument("--n-wt", type=int, default=3, help="WT/susceptible control isolates (default 3).")
    parser.add_argument("--pool-size", type=int, default=500,
                        help="Labelled samples to genotype when sourcing isolates (default 500).")
    parser.add_argument("--diverse-hotspots", action=argparse.BooleanOptionalAction, default=True,
                        help="Span distinct hotspot codon-sets for the resistant picks (default on; "
                             "--no-diverse-hotspots takes the natural distribution, for the n=100 proof).")
    parser.add_argument("--min-protein-length", type=int, default=30,
                        help="0B: exclude proteins shorter than this from the cross-genome ranking (default 30).")
    parser.add_argument("--top-n", type=int, default=20, help="0B: report the top-N flagged proteins per stat.")
    parser.add_argument("--max-proteins-per-genome", type=int, default=None,
                        help="0B: cap proteins scored per genome (CPU smoke; default all).")
    args = parser.parse_args()

    run_probe(
        ast_sheet_path=args.ast_sheet_path,
        parquet_dir=args.parquet_dir,
        drug=args.drug,
        phase=args.phase,
        device=args.device,
        scope=args.masked_scope,
        window=args.window,
        n_resistant=args.n_resistant,
        n_wt=args.n_wt,
        pool_size=args.pool_size,
        diverse=args.diverse_hotspots,
        min_length=args.min_protein_length,
        top_n=args.top_n,
        max_proteins=args.max_proteins_per_genome,
        out_json=args.output_json,
        out_dir=args.output_dir or args.output_json.parent,
        qc_log_path=args.output_json.with_name("rpob_copy_qc.log"),
    )


if __name__ == "__main__":
    main()

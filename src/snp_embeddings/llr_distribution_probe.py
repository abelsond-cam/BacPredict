r"""Phase 0 surprise diagnostic — validate the cheap proxy, then map the distributions.

Experiment 4 wants to hand each protein an undiluted **per-protein anomaly feature**
(the most-surprising-residue log-prob) and let an attention pool upweight the
anomalous protein. Two questions block the architecture, and this one read-only
probe answers both:

**0A — windowed masked-vs-unmasked proxy test.** For a resistant isolate's rpoB,
take the resistance hotspot codon(s) (where the sample differs from H37Rv) and
``±W`` residues either side. At each window position compute the **masked**
surprise (``log P(observed | context\\i)``, ablation-by-masking; the gold standard,
one forward per position) and the **unmasked** surprise (``log P(observed | full
context)``; cheap, one forward for the whole protein). Then:

- show the masked signal is a sharp **trough only at the SNP**, ~flat across the WT
  neighbours → "the only ablation anomaly is the amino-acid SNP";
- **correlate** masked vs unmasked across the window (Pearson + Spearman) → how well
  the cheap unmasked surprise proxies the expensive masked ablation;
- report the SNP's **z-score / rank** in each profile → how much it stands out.

A WT/susceptible isolate's same window is the negative control (no SNP → no trough).

**0B — full-gene + 2-neighbour distributions.** Run **unmasked** surprise across all
positions of rpoB and its two flat-adjacent neighbour proteins (``records[F±1]`` —
genomic neighbours, no resistance mutation), for the resistant and WT isolates.
Summarise each (min / p1 / p5 / mean-of-bottom-3 / skew / kurtosis, and the
resistance residue's rank within rpoB) and plot the distributions. This picks the
per-protein summary statistic (max vs top-k vs tail/skew) → the feature dimension.

Reuses :func:`snp_embeddings.snp_vs_esm_prediction.resolve_clean_splits` (canonical
labels), :func:`snp_embeddings.rpob_genotype.build_genotype_table` (single-copy rpoB
genotype + flat index), :func:`~snp_embeddings.rpob_genotype.sample_codon_positions`
(hotspot → sample coordinate), :func:`snp_embeddings.locate_gene.flatten_proteins`
(rpoB flat index + neighbours) and the pinned ESM-C MLM
(:func:`tl.embed.esm_residue_level.load_esmc_mlm`). Read-only diagnostics — no
training, no embedding store needed (works straight from the protein parquets).
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
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

# Columns the neighbour re-read needs (matches build_genotype_table's subset read).
_NEEDED_COLS = ["contig_idx", "gene_name", "protein_sequence"]


# ---------------------------------------------------------------------------
# Isolate selection + genotype helpers
# ---------------------------------------------------------------------------


def select_isolates(
    genotype: pd.DataFrame,
    label_map: dict[str, int],
    *,
    n_resistant: int,
    n_wt: int,
) -> tuple[list[str], list[str]]:
    """Pick resistant rpoB-mutant isolates and WT/susceptible controls from the genotype.

    Resistant = label 1 with ≥1 RRDR substitution (a hotspot to centre 0A on); WT =
    label 0 with zero RRDR substitutions (no trough expected). Taken in table order
    for reproducibility.
    """
    resistant: list[str] = []
    wt: list[str] = []
    for sample_id, row in genotype.iterrows():
        sid = str(sample_id)
        label = label_map.get(sid)
        n_sub = int(row["n_rrdr_substitutions"])
        if label == 1 and n_sub >= 1 and len(resistant) < n_resistant:
            resistant.append(sid)
        elif label == 0 and n_sub == 0 and len(wt) < n_wt:
            wt.append(sid)
        if len(resistant) >= n_resistant and len(wt) >= n_wt:
            break
    return resistant, wt


def hotspot_codons(genotype_row: pd.Series, reference: str) -> list[tuple[int, str, str]]:
    """RRDR codons where this sample differs from wild-type → ``(codon, wt, observed)``."""
    hotspots: list[tuple[int, str, str]] = []
    for codon in range(RRDR_FIRST_CODON, RRDR_LAST_CODON + 1):
        observed = genotype_row[f"codon_{codon}"]
        wt = reference[ref_index_for_codon(reference, codon)]
        if observed not in ("-", wt):
            hotspots.append((codon, wt, observed))
    return hotspots


def _neighbour_records(parquet_dir: Path, sample_id: str, flat_index: int) -> dict:
    """Flatten one sample's parquet and return its rpoB record + the two flat neighbours.

    Returns ``{"rpob": rec, "neighbour_minus1": rec|None, "neighbour_plus1": rec|None}``
    where each rec is a :func:`~snp_embeddings.locate_gene.flatten_proteins` dict.
    """
    parquet_path = parquet_dir / f"{sample_id}_protein_sequences.parquet"
    df_p = pd.read_parquet(parquet_path, columns=_NEEDED_COLS)
    records = flatten_proteins(df_p)
    rpob = records[flat_index]
    if rpob["gene_name"] is None or str(rpob["gene_name"]).lower() != "rpob":
        raise ValueError(
            f"{sample_id}: flat index {flat_index} is {rpob['gene_name']!r}, not rpoB — flat order misaligned."
        )
    return {
        "rpob": rpob,
        "neighbour_minus1": records[flat_index - 1] if flat_index - 1 >= 0 else None,
        "neighbour_plus1": records[flat_index + 1] if flat_index + 1 < len(records) else None,
    }


# ---------------------------------------------------------------------------
# 0A — windowed masked-vs-unmasked proxy test
# ---------------------------------------------------------------------------


def _window_positions(centres: list[int], half: int, length: int) -> list[int]:
    """Sorted unique 0-based positions within ``±half`` of any centre, clipped to the protein."""
    positions: set[int] = set()
    for c in centres:
        positions.update(range(max(0, c - half), min(length, c + half + 1)))
    return sorted(positions)


def _outlier_stats(values: np.ndarray, idx: int) -> dict:
    """z-score and ascending rank (1 = lowest = most surprising) of ``values[idx]``."""
    mean, std = float(values.mean()), float(values.std())
    z = float((values[idx] - mean) / std) if std > 0 else 0.0
    rank = int((values < values[idx]).sum()) + 1
    return {"value": float(values[idx]), "z": z, "rank": rank, "n": int(values.size)}


def windowed_proxy(
    model,
    tokenizer,
    seq: str,
    reference: str,
    *,
    role: str,
    sample_id: str,
    hotspots: list[tuple[int, str, str]],
    primary_codon: int,
    window: int,
    device: str,
) -> dict:
    """Masked + unmasked surprise across a hotspot window for one isolate.

    For a resistant isolate the window is centred on its hotspot codon(s); for the
    WT control it is centred on ``primary_codon`` (the resistant isolate's hotspot,
    mapped into the WT sample's own coordinates) so the same biological region is
    compared. Returns per-position profiles, the masked-vs-unmasked correlation, and
    each hotspot's outlier z-score / rank in both profiles.
    """
    length = len(seq)
    # Hotspot positions in the sample's own rpoB coordinates (None where gapped).
    hotspot_codon_list = [c for c, _wt, _alt in hotspots]
    codon_to_pos = sample_codon_positions(seq, reference, hotspot_codon_list) if hotspot_codon_list else {}
    hotspot_pos = {c: p for c, p in codon_to_pos.items() if p is not None}

    # Window centres: the hotspots (resistant) or the primary codon mapped in (WT).
    if hotspot_pos:
        centres = sorted(hotspot_pos.values())
    else:
        primary_pos = sample_codon_positions(seq, reference, [primary_codon]).get(primary_codon)
        if primary_pos is None:
            raise ValueError(f"{sample_id}: primary codon {primary_codon} maps to a gap — cannot centre WT window.")
        centres = [primary_pos]
    primary_pos = centres[0]

    positions = _window_positions(centres, window, length)
    masked = masked_logprobs(model, tokenizer, seq, positions=positions, device=device).numpy()
    unmasked_full = unmasked_logprobs(model, tokenizer, seq, device=device).numpy()
    unmasked = unmasked_full[positions]

    hotspot_pos_set = set(hotspot_pos.values())
    profile = [
        {
            "position": int(p),
            "offset_from_primary": int(p - primary_pos),
            "residue": seq[p],
            "masked_logp": float(masked[i]),
            "unmasked_logp": float(unmasked[i]),
            "is_hotspot": p in hotspot_pos_set,
        }
        for i, p in enumerate(positions)
    ]

    snp_outliers = []
    pos_to_window_idx = {p: i for i, p in enumerate(positions)}
    for codon, wt, alt in hotspots:
        p = hotspot_pos.get(codon)
        if p is None:
            continue
        i = pos_to_window_idx[p]
        snp_outliers.append({
            "codon": codon, "wt": wt, "alt": alt, "position": int(p),
            "masked": _outlier_stats(masked, i),
            "unmasked": _outlier_stats(unmasked, i),
        })

    pearson_r = float(pearsonr(masked, unmasked)[0]) if masked.size > 2 else None
    spearman_r = float(spearmanr(masked, unmasked)[0]) if masked.size > 2 else None
    return {
        "sample": sample_id,
        "role": role,
        "primary_codon": int(primary_codon),
        "window": window,
        "n_positions": len(positions),
        "hotspots": [{"codon": c, "wt": w, "alt": a} for c, w, a in hotspots],
        "pearson_masked_vs_unmasked": pearson_r,
        "spearman_masked_vs_unmasked": spearman_r,
        "snp_outliers": snp_outliers,
        "profile": profile,
    }


# ---------------------------------------------------------------------------
# 0B — full-gene + 2-neighbour unmasked distributions
# ---------------------------------------------------------------------------


def _distribution_stats(logp: np.ndarray) -> dict:
    """Summary of a per-residue ``log P(observed)`` vector (low = surprising)."""
    order = np.sort(logp)  # ascending → most-surprising first
    return {
        "length": int(logp.size),
        "min_logp": float(order[0]),
        "p1_logp": float(np.percentile(logp, 1)),
        "p5_logp": float(np.percentile(logp, 5)),
        "mean_logp": float(logp.mean()),
        "std_logp": float(logp.std()),
        "mean_bottom3_logp": float(order[: min(3, logp.size)].mean()),
        "skew_logp": float(skew(logp)),
        "kurtosis_logp": float(kurtosis(logp)),
    }


def gene_distribution(
    model,
    tokenizer,
    seq: str,
    *,
    sample_id: str,
    role: str,
    gene_slot: str,
    gene_name: str | None,
    flat_index: int,
    device: str,
    resistance_position: int | None = None,
) -> tuple[dict, np.ndarray]:
    """Unmasked surprise over a whole protein → stats + (for rpoB) the SNP's rank.

    Returns ``(record, logp_array)`` — the array is kept for the pooled violin plot.
    """
    logp = unmasked_logprobs(model, tokenizer, seq, device=device).numpy()
    stats = _distribution_stats(logp)
    resistance_rank = None
    resistance_logp = None
    if resistance_position is not None and 0 <= resistance_position < logp.size:
        resistance_rank = int((logp < logp[resistance_position]).sum()) + 1
        resistance_logp = float(logp[resistance_position])
    record = {
        "sample": sample_id,
        "role": role,
        "gene_slot": gene_slot,
        "gene_name": gene_name,
        "flat_index": int(flat_index),
        "stats": stats,
        "resistance_residue_rank": resistance_rank,
        "resistance_residue_logp": resistance_logp,
    }
    return record, logp


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_window(window_result: dict, out_path: Path) -> None:
    """0A: masked + unmasked log-prob across the window, SNP marked (offset axis)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    profile = window_result["profile"]
    offsets = [e["offset_from_primary"] for e in profile]
    masked = [e["masked_logp"] for e in profile]
    unmasked = [e["unmasked_logp"] for e in profile]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(offsets, masked, marker="o", ms=3, color="C3", label="masked log P(obs)")
    ax.plot(offsets, unmasked, marker="s", ms=3, color="C0", label="unmasked log P(obs)")
    for e in profile:
        if e["is_hotspot"]:
            ax.axvline(e["offset_from_primary"], ls="--", lw=1, color="grey")
    ax.set_xlabel("residue offset from primary hotspot")
    ax.set_ylabel("log P(observed)  (low = surprising)")
    ax.set_title(f"0A window — {window_result['role']} {window_result['sample']} (codon {window_result['primary_codon']})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_proxy_scatter(windows: list[dict], out_path: Path) -> None:
    """0A: masked vs unmasked across all resistant window positions, with hotspots flagged."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    masked, unmasked, is_hot = [], [], []
    for w in windows:
        if w["role"] != "resistant":
            continue
        for e in w["profile"]:
            masked.append(e["masked_logp"])
            unmasked.append(e["unmasked_logp"])
            is_hot.append(e["is_hotspot"])
    if not masked:
        return
    masked, unmasked, is_hot = np.array(masked), np.array(unmasked), np.array(is_hot)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(masked[~is_hot], unmasked[~is_hot], s=12, alpha=0.5, color="C0", label="WT window residue")
    ax.scatter(masked[is_hot], unmasked[is_hot], s=60, color="C3", marker="*", label="resistance SNP")
    pr = pearsonr(masked, unmasked)[0]
    sr = spearmanr(masked, unmasked)[0]
    ax.set_xlabel("masked log P(observed)")
    ax.set_ylabel("unmasked log P(observed)")
    ax.set_title(f"0A masked vs unmasked proxy — Pearson {pr:.3f}, Spearman {sr:.3f}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_distributions(groups: dict[str, list[float]], out_path: Path) -> None:
    """0B: violin of per-residue log P pooled by (role, gene-category)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [k for k, v in groups.items() if v]
    data = [np.array(groups[k]) for k in labels]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 3, 5))
    parts = ax.violinplot(data, showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.6)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("log P(observed)  (low = surprising)")
    ax.set_title("0B per-residue surprise distributions (unmasked)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_probe(
    *,
    ast_sheet_path: Path,
    parquet_dir: Path,
    drug: str,
    device: str,
    window: int,
    n_resistant: int,
    n_wt: int,
    pool_size: int,
    out_json: Path,
    out_dir: Path,
    qc_log_path: Path,
) -> dict:
    """Run 0A + 0B over a few resistant + WT isolates; write JSON + plots."""
    reference = load_reference()
    label_map, train_ids, validate_ids, evaluate_ids, split_info = resolve_clean_splits(ast_sheet_path, drug)
    pool = [*train_ids, *validate_ids, *evaluate_ids][:pool_size]
    logger.info("Genotyping a pool of %d labelled samples to source isolates", len(pool))
    genotype = build_genotype_table(pool, parquet_dir, reference, qc_log_path=qc_log_path)

    resistant_ids, wt_ids = select_isolates(genotype, label_map, n_resistant=n_resistant, n_wt=n_wt)
    if not resistant_ids:
        raise RuntimeError("No resistant rpoB-mutant isolate found in the pool — raise --pool-size.")
    logger.info("Selected resistant=%s wt=%s", resistant_ids, wt_ids)

    model, tokenizer = load_esmc_mlm(device=device)

    # The primary codon controls where the WT window is centred (first resistant hotspot).
    first_hotspots = hotspot_codons(genotype.loc[resistant_ids[0]], reference)
    primary_codon = first_hotspots[0][0]

    # --- 0A: windowed masked-vs-unmasked proxy --------------------------------
    windows: list[dict] = []
    for sid in resistant_ids:
        hotspots = hotspot_codons(genotype.loc[sid], reference)
        logger.info("0A resistant %s hotspots=%s", sid, hotspots)
        windows.append(windowed_proxy(
            model, tokenizer, genotype.loc[sid, "rpob_sequence"], reference,
            role="resistant", sample_id=sid, hotspots=hotspots,
            primary_codon=hotspots[0][0], window=window, device=device,
        ))
    for sid in wt_ids:
        logger.info("0A wt control %s (codon %d)", sid, primary_codon)
        windows.append(windowed_proxy(
            model, tokenizer, genotype.loc[sid, "rpob_sequence"], reference,
            role="wt", sample_id=sid, hotspots=[], primary_codon=primary_codon,
            window=window, device=device,
        ))

    # --- 0B: full-gene + 2-neighbour distributions ----------------------------
    distributions: list[dict] = []
    pooled: dict[str, list[float]] = {
        "rpoB (resistant)": [], "rpoB (WT)": [], "neighbours (resistant)": [], "neighbours (WT)": [],
    }
    for role, ids in (("resistant", resistant_ids), ("wt", wt_ids)):
        for sid in ids:
            flat_index = int(genotype.loc[sid, "rpob_flat_index"])
            neigh = _neighbour_records(parquet_dir, sid, flat_index)
            # rpoB — with the resistance residue's rank, if this isolate has a hotspot.
            res_pos = None
            if role == "resistant":
                hs = hotspot_codons(genotype.loc[sid], reference)
                res_pos = sample_codon_positions(
                    genotype.loc[sid, "rpob_sequence"], reference, [hs[0][0]]
                ).get(hs[0][0])
            rec, logp = gene_distribution(
                model, tokenizer, genotype.loc[sid, "rpob_sequence"], sample_id=sid, role=role,
                gene_slot="rpoB", gene_name="rpoB", flat_index=flat_index, device=device,
                resistance_position=res_pos,
            )
            distributions.append(rec)
            pooled[f"rpoB ({'resistant' if role == 'resistant' else 'WT'})"].extend(logp.tolist())
            for slot in ("neighbour_minus1", "neighbour_plus1"):
                nrec = neigh[slot]
                if nrec is None:
                    continue
                rec_n, logp_n = gene_distribution(
                    model, tokenizer, nrec["protein_sequence"], sample_id=sid, role=role,
                    gene_slot=slot, gene_name=nrec["gene_name"], flat_index=nrec["flat_index"], device=device,
                )
                distributions.append(rec_n)
                pooled[f"neighbours ({'resistant' if role == 'resistant' else 'WT'})"].extend(logp_n.tolist())

    payload: dict = {
        "schema_version": "1.0",
        "task": "snp_embeddings",
        "analysis": "llr_distribution_probe",
        "drug": drug,
        "device": device,
        "window": window,
        "reference": "UniProt P9WGY9 (H37Rv rpoB)",
        "split": split_info,
        "primary_codon": int(primary_codon),
        "selected": {"resistant": resistant_ids, "wt": wt_ids, "pool_size": len(pool),
                     "n_single_copy_genotyped": int(len(genotype))},
        "phase_0a": {"windows": windows},
        "phase_0b": {"distributions": distributions},
    }

    # Plots.
    out_dir.mkdir(parents=True, exist_ok=True)
    for w in windows:
        plot_window(w, out_dir / f"phase0a_window_{w['role']}_{w['sample']}.png")
    plot_proxy_scatter(windows, out_dir / "phase0a_masked_vs_unmasked_scatter.png")
    plot_distributions(pooled, out_dir / "phase0b_distributions.png")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    payload["host"] = socket.gethostname()
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
    parser.add_argument("--output-dir", type=Path, default=None, help="Dir for the plots (default: JSON's dir).")
    parser.add_argument("--drug", type=str, default="rifampin", help="Phenotype column (default rifampin).")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device (default cpu — Stage-A smoke).")
    parser.add_argument("--window", type=int, default=25,
                        help="±W residues around the hotspot for 0A (default 25; 100 for a stronger test).")
    parser.add_argument("--n-resistant", type=int, default=3, help="Resistant rpoB-mutant isolates (default 3).")
    parser.add_argument("--n-wt", type=int, default=1, help="WT/susceptible control isolates (default 1).")
    parser.add_argument("--pool-size", type=int, default=300,
                        help="Labelled samples to genotype when sourcing isolates (default 300).")
    parser.add_argument("--qc-log", type=Path, default=Path("rpob_copy_qc.log"),
                        help="Where to write the rpoB-copy QC log (default: ./rpob_copy_qc.log).")
    args = parser.parse_args()

    run_probe(
        ast_sheet_path=args.ast_sheet_path,
        parquet_dir=args.parquet_dir,
        drug=args.drug,
        device=args.device,
        window=args.window,
        n_resistant=args.n_resistant,
        n_wt=args.n_wt,
        pool_size=args.pool_size,
        out_json=args.output_json,
        out_dir=args.output_dir or args.output_json.parent,
        qc_log_path=args.qc_log,
    )


if __name__ == "__main__":
    main()

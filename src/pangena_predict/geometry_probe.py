"""Step 3b — per-residue / per-layer geometry probe (labels-free, Stage-A CPU).

Step 2 shows the frozen ESM-C mean-pooled rpoB vector loses the rifampicin
signal. Step 3b asks *where in ESM-C* the signal lives before that mean — is it
encoded per-residue and merely crushed by the pool (recoverable by an attention
pool), or never encoded at all (needs domain-adaptive pretraining)?

For each canonical RRDR substitution we build an **in-silico** single-residue
WT→mutant rpoB pair (apply the substitution to H37Rv, one at a time) and, layer
by layer, measure how far the ESM-C representation moves:

- ``d_site``   — at the mutated residue itself,
- ``d_window`` — averaged over the ±k neighbours (ESM-C is contextual, so the
  perturbation bleeds into neighbours),
- ``d_pool``   — between the two **production mean-pooled** protein vectors
  (:func:`tl.embed.esm_residue_level.production_mean_pool`, the real pool),
- ``d_max``    — the single most-perturbed residue anywhere,
- ``d_cls``    — at the ``<cls>`` token.

``d_site ≳ d_window ≫ d_pool`` ⇒ the residue *is* represented per-residue and the
mean washes it out ⇒ an attention pool could recover it; we report the
best-preserving layer (max ``d_site / d_pool``). Alongside, the **masked-LM LLR
profile** along the sequence shows the causal residue as a single sharp outlier
(others ~flat) — ESM-C "knows" the site is anomalous.

In-silico contrasts are primary; ``--real-rpob-fasta`` adds a few real resistant
rpoB sequences as confirmation (aligned, windowed LLR only — full per-position
geometry on real isolates is deferred). ``--validate-pool`` checks
``production_mean_pool`` reproduces a stored pooled rpoB vector (proves ``d_pool``
measures the deployed pool). Reference / provenance: see
:mod:`pangena_predict.rpob_genotype`.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch

from pangena_predict.rpob_genotype import (
    RRDR_FIRST_CODON,
    RRDR_LAST_CODON,
    RRDR_PANEL,
    assert_reference_panel,
    load_reference,
    ref_index_for_codon,
)
from tl.embed.esm_residue_level import (
    apply_point_mutation,
    load_esmc_mlm,
    masked_marginals,
    production_mean_pool,
    residue_states,
    substitution_llr,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Production ESM-C truncation (compute_genome_protein_embeddings max_prot_seq_len).
PRODUCTION_MAX_RESIDUES = 1024
METRICS = ("cosine", "euclidean")


def _distance(a: torch.Tensor, b: torch.Tensor, metric: str) -> float:
    """Scalar distance between two vectors (``cosine`` = 1 − cos sim, or euclidean)."""
    if metric == "cosine":
        return float(1.0 - torch.nn.functional.cosine_similarity(a, b, dim=-1))
    return float(torch.linalg.vector_norm(a - b))


def _per_position_distances(wt_layer: torch.Tensor, mut_layer: torch.Tensor, metric: str) -> torch.Tensor:
    """Per-residue distance between two ``[L, dim]`` layers → ``[L]``."""
    if metric == "cosine":
        return 1.0 - torch.nn.functional.cosine_similarity(wt_layer, mut_layer, dim=-1)
    return torch.linalg.vector_norm(wt_layer - mut_layer, dim=-1)


def geometry_for_substitution(
    model,
    tokenizer,
    reference: str,
    codon: int,
    wt: str,
    alt: str,
    *,
    device: str,
    window_k: int,
    pool_cap: int = PRODUCTION_MAX_RESIDUES,
) -> dict:
    """Per-layer WT→mutant ESM-C geometry for one in-silico RRDR substitution."""
    idx = ref_index_for_codon(reference, codon)
    mut_seq = apply_point_mutation(reference, idx, alt, expected_wt=wt)

    wt_res, wt_cls = residue_states(model, tokenizer, reference, device=device, all_layers=True, return_cls=True)
    mut_res, mut_cls = residue_states(model, tokenizer, mut_seq, device=device, all_layers=True, return_cls=True)
    n_layers, length, _dim = wt_res.shape
    lo, hi = max(0, idx - window_k), min(length, idx + window_k + 1)

    per_layer: list[dict] = []
    for layer in range(n_layers):
        entry: dict = {"layer": layer}
        for metric in METRICS:
            posd = _per_position_distances(wt_res[layer], mut_res[layer], metric)
            d_site = float(posd[idx])
            d_pool = _distance(
                production_mean_pool(wt_res[layer], max_residues=pool_cap),
                production_mean_pool(mut_res[layer], max_residues=pool_cap),
                metric,
            )
            entry[metric] = {
                "d_site": d_site,
                "d_window": float(posd[lo:hi].mean()),
                "d_pool": d_pool,
                "d_max": float(posd.max()),
                "d_cls": _distance(wt_cls[layer], mut_cls[layer], metric),
                "site_over_pool": (d_site / d_pool) if d_pool > 0 else None,
            }
        per_layer.append(entry)

    def _ratio(e: dict) -> float:
        r = e["cosine"]["site_over_pool"]
        return r if (r is not None and math.isfinite(r)) else -math.inf

    best_layer = max(per_layer, key=_ratio)["layer"]
    return {
        "codon": codon,
        "wt": wt,
        "alt": alt,
        "ref_index": idx,
        "n_layers": n_layers,
        "window_k": window_k,
        "best_layer_cosine_site_over_pool": best_layer,
        "per_layer": per_layer,
    }


def llr_profile(
    model,
    tokenizer,
    seq: str,
    reference_at: dict[int, str],
    positions: list[int],
    *,
    device: str,
) -> list[dict]:
    """Masked-LM LLR (``log P(observed) − log P(wt)``) at each position in ``seq``.

    ``reference_at`` maps a position in ``seq``'s own coordinate to the wild-type
    residue there. For an in-silico mutant this is the H37Rv residue; positions
    that match wild-type score ~0, the substituted site a sharp negative outlier.
    """
    logps = masked_marginals(
        model, tokenizer, seq, positions=positions, device=device,
        expected_residues={p: seq[p] for p in positions},
    )
    profile = []
    for p in positions:
        wt = reference_at[p]
        profile.append({
            "position": p,
            "wt": wt,
            "observed": seq[p],
            "llr": substitution_llr(logps[p], tokenizer, wt=wt, observed=seq[p]),
        })
    return profile


def validate_production_pool(
    model,
    tokenizer,
    reference: str,
    stored_vector: np.ndarray,
    *,
    device: str,
    atol: float = 1e-2,
) -> dict:
    """Check ``production_mean_pool`` of the final layer reproduces a stored pooled vector."""
    last = residue_states(model, tokenizer, reference, device=device, all_layers=False)
    pooled = production_mean_pool(last, max_residues=PRODUCTION_MAX_RESIDUES).numpy()
    stored = np.asarray(stored_vector, dtype=float).ravel()
    max_abs = float(np.max(np.abs(pooled - stored))) if stored.shape == pooled.shape else float("nan")
    ok = bool(stored.shape == pooled.shape and np.allclose(pooled, stored, atol=atol))
    logger.info("production-pool validation: max|Δ|=%.3e shape=%s match=%s", max_abs, pooled.shape, ok)
    return {"max_abs_diff": max_abs, "atol": atol, "match": ok, "dim": int(pooled.shape[0])}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_layer_distances(results: list[dict], out_path: Path, metric: str = "cosine") -> None:
    """Mean per-layer d_site / d_window / d_pool / d_max / d_cls across the panel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_layers = results[0]["n_layers"]
    keys = ["d_site", "d_window", "d_pool", "d_max", "d_cls"]
    means = {k: np.array([
        np.mean([r["per_layer"][layer][metric][k] for r in results]) for layer in range(n_layers)
    ]) for k in keys}

    fig, ax = plt.subplots(figsize=(8, 5))
    for k in keys:
        ax.plot(range(n_layers), means[k], marker="o", ms=3, label=k)
    ax.set_xlabel("ESM-C layer")
    ax.set_ylabel(f"{metric} distance (WT → mutant)")
    ax.set_title(f"Per-layer WT→mutant movement, mean over {len(results)} RRDR substitutions")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_llr_profile(profile: list[dict], title: str, out_path: Path, anchor_codon: int, anchor_index: int) -> None:
    """Per-position masked-LM LLR — the causal residue as a single sharp outlier."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Convert reference-string positions to Mtb codon numbers for a readable axis.
    positions = [e["position"] for e in profile]
    codons = [anchor_codon + (p - anchor_index) for p in positions]
    llrs = [e["llr"] for e in profile]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axhline(0.0, ls="--", lw=1, color="grey")
    ax.bar(codons, llrs, color="C3")
    ax.set_xlabel("Mtb rpoB codon")
    ax.set_ylabel("masked-LM LLR (observed − WT)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_geometry_probe(
    *,
    device: str,
    window_k: int,
    full_profile: bool,
    out_json: Path,
    out_dir: Path,
    validate_pool_npy: Path | None,
) -> dict:
    """Run the in-silico geometry probe over the RRDR panel; write JSON + plots."""
    reference = load_reference()
    assert_reference_panel(reference)
    model, tokenizer = load_esmc_mlm(device=device)

    # Profile positions: the RRDR window by default; the whole protein with --full-profile (GPU).
    if full_profile:
        profile_positions = list(range(len(reference)))
    else:
        lo = ref_index_for_codon(reference, RRDR_FIRST_CODON)
        hi = ref_index_for_codon(reference, RRDR_LAST_CODON)
        profile_positions = list(range(lo, hi + 1))

    geometry: list[dict] = []
    profiles: dict[str, list[dict]] = {}
    for codon, wt, alt in RRDR_PANEL:
        label = f"{wt}{codon}{alt}"
        logger.info("Geometry for %s", label)
        geom = geometry_for_substitution(
            model, tokenizer, reference, codon, wt, alt, device=device, window_k=window_k
        )
        geometry.append(geom)
        idx = ref_index_for_codon(reference, codon)
        mut_seq = apply_point_mutation(reference, idx, alt, expected_wt=wt)
        reference_at = {p: reference[p] for p in profile_positions}
        profiles[label] = llr_profile(model, tokenizer, mut_seq, reference_at, profile_positions, device=device)

    payload: dict = {
        "schema_version": "1.0",
        "task": "pangena_predict",
        "analysis": "geometry_probe",
        "reference": "UniProt P9WGY9 (H37Rv rpoB)",
        "device": device,
        "window_k": window_k,
        "full_profile": full_profile,
        "production_max_residues": PRODUCTION_MAX_RESIDUES,
        "panel": [{"codon": c, "wt": w, "alt": a} for c, w, a in RRDR_PANEL],
        "geometry": geometry,
        "llr_profiles": profiles,
    }

    if validate_pool_npy is not None:
        stored = np.load(validate_pool_npy)
        payload["production_pool_validation"] = validate_production_pool(
            model, tokenizer, reference, stored, device=device
        )

    # Plots: per-layer distances + a representative LLR profile (last panel codon).
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_layer_distances(geometry, out_dir / "geometry_layer_distances.png")
    rep_codon, rep_wt, rep_alt = RRDR_PANEL[-1]
    rep_label = f"{rep_wt}{rep_codon}{rep_alt}"
    plot_llr_profile(
        profiles[rep_label],
        f"ESM-C masked-LM LLR profile — in-silico {rep_label}",
        out_dir / "geometry_llr_profile.png",
        anchor_codon=rep_codon,
        anchor_index=ref_index_for_codon(reference, rep_codon),
    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %s + 2 plots in %s", out_json, out_dir)
    return payload


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-json", type=Path, required=True, help="Where to write the geometry JSON.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Dir for the plots (default: JSON's dir).")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device (default cpu — Stage-A smoke).")
    parser.add_argument("--window-k", type=int, default=5, help="±k neighbours for d_window (default 5).")
    parser.add_argument("--full-profile", action="store_true",
                        help="Profile every residue (all ~1,178 positions; GPU) instead of just the RRDR window.")
    parser.add_argument("--validate-pool", type=Path, default=None,
                        help="NPY of a stored pooled rpoB vector — assert production_mean_pool reproduces it.")
    args = parser.parse_args()

    run_geometry_probe(
        device=args.device,
        window_k=args.window_k,
        full_profile=args.full_profile,
        out_json=args.output_json,
        out_dir=args.output_dir or args.output_json.parent,
        validate_pool_npy=args.validate_pool,
    )


if __name__ == "__main__":
    main()

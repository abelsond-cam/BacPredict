"""Head-pool diagnostic — does the *predictive head's* learned pool attend to rpoB?

The companion :mod:`snp_embeddings.intrinsic_attention_probe` measures Bacformer's **own**
self-attention (between protein tokens, *inside* the backbone) and finds rpoB strongly
attended. But that is **not** what the classifier reads: a genome head must collapse the
~4,000 contextualised tokens into one vector, and *that* pooling step is a separate attention.

The paradox this probe resolves: the frozen Bacformer **rpoB token** alone scores ~0.953, the
fine-tuned **mean-pool** model scores ~0.905, yet our learned **gated-MIL attention pool** scores
**0.868 — below the mean**. If the head's pool were routing to rpoB it could not lose to the mean.
So we measure it directly: load a *trained* :class:`BacformerAttnPoolForGenomeClassification`
checkpoint, run it over the manifest genomes, read the pool's per-protein weight
(``model.last_attention_weights``), and rank **rpoB's** weight among the genome's proteins,
resistant vs wild-type.

Read-out:

- If rpoB's head-pool weight clusters near percentile **0.5** (a plain mean is uniform → rpoB at
  exactly 0.5), with no R-vs-WT separation ⇒ the learned head **did not route to rpoB** — it
  collapsed to something mean-like, explaining 0.868 < 0.905.
- If rpoB's weight is high ⇒ the head *does* attend rpoB and the deficit lives elsewhere (token
  content / optimisation), not in the pooling.

Label-blind (uses only the manifest's rpoB flat index), read-only, GPU. Reuses the manifest +
flat-index machinery of :mod:`snp_embeddings.frozen_bacformer_rpob_vectors` and the ``_rank_stats``
ranking of :mod:`snp_embeddings.intrinsic_attention_probe`.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from snp_embeddings.frozen_bacformer_rpob_vectors import _forward_inputs
from snp_embeddings.intrinsic_attention_probe import _rank_stats, _resolve_weight_checkpoint, load_attn_pool_wrapper
from snp_embeddings.snp_vs_esm_prediction import _real_protein_indices

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_attn_pool_checkpoint(checkpoint_dir: str, device: str) -> torch.nn.Module:
    """Load a trained ``BacformerAttnPoolForGenomeClassification`` checkpoint (eval, on device).

    ``_resolve_weight_checkpoint`` finds the best ``checkpoint-<step>/`` inside a run dir (keyed on
    saved weights, since the gated-MIL checkpoint has no ``config.json``); the wrapper is then rebuilt
    by :func:`~snp_embeddings.intrinsic_attention_probe.load_attn_pool_wrapper`
    (``from_pretrained_backbone`` + ``load_state_dict``) — the custom pool is a local ``nn.Module``,
    not HF remote code, so ``AutoModel.from_pretrained`` cannot reconstruct it.
    """
    model_dir = _resolve_weight_checkpoint(Path(checkpoint_dir))
    return load_attn_pool_wrapper(model_dir, device)


def _head_pool_weights(model: torch.nn.Module, inputs: dict, real_idx: torch.Tensor) -> np.ndarray:
    """Per-real-protein head-pool weight from one forward pass.

    Forwards the trained wrapper (which stashes ``self.last_attention_weights`` of shape
    ``(1, N)`` summing to 1 over valid tokens), then restricts to the real protein tokens —
    so the returned ``(n_real,)`` vector aligns with the manifest's flat rpoB index.
    """
    with torch.inference_mode():
        model(**inputs)
    weights = model.last_attention_weights  # (1, N)
    w = weights[0].float().index_select(0, real_idx.to(weights.device))
    return w.cpu().numpy()


def _concentration_stats(weights: np.ndarray, *, top_k: int = 20) -> dict:
    """How *few* proteins the pool routes to, from one genome's (normalised) weight vector.

    ``weights`` sums to ~1 over the genome's real proteins. The question is not where rpoB ranks
    but how concentrated the whole distribution is: a plain mean over ``n`` proteins has
    ``eff_n == n`` and ``topK_mass == K/n``; a pool that picks ~50 genes has ``eff_n ≈ 50`` and a
    large ``top50_mass``. We also keep the top-``top_k`` flat indices + weights so the dominant
    genes can be *named* afterwards (the confound test: rpoB, or katG/embB/lineage markers?).

    Returns ``eff_n`` (inverse participation ratio ``1/Σpᵢ²``), ``perplexity`` (entropy effective
    count ``exp(−Σpᵢ log pᵢ)``), ``max_weight``, cumulative ``topK_mass`` for K∈{1,10,50,100,200},
    and the ``top{top_k}_flat_idx`` / ``top{top_k}_weight`` of the heaviest proteins.
    """
    w = weights.astype(np.float64)
    n = int(w.size)
    s = w.sum()
    p = w / s if s > 0 else w
    nz = p[p > 0]
    order = np.argsort(w)[::-1]
    cmass = np.cumsum(np.sort(p)[::-1])
    return {
        "n_proteins": n,
        "eff_n": float(1.0 / np.square(p).sum()) if s > 0 else None,
        "perplexity": float(np.exp(-(nz * np.log(nz)).sum())) if nz.size else None,
        "max_weight": float(p.max()) if n else None,
        "top1_mass": float(cmass[0]) if n else None,
        "top10_mass": float(cmass[min(10, n) - 1]) if n else None,
        "top50_mass": float(cmass[min(50, n) - 1]) if n else None,
        "top100_mass": float(cmass[min(100, n) - 1]) if n else None,
        "top200_mass": float(cmass[min(200, n) - 1]) if n else None,
        f"top{top_k}_flat_idx": order[:top_k].astype(int).tolist(),
        f"top{top_k}_weight": w[order[:top_k]].astype(float).tolist(),
    }


def _rank_profile(weight_arrays: list[np.ndarray], cap: int = 4000) -> dict:
    """Rank-aligned **mean sorted head-pool weight** across genomes — the cumulative-attention curve.

    Each genome's weights are normalised (sum→1) and sorted descending; rank *r* is then averaged
    over the genomes that *have* an *r*-th protein (genomes vary in length). The result says, at
    each rank, how much mean attention mass a gene receives — heavy head, long dead tail. Capped at
    ``cap`` ranks to fix the array size. A uniform mean over ``n`` proteins gives a flat
    ``mean_sorted_weight == 1/n``; a pool routing to ~50 genes gives a steep curve whose cumulative
    sum saturates by rank ~50.

    Returns ``mean_sorted_weight`` ``(cap,)``, ``n_at_rank`` ``(cap,)`` (genomes contributing to each
    rank), and ``n_genomes``.
    """
    prof_sum = np.zeros(cap, dtype=np.float64)
    prof_cnt = np.zeros(cap, dtype=np.float64)
    for w in weight_arrays:
        w = np.asarray(w, dtype=np.float64)
        s = w.sum()
        if s <= 0 or w.size == 0:
            continue
        sorted_desc = np.sort(w / s)[::-1]
        k = min(sorted_desc.size, cap)
        prof_sum[:k] += sorted_desc[:k]
        prof_cnt[:k] += 1.0
    mean_sorted = np.divide(prof_sum, prof_cnt, out=np.zeros_like(prof_sum), where=prof_cnt > 0)
    return {"mean_sorted_weight": mean_sorted, "n_at_rank": prof_cnt, "n_genomes": len(weight_arrays)}


def probe_head_pool(
    manifest: pd.DataFrame,
    esm_store_dir: Path,
    checkpoint_dir: str,
    *,
    device: str,
    pt_suffix: str = "_esm_embeddings.pt",
    max_proteins: int | None = None,
    profile_cap: int = 4000,
) -> tuple[pd.DataFrame, dict]:
    """Per-genome rpoB head-pool-weight rank + pool concentration for a trained attention-pool checkpoint.

    Returns ``(df, profile)``. ``df`` has one row per kept genome: rpoB's pool weight, its percentile
    (fraction of proteins it exceeds; 1.0 = the single most-weighted protein) and rank (0 = top), plus
    the :func:`_concentration_stats` of the whole weight vector (``eff_n``, top-K mass, the dominant
    proteins' flat indices). ``profile`` holds the rank-aligned :func:`_rank_profile` (mean sorted
    weight vs rank, capped at ``profile_cap``) for ``all`` / ``resistant`` / ``wt`` — the input to the
    cumulative-attention figure.
    """
    model = load_attn_pool_checkpoint(checkpoint_dir, device)
    model_dtype = next(model.parameters()).dtype
    if getattr(model.config, "panel_mode", "none") not in (None, "none"):
        raise ValueError(
            f"checkpoint panel_mode={model.config.panel_mode!r}; this probe handles bare "
            "(panel_mode=none) gated-MIL/MHA checkpoints — no panel is wired for the manifest genomes."
        )

    rows: list[dict] = []
    weights_by_role: dict[str, list[np.ndarray]] = {"all": [], "resistant": [], "wt": []}
    skips: dict[str, int] = {}
    for _, m in manifest.iterrows():
        sample_id = str(m["sample"])
        role = str(m["role"])
        rpob_flat = int(m["rpob_flat_index"])
        pt_path = esm_store_dir / f"{sample_id}{pt_suffix}"
        if not pt_path.exists():
            skips["missing_pt"] = skips.get("missing_pt", 0) + 1
            continue
        store = torch.load(pt_path, map_location="cpu")
        input_len = store["protein_embeddings"].shape[1]
        real_idx = _real_protein_indices(store, input_len)
        if rpob_flat >= real_idx.numel():
            skips["out_of_range"] = skips.get("out_of_range", 0) + 1
            continue
        if max_proteins is not None and real_idx.numel() > max_proteins:
            skips["too_many_proteins"] = skips.get("too_many_proteins", 0) + 1
            continue

        inputs = _forward_inputs(store, device, model_dtype)
        weights = _head_pool_weights(model, inputs, real_idx)
        pct, rank = _rank_stats(weights, rpob_flat)
        rows.append({
            "sample": sample_id, "role": role,
            "rpob_pool_weight": float(weights[rpob_flat]),
            "weight_sum": float(weights.sum()),  # sanity: ~1.0 over real tokens
            "rpob_pool_pct": pct, "rpob_pool_rank": rank,
            **_concentration_stats(weights),
        })
        weights_by_role["all"].append(weights)
        if role in ("resistant", "wt"):
            weights_by_role[role].append(weights)

    if skips:
        logger.warning("head-pool probe: skipped %s", skips)
    profile = {key: _rank_profile(arrs, cap=profile_cap) for key, arrs in weights_by_role.items()}
    return pd.DataFrame(rows), profile


def summarise(df: pd.DataFrame, checkpoint_label: str) -> dict:
    """Summarise rpoB pooling rank **and pool concentration**, resistant vs WT.

    Two readings combine here. (1) rpoB's percentile/rank — but on its own the percentile misleads
    (a heavy-tailed pool can leave rpoB at the 95th pct while weighting it *below* a uniform mean).
    (2) The decisive metric is the rpoB *enrichment* (its weight ÷ the ``1/n`` a flat mean gives:
    >1 = up-weighted, <1 = suppressed) plus how concentrated the whole pool is (``eff_n``, top-K
    mass). A uniform mean has ``eff_n == n`` and ``topK_mass == K/n``; a pool routing to ~50 genes
    has ``eff_n ≈ 50`` and a large ``top50_mass`` — "doing well" only if rpoB is one of them.
    """

    def _med(s: pd.Series) -> float | None:
        return float(s.median()) if len(s) else None

    out: dict = {
        "checkpoint": checkpoint_label,
        "n_genomes": int(df["sample"].nunique()) if len(df) else 0,
        "mean_pool_reference_pct": 0.5,
        "weight_sum_median": _med(df["weight_sum"]) if len(df) else None,
        "uniform_reference": {
            "n_proteins_median": _med(df["n_proteins"]) if len(df) else None,
            "eff_n": _med(df["n_proteins"]) if len(df) else None,  # uniform pool → eff_n == n
            "top50_mass": float(50.0 / df["n_proteins"].median()) if len(df) else None,
        },
        "by_role": {},
    }
    for role in ("resistant", "wt"):
        g = df[df["role"] == role]
        ranks = g["rpob_pool_rank"]
        # enrichment: rpoB's weight relative to a flat 1/n mean (>1 up-weighted, <1 suppressed).
        enrich = g["rpob_pool_weight"] * g["n_proteins"]
        out["by_role"][role] = {
            "n": int(len(g)),
            "median_pct": _med(g["rpob_pool_pct"]),
            "top1_frac": float((ranks == 0).mean()) if len(ranks) else None,  # rpoB = most-weighted
            "rpob_rank_median": _med(ranks),
            "rpob_weight_median": _med(g["rpob_pool_weight"]),
            "rpob_enrichment_median": _med(enrich),
            "eff_n_median": _med(g["eff_n"]),
            "perplexity_median": _med(g["perplexity"]),
            "max_weight_median": _med(g["max_weight"]),
            "top10_mass_median": _med(g["top10_mass"]),
            "top50_mass_median": _med(g["top50_mass"]),
            "top100_mass_median": _med(g["top100_mass"]),
        }
    return out


def plot_probe(df: pd.DataFrame, out_path: Path, checkpoint_label: str) -> None:
    """Box of rpoB head-pool-weight percentile, resistant (red) vs WT (blue), vs the mean line."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5.5))
    data, labels, colors = [], [], []
    for role, color in (("resistant", "#d62728"), ("wt", "#1f77b4")):
        data.append(df[df["role"] == role]["rpob_pool_pct"].to_numpy())
        labels.append(role)
        colors.append(color)
    bp = ax.boxplot(data, positions=[0, 1], widths=0.5, patch_artist=True, showfliers=False)
    for box, color in zip(bp["boxes"], colors, strict=False):
        box.set(facecolor=color, alpha=0.55)
    for med in bp["medians"]:
        med.set(color="black")
    ax.axhline(0.5, ls="--", lw=1.5, color="grey", label="mean pool (uniform → pct 0.5)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylabel("rpoB head-pool-weight percentile\n(1.0 = most-weighted protein in the genome)")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Does the predictive head pool attend to rpoB?\n{checkpoint_label}")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cumulative_attention(df: pd.DataFrame, profile: dict, out_path: Path, checkpoint_label: str) -> None:
    """Two-panel head-pool weight distribution with rpoB's rank marked (the headline pp figure).

    Panel A — mean sorted head-pool weight vs gene rank (log-y): the heavy head / dead tail shape,
    with a dotted line at the uniform weight ``1/n`` (a flat mean). The rank where the curve drops
    below that line is how few genes are *up-weighted* vs a plain mean; rpoB's red line sits to its
    right (suppressed below uniform) yet far left of the tail (prioritised above the bulk).

    Panel B — cumulative attention mass vs rank: how much of the pool's mass is spent by each rank.
    The red line at rpoB's median rank, annotated with the mass already allocated to genes above it,
    is the headline — most of the attention is gone before rpoB.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean_w = np.asarray(profile["all"]["mean_sorted_weight"], dtype=np.float64)
    cap = mean_w.size
    ranks = np.arange(1, cap + 1)
    cum = np.cumsum(mean_w)
    n_med = float(df["n_proteins"].median()) if len(df) else cap
    uniform = 1.0 / n_med

    def _rank(role: str) -> float | None:
        g = df[df["role"] == role]["rpob_pool_rank"]
        return float(g.median()) if len(g) else None

    r_rank, w_rank = _rank("resistant"), _rank("wt")

    def _cum_at(rk: float | None) -> float | None:
        if rk is None or np.isnan(rk):
            return None
        return float(cum[min(int(rk), cap - 1)])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    ax.plot(ranks, mean_w, color="#333333", lw=1.3)
    ax.axhline(uniform, ls=":", lw=1.5, color="grey", label=f"uniform mean (1/n ≈ {uniform:.2e})")
    for rk, ls, lbl in ((r_rank, "-", "rpoB rank (R)"), (w_rank, "--", "rpoB rank (WT)")):
        if rk is not None and not np.isnan(rk):
            ax.axvline(rk, color="#d62728", ls=ls, lw=1.7, label=f"{lbl} ≈ {rk:.0f}")
    ax.set_yscale("log")
    ax.set_xlabel("gene rank (1 = most-weighted)")
    ax.set_ylabel("mean head-pool weight")
    ax.set_title("Sorted attention weights")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(ranks, cum, color="#1f77b4", lw=2.0)
    for rk, ls in ((r_rank, "-"), (w_rank, "--")):
        if rk is not None and not np.isnan(rk):
            ax.axvline(rk, color="#d62728", ls=ls, lw=1.7)
    cum_r = _cum_at(r_rank)
    if cum_r is not None:
        ax.annotate(
            f"≈{cum_r * 100:.0f}% of attention mass\nspent before rpoB (R, rank {r_rank:.0f})",
            xy=(r_rank, cum_r), xytext=(0.42, 0.45), textcoords="axes fraction", fontsize=9.5,
            arrowprops={"arrowstyle": "->", "color": "#d62728", "lw": 1.3}, color="#d62728",
        )
    ax.set_xlabel("gene rank (1 = most-weighted)")
    ax.set_ylabel("cumulative attention mass")
    ax.set_ylim(0, 1.02)
    ax.set_title("Cumulative attention mass")
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Head-pool prioritises rpoB above the bulk, but ~{r_rank:.0f} genes outweigh it"
        if r_rank is not None and not np.isnan(r_rank) else "Head-pool attention distribution",
        fontsize=12,
    )
    fig.text(0.5, 0.93, checkpoint_label, ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """CLI entry point — trained attention-pool head, rpoB pooling-weight rank over the manifest."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest-csv", type=Path, required=True, help="manifest.csv (sample/role/rpob_flat_index).")
    parser.add_argument("--esm-store-dir", type=Path, required=True, help="Dir of {sample}_esm_embeddings.pt.")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Trained attention-pool checkpoint dir.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", type=str, default=None, help="Checkpoint label for titles/JSON (default: dir name).")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap genomes (smoke).")
    parser.add_argument("--max-proteins", type=int, default=None, help="Skip genomes with more proteins (memory).")
    args = parser.parse_args()

    label = args.label or Path(args.checkpoint_dir).name
    manifest = pd.read_csv(args.manifest_csv)
    if args.max_samples is not None:
        # Keep the R/WT balance when capping: interleave by role.
        r = manifest[manifest["role"] == "resistant"].head(args.max_samples // 2)
        w = manifest[manifest["role"] == "wt"].head(args.max_samples - len(r))
        manifest = pd.concat([r, w]).reset_index(drop=True)
    logger.info("Probing head-pool attention over %d genomes (ckpt=%s) on %s", len(manifest), label, args.device)

    df, profile = probe_head_pool(
        manifest, args.esm_store_dir, args.checkpoint_dir, device=args.device, max_proteins=args.max_proteins
    )
    if df.empty:
        raise RuntimeError("No genomes scored — check esm_store_dir / .pt suffix / manifest.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output_dir / "head_pool_attention_rows.parquet", index=False)
    summary = summarise(df, label)
    (args.output_dir / "head_pool_attention.json").write_text(json.dumps(summary, indent=2))
    plot_probe(df, args.output_dir / "head_pool_attention.png", label)
    plot_cumulative_attention(df, profile, args.output_dir / "head_pool_cumulative_attention.png", label)
    # Regen-friendly profile bundle: restyle the pp figure on the login node (CPU) without re-running.
    np.savez(
        args.output_dir / "head_pool_profile.npz",
        mean_sorted_weight_all=profile["all"]["mean_sorted_weight"],
        mean_sorted_weight_resistant=profile["resistant"]["mean_sorted_weight"],
        mean_sorted_weight_wt=profile["wt"]["mean_sorted_weight"],
        n_at_rank_all=profile["all"]["n_at_rank"],
        rpob_rank_median_resistant=df[df["role"] == "resistant"]["rpob_pool_rank"].median(),
        rpob_rank_median_wt=df[df["role"] == "wt"]["rpob_pool_rank"].median(),
        uniform_weight=1.0 / df["n_proteins"].median(),
    )
    logger.info("Wrote rows + JSON + 2 figures + profile npz to %s", args.output_dir)
    ref = summary["uniform_reference"]
    for role in ("resistant", "wt"):
        b = summary["by_role"][role]
        logger.info(
            "%s: rpoB enrichment=%.2fx (rank %.0f, pct %.3f) | pool eff_n=%.0f vs uniform %.0f, "
            "top50_mass=%.2f vs uniform %.3f",
            role, b["rpob_enrichment_median"] or float("nan"), b["rpob_rank_median"] or float("nan"),
            b["median_pct"] or float("nan"), b["eff_n_median"] or float("nan"),
            ref["eff_n"] or float("nan"), b["top50_mass_median"] or float("nan"),
            ref["top50_mass"] or float("nan"),
        )


if __name__ == "__main__":
    main()

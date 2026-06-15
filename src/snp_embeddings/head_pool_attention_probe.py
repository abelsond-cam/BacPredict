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
from snp_embeddings.intrinsic_attention_probe import _rank_stats, load_attn_pool_wrapper
from snp_embeddings.snp_vs_esm_prediction import _real_protein_indices
from tl.train.evaluate import resolve_checkpoint_dir

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_attn_pool_checkpoint(checkpoint_dir: str, device: str) -> torch.nn.Module:
    """Load a trained ``BacformerAttnPoolForGenomeClassification`` checkpoint (eval, on device).

    ``resolve_checkpoint_dir`` finds the best ``checkpoint-<step>/`` inside a run dir; the wrapper is
    then rebuilt by :func:`~snp_embeddings.intrinsic_attention_probe.load_attn_pool_wrapper`
    (``from_pretrained_backbone`` + ``load_state_dict``) — the custom pool is a local ``nn.Module``,
    not HF remote code, so ``AutoModel.from_pretrained`` cannot reconstruct it.
    """
    model_dir = resolve_checkpoint_dir(Path(checkpoint_dir))
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


def probe_head_pool(
    manifest: pd.DataFrame,
    esm_store_dir: Path,
    checkpoint_dir: str,
    *,
    device: str,
    pt_suffix: str = "_esm_embeddings.pt",
    max_proteins: int | None = None,
) -> pd.DataFrame:
    """Per-genome rpoB head-pool-weight rank (R vs WT) for a trained attention-pool checkpoint.

    Returns one row per kept genome with rpoB's pool weight, its percentile (fraction of
    proteins it exceeds; 1.0 = the single most-weighted protein) and rank (0 = top).
    """
    model = load_attn_pool_checkpoint(checkpoint_dir, device)
    model_dtype = next(model.parameters()).dtype
    if getattr(model.config, "panel_mode", "none") not in (None, "none"):
        raise ValueError(
            f"checkpoint panel_mode={model.config.panel_mode!r}; this probe handles bare "
            "(panel_mode=none) gated-MIL/MHA checkpoints — no panel is wired for the manifest genomes."
        )

    rows: list[dict] = []
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
            "sample": sample_id, "role": role, "n_proteins": int(real_idx.numel()),
            "rpob_pool_weight": float(weights[rpob_flat]),
            "weight_sum": float(weights.sum()),  # sanity: ~1.0 over real tokens
            "rpob_pool_pct": pct, "rpob_pool_rank": rank,
        })

    if skips:
        logger.warning("head-pool probe: skipped %s", skips)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, checkpoint_label: str) -> dict:
    """Median rpoB head-pool-weight percentile + top-1 fraction, resistant vs WT.

    A plain mean pool would put every protein at percentile 0.5 (uniform weight); separation
    from 0.5 — especially R > WT — is the signature of a head that actually routes to rpoB.
    """
    out: dict = {
        "checkpoint": checkpoint_label,
        "n_genomes": int(df["sample"].nunique()) if len(df) else 0,
        "mean_pool_reference_pct": 0.5,
        "weight_sum_median": float(df["weight_sum"].median()) if len(df) else None,
        "by_role": {},
    }
    for role in ("resistant", "wt"):
        g = df[df["role"] == role]["rpob_pool_pct"]
        ranks = df[df["role"] == role]["rpob_pool_rank"]
        out["by_role"][role] = {
            "median_pct": float(g.median()) if len(g) else None,
            "top1_frac": float((ranks == 0).mean()) if len(ranks) else None,  # rpoB = most-weighted
            "n": int(len(g)),
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

    df = probe_head_pool(
        manifest, args.esm_store_dir, args.checkpoint_dir, device=args.device, max_proteins=args.max_proteins
    )
    if df.empty:
        raise RuntimeError("No genomes scored — check esm_store_dir / .pt suffix / manifest.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output_dir / "head_pool_attention_rows.parquet", index=False)
    summary = summarise(df, label)
    (args.output_dir / "head_pool_attention.json").write_text(json.dumps(summary, indent=2))
    plot_probe(df, args.output_dir / "head_pool_attention.png", label)
    logger.info("Wrote rows + JSON + figure to %s", args.output_dir)
    logger.info(
        "rpoB head-pool pct — resistant=%.3f wt=%.3f (mean-pool reference=0.500)",
        summary["by_role"]["resistant"]["median_pct"] or float("nan"),
        summary["by_role"]["wt"]["median_pct"] or float("nan"),
    )


if __name__ == "__main__":
    main()

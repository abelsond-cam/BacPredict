"""Intrinsic-attention diagnostic — does the *pretrained* Bacformer attend to rpoB?

Our learned gated-attention MIL pool underperforms the plain mean (e2e ≈ 0.868 < 0.905
baseline), so before building more pooling heads we ask the prior question with Maciej's
method (``run_collect_attention_weight_layers_by_distance`` in Bacformer-internal): read the
model's **own** self-attention (``return_attn_weights=True`` → ``BacformerModelOutput.attentions``),
average over the 15 heads, and measure **how much each protein token is attended to** — then
look at where **rpoB** sits, **resistant vs wild-type**.

Read-out:

- If the frozen model already concentrates attention on rpoB (high received-attention rank,
  especially in R genomes) → a learned pool *can* find it; our gated-MIL is just under-capacity,
  so a multi-head pool / the surprisal panel should help.
- If it does **not** → no label-trained pool can find rpoB unaided ⇒ the explicit surprisal
  panel is essential, and swapping pooling mechanisms won't rescue it.

Label-blind (uses only the manifest's rpoB flat index), read-only, GPU. Modelled on
:mod:`snp_embeddings.frozen_bacformer_rpob_vectors` — same frozen-forward + flat-index pattern,
reading the existing 1000-genome ``manifest.csv`` (``sample`` / ``role`` / ``rpob_flat_index``).
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
from snp_embeddings.snp_vs_esm_prediction import _real_protein_indices
from tl.embed.generate_embeddings import bacformer_attention_weights, load_bacformer_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _received_attention_by_layer(attentions: list[torch.Tensor], real_idx: torch.Tensor) -> np.ndarray:
    """Per-layer mean attention **received** by each real protein token.

    ``attentions[ℓ]`` is ``(1, H, N, N)`` (query dim −2 → key dim −1). For each layer: average
    over heads, restrict to the real protein tokens, then average over queries → the mean
    attention each protein *receives*. Returns ``(n_layers, n_real)`` (float32, on CPU).
    """
    rows: list[np.ndarray] = []
    for att in attentions:
        a = att[0].float().mean(dim=0)          # (N, N) mean over heads
        a = a.index_select(0, real_idx).index_select(1, real_idx)  # (R, R)
        rows.append(a.mean(dim=0).cpu().numpy())  # (R,) mean over queries = received
    return np.stack(rows)


def _rank_stats(received: np.ndarray, rpob_pos: int) -> tuple[float, int]:
    """Return ``(percentile, rank)`` of rpoB's received attention among the genome's proteins.

    ``percentile`` = fraction of proteins rpoB's received attention exceeds (1.0 = the single
    most-attended protein); ``rank`` = number of proteins more attended than rpoB (0 = top).
    """
    v = received[rpob_pos]
    return float((received < v).mean()), int((received > v).sum())


def probe_intrinsic_attention(
    manifest: pd.DataFrame,
    esm_store_dir: Path,
    *,
    device: str,
    pt_suffix: str = "_esm_embeddings.pt",
    max_proteins: int | None = None,
) -> pd.DataFrame:
    """Per-genome rpoB received-attention rank, per layer + mean-over-layers (R vs WT).

    Returns one row per ``(sample, layer)`` with rpoB's received-attention percentile/rank.
    """
    model = load_bacformer_model(device, dtype="auto")
    model_dtype = next(model.parameters()).dtype

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
        attentions = bacformer_attention_weights(model, inputs)
        received = _received_attention_by_layer(attentions, real_idx.to(attentions[0].device))
        del attentions
        n_layers, n_real = received.shape
        received_mean = received.mean(axis=0)

        for layer in range(n_layers):
            pct, rank = _rank_stats(received[layer], rpob_flat)
            rows.append({
                "sample": sample_id, "role": role, "layer": str(layer + 1), "n_proteins": int(n_real),
                "rpob_received": float(received[layer, rpob_flat]), "rpob_received_pct": pct,
                "rpob_received_rank": rank,
            })
        pct, rank = _rank_stats(received_mean, rpob_flat)
        rows.append({
            "sample": sample_id, "role": role, "layer": "mean", "n_proteins": int(n_real),
            "rpob_received": float(received_mean[rpob_flat]), "rpob_received_pct": pct,
            "rpob_received_rank": rank,
        })

    if skips:
        logger.warning("intrinsic-attention probe: skipped %s", skips)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> dict:
    """Per-layer median rpoB received-attention percentile, resistant vs WT."""
    out: dict = {"n_genomes": int(df["sample"].nunique()), "by_layer": {}}
    for layer, g in df.groupby("layer"):
        r = g[g["role"] == "resistant"]["rpob_received_pct"]
        w = g[g["role"] == "wt"]["rpob_received_pct"]
        out["by_layer"][str(layer)] = {
            "resistant_median_pct": float(r.median()) if len(r) else None,
            "wt_median_pct": float(w.median()) if len(w) else None,
            "resistant_top1_frac": float((r >= 0.999).mean()) if len(r) else None,  # rpoB = most-attended
            "n_resistant": int(len(r)), "n_wt": int(len(w)),
        }
    return out


def plot_probe(df: pd.DataFrame, out_path: Path) -> None:
    """Box-by-layer of rpoB received-attention percentile, resistant (red) vs WT (blue)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = [c for c in df["layer"].unique() if c != "mean"]
    layers = sorted(layers, key=int) + (["mean"] if "mean" in df["layer"].unique() else [])
    fig, ax = plt.subplots(figsize=(max(9, 1.0 * len(layers) + 2), 5.5))
    for offset, role, color in ((-0.18, "resistant", "#d62728"), (0.18, "wt", "#1f77b4")):
        data = [df[(df["layer"] == ly) & (df["role"] == role)]["rpob_received_pct"].to_numpy() for ly in layers]
        positions = [i + offset for i in range(len(layers))]
        bp = ax.boxplot(data, positions=positions, widths=0.32, patch_artist=True, showfliers=False)
        for box in bp["boxes"]:
            box.set(facecolor=color, alpha=0.55)
        for med in bp["medians"]:
            med.set(color="black")
        ax.plot([], [], color=color, lw=6, alpha=0.55, label=f"{role} rpoB")
    ax.axhline(0.5, ls="--", lw=1, color="grey", label="median protein (pct 0.5)")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers)
    ax.set_xlabel("Bacformer attention layer")
    ax.set_ylabel("rpoB received-attention percentile\n(1.0 = most-attended protein in the genome)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Does pretrained Bacformer attend to rpoB? (intrinsic self-attention, R vs WT)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """CLI entry point — frozen Bacformer intrinsic-attention probe over the manifest genomes."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest-csv", type=Path, required=True, help="manifest.csv (sample/role/rpob_flat_index).")
    parser.add_argument("--esm-store-dir", type=Path, required=True, help="Dir of {sample}_esm_embeddings.pt.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap genomes (smoke).")
    parser.add_argument("--max-proteins", type=int, default=None, help="Skip genomes with more proteins (memory).")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest_csv)
    if args.max_samples is not None:
        # Keep the R/WT balance when capping: interleave by role.
        r = manifest[manifest["role"] == "resistant"].head(args.max_samples // 2)
        w = manifest[manifest["role"] == "wt"].head(args.max_samples - len(r))
        manifest = pd.concat([r, w]).reset_index(drop=True)
    logger.info("Probing intrinsic attention over %d genomes on %s", len(manifest), args.device)

    df = probe_intrinsic_attention(
        manifest, args.esm_store_dir, device=args.device, max_proteins=args.max_proteins
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output_dir / "intrinsic_attention_rows.parquet", index=False)
    summary = summarise(df)
    (args.output_dir / "intrinsic_attention_probe.json").write_text(json.dumps(summary, indent=2))
    plot_probe(df, args.output_dir / "intrinsic_attention_probe.png")
    logger.info("Wrote rows + JSON + figure to %s", args.output_dir)
    logger.info("mean-layer rpoB pct — resistant=%.3f wt=%.3f",
                summary["by_layer"].get("mean", {}).get("resistant_median_pct") or float("nan"),
                summary["by_layer"].get("mean", {}).get("wt_median_pct") or float("nan"))


if __name__ == "__main__":
    main()

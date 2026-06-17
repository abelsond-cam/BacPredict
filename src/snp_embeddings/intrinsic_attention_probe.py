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
:mod:`snp_embeddings.bacformer_genome_vectors` — same frozen-forward + flat-index pattern,
reading the existing 1000-genome ``manifest.csv`` (``sample`` / ``role`` / ``rpob_flat_index``).

Two extensions:

- **D2 — ``--checkpoint-dir``:** probe a *fine-tuned* backbone's ``.bacformer`` encoder instead of
  the frozen model, to test whether fine-tuning *keeps* rpoB attended internally while the mean
  pool obliterates it (companion to :mod:`snp_embeddings.head_pool_attention_probe`, which measures
  the *head's* pool directly).
- **D3 — ``--protein-parquet-dir``:** name the **top-K most-attended genes** per genome
  (``flatten_proteins``), to ask whether rpoB is *by far* the strongest and what else recurs at the
  top — the evidence for a top-K-attended-gene selection head.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification

from snp_embeddings.bacformer_genome_vectors import _forward_inputs
from snp_embeddings.locate_gene import flatten_proteins
from snp_embeddings.snp_vs_esm_prediction import _real_protein_indices
from tl.embed.generate_embeddings import BACFORMER_MODEL_ID, bacformer_attention_weights, load_bacformer_model
from tl.train.attention_pool import BacformerAttnPoolForGenomeClassification

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


def _load_attn_pool_state_dict(model_dir: Path) -> dict:
    """Load a saved ``state_dict`` from a checkpoint dir (safetensors preferred, then ``.bin``)."""
    safetensors_path = model_dir / "model.safetensors"
    if safetensors_path.exists():
        from safetensors.torch import load_file

        return load_file(str(safetensors_path))
    bin_path = model_dir / "pytorch_model.bin"
    if bin_path.exists():
        return torch.load(str(bin_path), map_location="cpu")
    raise FileNotFoundError(f"no model.safetensors or pytorch_model.bin in {model_dir}")


def _resolve_weight_checkpoint(checkpoint: Path) -> Path:
    """Like :func:`tl.train.evaluate.resolve_checkpoint_dir` but keyed on saved **weights**.

    A plain-``nn.Module`` (gated-MIL) checkpoint is written by HF Trainer as ``model.safetensors``
    with **no** ``config.json``, so ``resolve_checkpoint_dir`` (which requires ``config.json``) cannot
    find it. Resolve over ``checkpoint-*/`` dirs that hold model weights, mirroring the Trainer's
    ``best_model_checkpoint`` choice (else the highest step).
    """
    checkpoint = Path(checkpoint)

    def _has_weights(p: Path) -> bool:
        return (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists()

    if _has_weights(checkpoint):
        return checkpoint

    def _step(p: Path) -> int:
        tail = p.name.rsplit("-", 1)[-1]
        return int(tail) if tail.isdigit() else -1

    candidates = sorted((p for p in checkpoint.glob("checkpoint-*") if _has_weights(p)), key=_step)
    if not candidates:
        raise FileNotFoundError(f"No model weights in {checkpoint} or any checkpoint-*/ subdir.")
    by_name = {p.name: p for p in candidates}
    for c in candidates:
        state = c / "trainer_state.json"
        if not state.exists():
            continue
        try:
            best = json.loads(state.read_text()).get("best_model_checkpoint")
        except (json.JSONDecodeError, OSError):
            best = None
        if best and Path(best).name in by_name:
            return by_name[Path(best).name]
        break
    return candidates[-1]


def load_attn_pool_wrapper(model_dir: Path, device: str, *, dtype: str = "auto") -> torch.nn.Module:
    """Reconstruct a trained ``BacformerAttnPoolForGenomeClassification`` from a resolved checkpoint dir.

    The wrapper is a local ``nn.Module``, **not** HF remote code, and HF Trainer saves it as a bare
    ``model.safetensors`` with **no** ``config.json`` — so ``AutoModel.from_pretrained`` cannot rebuild
    it and there is no stamped config to read (this is what silently failed the gated-MIL probe jobs).
    Build the architecture with :meth:`BacformerAttnPoolForGenomeClassification.from_pretrained_backbone`
    (stamped sizes when a ``config.json`` is present, else the training defaults — gated-MIL,
    ``panel_mode="none"``, ``attn_dim=128``), then overwrite **every** weight (backbone + pool + head)
    with the checkpoint ``state_dict``; a wrong config surfaces as a loud ``load_state_dict`` shape error.

    ``model_dir`` must already be a leaf checkpoint dir with weights — resolve a run dir with
    :func:`_resolve_weight_checkpoint` first.
    """
    model_dir = Path(model_dir)
    if (model_dir / "config.json").exists():  # stamped sizes available (e.g. future panel checkpoints)
        cfg = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
        attn_dim = getattr(cfg, "attn_dim", 128)
        panel_mode = getattr(cfg, "panel_mode", "none")
        panel_dim = getattr(cfg, "panel_dim", 9)
    else:  # plain-nn.Module gated-MIL checkpoint saves no config.json — use the training defaults
        attn_dim, panel_mode, panel_dim = 128, "none", 9
    logger.info(
        "Rebuilding attention-pool wrapper from %s (panel_mode=%s, attn_dim=%s)", model_dir, panel_mode, attn_dim
    )
    model = BacformerAttnPoolForGenomeClassification.from_pretrained_backbone(
        BACFORMER_MODEL_ID,
        num_labels=1,
        attn_dim=attn_dim,
        panel_mode=panel_mode,
        panel_dim=panel_dim,
        dtype=dtype,
    )
    missing, unexpected = model.load_state_dict(_load_attn_pool_state_dict(model_dir), strict=False)
    if unexpected:  # every saved weight must map to a param → guarantees backbone+pool+head all loaded
        raise RuntimeError(f"checkpoint {model_dir} has unexpected keys (arch mismatch): {unexpected[:6]}")
    substantive_missing = [k for k in missing if not k.endswith("position_ids")]
    if substantive_missing:  # a fine-tuned backbone left at base weights would fake a frozen-like result
        raise RuntimeError(f"checkpoint {model_dir} left params unloaded: {substantive_missing[:6]}")
    if missing:
        logger.info("attn-pool load: %d benign buffer(s) absent from checkpoint (e.g. %s)", len(missing), missing[:2])
    if device == "cpu":
        model = model.float()
    return model.to(device).eval()


def load_attention_encoder(device: str, checkpoint_dir: str | None = None) -> torch.nn.Module:
    """Return the attention-capable Bacformer encoder, frozen-pretrained or from a checkpoint.

    With ``checkpoint_dir=None`` this is the frozen complete-genomes model
    (:func:`load_bacformer_model`). Given a fine-tuned checkpoint it returns the ``.bacformer`` encoder
    submodule — the same ``BacformerLargeModel`` class, so :func:`bacformer_attention_weights` (which
    sets ``return_attn_weights=True``) works identically. This is how D2 asks "does a *fine-tuned*
    backbone still attend rpoB internally, while its mean pool obliterates it?".

    A stock **mean-pool** classifier loads via ``AutoModel`` (its remote code is cached). Our custom
    **attention-pool** wrapper (config carries ``attn_dim``) is not HF-loadable, so it is rebuilt with
    :func:`load_attn_pool_wrapper` and its backbone returned.
    """
    if checkpoint_dir is None:
        return load_bacformer_model(device, dtype="auto")
    model_dir = _resolve_weight_checkpoint(Path(checkpoint_dir))
    if not (model_dir / "config.json").exists():  # our gated-MIL wrapper saves weights but no config.json
        return load_attn_pool_wrapper(model_dir, device).bacformer
    logger.info("Loading fine-tuned (mean-pool) backbone from %s", model_dir)
    clf = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir),
        num_labels=1,
        problem_type="binary_classification",
        return_dict=True,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    if device == "cpu":
        clf = clf.float()
    clf = clf.to(device).eval()
    return getattr(clf, "bacformer", clf)


def _flat_to_gene(parquet_dir: Path, sample_id: str) -> dict[int, str | None]:
    """Map every real-protein flat index → gene name for one sample (via ``flatten_proteins``)."""
    df = pd.read_parquet(parquet_dir / f"{sample_id}_protein_sequences.parquet")
    return {r["flat_index"]: r["gene_name"] for r in flatten_proteins(df)}


def probe_intrinsic_attention(
    manifest: pd.DataFrame,
    esm_store_dir: Path,
    *,
    device: str,
    pt_suffix: str = "_esm_embeddings.pt",
    max_proteins: int | None = None,
    checkpoint_dir: str | None = None,
    parquet_dir: Path | None = None,
    top_k: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-genome rpoB received-attention rank, per layer + mean-over-layers (R vs WT).

    Returns ``(df, top_df)``:

    - ``df`` — one row per ``(sample, layer)`` with rpoB's received-attention percentile/rank
      (the mean-over-layers row also carries ``rpob_gap_ratio`` = rpoB / the most-attended protein).
    - ``top_df`` — when ``parquet_dir`` is given, the **top-K most-attended genes** per genome by
      mean-over-layers received attention, named via ``flatten_proteins`` (empty otherwise). This is
      D3: is rpoB *by far* the strongest, and what else is up there?

    ``checkpoint_dir`` selects a fine-tuned backbone (D2); ``None`` = the frozen pretrained model.
    """
    model = load_attention_encoder(device, checkpoint_dir)
    model_dtype = next(model.parameters()).dtype

    rows: list[dict] = []
    top_rows: list[dict] = []
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
        received_mean_max = float(received_mean.max())

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
            "rpob_gap_ratio": float(received_mean[rpob_flat] / received_mean_max) if received_mean_max else None,
        })

        if parquet_dir is not None:
            try:
                gene_map = _flat_to_gene(parquet_dir, sample_id)
            except FileNotFoundError:
                skips["missing_parquet"] = skips.get("missing_parquet", 0) + 1
                gene_map = {}
            order = np.argsort(received_mean)[::-1][:top_k]
            for top_rank, idx in enumerate(order):
                idx = int(idx)
                top_rows.append({
                    "sample": sample_id, "role": role, "rank": top_rank, "flat_index": idx,
                    "gene_name": gene_map.get(idx), "received_mean": float(received_mean[idx]),
                    "is_rpob": bool(idx == rpob_flat),
                })

    if skips:
        logger.warning("intrinsic-attention probe: skipped %s", skips)
    return pd.DataFrame(rows), pd.DataFrame(top_rows)


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


def summarise_topk(top_df: pd.DataFrame, df: pd.DataFrame) -> dict:
    """D3 summary — is rpoB *by far* the strongest, and what recurs at the top?

    Combines the per-genome top-K gene list (``top_df``) with the rpoB rank/gap from the
    mean-over-layers rows of ``df``: how often rpoB lands in the top 1/5/10/20, the median
    gap ratio (rpoB / most-attended), and the genes that most frequently occupy the top-K.
    """
    mean_rows = df[df["layer"] == "mean"]
    ranks = mean_rows["rpob_received_rank"]
    out: dict = {
        "n_genomes": int(top_df["sample"].nunique()) if len(top_df) else 0,
        "rpob_top1_frac": float((ranks == 0).mean()) if len(ranks) else None,
        "rpob_in_top5_frac": float((ranks < 5).mean()) if len(ranks) else None,
        "rpob_in_top10_frac": float((ranks < 10).mean()) if len(ranks) else None,
        "rpob_in_top20_frac": float((ranks < 20).mean()) if len(ranks) else None,
        "rpob_gap_ratio_median": (
            float(mean_rows["rpob_gap_ratio"].median()) if "rpob_gap_ratio" in mean_rows else None
        ),
        "top_genes_overall": (
            {str(k): int(v) for k, v in top_df["gene_name"].value_counts().head(25).items()}
            if len(top_df) else {}
        ),
    }
    return out


def plot_topk(top_df: pd.DataFrame, df: pd.DataFrame, out_path: Path) -> None:
    """Two panels: top-gene frequency across genomes (left) + rpoB gap-ratio histogram (right)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5.5))

    counts = top_df["gene_name"].fillna("(unnamed)").value_counts().head(20)[::-1]
    ax0.barh(range(len(counts)), counts.to_numpy(), color="#4c72b0")
    ax0.set_yticks(range(len(counts)))
    ax0.set_yticklabels(counts.index)
    ax0.set_xlabel(f"# genomes with gene in top-{int(top_df['rank'].max()) + 1} attended")
    ax0.set_title("Most frequently top-attended genes")
    ax0.grid(axis="x", alpha=0.3)

    mean_rows = df[df["layer"] == "mean"]
    if "rpob_gap_ratio" in mean_rows:
        ax1.hist(mean_rows["rpob_gap_ratio"].dropna().to_numpy(), bins=30, color="#d62728", alpha=0.8)
    ax1.axvline(1.0, ls="--", lw=1.5, color="black", label="rpoB = most-attended")
    ax1.set_xlabel("rpoB received attention / most-attended protein")
    ax1.set_ylabel("# genomes")
    ax1.set_title("Is rpoB by far the strongest?")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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
    """CLI entry point — Bacformer intrinsic-attention probe (frozen or fine-tuned) over the manifest."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest-csv", type=Path, required=True, help="manifest.csv (sample/role/rpob_flat_index).")
    parser.add_argument("--esm-store-dir", type=Path, required=True, help="Dir of {sample}_esm_embeddings.pt.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="D2: fine-tuned checkpoint whose .bacformer backbone to probe (default: frozen model).")
    parser.add_argument("--protein-parquet-dir", type=Path, default=None,
                        help="D3: dir of {sample}_protein_sequences.parquet — enables top-K gene naming.")
    parser.add_argument("--top-k", type=int, default=20, help="D3: number of top-attended genes to record per genome.")
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
    backbone = "frozen" if args.checkpoint_dir is None else args.checkpoint_dir
    logger.info("Probing intrinsic attention over %d genomes (backbone=%s) on %s",
                len(manifest), backbone, args.device)

    df, top_df = probe_intrinsic_attention(
        manifest, args.esm_store_dir, device=args.device, max_proteins=args.max_proteins,
        checkpoint_dir=args.checkpoint_dir, parquet_dir=args.protein_parquet_dir, top_k=args.top_k,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output_dir / "intrinsic_attention_rows.parquet", index=False)
    summary = summarise(df)
    (args.output_dir / "intrinsic_attention_probe.json").write_text(json.dumps(summary, indent=2))
    plot_probe(df, args.output_dir / "intrinsic_attention_probe.png")
    if not top_df.empty:
        top_df.to_parquet(args.output_dir / "intrinsic_attention_topk.parquet", index=False)
        topk_summary = summarise_topk(top_df, df)
        (args.output_dir / "intrinsic_attention_topk.json").write_text(json.dumps(topk_summary, indent=2))
        plot_topk(top_df, df, args.output_dir / "intrinsic_attention_topk.png")
        logger.info("rpoB top1=%.2f top5=%.2f top10=%.2f gap_ratio_median=%.3f",
                    topk_summary["rpob_top1_frac"] or float("nan"),
                    topk_summary["rpob_in_top5_frac"] or float("nan"),
                    topk_summary["rpob_in_top10_frac"] or float("nan"),
                    topk_summary["rpob_gap_ratio_median"] or float("nan"))
    logger.info("Wrote rows + JSON + figure to %s", args.output_dir)
    logger.info("mean-layer rpoB pct — resistant=%.3f wt=%.3f",
                summary["by_layer"].get("mean", {}).get("resistant_median_pct") or float("nan"),
                summary["by_layer"].get("mean", {}).get("wt_median_pct") or float("nan"))


if __name__ == "__main__":
    main()

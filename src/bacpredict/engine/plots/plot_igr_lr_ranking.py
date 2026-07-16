"""Results plots for a per-region baclm LR ranking: top-N bars + a score-density KDE.

For one ``(species, drug, method)`` this reads the embedding ranking CSV (+ an optional presence-one-hot
ranking) and renders two figures under ``<out-dir>/<species>/<display_drug>/<method>/``:

* ``top10.png`` — the N best regions by embedding AUROC as **split bars**: a grey **presence one-hot**
  bar (≈0.5 — carriage alone is ~chance) beside the **baclm embedding** bar, whose colour encodes the
  region's **prevalence** on a white→red scale (solid red for near-ubiquitous TB regions, pale for the
  low-prevalence Kp hits). The embedding bar is **hatched** iff the region is a known catalogue
  determinant. x-labels are the flanking-gene pair ``left→right`` with a ``(rRNA)``/``(CRISPR)``/
  ``(regulatory)``/… suffix for a named-feature unit.
* ``density.png`` — a gaussian-KDE of **all** the region AUROCs with a dashed line at the **N-th best**
  score, showing how far the top-N sit above the bulk.

Organism-agnostic: the causal determinants come in via ``--causal-genes`` / ``--causal-csv`` (the TB WHO
driver CSV's ``gene_name`` column, or the Kp CARD causal set supplied by a thin app driver), so the
engine never imports the app annotation layer. Login/CPU, pure matplotlib.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

from bacpredict.engine.plots.labels import display_name  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_FEATURE_SUFFIX = {
    "rrna": "(rRNA)", "trna": "(tRNA)", "tmrna": "(tmRNA)", "ncrna": "(ncRNA)",
    "crispr": "(CRISPR)", "regulatory_region": "(regulatory)", "oric": "(oriC)",
}
_PRESENCE_COLOUR = "#b8bcc4"
_CHANCE = 0.5


def _auroc_col(df: pd.DataFrame, prefix: str = "lr_auroc_") -> str:
    """The single ``<prefix><drug>`` AUROC column in a ranking table."""
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"no {prefix}* column in {list(df.columns)}")
    return cols[0]


def _region_label(row: pd.Series) -> str:
    """``left→right`` for an IGR, or ``<name> (rRNA/CRISPR/…)`` for a named-feature unit."""
    ftype = str(row.get("feature_type", "") or "")
    if ftype in _FEATURE_SUFFIX:
        name = str(row.get("feature_name", "") or row.get("igr_pair", "") or ftype)
        return f"{name} {_FEATURE_SUFFIX[ftype]}"
    pair = row.get("igr_pair")
    if isinstance(pair, str) and pair:
        return pair
    return f"{row.get('left_gene', '')}→{row.get('right_gene', '')}"


def _is_causal(row: pd.Series, causal_lower: set[str]) -> bool:
    """True if any of the region's flanking genes / feature name is a known causal determinant."""
    if not causal_lower:
        return False
    cands: list[str] = []
    for key in ("left_gene", "right_gene", "feature_name"):
        v = row.get(key)
        if isinstance(v, str) and v:
            cands.append(v.lower())
    pair = str(row.get("igr_pair", "") or "")
    if "→" in pair:
        cands += [p.lower() for p in pair.split("→")]
    return any(c in causal_lower for c in cands)


def load_causal(genes: list[str] | None, csv: str | Path | None) -> set[str]:
    """Lower-cased causal-gene set from an explicit list and/or a CSV with a ``gene_name`` column."""
    out = {g.lower() for g in (genes or [])}
    if csv and Path(csv).exists():
        df = pd.read_csv(csv)
        col = "gene_name" if "gene_name" in df.columns else df.columns[0]
        out |= {str(g).lower() for g in df[col] if str(g) != "__ALL_WHO_one_hot__"}
    return out


def plot_top10(rank: pd.DataFrame, presence: pd.DataFrame | None, *, drug: str, method: str,
               species: str, out_path: Path, causal_lower: set[str], top_n: int = 10) -> None:
    """Top-N split bars: presence one-hot vs prevalence-coloured embedding, hatched if causal."""
    au = _auroc_col(rank)
    top = rank.sort_values(au, ascending=False).head(top_n).reset_index(drop=True)
    labels = [_region_label(r) for _, r in top.iterrows()]
    emb = top[au].to_numpy(dtype=float)
    prev = top["prevalence"].to_numpy(dtype=float) if "prevalence" in top else np.zeros(len(top))
    causal_flags = [_is_causal(r, causal_lower) for _, r in top.iterrows()]

    pres_au = None
    if presence is not None and "igr_pair" in top.columns:
        pcol = _auroc_col(presence, "presence_lr_auroc_")
        pmap = dict(zip(presence["igr_pair"], presence[pcol], strict=False))
        pres_au = np.array([pmap.get(ip, np.nan) for ip in top["igr_pair"]], dtype=float)

    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = matplotlib.colormaps["Reds"]
    ecolours = [cmap(norm(0.0 if np.isnan(p) else p)) for p in prev]

    fig, ax = plt.subplots(figsize=(max(7.0, 0.95 * len(top)), 4.7))
    x = np.arange(len(top))
    if pres_au is not None:
        w = 0.38
        ax.bar(x - w / 2, np.nan_to_num(pres_au, nan=0.0), w, color=_PRESENCE_COLOUR,
               edgecolor="black", linewidth=0.5)
        ebars = ax.bar(x + w / 2, emb, w, color=ecolours, edgecolor="black", linewidth=0.6)
    else:
        ebars = ax.bar(x, emb, 0.6, color=ecolours, edgecolor="black", linewidth=0.6)
    for b, causal in zip(ebars, causal_flags, strict=True):
        if causal:
            b.set_hatch("////")

    ax.axhline(_CHANCE, color="0.6", linestyle=":", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("out-of-fold AUROC")
    ax.set_title(f"{species.upper()} {display_name(drug)} — top {len(top)} {method}")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.045).set_label("prevalence")
    handles = [Patch(facecolor=cmap(0.7), edgecolor="black", label="baclm embedding"),
               Patch(facecolor="white", edgecolor="black", hatch="////", label="catalogue-causal")]
    if pres_au is not None:
        handles.insert(0, Patch(facecolor=_PRESENCE_COLOUR, edgecolor="black", label="presence one-hot"))
    ax.legend(handles=handles, fontsize=7, loc="lower left", framealpha=0.9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_density(rank: pd.DataFrame, *, drug: str, method: str, species: str,
                 out_path: Path, top_n: int = 10) -> None:
    """KDE of all region AUROCs with a dashed line at the N-th best."""
    au = _auroc_col(rank)
    vals = rank[au].dropna().to_numpy(dtype=float)
    if len(vals) < 5:
        logger.warning("%s %s %s: only %d scores — skipping density", species, drug, method, len(vals))
        return
    cut = float(np.sort(vals)[::-1][: top_n][-1])  # the N-th best score

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    xs = np.linspace(float(vals.min()), float(max(vals.max(), 1.0)), 256)
    dens = gaussian_kde(vals)(xs)
    ax.fill_between(xs, dens, color="#4c72b0", alpha=0.35)
    ax.plot(xs, dens, color="#2f4b7c", linewidth=1.4)
    ax.axvline(cut, color="#c0392b", linestyle="--", linewidth=1.4, label=f"{top_n}th-best = {cut:.3f}")
    ax.axvline(_CHANCE, color="0.6", linestyle=":", linewidth=1.0)
    ax.set_xlabel("out-of-fold AUROC")
    ax.set_ylabel("density")
    ax.set_title(f"{species.upper()} {display_name(drug)} — all {len(vals):,} {method} scores")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(*, species: str, drug: str, method: str, csv: Path, presence_csv: Path | None, out_dir: Path,
        causal_genes: list[str] | None = None, causal_csv: Path | None = None, top_n: int = 10) -> Path:
    """Render both figures for one (species, drug, method); returns the output dir."""
    rank = pd.read_csv(csv)
    presence = pd.read_csv(presence_csv) if presence_csv and Path(presence_csv).exists() else None
    causal_lower = load_causal(causal_genes, causal_csv)
    base = Path(out_dir) / species / display_name(drug) / method
    plot_top10(rank, presence, drug=drug, method=method, species=species,
               out_path=base / "top10.png", causal_lower=causal_lower, top_n=top_n)
    plot_density(rank, drug=drug, method=method, species=species, out_path=base / "density.png", top_n=top_n)
    logger.info("%s %s %s: wrote %s/{top10,density}.png (%d causal genes, presence=%s)",
                species, drug, method, base, len(causal_lower), presence is not None)
    return base


def main() -> None:
    """CLI (a thin per-species driver supplies --causal-genes/--causal-csv)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--species", required=True, choices=["tb", "kp"])
    p.add_argument("--drug", required=True)
    p.add_argument("--method", default="per_igr", help="ranking method label (per_igr | whole_igr | per_unit).")
    p.add_argument("--csv", type=Path, required=True, help="embedding ranking per_<method>_lr_<drug>.csv")
    p.add_argument("--presence-csv", type=Path, default=None, help="presence ranking per_<method>_presence_lr_<drug>.csv")
    p.add_argument("--out-dir", type=Path, required=True, help="base dir; figures -> <out>/<species>/<drug>/<method>/")
    p.add_argument("--causal-genes", nargs="*", default=None, help="known causal gene names for the drug (hatch).")
    p.add_argument("--causal-csv", type=Path, default=None, help="CSV with a gene_name column of causal genes.")
    p.add_argument("--top-n", type=int, default=10)
    args = p.parse_args()
    run(species=args.species, drug=args.drug, method=args.method, csv=args.csv,
        presence_csv=args.presence_csv, out_dir=args.out_dir, causal_genes=args.causal_genes,
        causal_csv=args.causal_csv, top_n=args.top_n)


if __name__ == "__main__":
    main()

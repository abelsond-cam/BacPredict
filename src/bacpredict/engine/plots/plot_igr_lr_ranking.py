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

from bacpredict.engine.config import visualisations_dir  # noqa: E402
from bacpredict.engine.plots.labels import display_name  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_FEATURE_SUFFIX = {
    "rrna": "(rRNA)", "trna": "(tRNA)", "tmrna": "(tmRNA)", "ncrna": "(ncRNA)",
    "crispr": "(CRISPR)", "regulatory_region": "(regulatory)", "oric": "(oriC)",
}
_PRESENCE_COLOUR = "#b8bcc4"
_IMPUTED_COLOUR = "#2e8b57"  # sea green — the zero-imputed embedding series
_CHANCE = 0.5
_KEY_COLS = ("igr_pair", "upstream_gene", "unit", "gene_name")


def _key_col(df: pd.DataFrame) -> str | None:
    """The ranking table's region-identity column (``igr_pair`` / ``upstream_gene`` / ``unit`` / ``gene_name``).

    Lets the presence/zero-imputed rankings join back onto the carrier table regardless of key scheme —
    flank-pair (per-IGR), synteny-anchor (upstream), or body (per-unit).
    """
    return next((c for c in _KEY_COLS if c in df.columns), None)


def _auroc_col(df: pd.DataFrame, prefix: str = "lr_auroc_") -> str:
    """The single ``<prefix><drug>`` AUROC column in a ranking table."""
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"no {prefix}* column in {list(df.columns)}")
    return cols[0]


def _region_label(row: pd.Series) -> str:
    """Human x-tick label across every key scheme: ``<name> (rRNA/CRISPR/…)`` for a named body, the
    ``upstream:<gene>`` anchor, the ``left→right`` flank pair, or the raw ``unit`` key — capped in length
    (the CRISPR/regulatory feature names run to 100+ chars and otherwise blow out the axis)."""
    ftype = str(row.get("feature_type", "") or "").strip().lower()
    if ftype in _FEATURE_SUFFIX:
        name = str(row.get("feature_name", "") or "").strip() or ftype
        return _cap(f"{name} {_FEATURE_SUFFIX[ftype]}")
    unit = row.get("unit")  # per-unit body key <type>:<name> (feature types outside the suffix map)
    if isinstance(unit, str) and unit:
        return _cap(unit)
    for k in ("upstream_gene", "gene", "gene_name"):  # promoter anchor (upstream:<gene>) or coding gene
        v = row.get(k)
        if isinstance(v, str) and v:
            return _cap(v)
    pair = row.get("igr_pair")
    if isinstance(pair, str) and pair:
        return _cap(pair)
    return _cap(f"{row.get('left_gene', '')}→{row.get('right_gene', '')}")


def _cap(label: str, n: int = 30) -> str:
    """Truncate a long region label for the x-axis (…) — CRISPR/regulatory names run to 100+ chars."""
    label = label.strip()
    return label if len(label) <= n else label[: n - 1].rstrip() + "…"


def _is_causal(row: pd.Series, causal_lower: set[str]) -> bool:
    """True if any of the region's flanking genes / feature name is a known causal determinant."""
    if not causal_lower:
        return False
    cands: list[str] = []
    for key in ("left_gene", "right_gene", "feature_name", "gene", "gene_name"):
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


def _joined_auroc(top: pd.DataFrame, other: pd.DataFrame | None, prefix: str, key: str | None) -> np.ndarray | None:
    """Align ``other``'s ``<prefix><drug>`` AUROC onto the top-N rows by their shared identity ``key``."""
    if other is None or not key or key not in other.columns:
        return None
    try:
        ocol = _auroc_col(other, prefix)
    except ValueError:
        return None
    omap = dict(zip(other[key], other[ocol], strict=False))
    return np.array([omap.get(k, np.nan) for k in top[key]], dtype=float)


def plot_top10(rank: pd.DataFrame, presence: pd.DataFrame | None, *, imputed: pd.DataFrame | None = None,
               drug: str, method: str, species: str, out_path: Path, causal_lower: set[str],
               top_n: int = 10, min_n_pos: int = 20) -> None:
    """Top-N grouped bars: presence one-hot → prevalence-coloured carrier embedding → zero-imputed embedding.

    ``imputed`` (the zero-imputed ranking the concat actually consumes) adds a third bar per region, joined
    to the carrier top-N on the shared identity column; ``None`` keeps the presence-vs-carrier pair (or the
    carrier bar alone). Embedding bars are hatched where the region is a known catalogue determinant.
    ``min_n_pos`` drops the low-n conditional-on-carriage artifacts (a body at prevalence 0.003 scoring 1.0
    on n=6 resistant carriers) so the top-N shows regions with real support — the same guard the ladder uses.
    """
    au = _auroc_col(rank)
    ranked = rank
    if min_n_pos > 0 and "n_pos" in ranked.columns:
        floored = ranked[ranked["n_pos"] >= min_n_pos]
        ranked = floored if not floored.empty else ranked  # fall back if the floor removes everything
    top = ranked.sort_values(au, ascending=False).head(top_n).reset_index(drop=True)
    labels = [_region_label(r) for _, r in top.iterrows()]
    emb = top[au].to_numpy(dtype=float)
    prev = top["prevalence"].to_numpy(dtype=float) if "prevalence" in top else np.zeros(len(top))
    causal_flags = [_is_causal(r, causal_lower) for _, r in top.iterrows()]

    key = _key_col(top)
    pres_au = _joined_auroc(top, presence, "presence_lr_auroc_", key)
    imp_au = _joined_auroc(top, imputed, "lr_auroc_", key)

    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = matplotlib.colormaps["Reds"]
    ecolours = [cmap(norm(0.0 if np.isnan(p) else p)) for p in prev]

    # Bar groups in read order: presence baseline → carrier embedding → zero-imputed embedding.
    groups: list[tuple[np.ndarray, object, bool]] = []
    if pres_au is not None:
        groups.append((pres_au, _PRESENCE_COLOUR, False))
    groups.append((emb, ecolours, True))
    if imp_au is not None:
        groups.append((imp_au, _IMPUTED_COLOUR, True))

    fig, ax = plt.subplots(figsize=(max(7.0, 0.95 * len(top)), 4.7))
    x = np.arange(len(top))
    n = len(groups)
    w = 0.82 / n
    for j, (vals, colour, is_emb) in enumerate(groups):
        bars = ax.bar(x + (j - (n - 1) / 2) * w, np.nan_to_num(vals, nan=0.0), w, color=colour,
                      edgecolor="black", linewidth=0.6)
        if is_emb:
            for b, causal in zip(bars, causal_flags, strict=True):
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
    fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.045).set_label("carrier prevalence")
    handles = []
    if pres_au is not None:
        handles.append(Patch(facecolor=_PRESENCE_COLOUR, edgecolor="black", label="presence one-hot"))
    handles.append(Patch(facecolor=cmap(0.7), edgecolor="black", label="baclm carrier (prevalence)"))
    if imp_au is not None:
        handles.append(Patch(facecolor=_IMPUTED_COLOUR, edgecolor="black", label="baclm zero-imputed"))
    handles.append(Patch(facecolor="white", edgecolor="black", hatch="////", label="catalogue-causal"))
    ax.legend(handles=handles, fontsize=7, loc="lower left", framealpha=0.9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_density(rank: pd.DataFrame, *, drug: str, method: str, species: str, out_path: Path,
                 top_n: int = 10, overlays: list[tuple[str, pd.DataFrame, str, str]] | None = None,
                 min_n_pos: int = 20) -> None:
    """KDE of the carrier region AUROCs (N-th-best marked), overlaid with the zero-imputed + presence KDEs.

    ``overlays`` is ``[(label, ranking_df, auroc_prefix, colour)]`` — typically the zero-imputed
    (``lr_auroc_``) and presence (``presence_lr_auroc_``) rankings. This is the accessory-vs-core figure:
    if the zero-imputed distribution collapses onto the presence baseline the embedding adds nothing once
    mostly-zero (select **core-only**); if it stays above, accessory regions carry real sequence signal.
    A near-constant series (e.g. an all-chance presence one-hot) is dropped — a zero-variance KDE is
    undefined. The KDE spans **all** regions (the low-n artifact spike near 1.0 is part of the distribution),
    but the N-th-best marker is taken over the ``min_n_pos``-floored regions so it reflects real support.
    """
    au = _auroc_col(rank)
    vals = rank[au].dropna().to_numpy(dtype=float)
    if len(vals) < 5:
        logger.warning("%s %s %s: only %d scores — skipping density", species, drug, method, len(vals))
        return
    cut_pool = rank
    if min_n_pos > 0 and "n_pos" in rank.columns:
        floored = rank[rank["n_pos"] >= min_n_pos]
        cut_pool = floored if not floored.empty else rank
    cvals = cut_pool[au].dropna().to_numpy(dtype=float)
    cut = float(np.sort(cvals)[::-1][: top_n][-1]) if len(cvals) else float(np.sort(vals)[::-1][: top_n][-1])

    # (label, values, line-colour, fill-colour|None). The carrier series is filled; overlays are lines.
    series: list[tuple[str, np.ndarray, str, str | None]] = [("carrier-only", vals, "#2f4b7c", "#4c72b0")]
    for label, df, prefix, colour in overlays or []:
        try:
            acol = _auroc_col(df, prefix)
        except ValueError:
            continue
        ov = df[acol].dropna().to_numpy(dtype=float)
        if len(ov) >= 5 and float(np.ptp(ov)) > 0:
            series.append((label, ov, colour, None))

    allv = np.concatenate([sv for _, sv, _, _ in series])
    xs = np.linspace(min(float(allv.min()), 0.45), max(float(allv.max()), 1.0), 256)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for label, sv, line, fill in series:
        dens = gaussian_kde(sv)(xs)
        if fill is not None:
            ax.fill_between(xs, dens, color=fill, alpha=0.30)
        ax.plot(xs, dens, color=line, linewidth=1.5, label=f"{label} (n={len(sv):,})")
    ax.axvline(cut, color="#c0392b", linestyle="--", linewidth=1.4, label=f"{top_n}th-best carrier = {cut:.3f}")
    ax.axvline(_CHANCE, color="0.6", linestyle=":", linewidth=1.0)
    ax.set_xlabel("out-of-fold AUROC")
    ax.set_ylabel("density")
    ax.set_title(f"{species.upper()} {display_name(drug)} — {method} score density")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(*, species: str, drug: str, method: str, csv: Path, presence_csv: Path | None, out_dir: Path,
        causal_genes: list[str] | None = None, causal_csv: Path | None = None, top_n: int = 10,
        imputed_csv: Path | None = None, min_n_pos: int = 20) -> Path:
    """Render both figures for one (species, drug, method); returns the output dir.

    ``imputed_csv`` (the zero-imputed ranking) adds the third top-10 bar + the third density KDE — the
    accessory-vs-core comparison. ``None`` keeps the carrier-vs-presence pair. ``min_n_pos`` floors the
    top-10 selection + the density N-th-best marker so low-n carriage artifacts don't dominate.
    """
    rank = pd.read_csv(csv)
    presence = pd.read_csv(presence_csv) if presence_csv and Path(presence_csv).exists() else None
    imputed = pd.read_csv(imputed_csv) if imputed_csv and Path(imputed_csv).exists() else None
    causal_lower = load_causal(causal_genes, causal_csv)
    base = Path(out_dir) / species / display_name(drug) / method
    plot_top10(rank, presence, imputed=imputed, drug=drug, method=method, species=species,
               out_path=base / "top10.png", causal_lower=causal_lower, top_n=top_n, min_n_pos=min_n_pos)
    overlays: list[tuple[str, pd.DataFrame, str, str]] = []
    if imputed is not None:
        overlays.append(("zero-imputed", imputed, "lr_auroc_", _IMPUTED_COLOUR))
    if presence is not None:
        overlays.append(("presence one-hot", presence, "presence_lr_auroc_", _PRESENCE_COLOUR))
    plot_density(rank, drug=drug, method=method, species=species, out_path=base / "density.png",
                 top_n=top_n, overlays=overlays or None, min_n_pos=min_n_pos)
    logger.info("%s %s %s: wrote %s/{top10,density}.png (%d causal genes, presence=%s, imputed=%s)",
                species, drug, method, base, len(causal_lower), presence is not None, imputed is not None)
    return base


def main() -> None:
    """CLI (a thin per-species driver supplies --causal-genes/--causal-csv)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--species", required=True, choices=["tb", "kp"])
    p.add_argument("--drug", required=True)
    p.add_argument("--method", default="per_igr", help="ranking method label (per_igr | whole_igr | per_unit).")
    p.add_argument("--csv", type=Path, required=True, help="embedding ranking per_<method>_lr_<drug>.csv")
    p.add_argument("--presence-csv", type=Path, default=None, help="presence ranking per_<method>_presence_lr_<drug>.csv")
    p.add_argument("--imputed-csv", type=Path, default=None,
                   help="zero-imputed ranking per_<method>_lr_<drug>.csv from the imputed out-dir "
                        "(adds the 3rd top-10 bar + density KDE — the accessory-vs-core comparison).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="base dir; figures -> <out>/<species>/<drug>/<method>/ "
                   "(default: the repo src/bacpredict/visualisations/ tree).")
    p.add_argument("--causal-genes", nargs="*", default=None, help="known causal gene names for the drug (hatch).")
    p.add_argument("--causal-csv", type=Path, default=None, help="CSV with a gene_name column of causal genes.")
    p.add_argument("--top-n", type=int, default=10)
    args = p.parse_args()
    # Default to the repo visualisations tree; run() appends <species>/<drug>/<method>/, so pass its parent.
    out_dir = args.out_dir or visualisations_dir(args.species).parent
    run(species=args.species, drug=args.drug, method=args.method, csv=args.csv,
        presence_csv=args.presence_csv, out_dir=out_dir, causal_genes=args.causal_genes,
        causal_csv=args.causal_csv, top_n=args.top_n, imputed_csv=args.imputed_csv)


if __name__ == "__main__":
    main()

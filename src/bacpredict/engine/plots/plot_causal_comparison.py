"""Causal comparison: catalogue-called determinants vs the LR's top-ranked regions, per drug.

Two **stacked panels** for one ``(species, drug)``, so the reader sees *which regime we route into at the
concat head*:

* **top — presence-imputed** (the metric the *linear* concat head selects on): each catalogue determinant
  and LR-only region at its **zero-imputed whole-cohort** AUROC. Acquired determinants (``tet(d)``,
  ``aac(6')-Ib``) recover here — they sit level with their red catalogue one-hot tick — because the
  imputed LR sees the *presence* signal that is their mechanism. The concat's routed-in gene (the imputed
  rung-2 pick) is flagged with a ◆.
* **bottom — carrier-only** (the candidate pool the GBDT / tree approach will exploit): the same
  determinants at their **drop-absent** AUROC + a **recomputed** carrier-only LR-only top-10 — different
  regions surface (a rare gene that separates well *among its carriers* rises here but is presence-blind
  in the imputed panel).

**Penetrance → bar opacity.** Every filled bar's alpha tracks its carrier prevalence (a rare region is
translucent), so a *tall-but-faint* bar flags a low-penetrance region scoring high — the confound to eye.
The colourbar is that opacity scale; prevalence is carrier fraction on the resistance-balanced train
subsample, **not** the raw cohort rate. Dark blue = catalogue determinant, light blue = LR-only, red =
catalogue one-hot (per-determinant tick + dashed all-determinant ceiling).

A curated synonym map + the data-driven CARD→Bakta alias map bridge the catalogue↔LR naming gap (see
:func:`_load_alias_map`) — the catalogue names a determinant ``TetD`` / by the regulated gene, the LR
anchors it ``tet(d)`` / at the adjacent gene. Login/CPU.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from bacpredict.engine.config import visualisations_dir
from bacpredict.engine.plots.driver_panel import parse_driver_csv
from bacpredict.engine.plots.labels import PROMOTER_GENE_TO_DETERMINANT, display_name, region_label
from bacpredict.engine.plots.plot_igr_lr_ranking import _auroc_col, _cap

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Curated catalogue-name → LR region-key synonyms (lower-case), derived from the shared promoter map so the
# label helper and the causal-plot join stay in one source of truth. The catalogue names some determinants by
# the regulated gene while the LR anchors the non-coding region at the adjacent gene; the synonym lets a
# catalogue name (``inha``) match the promoter LR key anchored at the adjacent gene (``fabg1``).
_SYNONYMS: dict[str, list[str]] = {}
for _anchor, _det in PROMOTER_GENE_TO_DETERMINANT.items():
    _SYNONYMS.setdefault(_det.lower(), []).append(_anchor.lower())

# CARD names a chromosomal determinant twice — "GyrA (mut)" and "GyrA (WT)" — but the LR anchors one region
# per gene ("gyra"). Strip ONLY the trailing allele-status marker so the join works; never touch internal
# parens, which are part of real acquired-gene names (e.g. AAC(6')-Ib-cr).
_STATUS_SUFFIX = re.compile(r"\s*\((?:mut|wt)\)\s*$", re.IGNORECASE)

_CAUSAL = "#4a1486"     # deep purple (blue/red mix): catalogue region — penetrance now rides on alpha
_LRONLY = "#08306b"     # deep blue (same base darkness as _CAUSAL): non-catalogue genomic region
_CAT_RED = "#c0392b"    # catalogue reference — per-determinant one-hot tick + all-determinant ceiling
_CODING_MARK = "#000000"  # ◆ over the coding gene the concat routes in (rung 2) — black reads on both bar hues
_NC_MARK = "#d94801"    # ★ over the non-coding region the concat routes in (rung 3)
_NC_HATCH = "////"      # line hatch = "includes IGR" (a non-coding promoter/RNA region)
_CHANCE = 0.5
SPECIES_LABEL = {"tb": "TB", "kp": "Kp"}
# Neutral-grey opacity ramp for the penetrance colourbar (faint = rare → solid = near-universal); the alpha,
# not the hue, is the message now that two bar colours carry it. Matches _prev_alpha.
_ALPHA_RAMP = LinearSegmentedColormap.from_list("prev_opacity", [(0.35, 0.35, 0.35, 0.2), (0.15, 0.15, 0.15, 1.0)])


def _prev_alpha(prev: float | None) -> float:
    """Carrier prevalence → bar opacity in [0.2, 1.0] (a floor so a rare region stays visible); NaN → 1.0."""
    if prev is None or (isinstance(prev, float) and np.isnan(prev)):
        return 1.0
    return 0.2 + 0.8 * float(np.clip(prev, 0.0, 1.0))


def _norm_determinant(name: str) -> str:
    """Lower-cased catalogue determinant name with the trailing ``(mut)``/``(WT)`` status marker removed."""
    return _STATUS_SUFFIX.sub("", str(name).strip()).strip().lower()


def _synonyms(name: str) -> set[str]:
    """A catalogue gene name plus its curated LR-key synonyms (all lower-case)."""
    n = name.strip().lower()
    return {n} | {s.lower() for s in _SYNONYMS.get(n, [])}


def _load_alias_map(csv: Path | None) -> dict[str, tuple[frozenset[str], float]]:
    """``{normalised determinant → (frozenset of normalised LR-region aliases, coverage)}`` from an alias CSV.

    Generic — reads the CARD→Bakta map (``card_family``, ``bakta_gene_name``, ``bakta_gene_set``,
    ``bakta_coverage``) so the catalogue's CARD names (``TetA``, ``AAC(6')``) join to the Bakta-named LR
    region keys (``tet(a)``, ``aac(6')-Ib``) by *empirical minimap overlap*, not string parsing. Empty when
    the CSV is absent → callers fall back to name-string matching (the TB path passes no map).
    """
    if not csv or not Path(csv).exists():
        return {}
    m = pd.read_csv(csv)
    out: dict[str, tuple[frozenset[str], float]] = {}
    for _, r in m.iterrows():
        key = str(r.get("card_family", "")).strip().lower()
        aliases = {x.strip().lower() for col in ("bakta_gene_name", "bakta_gene_set")
                   if isinstance(r.get(col), str) for x in str(r[col]).split("|") if x.strip()}
        cov = float(r["bakta_coverage"]) if "bakta_coverage" in m.columns and pd.notna(
            r.get("bakta_coverage")) else float("nan")
        if key and aliases:
            out[key] = (frozenset(aliases), cov)
    return out


def _join_keys(name: str, alias_map: dict[str, tuple[frozenset[str], float]]) -> set[str]:
    """LR-region keys a catalogue determinant may match: itself + curated synonyms + data-driven aliases."""
    keys = _synonyms(name)
    entry = alias_map.get(name.strip().lower())
    return keys | set(entry[0]) if entry else keys


def _catalogue(csv: Path | None) -> tuple[set[str], dict[str, float], float | None]:
    """``(determinant set, {gene → best catalogue one-hot AUROC}, __ALL__ ceiling AUROC)`` from a driver CSV.

    Reuses :func:`driver_panel.parse_driver_csv`, which reads both the TB-Profiler and CARD schemas
    (``gene_name``/``mut_auroc``) and splits out the all-determinant ``__ALL__`` ceiling row.
    """
    if not csv or not Path(csv).exists():
        return set(), {}, None
    drivers, ceiling = parse_driver_csv(Path(csv))
    determinants = {_norm_determinant(g) for g in drivers["gene_name"]}
    au: dict[str, float] = {}
    if "mut_auroc" in drivers.columns:
        for _, r in drivers.iterrows():
            g = _norm_determinant(r["gene_name"])  # "GyrA (mut)"/"GyrA (WT)" → one "gyra" region, best AUROC
            v = r.get("mut_auroc")
            if pd.notna(v):
                au[g] = max(au.get(g, float("-inf")), float(v))
    return determinants, au, (ceiling.get("auroc") if ceiling else None)


def _best_by_key(rankings: list[tuple[pd.DataFrame | None, str, str]],
                 min_n_pos: int) -> dict[str, tuple[float, str, float, str]]:
    """``{key → (auroc, source, prevalence, raw_key)}`` — best LR AUROC per region key across the rankings.

    ``rankings`` is ``[(df, key_col, source_label)]``. Rows below ``min_n_pos`` resistant carriers are
    dropped (the low-n carriage artifacts). An ``upstream:<gene>`` key is *also* indexed by the bare
    ``<gene>`` so a catalogue gene name can match the promoter region anchored at it; both the prefixed and
    bare index carry the same ``raw_key`` (the full prefixed key, e.g. ``upstream:fabg1``) + ``source`` so a
    bar can be hatched and relabelled (promoter/RNA) by what actually won it.
    """
    best: dict[str, tuple[float, str, float, str]] = {}
    for df, kc, source in rankings:
        if df is None or kc not in df.columns:
            continue
        try:
            acol = _auroc_col(df)
        except ValueError:
            continue
        sub = df[df["n_pos"] >= min_n_pos] if "n_pos" in df.columns else df
        for _, r in sub.iterrows():
            au = r.get(acol)
            if pd.isna(au):
                continue
            au, prev = float(au), float(r.get("prevalence", np.nan))
            raw = str(r[kc]).strip().lower()
            keys = {raw} | ({raw.split(":", 1)[1]} if ":" in raw else set())
            for k in keys:
                if k not in best or au > best[k][0]:
                    best[k] = (au, source, prev, raw)
    return best


def _top_gene(coding_df: pd.DataFrame | None) -> str | None:
    """The (imputed) coding ranking's top gene by held-out/OOF AUROC = the concat gene rung's pick; else None."""
    if coding_df is None or "gene_name" not in coding_df.columns:
        return None
    acols = ([c for c in coding_df.columns if c.startswith("eval_auroc_") and coding_df[c].notna().any()]
             or [c for c in coding_df.columns if c.startswith("lr_auroc_")])
    if not acols:
        return None
    d = coding_df[coding_df[acols[0]].notna()]
    if d.empty:
        return None
    return str(d.sort_values(acols[0], ascending=False).iloc[0]["gene_name"]).strip().lower()


# (name, lr_auroc|nan, catalogue_ref|nan, coverage|nan, prevalence|nan, win_source, win_raw_key)
_CatBar = tuple[str, float, float, float, float, str, str]
# (name, lr_auroc, prevalence, win_source, win_raw_key)
_LrBar = tuple[str, float, float, str, str]

# LR sources that name a non-coding region (promoter / named body / convergent flank-pair) — these bars are
# hatched and relabelled by :func:`region_label` so a promoter/RNA reads as distinct from a coding gene.
_NONCODING_SOURCES = frozenset({"upstream", "per_unit", "between"})


def _panel_data(rankings: list[tuple[pd.DataFrame | None, str, str]], catalogue_lower: set[str],
                cat_auroc: dict[str, float], alias_map: dict[str, tuple[frozenset[str], float]], *,
                top_n_lr: int, min_n_pos: int) -> tuple[list[_CatBar], list[_LrBar]]:
    """One panel's ``(catalogue determinant bars, LR-only bars)`` from its ranking set.

    ``catalogue`` bars carry every determinant at its best LR AUROC via the CARD→Bakta alias map (+
    synonyms), plus its own catalogue one-hot AUROC, Bakta coverage, the matched region's prevalence, and
    the winning ranking's ``source``/``raw_key`` (so a determinant whose best signal is a promoter/RNA — e.g.
    ``inha`` won by ``upstream:fabg1`` — is hatched and relabelled "inhA promoter"). A determinant absent
    from every ranking is flagged with ``nan`` AUROC. ``LR-only`` bars are the strongest ``top_n_lr`` regions
    the catalogue does not claim, each with its own source/raw_key. Empty lists when there are no rankings.
    """
    best = _best_by_key(rankings, min_n_pos)
    if not best:
        return [], []
    all_au = sorted((v[0] for v in best.values()), reverse=True)
    cutoff = all_au[min(top_n_lr, len(all_au)) - 1] if all_au else _CHANCE

    cat: list[_CatBar] = []
    matched_keys: set[str] = set()
    for d in sorted(catalogue_lower):
        jk = _join_keys(d, alias_map)
        cands = [(best[s][0], best[s][2], s) for s in jk if s in best]  # (auroc, prevalence, join-key)
        ref = float(cat_auroc.get(d, float("nan")))
        cov = alias_map[d][1] if d in alias_map else float("nan")
        if not cands:
            cat.append((d, float("nan"), ref, cov, float("nan"), "", ""))
            continue
        au, prev, key = max(cands)
        src, raw = best[key][1], best[key][3]
        matched_keys |= {s for s in jk if s in best}
        cat.append((d, au, ref, cov, prev, src, raw))
    cat.sort(key=lambda t: -(t[1] if not np.isnan(t[1]) else -1.0))  # ranked determinants first, absent last

    # LR-only: strongest regions the catalogue does not claim. _best_by_key indexes each prefixed key
    # (upstream:/…) *also* under its bare suffix; collapse to one entry per bare name, drop matched ones.
    matched_bare = {k.split(":", 1)[1] if ":" in k else k for k in matched_keys}
    canon_best: dict[str, tuple[float, float, str, str]] = {}
    for k, (au, src, prev, raw) in best.items():
        name = k.split(":", 1)[1] if ":" in k else k
        if name not in canon_best or au > canon_best[name][0]:
            canon_best[name] = (au, prev, src, raw)
    lr_only = sorted(((n, au, prev, src, raw) for n, (au, prev, src, raw) in canon_best.items()
                      if n not in matched_bare), key=lambda t: -t[1])
    lr_only = [t for t in lr_only if t[1] >= cutoff][:top_n_lr]
    return cat, lr_only


def _bar_label(name: str, source: str, raw_key: str) -> str:
    """x-tick label for one bar, capped so long picks don't collide.

    A non-coding source is relabelled via :func:`region_label` (promoter/RNA/convergent); a coding gene /
    catalogue determinant keeps its name.
    """
    return _cap(region_label(raw_key) if (source in _NONCODING_SOURCES and raw_key) else name, 22)


def _draw_marks(ax, xi: float, top: float, name: str, coding_mark: str | None, noncoding_mark: str | None) -> None:
    """Mark the concat's routed rungs: ◆ over the coding gene (rung 2), ★ over the non-coding region (rung 3)."""
    if coding_mark and name == coding_mark:
        ax.plot(xi, top + 0.032, marker="D", color=_CODING_MARK, markersize=7, markeredgecolor="white",
                markeredgewidth=0.6, zorder=7)
    if noncoding_mark and name == noncoding_mark:
        ax.plot(xi, top + 0.034, marker="*", color=_NC_MARK, markersize=13, markeredgecolor="white",
                markeredgewidth=0.6, zorder=7)


def _draw_panel(ax, cat: list[_CatBar], lr_only: list[_LrBar], ceiling_auroc: float | None, *,
                coding_mark: str | None = None, noncoding_mark: str | None = None) -> bool:
    """Draw one panel (determinant + LR-only bars, penetrance-alpha, red ticks, ceiling). Returns low_cov.

    Non-coding bars (promoter / named RNA body / convergent flank-pair) are dot-hatched and relabelled via
    :func:`region_label`; ``coding_mark``/``noncoding_mark`` place the ◆/★ over the two rungs the concat routes.
    """
    n_cat, n_lr = len(cat), len(lr_only)
    gap = 0.9 if (n_cat and n_lr) else 0.0
    x_cat = list(range(n_cat))
    x_lr = [n_cat + gap + j for j in range(n_lr)]
    labels: list[str] = []
    low_cov = False
    for xi, (name, au, ref, cov, prev, src, raw) in zip(x_cat, cat, strict=True):
        absent = np.isnan(au)
        if absent:  # catalogue determinant absent from every ranking — same colour as the rest, just hollow
            ax.bar(xi, _CHANCE, width=0.66, color="none", edgecolor=_CAUSAL, linewidth=1.1, zorder=3)
            ax.text(xi, _CHANCE + 0.008, "not ranked", ha="center", va="bottom", fontsize=6.5,
                    color="#666", rotation=90)
        else:
            bars = ax.bar(xi, au, width=0.66, color=to_rgba(_CAUSAL, _prev_alpha(prev)), edgecolor="black",
                          linewidth=0.7, zorder=3)
            if src in _NONCODING_SOURCES:
                bars[0].set_hatch(_NC_HATCH)  # a determinant won by its promoter/RNA reads as non-coding
            ax.text(xi, au + 0.006, f"{au:.2f}", ha="center", va="bottom", fontsize=7.5)
        if not np.isnan(ref):  # the determinant's own catalogue one-hot AUROC, as a red reference tick
            ax.plot([xi - 0.36, xi + 0.36], [ref, ref], color=_CAT_RED, lw=2.2, solid_capstyle="butt", zorder=6)
        _draw_marks(ax, xi, _CHANCE if absent else au, name, coding_mark, noncoding_mark)
        flag = not np.isnan(cov) and cov < 0.9  # Bakta under-annotates this CARD determinant
        low_cov = low_cov or flag
        labels.append(_bar_label(name, src, raw) + (" ‡" if flag else ""))
    for xi, (name, au, prev, src, raw) in zip(x_lr, lr_only, strict=True):
        bars = ax.bar(xi, au, width=0.66, color=to_rgba(_LRONLY, _prev_alpha(prev)), edgecolor="black",
                      linewidth=0.7, zorder=3)
        if src in _NONCODING_SOURCES:
            bars[0].set_hatch(_NC_HATCH)
        ax.text(xi, au + 0.006, f"{au:.2f}", ha="center", va="bottom", fontsize=7.5)
        _draw_marks(ax, xi, au, name, coding_mark, noncoding_mark)
        labels.append(_bar_label(name, src, raw))

    if ceiling_auroc is not None and not np.isnan(ceiling_auroc):
        ax.axhline(ceiling_auroc, color=_CAT_RED, ls="--", lw=1.3)
    ax.axhline(_CHANCE, color="0.6", ls=":", lw=1.0)
    ax.set_xticks(x_cat + x_lr)
    ax.set_xticklabels(labels or [""], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0.45, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    return low_cov


def plot_causal_comparison(*, imputed: tuple[list[_CatBar], list[_LrBar]],
                           carrier: tuple[list[_CatBar], list[_LrBar]], ceiling_auroc: float | None,
                           drug: str, species: str, out_path: Path, coding_mark: str | None = None,
                           noncoding_mark: str | None = None) -> None:
    """Draw the two-panel (imputed top / carrier-only bottom) causal comparison → ``out_path``.

    ``coding_mark``/``noncoding_mark`` (bar names resolved from the ladder table) place the ◆/★ over the
    coding gene and non-coding region the concat routes — on the imputed panel only (what it selects on).
    """
    (cat_i, lr_i), (cat_c, lr_c) = imputed, carrier
    if not (cat_i or lr_i or cat_c or lr_c):
        logger.warning("%s %s: no rankings on either panel — skipping causal comparison", species, drug)
        return
    n_max = max(len(cat_i) + len(lr_i), len(cat_c) + len(lr_c), 1)
    width = max(9.0, 0.52 * n_max + 2.6)
    # Extra hspace so the top panel's rotated (long "inhA promoter"/"ogt→mura") x-labels clear the bottom title.
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(width, 11.4), gridspec_kw={"hspace": 0.62})
    low_cov = _draw_panel(ax_top, cat_i, lr_i, ceiling_auroc, coding_mark=coding_mark, noncoding_mark=noncoding_mark)
    low_cov |= _draw_panel(ax_bot, cat_c, lr_c, ceiling_auroc)
    ax_top.set_ylabel("best imputed LR AUROC")
    ax_bot.set_ylabel("best carrier-only LR AUROC")
    ax_top.set_title("LR on all genomes (absence imputed as zero embedding)", fontsize=9.5, loc="left")
    ax_bot.set_title("LR on carrier only (drop-absent)", fontsize=9.5, loc="left")

    # penetrance → opacity colourbar (the alpha scale shared by both panels).
    sm = ScalarMappable(norm=Normalize(0.0, 1.0), cmap=_ALPHA_RAMP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_top, ax_bot], pad=0.012, fraction=0.032)
    cbar.set_label("Carrier prevalence annotated by grading bar opacity", fontsize=8)

    fig.suptitle("Logistic Regression on bacLM Genomic Regions (defined by Bakta)",
                 fontsize=12.5, fontweight="bold", x=0.5, y=0.995)
    fig.text(0.5, 0.955, f"{SPECIES_LABEL.get(species, species.upper())} {display_name(drug)} — "
             "catalogue regions (purple) vs non-catalogue regions (deep blue)", ha="center", fontsize=9.5)
    handles = [Patch(facecolor=_CAUSAL, edgecolor="black", label="Catalogue regions"),
               Patch(facecolor=_LRONLY, edgecolor="black", label="Non-catalogue genomic regions"),
               Patch(facecolor="0.8", edgecolor="black", hatch=_NC_HATCH,
                     label="non-coding IGR (promoter / RNA)"),
               Line2D([0], [0], color=_CAT_RED, lw=2.2, label="catalogue one-hot AUROC (per determinant)"),
               Line2D([0], [0], color=_CAT_RED, lw=1.3, ls="--", label="all-determinant catalogue ceiling"),
               Line2D([0], [0], marker="D", color=_CODING_MARK, lw=0, markersize=7,
                      label="concat gene rung (routed in)"),
               Line2D([0], [0], marker="*", color=_NC_MARK, lw=0, markersize=12,
                      label="concat non-coding rung (routed in)")]
    ax_top.legend(handles=handles, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.02, 1.0), framealpha=0.95)
    if low_cov:
        fig.text(0.01, 0.004, "‡ Bakta under-annotates this CARD determinant (overlap coverage <90%) — "
                 "still the region set the LR runs on", fontsize=6.5, color="#666")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _read(csv: Path | None) -> pd.DataFrame | None:
    return pd.read_csv(csv) if csv and Path(csv).exists() else None


def _routed_marks(ladder_csv: Path | None, cat: list[_CatBar],
                  lr_only: list[_LrBar]) -> tuple[str | None, str | None]:
    """``(coding_mark, noncoding_mark)`` bar-names from the ladder table's routed rung-2 / rung-3 blocks.

    Couples the ◆/★ to what the ladder actually concatenated (rung 2 ``block`` = coding gene, rung 3
    ``block`` = non-coding region key) rather than the tallest ungated bar — the ladder selects on
    ``eval_auroc``-first with a prevalence gate, the bars rank on ``lr_auroc`` ungated, so the two diverge.
    Each block is matched to a drawn bar on its winning ``raw_key`` (else the bar name / bare suffix, so
    ``upstream:fabg1`` lands on the ``inha`` determinant bar that folded it in). ``(None, None)`` if absent.
    """
    if not ladder_csv or not Path(ladder_csv).exists():
        return None, None
    df = pd.read_csv(ladder_csv)
    if "rung" not in df.columns or "block" not in df.columns:
        return None, None

    def _block(rung: int) -> str | None:
        r = df[df["rung"] == rung]
        b = str(r.iloc[0]["block"]).strip().lower() if not r.empty else ""
        return b or None

    bars = [(c[0], c[6]) for c in cat] + [(lb[0], lb[4]) for lb in lr_only]  # (name, raw_key)

    def _match(block: str | None) -> str | None:
        if not block:
            return None
        bare = block.split(":", 1)[1] if ":" in block else block
        for name, raw in bars:
            if block in (raw, name) or name == bare:
                return name
        return None

    return _match(_block(2)), _match(_block(3))


def run(*, species: str, drug: str, imputed_coding_csv: Path | None, carrier_coding_csv: Path | None,
        upstream_csv: Path | None, unit_csv: Path | None, catalogue_csv: Path | None, out_dir: Path,
        causal_genes: list[str] | None = None, top_n_lr: int = 10, min_n_pos: int = 20,
        card_bakta_map_csv: Path | None = None, ladder_table: Path | None = None) -> Path:
    """Render one drug's two-panel causal comparison → ``<out_dir>/<species>/<display_drug>/causal_comparison.png``.

    ``imputed_coding_csv`` (per_gene_lr_ranking_imputed_baclm) drives the **top** panel; ``carrier_coding_csv``
    (per_gene_lr_ranking_baclm) the **bottom**. The non-coding rankings (``upstream_csv``, ``unit_csv``) are
    shared by both panels — core non-coding regions are near-universal, so their imputed and carrier AUROC
    coincide; only the coding ranking differs meaningfully. ``ladder_table`` (the drug's
    ``<drug>_amr_ladder_table.csv``), when given, places the ◆ (coding rung) + ★ (non-coding rung) over the
    two regions the concat actually routes; without it the ◆ falls back to the imputed coding top gene.
    """
    determinants, cat_auroc, ceiling_auroc = _catalogue(catalogue_csv)
    determinants |= {g.strip().lower() for g in (causal_genes or [])}
    alias_map = _load_alias_map(card_bakta_map_csv)  # CARD→Bakta (Kp); {} for TB → name-string matching
    up, un = _read(upstream_csv), _read(unit_csv)
    imp_coding, car_coding = _read(imputed_coding_csv), _read(carrier_coding_csv)
    if imp_coding is None:
        logger.warning("%s %s: imputed coding ranking %s absent — top panel omits the imputed coding bars "
                       "(rebuild once per_gene_lr_ranking_imputed_baclm lands)", species, drug, imputed_coding_csv)
    imputed = _panel_data([(imp_coding, "gene_name", "coding"), (up, "upstream_gene", "upstream"),
                           (un, "unit", "per_unit")], determinants, cat_auroc, alias_map,
                          top_n_lr=top_n_lr, min_n_pos=min_n_pos)
    carrier = _panel_data([(car_coding, "gene_name", "coding"), (up, "upstream_gene", "upstream"),
                           (un, "unit", "per_unit")], determinants, cat_auroc, alias_map,
                          top_n_lr=top_n_lr, min_n_pos=min_n_pos)
    if ladder_table is not None and Path(ladder_table).exists():
        coding_mark, noncoding_mark = _routed_marks(ladder_table, imputed[0], imputed[1])
    else:
        coding_mark, noncoding_mark = _top_gene(imp_coding), None
    out = Path(out_dir) / species / display_name(drug) / "causal_comparison.png"
    plot_causal_comparison(imputed=imputed, carrier=carrier, ceiling_auroc=ceiling_auroc, drug=drug,
                           species=species, out_path=out, coding_mark=coding_mark, noncoding_mark=noncoding_mark)
    logger.info("%s %s: wrote %s (◆ coding=%s, ★ non-coding=%s, %d determinants, %d CARD→Bakta aliases, ceiling=%s)",
                species, drug, out, coding_mark, noncoding_mark, len(determinants), len(alias_map),
                None if ceiling_auroc is None else round(ceiling_auroc, 3))
    return out


def main() -> None:
    """CLI: one drug's two-panel causal comparison from imputed + carrier coding rankings + a catalogue CSV."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--species", required=True, choices=["tb", "kp"])
    p.add_argument("--drug", required=True)
    p.add_argument("--imputed-coding-csv", type=Path, default=None,
                   help="per_gene_lr_ranking_imputed_baclm/<drug>/per_gene_lr_<drug>.csv (top panel).")
    p.add_argument("--carrier-coding-csv", type=Path, default=None,
                   help="per_gene_lr_ranking_baclm/<drug>/per_gene_lr_<drug>.csv (bottom panel).")
    p.add_argument("--upstream-csv", type=Path, default=None, help="per_upstream_lr_<drug>.csv (shared).")
    p.add_argument("--unit-csv", type=Path, default=None, help="per_unit_lr_<drug>.csv (shared).")
    p.add_argument("--catalogue-csv", type=Path, required=True,
                   help="driver CSV with gene_name + mut_auroc (TB-Profiler or CARD schema)")
    p.add_argument("--out-dir", type=Path, default=None, help="default: the repo visualisations/ tree")
    p.add_argument("--card-bakta-map", type=Path, default=None,
                   help="CARD→Bakta alias map CSV (Kp); joins CARD determinant names to Bakta LR keys. "
                        "Omit for TB (TB-Profiler names already match Bakta).")
    p.add_argument("--ladder-table", type=Path, default=None,
                   help="<drug>_amr_ladder_table.csv — places the ◆ (coding rung) + ★ (non-coding rung) over "
                        "the regions the concat actually routes. Omit to fall back to the imputed coding top gene.")
    p.add_argument("--top-n-lr", type=int, default=10)
    p.add_argument("--min-n-pos", type=int, default=20)
    args = p.parse_args()
    out_dir = args.out_dir or visualisations_dir(args.species).parent
    run(species=args.species, drug=args.drug, imputed_coding_csv=args.imputed_coding_csv,
        carrier_coding_csv=args.carrier_coding_csv, upstream_csv=args.upstream_csv, unit_csv=args.unit_csv,
        catalogue_csv=args.catalogue_csv, out_dir=out_dir, top_n_lr=args.top_n_lr, min_n_pos=args.min_n_pos,
        card_bakta_map_csv=args.card_bakta_map, ladder_table=args.ladder_table)


if __name__ == "__main__":
    main()

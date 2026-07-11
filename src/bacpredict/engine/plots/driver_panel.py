"""Per-drug DRIVER PANEL: one-hot ceiling vs baclm vs ESM vs Bacformer for every driving mutation.

For each drug we read its driver list (TB: the TB-Profiler ``tbprofiler_gene_lr_<drug>.csv``; Kp: the
Kleborate ``kleborate_determinant_lr_<drug>.csv``) — one row per driving mutation/determinant, carrying
the **one-hot WHO/Kleborate ceiling** AUROC/AUPRC already computed for that driver. This module fills
the other three columns for the **coding** drivers by scoring each driver gene's pooled vector through
the same k-fold LR harness (``run_kfold_probe``, so the numbers are comparable to the CSV's one-hot):

    | driver | one-hot (from CSV) | baclm | ESM | Bacformer |

- **coding** drivers (``embeddable``) → baclm + ESM (always) + Bacformer (if a gene-token NPZ is passed).
- **non-coding / promoter / rRNA** drivers → one-hot only for now (baclm's non-coding channel needs the
  2d re-embed; ESM/Bacformer are protein models). Left **blank** — the panel is designed to show the
  gaps and backfill them later.

Emits a per-drug table CSV + a grouped column chart (AUROC and AUPRC). CPU except the optional Bacformer
gene-token vectors, which are precomputed on GPU by :mod:`bacpredict.engine.concat.bacformer_gene_panel_vectors`.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bacpredict.engine.gene_lr.card_gene_locator import build_card_presence, sidecar_dir_available
from bacpredict.engine.gene_lr.coding_amr_lr import (
    SpeciesPaths,
    build_multi_gene_presence,
    default_paths,
    load_baclm_gene_vectors,
)
from bacpredict.engine.gene_lr.kfold_probe import FeatureSpec, run_kfold_probe
from bacpredict.engine.gene_lr.snp_vs_esm_prediction import load_pooled_gene_vectors, resolve_clean_splits

logger = logging.getLogger(__name__)

# Drug name (CSV folder) -> AST column in binary_ast_with_split.csv when they differ.
DRUG_COLUMN_OVERRIDES = {"rifampicin": "rifampin"}

_METHOD_COLOURS = {  # keep consistent with the rest of the repo palette
    "one_hot": "#c0392b", "baclm": "#1e8449", "esm": "#7e3f9e", "bacformer": "#2e2a7a",
}
_ALL_ROW = "__ALL_WHO_one_hot__"


def parse_driver_csv(csv_path: Path) -> tuple[pd.DataFrame, dict | None]:
    """Read a driver CSV → ``(drivers, ceiling)``: per-driver rows + the ``__ALL__`` ceiling row (or None).

    Works for both the TB-Profiler and Kleborate schemas: both carry ``gene_name``/``region``/``site``/
    ``mut_auroc``/``mut_auprc`` (+ ``embeddable``/``is_rrna``/``is_noncoding`` for TB). The all-drivers
    ceiling row (``gene_name == __ALL_WHO_one_hot__`` or ``__ALL_Kleborate__``) is split out.
    """
    df = pd.read_csv(csv_path)
    is_all = df["gene_name"].astype(str).str.startswith("__ALL")
    ceiling = None
    if is_all.any():
        r = df[is_all].iloc[0]
        ceiling = {"auroc": float(r["mut_auroc"]), "auprc": float(r["mut_auprc"])}
    return df[~is_all].reset_index(drop=True), ceiling


def _is_coding(row: pd.Series) -> bool:
    """A driver we can score with a protein gene vector: flagged ``embeddable``, not promoter/rRNA.

    Schema-agnostic across TB (TB-Profiler: ``region`` coding/non-coding) and Kp (CARD: ``category``
    acquired_hgt/chromosomal_*). ``embeddable`` is the shared gate; ``is_noncoding``/``is_rrna`` exclude
    promoters and rRNA (which the baclm non-coding channel handles, not the protein path).
    """
    embeddable = bool(row.get("embeddable", str(row.get("region", "")).lower() == "coding"))
    return embeddable and not bool(row.get("is_noncoding", False)) and not bool(row.get("is_rrna", False))


def _score_gene(gene_table, paths: SpeciesPaths, label_map, bacformer_frame, *, n_folds, seeds, pool_workers):
    """k-fold AUROC/AUPRC for one gene's baclm + ESM (+ Bacformer if provided) vectors vs the label."""
    if gene_table.empty:
        return {"error": "no single-copy genomes"}
    esm = load_pooled_gene_vectors(gene_table, paths.esm_dir, pt_suffix=paths.esm_suffix, pool_workers=pool_workers)
    baclm = load_baclm_gene_vectors(gene_table, paths.baclm_dir, pt_suffix=paths.baclm_suffix, pool_workers=pool_workers)
    specs = {}
    if not esm.empty:
        specs["esm"] = FeatureSpec(frame=esm, kind="numeric", standardise=True)
    if not baclm.empty:
        specs["baclm"] = FeatureSpec(frame=baclm, kind="numeric", standardise=True)
    if bacformer_frame is not None and not bacformer_frame.empty:
        specs["bacformer"] = FeatureSpec(frame=bacformer_frame, kind="numeric", standardise=True)
    if not specs:
        return {"error": "no non-empty frames"}
    kfold = run_kfold_probe(specs, label_map, n_folds=n_folds, seeds=seeds)
    out = {}
    for name, f in kfold["frames"].items():
        agg = f["aggregate"]
        out[name] = {"auroc": agg["auroc"], "auprc": agg["auprc"]}
    out["n_evaluate"] = kfold["config"]["n_evaluate"]
    return out


def _bacformer_frame_for_gene(gene: str, npz) -> pd.DataFrame | None:
    """Pull one gene's Bacformer gene-token matrix from a sweep NPZ (keys ``<gene>__ids``/``__tok``)."""
    if npz is None:
        return None
    ids_key, tok_key = f"{gene}__ids", f"{gene}__tok"
    if ids_key not in npz or tok_key not in npz:
        return None
    ids = [str(x) for x in npz[ids_key]]
    return pd.DataFrame(npz[tok_key], index=pd.Index(ids, name="Sample"))


def run_drug_panel(
    drug: str,
    csv_path: Path,
    paths: SpeciesPaths,
    *,
    ast_column: str,
    bacformer_npz=None,
    card_sidecar_dir: Path | None = None,
    n_folds: int = 5,
    seeds: tuple[int, ...] = (1, 2, 3),
    pool_workers: int = 8,
) -> dict:
    """Build the per-driver [one-hot | baclm | ESM | Bacformer] table for one drug.

    Coding driver genes are located either by Bakta ``gene_name`` (default; works for core genes like
    *rpoB*/*gyrA*) or, when ``card_sidecar_dir`` is given, by CARD ``amr_gene_family`` from the AMR
    sidecar (needed for acquired Kp genes Bakta under-annotates — AAC(6'), bla_KPC, …).
    """
    drivers, ceiling = parse_driver_csv(csv_path)
    label_map, *_ = resolve_clean_splits(paths.ast_sheet, ast_column)
    sample_ids = sorted(label_map)
    logger.info("[%s] %d drivers, %d labelled samples (col=%s)", drug, len(drivers), len(sample_ids), ast_column)

    coding_genes = sorted({str(r["gene_name"]) for _, r in drivers.iterrows() if _is_coding(r)})
    if not coding_genes:
        presence = {}
    elif card_sidecar_dir is not None:
        presence = build_card_presence(
            sample_ids, card_sidecar_dir, paths.parquet_dir, [(g, ()) for g in coding_genes],
            parquet_suffix=paths.parquet_suffix, pool_workers=pool_workers,
        )
    else:
        presence = build_multi_gene_presence(
            sample_ids, paths.parquet_dir, [(g, ()) for g in coding_genes],
            parquet_suffix=paths.parquet_suffix, pool_workers=pool_workers,
        )

    rows = []
    for _, r in drivers.iterrows():
        gene = str(r["gene_name"])
        n_genomes = r.get("n_genomes_with_variant", r.get("n_genomes_with_determinant", 0))
        row = {
            "gene": gene, "site": r.get("site", gene),
            "region": r.get("region", r.get("category", "")),
            "onehot_auroc": float(r["mut_auroc"]), "onehot_auprc": float(r["mut_auprc"]),
            "n_genomes_with_variant": int(n_genomes or 0),
            "baclm_auroc": None, "baclm_auprc": None, "esm_auroc": None, "esm_auprc": None,
            "bacformer_auroc": None, "bacformer_auprc": None,
        }
        if _is_coding(r) and gene in presence:
            gene_table = presence[gene].loc[presence[gene].index.intersection(sample_ids)]
            bac = _bacformer_frame_for_gene(gene, bacformer_npz)
            scored = _score_gene(gene_table, paths, label_map, bac,
                                 n_folds=n_folds, seeds=seeds, pool_workers=pool_workers)
            for m in ("baclm", "esm", "bacformer"):
                if m in scored and scored[m].get("auroc"):
                    row[f"{m}_auroc"] = scored[m]["auroc"]["mean"]
                    row[f"{m}_auprc"] = scored[m]["auprc"]["mean"] if scored[m].get("auprc") else None
            row["n_evaluate"] = scored.get("n_evaluate")
        rows.append(row)

    table = pd.DataFrame(rows)
    return {"drug": drug, "ast_column": ast_column, "ceiling": ceiling, "table": table}


def plot_drug_panel(result: dict, png_path: Path) -> None:
    """Grouped column chart per driver: one-hot / baclm / ESM / Bacformer, AUROC (top) + AUPRC (bottom)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = result["table"]
    if table.empty:
        return
    # Order drivers by one-hot AUROC descending; cap to keep the chart legible.
    table = table.sort_values("onehot_auroc", ascending=False).head(20).reset_index(drop=True)
    labels = [f"{r.gene}\n{r.site}" if str(r.site) != str(r.gene) else str(r.gene) for r in table.itertuples()]
    methods = [("one-hot", "onehot"), ("baclm", "baclm"), ("ESM", "esm"), ("Bacformer", "bacformer")]
    colours = [_METHOD_COLOURS[k] for k in ("one_hot", "baclm", "esm", "bacformer")]

    x = np.arange(len(table))
    w = 0.8 / len(methods)
    fig, axes = plt.subplots(2, 1, figsize=(max(8, 0.9 * len(table)), 8), sharex=True)
    for ax, metric, title in ((axes[0], "auroc", "AUROC"), (axes[1], "auprc", "AUPRC")):
        for i, (name, key) in enumerate(methods):
            vals = [table[f"{key}_{metric}"].iloc[j] for j in range(len(table))]
            vals = [np.nan if v is None else v for v in vals]
            ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=name, color=colours[i], edgecolor="black", linewidth=0.3)
        if result.get("ceiling") and metric in result["ceiling"]:
            ax.axhline(result["ceiling"][metric], ls="--", c="#555", lw=1, label="all-drivers ceiling")
        ax.set_ylabel(title)
        ax.set_ylim(0, 1.02)
        ax.yaxis.grid(True, ls="--", alpha=0.5)
        ax.set_axisbelow(True)
    axes[0].set_title(f"{result['drug']} — driver panel (blank = not yet embedded)", fontweight="bold")
    axes[0].legend(ncol=5, fontsize=8, loc="lower right")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _resolve_ast_column(drug: str, ast_columns: set[str]) -> str | None:
    """Map a drug (CSV folder name) to its AST column, or None if the cohort has no such column."""
    col = DRUG_COLUMN_OVERRIDES.get(drug, drug)
    return col if col in ast_columns else None


def main() -> None:
    """CLI: build driver panels for a set of drugs (TB/Kp), writing per-drug tables + charts."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    ap = argparse.ArgumentParser(description="Per-drug driver panel: one-hot vs baclm/ESM/Bacformer.")
    ap.add_argument("--species", choices=["tb", "kp"], default="tb")
    ap.add_argument("--csv-dir", type=Path, required=True,
                    help="dir with <prefix>_<drug>/*_gene_lr_<drug>.csv driver lists (repo docs/visualisations).")
    ap.add_argument("--csv-prefix", default="tbprofiler_gene_lr", help="driver CSV filename stem (default TB).")
    ap.add_argument("--csv-suffix", default="", help="driver CSV filename suffix before .csv (Kp: _family).")
    ap.add_argument("--folder-prefix", default="tb", help="per-drug folder prefix (tb_/kp_).")
    ap.add_argument("--drugs", nargs="*", default=None, help="drugs to run (default: all folders present).")
    ap.add_argument("--bacformer-npz", type=Path, default=None, help="optional Bacformer gene-token sweep NPZ.")
    ap.add_argument("--amr-sidecar-dir", type=Path, default=None,
                    help="dir of {Sample}_amr.parquet CARD sidecars — locate coding drivers by CARD family "
                    "(Kp acquired genes Bakta misses) instead of Bakta gene_name.")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seeds", type=str, default="1,2,3")
    ap.add_argument("--pool-workers", type=int, default=8)
    ap.add_argument("--output", type=Path, required=True, help="output dir for per-drug tables + charts + summary.")
    args = ap.parse_args()

    paths = default_paths(args.species)
    ast_columns = set(pd.read_csv(paths.ast_sheet, nrows=0).columns)
    seeds = tuple(int(s) for s in args.seeds.split(","))
    npz = np.load(args.bacformer_npz) if args.bacformer_npz and args.bacformer_npz.exists() else None

    card_dir = args.amr_sidecar_dir
    if card_dir is not None:
        universe = pd.read_csv(paths.ast_sheet)["Sample"].astype(str).tolist()
        if not sidecar_dir_available(card_dir, universe):
            logger.warning("AMR sidecar dir %s not populated — falling back to Bakta gene_name locating", card_dir)
            card_dir = None

    # Discover drugs from the folder layout unless an explicit list is given.
    if args.drugs:
        drugs = args.drugs
    else:
        drugs = sorted(p.name[len(args.folder_prefix) + 1:] for p in args.csv_dir.glob(f"{args.folder_prefix}_*") if p.is_dir())

    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    for drug in drugs:
        csv_path = args.csv_dir / f"{args.folder_prefix}_{drug}" / f"{args.csv_prefix}_{drug}{args.csv_suffix}.csv"
        if not csv_path.exists():
            logger.warning("[%s] no CSV at %s — skipping", drug, csv_path)
            continue
        ast_col = _resolve_ast_column(drug, ast_columns)
        if ast_col is None:
            logger.warning("[%s] no AST column in the cohort — skipping (drivers-only, no labels)", drug)
            continue
        result = run_drug_panel(drug, csv_path, paths, ast_column=ast_col, bacformer_npz=npz,
                                card_sidecar_dir=card_dir, n_folds=args.n_folds, seeds=seeds,
                                pool_workers=args.pool_workers)
        result["table"].to_csv(args.output / f"driver_panel_{drug}.csv", index=False)
        plot_drug_panel(result, args.output / f"driver_panel_{drug}.png")
        n_coding = int(result["table"]["baclm_auroc"].notna().sum())
        summary.append({"drug": drug, "ast_column": ast_col, "n_drivers": len(result["table"]),
                        "n_coding_scored": n_coding,
                        "ceiling_auroc": (result["ceiling"] or {}).get("auroc")})
        logger.info("[%s] wrote table + chart (%d drivers, %d coding scored)", drug, len(result["table"]), n_coding)

    (args.output / "driver_panel_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("done: %d drugs -> %s", len(summary), args.output)


if __name__ == "__main__":
    main()

"""Per-determinant Kleborate LR — the *Klebsiella* resistance-catalogue "ceiling".

The Kp analogue of ``bacpredict.apps.tb.tbprofiler_gene_lr`` (the WHO one-hot ceiling for TB). Where TB's
ceiling is built from the WHO/TB-Profiler variant catalogue, Kp's is built from **Kleborate** determinant
calls already stored in ``metadata_v2``. For each drug we build a one-hot of the relevant Kleborate
determinants and score ``determinants → drug label`` through the same k-fold × m-seed harness
(:func:`bacpredict.engine.gene_lr.kfold_probe.run_kfold_probe`), so the AUROC is directly comparable to the
ESM-gene / Bacformer-mean / concat ladder.

Two grains are emitted per drug:

- one **bar per Kleborate column** (e.g. ``Bla_ESBL_acquired``, ``Flq_mutations``, ``Omp_mutations``),
  each scored on its own determinant one-hot and tagged by mechanism **category** — ``acquired_hgt``
  (a gene Bacformer/ESM can embed), ``chromosomal_coding`` (intrinsic gene), ``chromosomal_mutation``
  (point mutation — the signal ESM loses), ``porin_truncation``, ``truncation_lof``. This is the
  HGT-vs-chromosomal split, the programme's central axis;
- the **full per-drug one-hot** (every determinant for the drug together) — row ``__ALL_Kleborate__``,
  the catalogue ceiling the protein-only concat is measured against.

**Why Kleborate alone, not a re-run of CARD/AMRFinderPlus/ResFinder.** For the KpSC module Kleborate v3
is CARD-derived (CARD v3.2.9) and detects the most ARGs of any tool; a 2025 benchmark (Sci Rep
s41598-025-24333-9) found integrating other tools does *not* improve Kp determinant-based prediction
("fewer, well-curated features outperform quantity"), and the weak β-lactam/tetracycline drugs are
literature-wide catalogue knowledge gaps — so a *low* Kleborate ceiling there is faithful, not an
artefact. See ``CLAUDE.md`` and the ``kleborate-ceiling-vs-amr-tools`` memory.

sklearn over a sparse binary matrix — light; a short CPU job (login node / small sbatch, no GPU).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd

from bacpredict.engine.catalogue.base import score_onehot_frame

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Canonical metadata_v2 row key and the AST-sheet sample-id aliases that map onto it.
META_SAMPLE_COL = "Sample"
AST_SAMPLE_ALIASES = ("Sample", "phenotype-BioSample_ID", "sample_accession")

# A determinant must be carried by ≥ this many genomes for its Kleborate column to get a scored bar
# (mirrors TB's MIN_VARIANT_GENOMES). The full one-hot ceiling uses every determinant regardless.
MIN_DETERMINANT_GENOMES = 10

# ---------------------------------------------------------------------------
# Vendored Kleborate cell semantics (mirror BacHGT bac_kleborate.parsing — not importable here, that
# package lives in the sibling repo and is not a dependency of this env; kept faithful to its rules).
# ---------------------------------------------------------------------------
KLEBORATE_ABSENT_TOKENS: frozenset[str] = frozenset(
    {"-", "0", "0.0", "", "NA", "na", "nan", "None", "none"}
)
_COVERAGE_SUFFIX = re.compile(r"-?\d+(?:\.\d+)?%$")  # trailing '-42%' partial-coverage marker
_QUALITY_MARKERS = re.compile(r"[*^?]")              # imperfect-match flags Kleborate appends


def _clean_token(tok: str) -> str:
    """Strip Kleborate quality markers (``*^?``) and a trailing ``-NN%`` coverage suffix from a token."""
    return _QUALITY_MARKERS.sub("", _COVERAGE_SUFFIX.sub("", tok.strip())).strip()


def tokenize_cell(val) -> list[str]:
    """``;``/``,``-separated Kleborate cell → list of cleaned determinant tokens (absent/NaN → ``[]``).

    Splits on ``;`` (classes) and ``,`` (multi-copy lists), drops the no-detection sentinels in
    :data:`KLEBORATE_ABSENT_TOKENS`, and removes imperfect-match / coverage annotations so ``CTX-M-15``
    and ``CTX-M-15*`` collapse to one feature.
    """
    if pd.isna(val):
        return []
    out: list[str] = []
    for chunk in str(val).replace(",", ";").split(";"):
        chunk = chunk.strip()
        if not chunk or chunk in KLEBORATE_ABSENT_TOKENS:
            continue
        cleaned = _clean_token(chunk)
        if cleaned and cleaned not in KLEBORATE_ABSENT_TOKENS:
            out.append(cleaned)
    return out


# ---------------------------------------------------------------------------
# Kleborate column → mechanism schema, and the per-drug determinant map.
# ---------------------------------------------------------------------------
# (category, embeddable): embeddable = can ESM/Bacformer represent the determinant's signal?
#   acquired genes + intrinsic chromosomal genes are real proteins (True); point mutations and
#   truncations/loss are the signals a protein-mean model loses (False) — the chromosomal regime.
COLUMN_SCHEMA: dict[str, tuple[str, bool]] = {
    "AGly_acquired": ("acquired_hgt", True),
    "Bla_acquired": ("acquired_hgt", True),
    "Bla_inhR_acquired": ("acquired_hgt", True),
    "Bla_ESBL_acquired": ("acquired_hgt", True),
    "Bla_ESBL_inhR_acquired": ("acquired_hgt", True),
    "Bla_Carb_acquired": ("acquired_hgt", True),
    "Col_acquired": ("acquired_hgt", True),       # mcr-1..-9
    "Tet_acquired": ("acquired_hgt", True),
    "MLS_acquired": ("acquired_hgt", True),
    "Tmt_acquired": ("acquired_hgt", True),
    "Sul_acquired": ("acquired_hgt", True),
    "Flq_acquired": ("acquired_hgt", True),        # qnr / oqxAB / aac(6')-Ib-cr
    "Bla_chr": ("chromosomal_coding", True),       # intrinsic SHV/OKP/LEN — a gene, embeddable
    "SHV_mutations": ("chromosomal_mutation", False),
    "Flq_mutations": ("chromosomal_mutation", False),   # GyrA/ParC QRDR codons
    "Col_mutations": ("truncation_lof", False),    # mgrB/pmrB truncation/disruption
    "Omp_mutations": ("porin_truncation", False),  # OmpK35/36 loss + GD/TD loop inserts
    "truncated_resistance_hits": ("truncation_lof", False),
}

# Mechanism groups → the Kleborate columns that carry their determinants. The β-lactam group is
# deliberately inclusive (all Bla_* + porin) because a ceiling should use every determinant that can
# plausibly drive resistance to the drug and let the LR weight them — under-including lowers the ceiling.
_BLA_ALL = [
    "Bla_acquired", "Bla_inhR_acquired", "Bla_ESBL_acquired", "Bla_ESBL_inhR_acquired",
    "Bla_Carb_acquired", "Bla_chr", "SHV_mutations", "Omp_mutations",
]
_FLQ = ["Flq_acquired", "Flq_mutations"]
_AGLY = ["AGly_acquired"]

# Per-drug determinant columns. Keys are the lowercase AST drug columns in binary_ast_with_split.csv.
# REVIEW POINT: this drug→Kleborate-column map is a clinical-pharmacology judgment — adjust as needed.
DRUG_COLUMNS: dict[str, list[str]] = {
    # β-lactams (penicillins/combos, cephalosporins, cephamycin, monobactam, carbapenems)
    "ampicillin-sulbactam": _BLA_ALL,
    "piperacillin-tazobactam": _BLA_ALL,
    "cefazolin": _BLA_ALL,
    "cefuroxime": _BLA_ALL,
    "cefoxitin": _BLA_ALL,
    "cefotaxime": _BLA_ALL,
    "ceftriaxone": _BLA_ALL,
    "ceftazidime": _BLA_ALL,
    "cefepime": _BLA_ALL,
    "aztreonam": _BLA_ALL,
    "ertapenem": _BLA_ALL,
    "imipenem": _BLA_ALL,
    "meropenem": _BLA_ALL,
    # fluoroquinolones
    "ciprofloxacin": _FLQ,
    "levofloxacin": _FLQ,
    # aminoglycosides
    "gentamicin": _AGLY,
    "amikacin": _AGLY,
    "tobramycin": _AGLY,
    # other classes
    "colistin": ["Col_acquired", "Col_mutations"],
    "tetracycline": ["Tet_acquired"],
    "azithromycin": ["MLS_acquired"],
    "trimethoprim-sulfamethoxazole": ["Tmt_acquired", "Sul_acquired"],
}

# Weakest-first default sweep (Bacformer held-out AUROC ascending) — where concat is most likely to help.
DEFAULT_DRUGS = ["colistin", "azithromycin", "cefepime", "aztreonam", "cefoxitin", "tetracycline"]

ALL_KEY = "__ALL_Kleborate__"


def _site_label(column: str) -> str:
    """Human-readable bar label for a Kleborate column, e.g. ``Bla_ESBL (acquired)``."""
    pretty = column.replace("_acquired", " (acquired)").replace("_mutations", " (mutations)")
    return pretty.replace("_", " ")


def load_labels(ast_sheet: Path, drug: str) -> dict[str, int]:
    """``Sample → 0/1`` for one drug from the AST sheet (drop NaN / ambiguous 0.5)."""
    df = pd.read_csv(ast_sheet)
    sample_col = next((c for c in AST_SAMPLE_ALIASES if c in df.columns), None)
    if sample_col is None:
        raise ValueError(f"{ast_sheet} has no sample-id column (looked for {AST_SAMPLE_ALIASES}).")
    if drug not in df.columns:
        raise ValueError(f"{ast_sheet} has no '{drug}' column.")
    df = df[[sample_col, drug]].rename(columns={sample_col: "Sample"}).dropna(subset=[drug])
    df = df[df[drug].isin([0, 1, 0.0, 1.0])]
    return {s: int(v) for s, v in zip(df["Sample"], df[drug], strict=True)}


def build_determinant_onehot(meta_labelled: pd.DataFrame, columns: list[str],
                             universe: list[str]) -> pd.DataFrame:
    """Genomes × determinant binary frame over ``universe`` for the given Kleborate ``columns``.

    Each feature is ``"<column>:<token>"`` so features never collide across columns and can be grouped
    back to their column for the per-mechanism bars. Genomes carrying no determinant become all-zero rows.
    """
    records: list[pd.DataFrame] = []
    for col in columns:
        if col not in meta_labelled.columns:
            logger.warning("column %s not in metadata — skipping", col)
            continue
        sub = meta_labelled[[META_SAMPLE_COL, col]].copy()
        sub["token"] = sub[col].apply(tokenize_cell)
        sub = sub.explode("token").dropna(subset=["token"])
        if sub.empty:
            continue
        sub["feature"] = col + ":" + sub["token"].astype(str)
        records.append(sub[[META_SAMPLE_COL, "feature"]])
    if not records:
        return pd.DataFrame(index=universe)
    long = pd.concat(records, ignore_index=True)
    oh = pd.crosstab(long[META_SAMPLE_COL], long["feature"]).clip(upper=1)
    return oh.reindex(universe).fillna(0).astype(int)


def _columns_for(frame: pd.DataFrame, column: str) -> list[str]:
    """The one-hot feature names belonging to one Kleborate ``column`` (prefix ``"<column>:"``)."""
    return [f for f in frame.columns if f.split(":", 1)[0] == column]


def run(metadata: Path, ast_sheet: Path, out_dir: Path, drugs: list[str],
        seeds: tuple[int, ...] = (1, 2, 3)) -> None:
    """Per drug: per-Kleborate-column determinant LR + the full one-hot ceiling, written as a CSV."""
    needed = sorted({c for d in drugs for c in DRUG_COLUMNS.get(d, [])})
    unknown = [d for d in drugs if d not in DRUG_COLUMNS]
    if unknown:
        raise ValueError(f"no Kleborate column map for drugs: {unknown} (add to DRUG_COLUMNS).")
    logger.info("reading metadata %s (cols: Sample + %d determinant columns)", metadata, len(needed))
    meta = pd.read_csv(metadata, sep="\t", usecols=[META_SAMPLE_COL, *needed], low_memory=False)
    out_dir.mkdir(parents=True, exist_ok=True)

    for drug in drugs:
        label_map = load_labels(ast_sheet, drug)
        meta_labelled = meta[meta[META_SAMPLE_COL].isin(label_map)]
        universe = sorted(set(meta_labelled[META_SAMPLE_COL]) & set(label_map))
        n_join_miss = len(label_map) - len(universe)
        logger.info("%s: %d labelled, %d joined to metadata (%d unmatched)",
                    drug, len(label_map), len(universe), n_join_miss)
        if n_join_miss > 0.5 * len(label_map):
            logger.warning("%s: >50%% of labelled samples have no metadata_v2 row — check the join key "
                           "(Sample vs assembly accession vs BioSample)", drug)
        if len(universe) < 6:
            logger.warning("%s: too few joined samples (%d) — skipping", drug, len(universe))
            continue

        columns = DRUG_COLUMNS[drug]
        oh = build_determinant_onehot(meta_labelled, columns, universe)
        rows = []
        for col in columns:
            feats = _columns_for(oh, col)
            if not feats:
                continue
            sub = oh[feats]
            n_genomes = int((sub.sum(axis=1) > 0).sum())
            if n_genomes < MIN_DETERMINANT_GENOMES:
                continue
            agg = score_onehot_frame(sub, label_map, seeds)
            if agg is None:
                continue
            category, embeddable = COLUMN_SCHEMA.get(col, ("other", False))
            rows.append({
                "gene_name": col, "site": _site_label(col), "category": category,
                "mut_auroc": agg["auroc"]["mean"], "mut_auroc_sd": agg["auroc"]["sd"],
                "mut_auprc": agg["auprc"]["mean"], "mut_auprc_sd": agg["auprc"]["sd"],
                "n_determinants": len(feats), "n_genomes_with_determinant": n_genomes,
                "embeddable": embeddable,
                # back-compat fields for the TB cause-histogram plotter:
                "is_rrna": False, "is_noncoding": not embeddable,
            })

        full = score_onehot_frame(oh, label_map, seeds)
        if full is not None:
            rows.append({
                "gene_name": ALL_KEY, "site": ALL_KEY, "category": "all",
                "mut_auroc": full["auroc"]["mean"], "mut_auroc_sd": full["auroc"]["sd"],
                "mut_auprc": full["auprc"]["mean"], "mut_auprc_sd": full["auprc"]["sd"],
                "n_determinants": oh.shape[1], "n_genomes_with_determinant": int((oh.sum(axis=1) > 0).sum()),
                "embeddable": False, "is_rrna": False, "is_noncoding": False,
            })
        if not rows:
            logger.warning("%s: no scorable determinant columns — skipping", drug)
            continue

        df = pd.DataFrame(rows).sort_values("mut_auroc", ascending=False)
        drug_dir = out_dir / f"kp_{drug}"
        drug_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(drug_dir / f"kleborate_determinant_lr_{drug}.csv", index=False)
        top = df[df.gene_name != ALL_KEY].head(3)
        logger.info("%s: %d columns scored | top: %s | ceiling %.3f", drug, len(df) - 1,
                    ", ".join(f"{r.site}={r.mut_auroc:.3f}" for r in top.itertuples()),
                    full["auroc"]["mean"] if full else float("nan"))

    done = sorted(str(p.relative_to(out_dir)) for p in out_dir.glob("kp_*/kleborate_determinant_lr_*.csv"))
    (out_dir / "kleborate_determinant_lr_manifest.json").write_text(json.dumps({"files": done}, indent=2))


def main() -> None:
    """CLI entry point."""
    rds = Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david")
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", type=Path,
                        default=rds / "final" / "metadata_v2_all_samples_and_columns.tsv",
                        help="metadata_v2_all_samples_and_columns.tsv (Kleborate determinant columns).")
    parser.add_argument("--ast-sheet", type=Path,
                        default=rds / "processed" / "train_kleb_ast" / "binary_ast_with_split.csv",
                        help="Kp binary_ast_with_split.csv (Sample + lowercase drug columns).")
    parser.add_argument("--out-dir", type=Path, default=here / "docs" / "visualisations",
                        help="Per-drug CSVs go to <out-dir>/kp_<drug>/kleborate_determinant_lr_<drug>.csv.")
    parser.add_argument("--drugs", type=str, nargs="+", default=DEFAULT_DRUGS,
                        help=f"AST drug columns to score (default weak-first: {DEFAULT_DRUGS}).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()
    run(args.metadata, args.ast_sheet, args.out_dir, args.drugs, tuple(args.seeds))


if __name__ == "__main__":
    main()

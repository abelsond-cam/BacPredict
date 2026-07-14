"""Validate the minimap-derived AMR sidecars against Kleborate's metadata_v2 calls + Bakta miss-rate.

Two checks, both the Phase-1 gate before trusting the reliable labels (and before any GPU spend):

1. **Carrier agreement vs Kleborate.** Our acquired-allele carrier sets (from the
   ``{Sample}_amr.parquet`` sidecars) should agree closely with Kleborate's own per-isolate calls
   already in ``metadata_v2`` (both are CARD-derived, so they *should* coincide). Per acquired allele
   we report genomes-with-it on each side and the overlap; aggregated to micro precision / recall over
   all ``(genome, allele)`` pairs. A low number flags an assembly-source or threshold mismatch to
   investigate **before** scaling the array.

2. **Bakta miss-rate.** The reason this whole effort exists: how often does Bakta fail to name an
   acquired AMR gene that CARD finds? We report, over acquired CARD calls: the fraction that landed on
   **no CDS at all** (``flat_index == -1`` — a hard miss, Bakta called no protein there), the fraction
   on a CDS that Bakta left **unnamed** (``bakta_gene_name`` null or equal to the locus tag), and the
   fraction whose Bakta name does **not** contain the CARD gene-family token (a softer "mislabelled"
   proxy). Together these quantify the bias that motivated re-labelling.

Light CPU (login node / small sbatch). Reads the sidecar dir + metadata_v2; writes a per-allele CSV +
a summary JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd

from bacpredict.apps.kleb.kleborate_determinant_lr import COLUMN_SCHEMA, tokenize_cell
from bacpredict.engine.config import final_root, resolve_data_root


def default_sidecar_dir() -> Path:
    """``<data-root>/processed/train_kleb_ast/amr_annotation`` — CARD AMR-call sidecars."""
    return resolve_data_root() / "processed" / "train_kleb_ast" / "amr_annotation"


def default_metadata() -> Path:
    """``<data-root>/final/metadata_v2_all_samples_and_columns.tsv`` — the curated metadata TSV."""
    return final_root() / "metadata_v2_all_samples_and_columns.tsv"

# Kleborate columns that carry *gene/allele* tokens — the grain our CARD calls match. This is the
# acquired-HGT columns PLUS Bla_chr: intrinsic SHV/OKP/LEN comes from the CARD ref (so we tag it
# amr_source="acquired"), but Kleborate files it in the chromosomal-coding Bla_chr column, so it must
# be in the comparison set or those genuine calls read as false positives. The *_mutations columns
# (codon strings) are a different grain and are reported separately, not compared here.
ACQUIRED_COLUMNS = [c for c, (cat, _emb) in COLUMN_SCHEMA.items()
                    if cat in ("acquired_hgt", "chromosomal_coding")]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_NORM = re.compile(r"[^a-z0-9]")


def _norm_allele(tok: str) -> str:
    """Lowercase + strip non-alphanumerics so ``aac(2')-Ia`` and ``aac(2)-ia`` compare equal."""
    return _NORM.sub("", str(tok).lower())


def load_sidecars(sidecar_dir: Path) -> pd.DataFrame:
    """Concatenate every ``{Sample}_amr.parquet`` sidecar under ``sidecar_dir`` (empty frame if none)."""
    files = sorted(sidecar_dir.glob("*_amr.parquet"))
    if not files:
        logger.warning("no sidecars under %s", sidecar_dir)
        return pd.DataFrame()
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    logger.info("loaded %d AMR calls from %d sidecars", len(df), len(files))
    return df


def our_acquired_carriers(calls: pd.DataFrame) -> dict[str, set[str]]:
    """``Sample → {normalised acquired allele}`` from our calls (includes orphan flat_index==-1)."""
    acq = calls[calls["amr_source"] == "acquired"]
    out: dict[str, set[str]] = {}
    for sample, allele in zip(acq["Sample"], acq["amr_allele"], strict=True):
        out.setdefault(str(sample), set()).add(_norm_allele(allele))
    return out


def kleborate_acquired_carriers(metadata: Path, samples: set[str]) -> dict[str, set[str]]:
    """``Sample → {normalised acquired allele}`` from metadata_v2 Kleborate acquired columns."""
    cols = list(ACQUIRED_COLUMNS)
    meta = pd.read_csv(metadata, sep="\t", usecols=["Sample", *cols], low_memory=False)
    meta["Sample"] = meta["Sample"].astype(str)
    meta = meta[meta["Sample"].isin(samples)]
    out: dict[str, set[str]] = {}
    for _, row in meta.iterrows():
        toks: set[str] = set()
        for c in cols:
            toks.update(_norm_allele(t) for t in tokenize_cell(row[c]))
        out[row["Sample"]] = {t for t in toks if t}
    return out


def carrier_agreement(ours: dict[str, set[str]], kleb: dict[str, set[str]]) -> dict:
    """Micro precision/recall over ``(genome, allele)`` pairs + per-genome exact-match rate."""
    shared = sorted(set(ours) & set(kleb))
    tp = fp = fn = 0
    exact = 0
    for s in shared:
        a, b = ours.get(s, set()), kleb.get(s, set())
        tp += len(a & b)
        fp += len(a - b)
        fn += len(b - a)
        if a == b:
            exact += 1
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    return {
        "n_genomes_compared": len(shared),
        "n_genomes_only_ours": len(set(ours) - set(kleb)),
        "n_genomes_only_kleborate": len(set(kleb) - set(ours)),
        "pairs_tp": tp, "pairs_fp_ours_only": fp, "pairs_fn_kleborate_only": fn,
        "allele_precision_vs_kleborate": round(precision, 4),
        "allele_recall_vs_kleborate": round(recall, 4),
        "genome_exact_match_rate": round(exact / len(shared), 4) if shared else float("nan"),
    }


def per_allele_table(ours: dict[str, set[str]], kleb: dict[str, set[str]],
                     calls: pd.DataFrame) -> pd.DataFrame:
    """Per acquired allele: carriers on each side + overlap (display name from the calls)."""
    display = {}
    for allele in calls.loc[calls["amr_source"] == "acquired", "amr_allele"]:
        display.setdefault(_norm_allele(allele), str(allele))
    shared = set(ours) & set(kleb)
    n_ours: dict[str, int] = {}
    n_kleb: dict[str, int] = {}
    n_both: dict[str, int] = {}
    for s in shared:
        a, b = ours.get(s, set()), kleb.get(s, set())
        for x in a:
            n_ours[x] = n_ours.get(x, 0) + 1
        for x in b:
            n_kleb[x] = n_kleb.get(x, 0) + 1
        for x in a & b:
            n_both[x] = n_both.get(x, 0) + 1
    alleles = sorted(set(n_ours) | set(n_kleb))
    rows = [{
        "allele": display.get(x, x),
        "n_ours": n_ours.get(x, 0),
        "n_kleborate": n_kleb.get(x, 0),
        "n_both": n_both.get(x, 0),
    } for x in alleles]
    return pd.DataFrame(rows).sort_values("n_kleborate", ascending=False)


def bakta_miss_report(calls: pd.DataFrame) -> dict:
    """Quantify how often Bakta fails to name an acquired gene that CARD finds."""
    acq = calls[calls["amr_source"] == "acquired"]
    n_total = len(acq)
    on_cds = acq[acq["flat_index"] >= 0]
    n_orphan = int((acq["flat_index"] < 0).sum())

    gene = on_cds["bakta_gene_name"].astype("string")
    locus = on_cds["bakta_locus_tag"].astype("string")
    unnamed = gene.isna() | (gene == locus)
    n_unnamed = int(unnamed.sum())

    def _named_correct(row) -> bool:
        g = row["bakta_gene_name"]
        if not isinstance(g, str) or not g:
            return False
        fam = _norm_allele(row["amr_gene_family"])
        return bool(fam) and (fam in _norm_allele(g) or _norm_allele(g) in _norm_allele(row["amr_allele"]))

    named = on_cds[~unnamed]
    n_named_correct = int(named.apply(_named_correct, axis=1).sum()) if len(named) else 0

    return {
        "n_acquired_calls": int(n_total),
        "n_orphan_no_cds": n_orphan,
        "n_on_cds": int(len(on_cds)),
        "n_on_cds_bakta_unnamed": n_unnamed,
        "n_on_cds_bakta_named": int(len(on_cds) - n_unnamed),
        "n_on_cds_bakta_named_matches_family": n_named_correct,
        "hard_miss_rate_no_cds": round(n_orphan / n_total, 4) if n_total else float("nan"),
        "miss_rate_no_cds_or_unnamed": round((n_orphan + n_unnamed) / n_total, 4) if n_total else float("nan"),
        "family_match_rate_among_calls": round(n_named_correct / n_total, 4) if n_total else float("nan"),
    }


def run(sidecar_dir: Path, metadata: Path, out_dir: Path) -> None:
    """Concatenate sidecars, compute carrier agreement + Bakta miss-rate, write CSV + JSON."""
    calls = load_sidecars(sidecar_dir)
    if calls.empty:
        logger.error("no sidecars to validate")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    ours = our_acquired_carriers(calls)
    kleb = kleborate_acquired_carriers(metadata, set(ours))
    agreement = carrier_agreement(ours, kleb)
    miss = bakta_miss_report(calls)

    chrom = calls[calls["amr_source"] == "chromosomal"]
    summary = {
        "n_samples_with_sidecar": int(calls["Sample"].nunique()),
        "n_calls_total": int(len(calls)),
        "n_acquired_calls": int((calls["amr_source"] == "acquired").sum()),
        "n_chromosomal_calls": int(len(chrom)),
        "chromosomal_genes_located": sorted(chrom["amr_allele"].dropna().unique().tolist()),
        "carrier_agreement_vs_kleborate": agreement,
        "bakta_miss": miss,
    }
    per_allele = per_allele_table(ours, kleb, calls)
    per_allele.to_csv(out_dir / "amr_carrier_agreement_per_allele.csv", index=False)
    (out_dir / "amr_validation_summary.json").write_text(json.dumps(summary, indent=2))

    logger.info("carrier agreement vs Kleborate: precision=%.3f recall=%.3f exact-genome=%.3f over %d genomes",
                agreement["allele_precision_vs_kleborate"], agreement["allele_recall_vs_kleborate"],
                agreement["genome_exact_match_rate"], agreement["n_genomes_compared"])
    logger.info("Bakta miss: %d/%d acquired calls had no CDS (hard miss %.1f%%); no-CDS-or-unnamed %.1f%%",
                miss["n_orphan_no_cds"], miss["n_acquired_calls"],
                100 * miss["hard_miss_rate_no_cds"], 100 * miss["miss_rate_no_cds_or_unnamed"])
    logger.info("wrote %s + amr_carrier_agreement_per_allele.csv", out_dir / "amr_validation_summary.json")


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sidecar-dir", type=Path, default=None,
                   help="CARD AMR-call sidecar dir (default: <data-root>/processed/train_kleb_ast/amr_annotation).")
    p.add_argument("--metadata", type=Path, default=None,
                   help="metadata_v2 TSV (default: <data-root>/final/metadata_v2_all_samples_and_columns.tsv).")
    p.add_argument("--out-dir", type=Path, default=None, help="default: <sidecar-dir>/validation")
    args = p.parse_args()
    sidecar_dir = args.sidecar_dir or default_sidecar_dir()
    metadata = args.metadata or default_metadata()
    out_dir = args.out_dir or sidecar_dir / "validation"
    run(sidecar_dir, metadata, out_dir)


if __name__ == "__main__":
    main()

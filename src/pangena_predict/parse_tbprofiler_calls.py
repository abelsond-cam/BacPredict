"""Parse TB-Profiler results JSONs → WHO-catalogue variant one-hot + native per-drug R/S calls.

TB-Profiler (``run_tbprofiler_cohort.sh``) writes one ``<Sample>.results.json`` per genome. Each
``dr_variants`` entry is a WHO-catalogue resistance variant with a ``gene_name``, a ``change``
(protein for coding genes, nucleotide for non-coding — incl. the inhA/fabG1 promoter and rrs/rrl), a
``type``, and a ``drugs`` list (the drugs it confers resistance to, with confidence). This module
collapses the cohort's JSONs into three tidy artifacts:

- ``tbprofiler_variants.parquet`` — long ``(Sample, variant_id, gene_name, change, type, drugs)``; the
  substrate for the per-drug WHO **one-hot** feature matrix (built downstream, fed to the LR comparator).
- ``tbprofiler_native_calls.parquet`` — wide ``Sample × <drug>`` native R/S call (1 if the genome has
  any catalogue resistance variant for that drug) + ``drtype``; the no-LR comparator vs concat.
- ``tbprofiler_lineage.csv`` — ``Sample, main_lineage, sub_lineage`` (covariate / confound check).

Crucially the one-hot captures the **non-embeddable** mechanisms (rrs/rrl, inhA promoter) that a
protein-only embedding can't, so ``one-hot − concat`` AUROC per drug measures that gap. Pure pandas/json
(``uv run python``) — no tb-profiler needed to parse. Login-node friendly.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# TB-Profiler drug names → our AST column names (binary_ast_with_split.csv). Only the genuine differences;
# everything else normalises by lower-casing + underscore→space.
DRUG_NAME_MAP = {
    "rifampicin": "rifampin",
    "para-aminosalicylic acid": "para-aminosalicylic acid",
}


def _norm_drug(drug: str) -> str:
    """Normalise a TB-Profiler drug name to our AST column convention (rifampicin→rifampin etc.)."""
    d = drug.strip().lower().replace("_", " ")
    return DRUG_NAME_MAP.get(d, d)


def _variant_id(v: dict) -> str:
    """Stable variant identity ``gene@change`` (protein change for CDS, nucleotide for non-coding)."""
    change = v.get("change") or v.get("protein_change") or v.get("nucleotide_change") or "?"
    return f"{v.get('gene_name', '?')}@{change}"


def _drugs_of(v: dict) -> set[str]:
    """Set of (normalised) drugs a dr_variant confers resistance to."""
    out: set[str] = set()
    for a in v.get("drugs") or []:
        if a.get("drug"):
            out.add(_norm_drug(a["drug"]))
    return out


def parse_one(path: str) -> tuple[list[dict], set[str], dict]:
    """Parse one results JSON → (variant rows, resistant-drug set, lineage/drtype meta)."""
    d = json.loads(Path(path).read_text())
    sample = d.get("id") or Path(path).name.split(".results.json")[0]
    rows, resistant = [], set()
    for v in d.get("dr_variants", []):
        vdrugs = _drugs_of(v)
        resistant |= vdrugs
        rows.append({
            "Sample": sample, "variant_id": _variant_id(v), "gene_name": v.get("gene_name"),
            "change": v.get("change"), "type": v.get("type"), "drugs": ";".join(sorted(vdrugs)),
        })
    meta = {"Sample": sample, "drtype": d.get("drtype"),
            "main_lineage": d.get("main_lineage"), "sub_lineage": d.get("sub_lineage")}
    return rows, resistant, meta


def run(results_dir: Path, out_dir: Path, drugs: list[str]) -> dict:
    """Parse every ``*.results.json`` under ``results_dir`` → the three artifacts in ``out_dir``."""
    files = sorted(glob.glob(str(results_dir / "*.results.json")))
    if not files:
        raise RuntimeError(f"No *.results.json under {results_dir}")
    logger.info("Parsing %d TB-Profiler result JSONs", len(files))

    all_rows, native, meta = [], [], []
    for i, f in enumerate(files):
        try:
            rows, resistant, m = parse_one(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("skip %s: %s", Path(f).name, exc)
            continue
        all_rows.extend(rows)
        native.append({"Sample": m["Sample"], "drtype": m["drtype"],
                       **{drug: (1 if drug in resistant else 0) for drug in drugs}})
        meta.append(m)
        if (i + 1) % 5000 == 0:
            logger.info("  parsed %d/%d", i + 1, len(files))

    out_dir.mkdir(parents=True, exist_ok=True)
    variants_df = pd.DataFrame(all_rows)
    variants_df.to_parquet(out_dir / "tbprofiler_variants.parquet", index=False)
    native_df = pd.DataFrame(native)
    native_df.to_parquet(out_dir / "tbprofiler_native_calls.parquet", index=False)
    pd.DataFrame(meta).to_csv(out_dir / "tbprofiler_lineage.csv", index=False)

    summary = {
        "n_genomes": len(native), "n_variant_rows": len(variants_df),
        "n_distinct_variants": int(variants_df["variant_id"].nunique()) if not variants_df.empty else 0,
        "per_drug_resistant": {drug: int(native_df[drug].sum()) for drug in drugs} if native else {},
    }
    (out_dir / "tbprofiler_parse_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Wrote variants/native/lineage to %s | %s", out_dir, json.dumps(summary["per_drug_resistant"]))
    return summary


def main() -> None:
    """CLI entry point."""
    default_drugs = ["rifampin", "isoniazid", "ethambutol", "pyrazinamide", "moxifloxacin",
                     "levofloxacin", "streptomycin", "ethionamide", "rifabutin", "kanamycin"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, required=True, help="Dir of <Sample>.results.json.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Where to write the parsed artifacts.")
    parser.add_argument("--drugs", type=str, nargs="+", default=default_drugs, help="AST drug columns to call.")
    args = parser.parse_args()
    run(args.results_dir, args.out_dir, args.drugs)


if __name__ == "__main__":
    main()

r"""Turn the lab-collection spreadsheet into one keyed manifest for invasion scoring.

The collection is a set of *physical* isolates a collaborator can test in an animal (Galleria)
model, supplied as an xlsx with two tabs: ``Files`` (identity + assembly/GFF paths) and
``Kleborate`` (the full Kleborate v3 output). This module joins them, attaches everything the
downstream scorers need, and — crucially — records what is *missing* rather than quietly dropping it.

Three joins matter and none of them are cosmetic:

**Sublineage.** The spreadsheet carries ``ST`` but not ``Sublineage``. They are not the same thing
and there is no ST→SL mapping in this repo (Pasteur LIN-typing produces ``Sublineage``, and the
mash-distance fallback is documented as collapsing on KpSC). So it is joined from ``metadata_v2``,
which covers ~656/677. The rest are labelled ``unknown`` and stay in the table.

**Split provenance.** Many of these genomes are already in the fine-tuning cohorts, and the model
memorises hard (train AUROC 0.959 vs 0.786 held out). A score for a genome the model trained on is
not a prediction. Both cohorts' splits are recorded — ``pooled_split`` and ``all_samples_split`` —
because which one counts depends on which model is finally used, and rebuilding the manifest after
that decision would be wasted work.

**True label.** Where a genome is in a cohort, its real blood/faeces label is known. Keeping score
and truth side by side is what lets the lab check that Galleria death actually tracks the
blood-vs-faeces phenotype before trusting predictions on the genomes that have no label.

Usage
-----
    python -m kleb_iso_source.build_lab_collection_manifest \
        --xlsx      <data>/processed/train_iso_source/lab_collection/Kleborate_labcollection.xlsx \
        --out-dir   <data>/processed/train_iso_source/lab_collection
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from bac_kleborate.parsing import KLEBORATE_VIRULENCE_LOCI, acquired_column_names

logger = logging.getLogger(__name__)

ID_COL = "sample_accession"
SAMPLE_COL = "Sample"
LABEL_COL = "blood_vs_faeces_label"
UNKNOWN_SL = "unknown"

# Identity columns carried through from the Files/Kleborate tabs for the lab to act on.
IDENTITY_COLS = ["strain", "strain_v2", "LabID", "Study", "species"]
# Kleborate summary columns the comparator recipes consume, plus the ones a human wants to eyeball.
KLEBORATE_SUMMARY_COLS = [
    "ST", "virulence_score", "resistance_score", "num_resistance_classes", "num_resistance_genes",
    "K_locus", "O_locus", "Yersiniabactin", "Colibactin", "Aerobactin", "Salmochelin", "RmpADC",
    "rmpA2", "contig_count", "N50", "QC_warnings",
]
VIRULENCE_ALLELE_COLS = sorted(
    {allele for info in KLEBORATE_VIRULENCE_LOCI.values() for allele in info["alleles"]}
)

# metadata_v2 is a wide TSV; only these are read.
METADATA_COLS = ["Sample", "Sublineage", "Clonal group", "country_parsed"]

COHORTS = {
    "pooled": "sampled_country_2_1_all",
    "all_samples": "all_samples",
}


JOIN_KEYS = [ID_COL, "strain"]


def load_spreadsheet(xlsx: Path) -> pd.DataFrame:
    """Join the ``Files`` and ``Kleborate`` tabs, one row per isolate.

    Joined on ``(sample_accession, strain)``, not on the accession alone: three accessions appear
    twice (the same deposit sequenced/assembled twice), so an accession-only merge fans 680 rows out
    to 684 and can pair a Files row with the *other* assembly's Kleborate output. ``strain`` is
    unique in both tabs, so the pair is a genuine one-to-one key — asserted by ``validate``, which
    turns any future duplication into an exception instead of silent row inflation.
    """
    files = pd.read_excel(xlsx, sheet_name="Files")
    kleb = pd.read_excel(xlsx, sheet_name="Kleborate")
    for name, df in (("Files", files), ("Kleborate", kleb)):
        missing = [c for c in JOIN_KEYS if c not in df.columns]
        if missing:
            raise ValueError(f"{xlsx} tab {name!r} is missing join column(s) {missing}")

    feature_cols = [c for c in (KLEBORATE_SUMMARY_COLS + VIRULENCE_ALLELE_COLS) if c in kleb.columns]
    acquired = acquired_column_names(list(kleb.columns))
    keep = [*JOIN_KEYS, *feature_cols, *acquired]
    kleb_sub = kleb[list(dict.fromkeys(keep))].copy()  # de-duplicate, preserving order

    merged = files.merge(kleb_sub, on=JOIN_KEYS, how="left", validate="one_to_one")
    if len(merged) != len(files):
        raise ValueError(f"join changed the row count: {len(files)} -> {len(merged)}")
    logger.info("spreadsheet: %d Files rows x %d Kleborate cols -> %d merged rows (%d acquired-AMR cols)",
                len(files), kleb.shape[1], len(merged), len(acquired))
    return merged


def resolve_paths(df: pd.DataFrame, project_k_root: Path) -> pd.DataFrame:
    """Prefix the spreadsheet's relative assembly/GFF paths with the project-k root."""
    out = df.copy()
    for src, dest in (("assembly_file", "assembly_path"), ("gff_file", "gff_path")):
        if src not in out.columns:
            out[dest] = pd.NA
            continue
        out[dest] = out[src].apply(lambda p: str(project_k_root / str(p)) if pd.notna(p) else pd.NA)
    return out


def attach_sublineage(df: pd.DataFrame, metadata_tsv: Path, *, min_coverage: float = 0.5) -> pd.DataFrame:
    """Join ``Sublineage`` / ``Clonal group`` / ``country_parsed`` from metadata_v2.

    Guarded by ``min_coverage`` in the spirit of ``sublineage_from_metadata.run``: a join that
    silently matches almost nothing would otherwise surface as every genome landing in ``unknown``,
    which looks like a biology result rather than a broken key.
    """
    usecols = METADATA_COLS
    meta = pd.read_csv(metadata_tsv, sep="\t", usecols=lambda c: c in usecols, dtype=str, low_memory=False)
    meta = meta.drop_duplicates(subset=[SAMPLE_COL])
    out = df.merge(meta, left_on=ID_COL, right_on=SAMPLE_COL, how="left")

    matched = out[SAMPLE_COL].notna().sum()
    coverage = matched / len(out) if len(out) else 0.0
    logger.info("metadata_v2 join: %d/%d rows matched (%.1f%%)", matched, len(out), 100 * coverage)
    if coverage < min_coverage:
        raise SystemExit(
            f"metadata_v2 join matched only {coverage:.1%} of rows (< {min_coverage:.0%}). "
            f"Check that {ID_COL} values are BioSample accessions matching metadata_v2's 'Sample'."
        )

    for col in ("Sublineage", "Clonal group"):
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = out[col].fillna(UNKNOWN_SL).replace({"": UNKNOWN_SL, "nan": UNKNOWN_SL})
    return out.drop(columns=[SAMPLE_COL])


def attach_cohort_splits(df: pd.DataFrame, train_root: Path, pair: str = "blood_faeces",
                         flavor: str = "kpsc_human") -> pd.DataFrame:
    """Add ``<cohort>_split`` for every cohort, plus a single ``true_label``.

    A genome's blood/faeces label is a property of the isolate, not of the cohort, so the label is
    taken from whichever cohort knows it and asserted consistent where both do.
    """
    out = df.copy()
    labels: dict[str, float] = {}
    for tag, cohort in COHORTS.items():
        csv_path = train_root / pair / cohort / flavor / "binary_blood_vs_faeces_with_split.csv"
        col = f"{tag}_split"
        if not csv_path.is_file():
            logger.warning("cohort split CSV missing, %s left empty: %s", col, csv_path)
            out[col] = "unseen"
            continue
        split = pd.read_csv(csv_path, usecols=[SAMPLE_COL, LABEL_COL, "train_val_eval"], low_memory=False)
        split = split[split[LABEL_COL].isin([0, 1])]
        out[col] = out[ID_COL].map(dict(zip(split[SAMPLE_COL], split["train_val_eval"], strict=True)))
        out[col] = out[col].fillna("unseen")
        for sample, lab in zip(split[SAMPLE_COL], split[LABEL_COL], strict=True):
            prev = labels.get(sample)
            if prev is not None and prev != lab:
                raise ValueError(f"{sample} has conflicting labels across cohorts: {prev} vs {lab}")
            labels[sample] = lab
        logger.info("%s: %d/%d lab genomes in cohort %s", col, (out[col] != "unseen").sum(), len(out), cohort)

    out["true_label"] = out[ID_COL].map(labels)
    return out


def flag_availability(df: pd.DataFrame, embeddings_dir: Path, *, check_assemblies: bool) -> pd.DataFrame:
    """Add ``has_embedding`` / ``has_assembly``, the two things that gate whether a genome scores."""
    out = df.copy()
    out["has_embedding"] = out[ID_COL].apply(
        lambda s: (embeddings_dir / f"{s}_esm_embeddings.pt").is_file() if pd.notna(s) else False
    )
    if check_assemblies:
        out["has_assembly"] = out["assembly_path"].apply(
            lambda p: Path(p).is_file() if pd.notna(p) else False
        )
    else:
        out["has_assembly"] = out["assembly_path"].notna()
    logger.info("availability: %d/%d have an ESM embedding, %d/%d have an assembly",
                out["has_embedding"].sum(), len(out), out["has_assembly"].sum(), len(out))
    return out


def resolve_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse duplicate accessions to one scoring row, keeping every original row as a record.

    The same ``sample_accession`` can appear twice with different ``strain``/``LabID`` (e.g. a v1 and
    v2 assembly of one deposit). The genome is the same, so scoring it twice would double-count it in
    the ranking; but the lab needs both rows to know which tube is which. Keep the most complete row
    and flag both.
    """
    out = df.copy()
    dup_mask = out[ID_COL].duplicated(keep=False) & out[ID_COL].notna()
    out["duplicate_accession"] = dup_mask

    completeness = out["has_embedding"].astype(int) + out["has_assembly"].astype(int)
    out = out.assign(_completeness=completeness)
    out = out.sort_values([ID_COL, "_completeness"], ascending=[True, False], kind="stable")
    out["is_scoring_row"] = ~out[ID_COL].duplicated(keep="first") & out[ID_COL].notna()
    out = out.drop(columns=["_completeness"])

    dropped = out[out["duplicate_accession"] & ~out["is_scoring_row"]]
    return out, dropped


def build_exclusions(df: pd.DataFrame) -> pd.DataFrame:
    """One row per genome that will not be fully scored, with the reason. Never silent."""
    reasons = []
    for row in df.itertuples():
        why = []
        if pd.isna(getattr(row, ID_COL)):
            why.append("no sample_accession")
        if not row.has_embedding:
            why.append("no ESM embedding (Bacformer score unavailable)")
        if not row.has_assembly:
            why.append("no assembly (unitig presence unavailable)")
        if getattr(row, "Sublineage", UNKNOWN_SL) == UNKNOWN_SL:
            why.append("no Sublineage in metadata_v2")
        if row.duplicate_accession and not row.is_scoring_row:
            why.append("duplicate accession, not the scoring row")
        if why:
            reasons.append({
                ID_COL: getattr(row, ID_COL), "strain": getattr(row, "strain", None),
                "LabID": getattr(row, "LabID", None), "reason": "; ".join(why),
            })
    return pd.DataFrame(reasons)


def build_manifest(xlsx: Path, metadata_tsv: Path, train_root: Path, embeddings_dir: Path,
                   project_k_root: Path, *, check_assemblies: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full pipeline: spreadsheet -> joined, flagged manifest + its exclusions table."""
    df = load_spreadsheet(xlsx)
    df = resolve_paths(df, project_k_root)
    df = attach_sublineage(df, metadata_tsv)
    df = attach_cohort_splits(df, train_root)
    df = flag_availability(df, embeddings_dir, check_assemblies=check_assemblies)
    df, _ = resolve_duplicates(df)
    exclusions = build_exclusions(df)
    return df, exclusions


def _main_cli() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xlsx", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--metadata", type=Path,
                   default=Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/final/"
                                "metadata_v2_all_samples_and_columns.tsv"))
    p.add_argument("--train-root", type=Path,
                   default=Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/"
                                "train_iso_source"))
    p.add_argument("--embeddings-dir", type=Path,
                   default=Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/"
                                "klebsiella_esm_embeddings"))
    p.add_argument("--project-k-root", type=Path,
                   default=Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw"))
    p.add_argument("--no-check-assemblies", action="store_true",
                   help="Skip stat-ing each assembly path (use when the store is not mounted).")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    manifest, exclusions = build_manifest(
        args.xlsx, args.metadata, args.train_root, args.embeddings_dir, args.project_k_root,
        check_assemblies=not args.no_check_assemblies,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "lab_collection_manifest.csv"
    excl_path = args.out_dir / "lab_collection_exclusions.csv"
    manifest.to_csv(manifest_path, index=False)
    exclusions.to_csv(excl_path, index=False)

    scoring = manifest[manifest["is_scoring_row"]]
    summary = {
        "n_rows": int(len(manifest)),
        "n_scoring_rows": int(len(scoring)),
        "n_with_embedding": int(scoring["has_embedding"].sum()),
        "n_with_assembly": int(scoring["has_assembly"].sum()),
        "n_with_sublineage": int((scoring["Sublineage"] != UNKNOWN_SL).sum()),
        "n_excluded_rows": int(len(exclusions)),
        "split_counts": {t: scoring[f"{t}_split"].value_counts().to_dict() for t in COHORTS},
        "n_with_true_label": int(scoring["true_label"].notna().sum()),
        "top_sublineages": scoring["Sublineage"].value_counts().head(12).to_dict(),
    }
    (args.out_dir / "lab_collection_manifest_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {manifest_path}\nWrote {excl_path}")


if __name__ == "__main__":
    _main_cli()

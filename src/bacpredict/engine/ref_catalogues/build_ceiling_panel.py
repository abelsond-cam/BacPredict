"""Build the per-organism catalogue-ceiling panel from the per-drug catalogue CSVs.

The panel (``visualisations/<organism>/catalogue_ceiling_panel.csv``) is what every downstream
consumer reads for "what can a known-determinant model do on this drug" — the ladder's red rung and
the ``ast_gwas`` comparison table. It exists because the thing they used to read,
``amr_summary_panel.csv``, also carried a fine-tune column that went stale and was quoted for weeks
after the runs behind it were redone.

**This module exists because the first version of that panel was assembled by hand.** A shared
artifact with no producer cannot be regenerated, cannot be diffed against its sources, and quietly
invites the one thing the TB rebuild must not do — copying numbers from the deprecated tree into the
canonical path, which launders them. Rebuilding the panel is now a command.

Two catalogue schemas feed it and they are **not** interchangeable:

===================  ==========================  ==================================
                     Kp / CARD                   TB / WHO + TB-Profiler
===================  ==========================  ==================================
all-determinant row  ``__ALL_CARD__``            ``__ALL_WHO_one_hot__``
determinant count    ``n_determinants``          ``n_variants``
carrier count        ``n_genomes_with_determinant``  ``n_genomes_with_variant``
extra column         ``is_causal``               (absent); has ``region``
===================  ==========================  ==================================

**On ``ceiling_estimator``.** It is a required argument, not an inference, because it is the field
that decides whether a ceiling may be compared to a fine-tune at all — and getting it wrong is
invisible. ``deployment_holdout`` means the determinant LR was fit on ``train`` and scored on the
``holdout`` from the same ``<drug>_split.csv`` the fine-tune used, so the two are commensurable.
``kfold_probe`` is the retired whole-cohort probe: a different estimator on a different evaluation
set, whose ceiling-vs-FT gap is not readable in either direction.

The caller declares it and this module **checks the declaration against the data**: the
deployment-holdout scorer fits once, so its ``mut_auroc_sd`` is exactly ``0.0``
(:func:`bacpredict.engine.ref_catalogues.base.score_onehot_frame` returns ``sd=0.0`` for precisely
this reason). A non-zero spread on a row claiming ``deployment_holdout`` means one of the two is
wrong, and that is worth stopping for. The converse is *not* checked: a k-fold probe can legitimately
report ``0.0`` where a determinant scores identically across folds, so zero spread proves nothing on
its own — which is why the label is declared rather than sniffed.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PANEL_COLUMNS = (
    "drug", "ceiling_auroc", "ceiling_auprc", "ceiling_auroc_sd",
    "n_determinants", "n_carriers",
    "ceiling_catalogue", "ceiling_grain", "ceiling_estimator", "ceiling_status",
)

ESTIMATORS = ("deployment_holdout", "kfold_probe")


@dataclass(frozen=True)
class CatalogueSchema:
    """How one catalogue's per-drug CSV names the things the panel needs."""

    all_row: str
    n_determinants: str
    n_carriers: str
    catalogue: str


SCHEMAS = {
    "card": CatalogueSchema(
        all_row="__ALL_CARD__",
        n_determinants="n_determinants",
        n_carriers="n_genomes_with_determinant",
        catalogue="CARD",
    ),
    "who": CatalogueSchema(
        all_row="__ALL_WHO_one_hot__",
        n_determinants="n_variants",
        n_carriers="n_genomes_with_variant",
        catalogue="WHO_tbprofiler",
    ),
}


def read_ceiling_row(csv: Path, schema: CatalogueSchema) -> dict | None:
    """Pull the all-determinant ceiling row out of one per-drug catalogue CSV.

    Returns ``None`` (with a warning) rather than raising when the row is absent: a drug whose
    catalogue model was never built is a real, expected state — TB is missing rifabutin — and it must
    surface as an absent row, not as a zero or a crash.
    """
    frame = pd.read_csv(csv)
    hit = frame.loc[frame["gene_name"] == schema.all_row]
    if hit.empty:
        logger.warning("%s has no %s row — skipping", csv.name, schema.all_row)
        return None
    row = hit.iloc[0]
    return {
        "ceiling_auroc": float(row["mut_auroc"]),
        "ceiling_auprc": float(row["mut_auprc"]),
        "ceiling_auroc_sd": float(row["mut_auroc_sd"]),
        "n_determinants": int(row[schema.n_determinants]),
        "n_carriers": int(row[schema.n_carriers]),
    }


def build_panel(
    per_drug: dict[str, Path],
    *,
    schema_key: str,
    grain: str,
    estimator: str,
    status: str,
) -> pd.DataFrame:
    """Collate per-drug catalogue CSVs into the panel, verifying the declared estimator.

    Raises
    ------
    ValueError
        If ``estimator`` is not one of :data:`ESTIMATORS`, or if a row declares
        ``deployment_holdout`` while reporting a non-zero AUROC spread — the scorer fits once, so a
        spread means the declaration and the data disagree.
    """
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {ESTIMATORS}, got {estimator!r}")
    schema = SCHEMAS[schema_key]

    rows = []
    for drug in sorted(per_drug):
        values = read_ceiling_row(per_drug[drug], schema)
        if values is None:
            continue
        if estimator == "deployment_holdout" and values["ceiling_auroc_sd"] != 0.0:
            raise ValueError(
                f"{drug}: declared estimator 'deployment_holdout' fits once and cannot have a "
                f"spread, but mut_auroc_sd is {values['ceiling_auroc_sd']}. Either the CSV came "
                "from the k-fold probe, or the scorer changed. Do not relabel to make this pass."
            )
        rows.append({
            "drug": drug, **values,
            "ceiling_catalogue": schema.catalogue, "ceiling_grain": grain,
            "ceiling_estimator": estimator, "ceiling_status": status,
        })

    if not rows:
        raise ValueError("no ceiling rows found — check the input paths and the catalogue schema")
    return pd.DataFrame(rows, columns=list(PANEL_COLUMNS))


def discover_card(ceiling_dir: Path, grain: str) -> dict[str, Path]:
    """Map drug -> CSV for the Kp CARD layout (``<dir>/<drug>/card_determinant_lr_<drug>_<grain>.csv``)."""
    found = {}
    for sub in sorted(p for p in ceiling_dir.iterdir() if p.is_dir()):
        csv = sub / f"card_determinant_lr_{sub.name}_{grain}.csv"
        if csv.is_file():
            found[sub.name] = csv
    return found


def discover_who(ceiling_dir: Path) -> dict[str, Path]:
    """Map drug -> CSV for the TB layout (``<dir>/tbprofiler_gene_lr_<drug>.csv``)."""
    return {
        p.stem.removeprefix("tbprofiler_gene_lr_"): p
        for p in sorted(ceiling_dir.glob("tbprofiler_gene_lr_*.csv"))
        if p.stem != "tbprofiler_gene_lr_manifest"
    }


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ceiling-dir", type=Path, required=True,
                   help="Directory of per-drug catalogue CSVs (card_ceiling/ or tbprofiler_gene_lr/).")
    p.add_argument("--out-csv", type=Path, required=True,
                   help="visualisations/<organism>/catalogue_ceiling_panel.csv")
    p.add_argument("--catalogue", choices=sorted(SCHEMAS), required=True)
    p.add_argument("--grain", default="allele",
                   help="CARD grain ('allele' or 'family'); ignored for WHO, which is one_hot.")
    p.add_argument("--estimator", choices=ESTIMATORS, required=True,
                   help="How the ceiling was scored. Declared, then checked against the data — see "
                        "the module docstring. Getting this wrong is invisible and makes the "
                        "ceiling-vs-fine-tune comparison unreadable.")
    p.add_argument("--status", choices=("current", "provisional"), required=True)
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.catalogue == "card":
        per_drug = discover_card(args.ceiling_dir, args.grain)
        grain = args.grain
    else:
        per_drug = discover_who(args.ceiling_dir)
        grain = "one_hot"

    panel = build_panel(
        per_drug, schema_key=args.catalogue, grain=grain,
        estimator=args.estimator, status=args.status,
    )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    # %.17g round-trips float64 exactly, so re-running this command reproduces the file byte for
    # byte. Without it the last significant figure drifts on the write and the artifact is only
    # nearly reproducible — which is not reproducible, and is precisely the kind of "close enough"
    # that let a hand-assembled panel sit unnoticed in the first place.
    panel.to_csv(args.out_csv, index=False, float_format="%.17g")
    logger.info("wrote %s (%d drugs, %s/%s)", args.out_csv, len(panel), args.estimator, args.status)
    if args.status == "provisional":
        logger.warning(
            "PROVISIONAL: these rows are not like-for-like with a fine-tune number. See "
            "visualisations/PROVENANCE.md before quoting any gap."
        )


if __name__ == "__main__":
    main()

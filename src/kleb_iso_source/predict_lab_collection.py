r"""Score the lab collection for invasion probability, with the comparator models alongside.

Produces the table a collaborator uses to pick isolates for animal-model (Galleria) testing: one row
per genome, ranked by Bacformer's P(blood | genome), with the Kleborate and unitig comparators in
adjacent columns and — where the genome happens to be in a fine-tuning cohort — its real
blood/faeces label.

Three stages, because they want different machines:

``bacformer``   GPU. Runs the fine-tuned head over the ESM embeddings. Both cohorts' models are
                scored, since which is "the" model is a separate decision and re-queuing a GPU job
                to get the other column is wasteful.
``kleborate``   CPU. Fits the annotation baselines on the cohort's train split and predicts on the
                lab frame. These are known-weak predictors, carried so that when animal results
                arrive it is possible to say what Bacformer got right that annotation did not.
``assemble``    CPU. Merges the stage outputs (plus unitig probabilities, if computed) into the
                ranked deliverable, the per-sublineage summary, and the backup AUROC.

**What the probabilities are not.** The model is calibrated to its cohort's ~52% blood prevalence,
which is not the probability of invasion given gut carriage in a patient. The ranking is the usable
product; the absolute number is not a clinical risk.

Usage
-----
    python -m kleb_iso_source.predict_lab_collection bacformer --manifest … --out …
    python -m kleb_iso_source.predict_lab_collection kleborate --manifest … --out …
    python -m kleb_iso_source.predict_lab_collection assemble  --manifest … --bacformer … [--unitig …]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ID_COL = "sample_accession"
LABEL_COL = "blood_vs_faeces_label"
UNKNOWN_SL = "unknown"

# Cohort tag -> directory name under processed/train_iso_source/blood_faeces/.
COHORTS = {"pooled": "sampled_country_2_1_all", "all_samples": "all_samples"}

# The three comparator recipes asked for, plus the richest Kleborate-only stack for free.
KLEBORATE_RECIPES = {
    "virulence_score": ["virulence_score"],
    "virulence_bsc": ["virulence_bsc"],
    "virulence_amr": ["virulence_bsc", "amr_class"],
    "kleborate_all": ["virulence_score", "virulence_bsc", "amr_score", "amr_class"],
}


def _logit(p: np.ndarray) -> np.ndarray:
    """Log-odds, clipped off the asymptotes so a saturated probability stays plottable."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load_manifest(path: Path) -> pd.DataFrame:
    """Read the manifest written by :mod:`kleb_iso_source.build_lab_collection_manifest`."""
    df = pd.read_csv(path, low_memory=False)
    for col in (ID_COL, "is_scoring_row", "has_embedding"):
        if col not in df.columns:
            raise ValueError(f"{path} is missing required column {col!r}")
    return df


# --------------------------------------------------------------------------- bacformer


def _cmd_bacformer(args: argparse.Namespace) -> None:
    import torch

    from bacpredict.engine.finetune.predict import predict_proba

    manifest = load_manifest(args.manifest)
    scoring = manifest[manifest["is_scoring_row"] & manifest["has_embedding"]]
    sample_ids = scoring[ID_COL].astype(str).tolist()
    logger.info("scoring %d lab genomes (%d rows have no embedding and are skipped)",
                len(sample_ids), int((manifest["is_scoring_row"] & ~manifest["has_embedding"]).sum()))

    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    out = pd.DataFrame({ID_COL: sample_ids})
    for tag, cohort in COHORTS.items():
        ckpt = args.train_root / "blood_faeces" / cohort / args.flavor / args.models_dir
        if not ckpt.exists():
            logger.warning("no checkpoint for %s at %s — column skipped", tag, ckpt)
            continue
        logger.info("scoring with the %s model: %s", tag, ckpt)
        probs = predict_proba(ckpt, sample_ids, args.embeddings_dir, device=device,
                              batch_size=args.batch_size, num_workers=args.num_workers)
        out[f"bacformer_{tag}_prob"] = probs
        out[f"bacformer_{tag}_logit"] = _logit(probs)
        logger.info("  %s: mean %.4f, min %.4f, max %.4f", tag, probs.mean(), probs.min(), probs.max())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(out)} genomes, {out.shape[1] - 1} score columns)")


# --------------------------------------------------------------------------- kleborate


def _cmd_kleborate(args: argparse.Namespace) -> None:
    from bacpredict.engine.finetune.linear_baselines import fit_and_predict_new

    manifest = load_manifest(args.manifest)
    lab = manifest[manifest["is_scoring_row"]].reset_index(drop=True)

    cohort_csv = args.train_root / "blood_faeces" / COHORTS[args.cohort] / args.flavor / \
        "binary_blood_vs_faeces_with_split.csv"
    cohort = pd.read_csv(cohort_csv, low_memory=False)
    cohort = cohort[cohort[LABEL_COL].isin([0, 1])].reset_index(drop=True)

    # The cohort split CSV carries the label and split but not every Kleborate column, so the
    # feature columns are joined in from metadata_v2 — the same source the fitted baselines used.
    logger.info("fitting recipes %s on cohort %s (%d labelled genomes)",
                sorted(KLEBORATE_RECIPES), args.cohort, len(cohort))
    meta_cols = None
    if args.metadata is not None:
        header = pd.read_csv(args.metadata, sep="\t", nrows=0)
        lab_cols = set(lab.columns)
        meta_cols = ["Sample"] + [c for c in header.columns if c not in lab_cols and c != "Sample"]
        meta = pd.read_csv(args.metadata, sep="\t", usecols=meta_cols, dtype=str, low_memory=False)
        meta = meta.drop_duplicates("Sample")
        cohort = cohort.merge(meta, on="Sample", how="left")
        logger.info("joined %d metadata columns onto the cohort frame", len(meta_cols) - 1)

    train_mask = (cohort["train_val_eval"] == "train").to_numpy()
    out = pd.DataFrame({ID_COL: lab[ID_COL]})
    for name, blocks in KLEBORATE_RECIPES.items():
        try:
            probs, n_feat = fit_and_predict_new(cohort, lab, blocks, LABEL_COL, train_mask)
        except (ValueError, KeyError) as exc:  # a block whose columns the cohort frame lacks
            logger.warning("recipe %s skipped: %s", name, exc)
            continue
        out[f"{name}_prob"] = probs
        logger.info("  %-16s %d features, mean prob %.4f", name, n_feat, probs.mean())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(out)} genomes, {out.shape[1] - 1} comparator columns)")


# --------------------------------------------------------------------------- assemble


def per_sublineage_summary(df: pd.DataFrame, prob_col: str, min_n: int = 10) -> pd.DataFrame:
    """Spread of predicted invasiveness within each sublineage.

    The within-clone spread is the point of the exercise: picking a high- and a low-scoring isolate
    from the *same* lineage is what makes an animal experiment a test of the model rather than a test
    of lineage. SD and IQR are both reported because a bimodal clone and a diffuse one can share an SD.
    """
    rows = []
    for sl, g in df.groupby("Sublineage"):
        if len(g) < min_n:
            continue
        p = g[prob_col].dropna()
        if p.empty:
            continue
        top = g.nlargest(3, prob_col)[["strain", "LabID", prob_col]].to_dict("records")
        bottom = g.nsmallest(3, prob_col)[["strain", "LabID", prob_col]].to_dict("records")
        rows.append({
            "Sublineage": sl, "n": len(g), "mean": p.mean(), "median": p.median(),
            "sd": p.std(ddof=1), "iqr": p.quantile(0.75) - p.quantile(0.25),
            "min": p.min(), "max": p.max(),
            "top3": json.dumps(top, default=float), "bottom3": json.dumps(bottom, default=float),
        })
    out = pd.DataFrame(rows)
    return out.sort_values("n", ascending=False).reset_index(drop=True) if len(out) else out


def backup_auroc(df: pd.DataFrame, prob_col: str, split_col: str) -> pd.DataFrame:
    """AUROC on the lab genomes that have a real label, overall and by split provenance.

    Reported per provenance because the overall figure mixes genomes the model was fitted on with
    genomes it never saw. The train rows are recall, not prediction; ``evaluate`` is the quotable
    number and the table says so rather than leaving a reader to work it out.
    """
    from sklearn.metrics import roc_auc_score

    from bacpredict.engine.finetune.stratified_metrics import bootstrap_auroc_ci

    labelled = df[df["true_label"].notna() & df[prob_col].notna()]
    rows = []
    groups = [("all_labelled", labelled)] + [
        (name, labelled[labelled[split_col] == name]) for name in ("train", "validate", "evaluate")
    ]
    for name, g in groups:
        if len(g) < 2 or g["true_label"].nunique() < 2:
            rows.append({"subset": name, "n": len(g), "auroc": np.nan, "ci_lo": np.nan,
                         "ci_hi": np.nan, "quotable": False, "note": "too few / single-class"})
            continue
        y = g["true_label"].astype(int).to_numpy()
        p = g[prob_col].to_numpy()
        lo, hi, _ = bootstrap_auroc_ci(y, p)
        rows.append({
            "subset": name, "n": len(g), "n_pos": int(y.sum()),
            "auroc": float(roc_auc_score(y, p)), "ci_lo": lo, "ci_hi": hi,
            "quotable": name == "evaluate",
            "note": {"train": "FITTED ON — recall, not a measurement",
                     "all_labelled": "mixes fitted-on and held-out genomes; inflated",
                     "validate": "model selection used this split; mildly optimistic",
                     "evaluate": "held out — the quotable figure"}[name],
        })
    return pd.DataFrame(rows)


def _cmd_assemble(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    df = manifest[manifest["is_scoring_row"]].copy()

    for path in [args.bacformer, args.kleborate, args.unitig]:
        if path is None:
            continue
        if not Path(path).is_file():
            logger.warning("input missing, its columns will be absent: %s", path)
            continue
        part = pd.read_csv(path)
        df = df.merge(part, on=ID_COL, how="left", validate="one_to_one")

    prob_col = f"bacformer_{args.model}_prob"
    if prob_col not in df.columns:
        raise SystemExit(f"{prob_col} not present — run the bacformer stage first")
    split_col = f"{args.model}_split"

    df = df.sort_values(prob_col, ascending=False, na_position="last").reset_index(drop=True)
    df["bacformer_rank"] = np.arange(1, len(df) + 1)

    front = [ID_COL, "strain", "LabID", "Study", "species", "Sublineage", "Clonal group", "ST",
             "bacformer_rank", prob_col]
    score_cols = [c for c in df.columns if c.endswith(("_prob", "_logit")) and c not in front]
    tail = ["virulence_score", split_col, "true_label", "has_embedding", "has_assembly",
            "duplicate_accession"]
    ordered = [c for c in front + score_cols + tail if c in df.columns]
    ordered += [c for c in df.columns if c not in ordered]
    df = df[ordered]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    main_path = args.out_dir / "lab_collection_invasion_predictions.csv"
    df.to_csv(main_path, index=False)

    summary = per_sublineage_summary(df, prob_col, min_n=args.min_group_n)
    summary.to_csv(args.out_dir / "lab_collection_per_sublineage.csv", index=False)

    auroc = backup_auroc(df, prob_col, split_col)
    auroc.to_csv(args.out_dir / "lab_collection_backup_auroc.csv", index=False)

    print(f"Wrote {main_path}  ({len(df)} genomes ranked by {prob_col})")
    print(f"\nPer-sublineage spread (n >= {args.min_group_n}):")
    if len(summary):
        print(summary[["Sublineage", "n", "mean", "median", "sd", "iqr", "min", "max"]].to_string(index=False))
    print("\nBackup AUROC on labelled lab genomes:")
    print(auroc.to_string(index=False))
    print("\nNOTE: probabilities are calibrated to the training cohort's ~52% blood prevalence, not "
          "to the real-world risk of invasion given carriage. Use the ranking, not the absolute value.")


def build_parser() -> argparse.ArgumentParser:
    """Three stages as subcommands, because they want different machines (GPU / CPU / CPU)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data = Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bacformer", help="GPU: score the lab genomes with both cohorts' models")
    b.add_argument("--manifest", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--train-root", type=Path, default=data / "processed/train_iso_source")
    b.add_argument("--embeddings-dir", type=Path, default=data / "processed/klebsiella_esm_embeddings")
    b.add_argument("--flavor", type=str, default="kpsc_human")
    b.add_argument("--models-dir", type=str, default="models", help="fp32 deployed models by default")
    b.add_argument("--batch-size", type=int, default=1)
    b.add_argument("--num-workers", type=int, default=8)
    b.add_argument("--no-cuda", action="store_true")
    b.set_defaults(func=_cmd_bacformer)

    k = sub.add_parser("kleborate", help="CPU: Kleborate annotation comparators")
    k.add_argument("--manifest", type=Path, required=True)
    k.add_argument("--out", type=Path, required=True)
    k.add_argument("--train-root", type=Path, default=data / "processed/train_iso_source")
    k.add_argument("--metadata", type=Path, default=data / "final/metadata_v2_all_samples_and_columns.tsv")
    k.add_argument("--cohort", type=str, default="pooled", choices=list(COHORTS))
    k.add_argument("--flavor", type=str, default="kpsc_human")
    k.set_defaults(func=_cmd_kleborate)

    a = sub.add_parser("assemble", help="CPU: merge stages into the ranked deliverable")
    a.add_argument("--manifest", type=Path, required=True)
    a.add_argument("--bacformer", type=Path, required=True)
    a.add_argument("--kleborate", type=Path, default=None)
    a.add_argument("--unitig", type=Path, default=None)
    a.add_argument("--out-dir", type=Path, required=True)
    a.add_argument("--model", type=str, default="pooled", choices=list(COHORTS),
                   help="Which cohort's Bacformer model is the headline ranking.")
    a.add_argument("--min-group-n", type=int, default=10)
    a.set_defaults(func=_cmd_assemble)
    return p


def _main_cli() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    _main_cli()

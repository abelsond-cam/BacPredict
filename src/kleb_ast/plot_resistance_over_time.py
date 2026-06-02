"""Per-drug rolling resistance-rate-over-time plots for the Kp panel.

For each panel drug, plot the rolling fraction of "R" calls over time using
``collection_date_parsed``. The window is a sample-count rolling mean (default
100 samples — the last *n* samples prior to the current one, regardless of
calendar gaps). Two lines per drug:

* **predicted** (thick C0)  — from ``predicted_<drug>_AST`` in the post-merge
  v2 metadata table; covers every ``kpsc_final_list`` isolate with an embedding
  on disk.
* **EBI ground truth** (dashed C3) — from ``EBI_<drug>_AST``; sparse pre-2010
  but a useful sanity overlay where present.

Login-node CPU; matplotlib only. Output: one ``{drug}.png`` per panel drug in
``--out-dir`` (default ``processed/train_kleb_ast/predicting_AST_over_time/``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Panel order matches eval_panel_on_slurm.sh + predict_amr_panel_on_slurm.sh.
DRUG_PANEL: list[str] = [
    "gentamicin", "ceftazidime", "meropenem", "ciprofloxacin",
    "trimethoprim-sulfamethoxazole", "amikacin", "ceftriaxone",
    "piperacillin-tazobactam", "cefoxitin", "aztreonam", "cefazolin",
    "tobramycin", "cefepime", "imipenem", "levofloxacin", "cefotaxime",
    "cefuroxime", "ampicillin-sulbactam", "ertapenem", "tetracycline",
    "azithromycin", "colistin",
]


def _rolling_r_rate(series: pd.Series, window: int) -> pd.Series:
    """Rolling fraction of "R" calls over the last ``window`` rows.

    Requires the input to be pre-sorted by date and to contain only ``"R"``/``"S"``
    (no NaN). ``min_periods = max(10, window // 4)`` so sparse eras still produce
    a value once a meaningful sample is accumulated.
    """
    is_r = (series == "R").astype(int)
    return is_r.rolling(window, min_periods=max(10, window // 4)).mean()


def _filtered_sorted(df: pd.DataFrame, drug_col: str, date_col: str) -> pd.DataFrame:
    """Drop rows missing the drug call or date; sort ascending by date."""
    sub = df[[drug_col, date_col]].copy()
    sub = sub.dropna(subset=[drug_col, date_col])
    sub = sub[sub[drug_col].isin(["R", "S"])]
    sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce")
    sub = sub.dropna(subset=[date_col])
    return sub.sort_values(date_col).reset_index(drop=True)


def plot_one_drug(
    df: pd.DataFrame,
    drug: str,
    out_dir: Path,
    window: int,
    date_col: str = "collection_date_parsed",
) -> Path | None:
    """Render and save the per-drug resistance-rate-over-time plot.

    Returns the output path, or ``None`` if there isn't enough data to plot.
    """
    pred_col = f"predicted_{drug}_AST"
    ebi_col = f"EBI_{drug}_AST"
    if pred_col not in df.columns:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred = _filtered_sorted(df, pred_col, date_col)
    if len(pred) < max(10, window // 4):
        return None

    pred["rate"] = _rolling_r_rate(pred[pred_col], window)

    ebi = _filtered_sorted(df, ebi_col, date_col) if ebi_col in df.columns else pd.DataFrame()
    if len(ebi):
        ebi["rate"] = _rolling_r_rate(ebi[ebi_col], window)

    pred_r_rate = float((pred[pred_col] == "R").mean())
    ebi_r_rate = float((ebi[ebi_col] == "R").mean()) if len(ebi) else float("nan")

    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    ax.plot(
        pred[date_col], pred["rate"],
        color="C0", lw=2,
        label=f"predicted (n={len(pred):,}, R rate {pred_r_rate:.2f})",
    )
    if len(ebi):
        ax.plot(
            ebi[date_col], ebi["rate"],
            color="C3", lw=1.2, ls="--",
            label=f"EBI actual (n={len(ebi):,}, R rate {ebi_r_rate:.2f})",
        )
    ax.set_xlabel("collection_date_parsed")
    ax.set_ylabel(f"rolling resistance rate (window = {window} samples)")
    ax.set_title(f"{drug} — rolling R rate over time")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{drug}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--metadata-tsv", required=True,
        help="Post-merge v2 metadata TSV (must contain predicted_<drug>_AST + EBI_<drug>_AST columns).",
    )
    p.add_argument(
        "--out-dir",
        default="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_kleb_ast/predicting_AST_over_time",
        help="Output directory for per-drug PNGs.",
    )
    p.add_argument("--window", type=int, default=100, help="Rolling-window size (samples). Default 100.")
    p.add_argument(
        "--date-column", default="collection_date_parsed",
        help="Datetime column to sort/plot by. Default collection_date_parsed.",
    )
    args = p.parse_args()

    print(f"Reading metadata: {args.metadata_tsv}")
    # Read only the columns we need to keep memory low.
    needed_cols = ["Sample", "kpsc_final_list", args.date_column]
    needed_cols += [f"predicted_{d}_AST" for d in DRUG_PANEL]
    needed_cols += [f"EBI_{d}_AST" for d in DRUG_PANEL]

    # Some columns may be missing (a drug failed to train); read with usecols-fallback.
    df_header = pd.read_csv(args.metadata_tsv, sep="\t", nrows=0)
    present_cols = [c for c in needed_cols if c in df_header.columns]
    missing_cols = [c for c in needed_cols if c not in df_header.columns]
    if missing_cols:
        print(f"WARNING: {len(missing_cols)} expected columns absent from TSV (will be skipped):")
        for c in missing_cols[:10]:
            print(f"  - {c}")
        if len(missing_cols) > 10:
            print(f"  ... and {len(missing_cols) - 10} more")

    df = pd.read_csv(args.metadata_tsv, sep="\t", low_memory=False, usecols=present_cols)
    df = df[df["kpsc_final_list"].astype(bool)]
    print(f"kpsc_final_list rows: {len(df):,}")

    out_dir = Path(args.out_dir)
    written, skipped = [], []
    for drug in DRUG_PANEL:
        path = plot_one_drug(df, drug, out_dir, args.window, date_col=args.date_column)
        if path is not None:
            written.append(drug)
        else:
            skipped.append(drug)

    print(f"Wrote {len(written)} PNG(s) to {out_dir}")
    if skipped:
        print(f"Skipped {len(skipped)} drug(s) for insufficient data: {', '.join(skipped)}")


if __name__ == "__main__":
    main()

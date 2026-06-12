"""Matched-protein LR-vs-SR recovery analysis for DefensePredictor outputs.

For each paired genome that scored on both arms, match proteins between the long-read (LR) and
short-read (SR) assemblies by **exact amino-acid sequence**, then — treating the LR calls as the
reference (the LR DefensePredictor AUROC is the published 0.975) — measure how well the SR arm
reproduces them on the shared protein set:

- **recovery** (sensitivity vs LR): of LR-defensive proteins that also appear (same sequence) in
  the SR assembly, the fraction the SR arm also calls defensive (``mean_log_odds >= 4``).
- **lost-to-assembly**: LR-defensive proteins with no exact SR sequence match (dropped/altered by
  short-read fragmentation+error — not a model miss but an assembly miss).
- **SR-only**: SR-defensive proteins not matched to an LR-defensive protein (false-positive-like).

Protein sequences are re-derived deterministically from the cached combined GFFs (same gene models
DefensePredictor scored) via DefensePredictor's own ``gff`` parser, keyed by ``locus_tag`` =
the ``product_accession`` in the prediction CSVs.

Run with the isolated DP venv python (it imports ``defense_predictor.gff``). Parallelises across
pairs; intended as a short icelake CPU job (translating ~2,580 genomes).

Outputs to ``--out-dir``:
- ``lr_vs_sr_recovery_per_pair.tsv`` — one row per pair.
- ``lr_vs_sr_recovery_summary.json`` — pooled metrics, overall and reference-only.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from defense_predictor.gff import build_cds_seq_df, parse_gff, translate_cds

DEFENSIVE_CUTOFF = 4.0


def _proteins_for(combined_gff: Path) -> dict[str, str]:
    """Return ``{locus_tag: protein_aa}`` for one combined GFF (DP's gene models)."""
    cds_records, contig_seqs = parse_gff(str(combined_gff))
    seq_df = build_cds_seq_df(cds_records, contig_seqs)
    return {lt: translate_cds(nt) for lt, nt in zip(seq_df["locus_tag"], seq_df["seq"], strict=False)}


def _arm_table(combined_gff: Path, pred_csv: Path) -> dict[str, tuple[str, float]]:
    """Return ``{locus_tag: (protein_aa, mean_log_odds)}`` for one arm."""
    aa = _proteins_for(combined_gff)
    pred = pd.read_csv(pred_csv, usecols=["product_accession", "mean_log_odds"])
    lo = dict(zip(pred["product_accession"].astype(str), pred["mean_log_odds"], strict=False))
    out = {}
    for lt, seq in aa.items():
        if lt in lo and seq:
            out[lt] = (seq, float(lo[lt]))
    return out


def analyse_pair(args: tuple) -> dict | None:
    """Compute matched-recovery metrics for one LR/SR pair. Returns None on read failure."""
    sample, is_reference, lr_gff, lr_csv, sr_gff, sr_csv = args
    try:
        lr = _arm_table(Path(lr_gff), Path(lr_csv))
        sr = _arm_table(Path(sr_gff), Path(sr_csv))
    except Exception:  # noqa: BLE001 — a single unreadable pair must not kill the batch
        return None

    # SR indexed by sequence: keep the strongest log-odds for duplicate sequences.
    sr_by_seq: dict[str, float] = {}
    for seq, lo in sr.values():
        if seq not in sr_by_seq or lo > sr_by_seq[seq]:
            sr_by_seq[seq] = lo

    n_shared = lr_def = lr_def_shared = recovered = sr_def = 0
    for seq, lo in lr.values():
        lr_pos = lo >= DEFENSIVE_CUTOFF
        lr_def += lr_pos
        if seq in sr_by_seq:
            n_shared += 1
            if lr_pos:
                lr_def_shared += 1
                if sr_by_seq[seq] >= DEFENSIVE_CUTOFF:
                    recovered += 1
    sr_def = sum(lo >= DEFENSIVE_CUTOFF for _, lo in sr.values())

    return {
        "Sample": sample,
        "is_reference": bool(is_reference),
        "n_lr": len(lr),
        "n_sr": len(sr),
        "n_shared_seq": n_shared,
        "lr_def": lr_def,
        "sr_def": sr_def,
        "lr_def_shared": lr_def_shared,  # LR-defensive proteins also present (seq) in SR
        "recovered": recovered,  # ...of which SR also calls defensive
        "lr_def_lost_unmatched": lr_def - lr_def_shared,  # LR-defensive with no SR seq match
    }


def _build_tasks(manifest: pd.DataFrame, pred_dir: Path, gff_dir: Path) -> list[tuple]:
    """Pairs that scored on both arms (both prediction CSVs exist)."""
    tasks = []
    for sample, g in manifest.groupby("Sample"):
        arms = {r["arm"]: r for _, r in g.iterrows()}
        if "lr" not in arms or "sr" not in arms:
            continue
        lr_label, sr_label = str(arms["lr"]["panaroo_label"]), str(arms["sr"]["panaroo_label"])
        lr_csv, sr_csv = pred_dir / "lr" / f"{lr_label}.csv", pred_dir / "sr" / f"{sr_label}.csv"
        lr_gff, sr_gff = gff_dir / "lr" / f"{lr_label}.gff", gff_dir / "sr" / f"{sr_label}.gff"
        if lr_csv.exists() and sr_csv.exists() and lr_gff.exists() and sr_gff.exists():
            tasks.append((sample, bool(arms["lr"]["is_reference"]), str(lr_gff), str(lr_csv), str(sr_gff), str(sr_csv)))
    return tasks


def _summarise(df: pd.DataFrame, label: str) -> dict:
    """Pool per-pair counts into recovery / loss rates (LR calls as truth)."""
    lr_def = int(df["lr_def"].sum())
    lr_def_shared = int(df["lr_def_shared"].sum())
    recovered = int(df["recovered"].sum())
    sr_def = int(df["sr_def"].sum())
    return {
        "slice": label,
        "n_pairs": len(df),
        "mean_shared_frac_of_lr": round((df["n_shared_seq"] / df["n_lr"]).mean(), 4),
        "lr_defensive_total": lr_def,
        "lr_defensive_with_sr_match": lr_def_shared,
        "recovered_defensive_in_sr": recovered,
        "recovery_rate_on_shared": round(recovered / lr_def_shared, 4) if lr_def_shared else None,
        "lost_to_assembly_rate": round((lr_def - lr_def_shared) / lr_def, 4) if lr_def else None,
        "sr_defensive_total": sr_def,
    }


def main() -> None:
    """CLI: run the matched-recovery analysis over the full sweep outputs."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path, help="dp_manifest_full.tsv")
    ap.add_argument("--results-dir", required=True, type=Path, help="full/ dir with predictions/ + combined_gff/")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest, sep="\t")
    tasks = _build_tasks(manifest, args.results_dir / "predictions", args.results_dir / "combined_gff")
    print(f"Analysing {len(tasks)} pairs that scored on both arms with {args.workers} workers...")

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, res in enumerate(ex.map(analyse_pair, tasks, chunksize=8)):
            if res is not None:
                rows.append(res)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(tasks)}", flush=True)

    df = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "lr_vs_sr_recovery_per_pair.tsv", sep="\t", index=False)

    summary = {
        "all": _summarise(df, "all"),
        "reference": _summarise(df[df["is_reference"]], "reference"),
    }
    (args.out_dir / "lr_vs_sr_recovery_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nWrote per-pair ({len(df)} pairs) + summary -> {args.out_dir}")
    for s in summary.values():
        print(
            f"  [{s['slice']:9s}] pairs={s['n_pairs']:4d}  shared_frac={s['mean_shared_frac_of_lr']:.2f}  "
            f"LR-def={s['lr_defensive_total']:5d}  recovery_on_shared={s['recovery_rate_on_shared']}  "
            f"lost_to_assembly={s['lost_to_assembly_rate']}"
        )


if __name__ == "__main__":
    main()

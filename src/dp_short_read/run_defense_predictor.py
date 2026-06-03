"""Run DefensePredictor over the LR/SR genome manifest, one combined GFF at a time.

For each genome arm in the manifest produced by :mod:`build_dp_cohort`:

1. Decompress the Bakta GFF + assembly if gzipped.
2. Fuse them into a Prokka-style GFF3 with an embedded ``##FASTA`` block using Panaroo's
   ``convert_bakta_to_prokka_gff.convert`` (loaded by file path from the sibling panaroo fork).
   The combined GFF is cached so re-runs skip conversion.
3. Run ``defense_predictor(gff=combined_gff)`` (ESM2-150M forward pass + 5-fold LightGBM
   ensemble) and write the per-protein log-odds table, with a ``predicted_defensive`` flag at
   the README-recommended ``mean_log_odds >= 4`` cutoff.

Output layout under ``--out-dir``::

    combined_gff/{lr,sr}/<label>.gff      # cached fused GFF+FASTA
    predictions/{lr,sr}/<label>.csv       # DefensePredictor output, one row per CDS
    run_manifest.tsv                      # per-arm status + timing (this shard)

SLURM arrays shard the manifest with ``--n-shards`` / ``--shard-index`` (round-robin by row).
DefensePredictor and the convert script live only in the isolated DP venv — run this with
``.venv-dp/bin/python`` (see ``scripts/setup_dp_env.sh``), not ``uv run``.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import re
import shutil
import tempfile
import time
import traceback
from pathlib import Path

import pandas as pd

DEFENSIVE_LOG_ODDS_CUTOFF = 4.0  # README: stringent cutoff to call a protein defensive

_ID_RE = re.compile(r"(?:^|;)ID=([^;]+)")
_LOCUS_RE = re.compile(r"locus_tag=([^;\n]+)")


def uniquify_locus_tags(gff_path: Path) -> int:
    """Rewrite ``gff_path`` in place so every CDS ``locus_tag`` is unique across distinct IDs.

    DefensePredictor keys both its per-protein FASTA (fed to ESM2) and its embedding merge on
    ``locus_tag``. Two CDS sharing a ``locus_tag`` but with different ``ID``s — which occurs in
    some RefSeq annotations — crash ESM (``Found duplicate sequence labels``) and would corrupt
    the feature pivot. Multi-segment CDS that share a single ``ID`` (programmed frameshifts)
    keep their one ``locus_tag``; only genuine cross-ID collisions get a ``_dup{n}`` suffix.
    The embedded ``##FASTA`` block (contig headers) is left untouched. Returns lines renamed.
    """
    lines = gff_path.read_text().splitlines(keepends=True)

    # Pass 1: first locus_tag seen for each distinct CDS ID.
    id_to_lt: dict[str, str] = {}
    in_fasta = False
    for line in lines:
        if in_fasta or line.startswith("##FASTA"):
            in_fasta = True
            continue
        if line.startswith("#") or "\tCDS\t" not in line:
            continue
        attrs = line.split("\t")[8]
        m_id, m_lt = _ID_RE.search(attrs), _LOCUS_RE.search(attrs)
        if m_id and m_lt:
            id_to_lt.setdefault(m_id.group(1), m_lt.group(1))

    # Assign a unique locus_tag per ID: first user of a tag keeps it, later distinct IDs suffix.
    seen: dict[str, int] = {}
    id_to_newlt: dict[str, str] = {}
    for cds_id, lt in id_to_lt.items():
        n = seen.get(lt, 0)
        id_to_newlt[cds_id] = lt if n == 0 else f"{lt}_dup{n}"
        seen[lt] = n + 1

    # Pass 2: rewrite CDS locus_tags by ID.
    renamed = 0
    out: list[str] = []
    in_fasta = False
    for line in lines:
        if in_fasta or line.startswith("##FASTA"):
            in_fasta = True
            out.append(line)
            continue
        if line.startswith("#") or "\tCDS\t" not in line:
            out.append(line)
            continue
        fields = line.split("\t")
        m_id = _ID_RE.search(fields[8])
        new_lt = id_to_newlt.get(m_id.group(1)) if m_id else None
        if new_lt is not None:
            new_attrs = _LOCUS_RE.sub(f"locus_tag={new_lt}", fields[8], count=1)
            if new_attrs != fields[8]:
                renamed += 1
                fields[8] = new_attrs
                line = "\t".join(fields)
        out.append(line)

    if renamed:
        gff_path.write_text("".join(out))
    return renamed


def _load_convert(panaroo_repo: Path):
    """Load Panaroo's ``convert`` function by file path (it is a script, not a package module)."""
    script = panaroo_repo / "scripts" / "convert_bakta_to_prokka_gff.py"
    if not script.is_file():
        raise FileNotFoundError(f"Expected Panaroo convert script at {script}")
    spec = importlib.util.spec_from_file_location("panaroo_convert_bakta", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.convert


def _maybe_gunzip(path: Path, workdir: Path) -> Path:
    """Return a plain (uncompressed) path, decompressing into ``workdir`` if ``path`` is .gz."""
    if path.suffix != ".gz":
        return path
    dst = workdir / path.with_suffix("").name
    with gzip.open(path, "rb") as fin, open(dst, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    return dst


def build_combined_gff(convert, gff_abs: Path, assembly_abs: Path, out_gff: Path) -> None:
    """Fuse a Bakta GFF + assembly FASTA into a combined GFF3 with ``##FASTA`` at ``out_gff``."""
    out_gff.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        gff_plain = _maybe_gunzip(gff_abs, tmpdir)
        asm_plain = _maybe_gunzip(assembly_abs, tmpdir)
        # is_ignore_overlapping=True matches the Panaroo run pipeline (panaroo_run_strain.py).
        convert(str(gff_plain), str(out_gff), str(asm_plain), is_ignore_overlapping=True)
    renamed = uniquify_locus_tags(out_gff)
    if renamed:
        print(f"    uniquified {renamed} duplicate locus_tag(s) in {out_gff.name}")


def run_one(dfp, convert, row: pd.Series, out_dir: Path, force: bool) -> dict:
    """Convert + score a single genome arm; return a status dict for the run manifest."""
    label = row["panaroo_label"]
    arm = row["arm"]
    combined_gff = out_dir / "combined_gff" / arm / f"{label}.gff"
    pred_csv = out_dir / "predictions" / arm / f"{label}.csv"
    status = {
        "panaroo_label": label,
        "arm": arm,
        "Sample": row["Sample"],
        "status": "ok",
        "n_proteins": 0,
        "n_defensive": 0,
        "seconds": 0.0,
        "error": "",
    }
    t0 = time.time()
    try:
        if pred_csv.exists() and not force:
            existing = pd.read_csv(pred_csv)
            status.update(
                status="cached",
                n_proteins=len(existing),
                n_defensive=int((existing["mean_log_odds"] >= DEFENSIVE_LOG_ODDS_CUTOFF).sum()),
                seconds=round(time.time() - t0, 2),
            )
            return status

        if not combined_gff.exists() or force:
            build_combined_gff(convert, Path(row["gff_abs"]), Path(row["assembly_abs"]), combined_gff)

        out_df, _ = dfp.defense_predictor(gff=str(combined_gff))
        out_df["predicted_defensive"] = (out_df["mean_log_odds"] >= DEFENSIVE_LOG_ODDS_CUTOFF).astype(int)
        out_df.insert(0, "arm", arm)
        out_df.insert(0, "panaroo_label", label)
        pred_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(pred_csv, index=False)

        status.update(
            n_proteins=len(out_df),
            n_defensive=int(out_df["predicted_defensive"].sum()),
            seconds=round(time.time() - t0, 2),
        )
    except RuntimeError as exc:
        # Known convert failure: GFF CDS seqids absent from the FASTA. Skip this arm gracefully.
        msg = "fasta_gff_mismatch" if "Mismatch between fasta and GFF!" in str(exc) else str(exc)
        status.update(status="failed", error=msg, seconds=round(time.time() - t0, 2))
    except Exception as exc:  # noqa: BLE001 — record any failure, keep the batch alive
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}", seconds=round(time.time() - t0, 2))
        traceback.print_exc()
    return status


def main() -> None:
    """CLI: run DefensePredictor across (a shard of) the manifest."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path, help="TSV from build_dp_cohort.py")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--panaroo-repo", required=True, type=Path, help="sibling panaroo fork checkout")
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="ignore cached combined GFFs / predictions")
    args = ap.parse_args()

    import defense_predictor as dfp  # imported here so --help works without the DP venv

    convert = _load_convert(args.panaroo_repo)
    manifest = pd.read_csv(args.manifest, sep="\t", dtype=str)
    shard = manifest.iloc[args.shard_index :: args.n_shards].reset_index(drop=True)
    print(f"Shard {args.shard_index}/{args.n_shards}: {len(shard)} of {len(manifest)} genome arms")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    statuses: list[dict] = []
    for i, row in shard.iterrows():
        print(f"[{i + 1}/{len(shard)}] {row['arm']}:{row['panaroo_label']}", flush=True)
        st = run_one(dfp, convert, row, args.out_dir, args.force)
        print(f"    -> {st['status']} | {st['n_proteins']} prot, {st['n_defensive']} defensive, {st['seconds']}s")
        statuses.append(st)

    run_manifest = pd.DataFrame(statuses)
    run_path = args.out_dir / f"run_manifest_shard{args.shard_index:03d}.tsv"
    run_manifest.to_csv(run_path, sep="\t", index=False)
    ok = (run_manifest["status"].isin(["ok", "cached"])).sum()
    print(f"\nDone: {ok}/{len(run_manifest)} arms scored. Run manifest -> {run_path}")
    if (run_manifest["status"] == "failed").any():
        print("Failures:")
        print(run_manifest[run_manifest["status"] == "failed"][["arm", "panaroo_label", "error"]].to_string(index=False))


if __name__ == "__main__":
    main()

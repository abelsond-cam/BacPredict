"""Stage 2c — does a baclm PROMOTER (intergenic) embedding carry AMR signal?

For a target *flanking gene* + drug this locates the intergenic region **immediately 5′ of the gene**
(the promoter), pulls that region's baclm ``intergenic_embeddings`` row, and scores it against the drug
label through the same learning-curve harness as the coding probe
(:func:`bacpredict.engine.gene_lr.coding_amr_lr.ladder_over_frames`, k-seed × training-size sweep on a fixed
evaluate holdout). Only baclm has a non-coding channel — ESM/Bacformer are protein models — so this is
a baclm-only, vs-chance read: is there promoter-mutation signal an LR can pick up?

Anchoring (per assembly, from the Bakta GFF — coords are per-assembly, NOT H37Rv):
  * the flank gene's ``(seqid, start, end, strand)`` come from its ``gene=`` CDS line;
  * the promoter IGR is the ``intergenic_embeddings`` row whose gene-side boundary abuts the gene's
    5′ end — ``end == gene_start-1`` on ``+``, ``start == gene_end+1`` on ``−`` (both self-describing
    in the ``.pt`` via ``intergenic_seqid/start/end``). No parquet needed.

Build-defect audit folded in (Stage 2b): each promoter's **far-flank feature type** (CDS vs RNA) and
**length** are recorded — a promoter abutting a tRNA/rRNA is a fragment baclm may have seen
out-of-distribution, and a region > 2048 bp was truncated then mean-pooled. Both are reported per
locus so a weak AUROC can be read against build quality rather than mistaken for absent signal.

CPU-only (GFF parse + sklearn LR). ``rrs``-driven drugs (streptomycin, amikacin) are **not** here — the
16S rRNA body is not an intergenic region and is missing from the current build (needs the 2d re-embed).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bacpredict.engine.embedding.extract_proteins_from_gff_fna import _open_text
from bacpredict.engine.gene_lr.coding_amr_lr import ladder_over_frames
from bacpredict.engine.gene_lr.snp_vs_esm_prediction import resolve_clean_splits

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SCHEMA_VERSION = 1

# GFF feature types treated as RNA for the far-flank audit (a promoter abutting one of these is a
# fragment under the current build). Everything else that occupies sequence is treated as CDS-like.
_RNA_TYPES = frozenset({"trna", "rrna", "rrna_16s", "rrna_23s", "rrna_5s", "tmrna", "ncrna",
                        "ncrna_region", "regulatory_region", "riboswitch"})
_NON_OCCUPYING = frozenset({"region", "databank_entry", "gap"})
_BACLM_CONTEXT = 2048  # baclm max context; longer regions were truncated + mean-pooled at embed time


@dataclass(frozen=True)
class IgrTarget:
    """One (flank gene → promoter, drug) probe. ``aliases`` are alternative gene symbols (lowercased)."""

    flank_gene: str
    drug: str
    aliases: tuple[str, ...] = ()
    region: str = ""
    note: str = ""


# Promoter loci that are true intergenic regions (CDS-flanked, in the current build). rrs-driven drugs
# excluded (RNA body, missing until 2d). fabG1 promoter drives the mabA-inhA operon (INH via inhA
# overexpression + ETH); eis promoter → kanamycin; pncA promoter → pyrazinamide.
IGR_PANEL: dict[str, list[IgrTarget]] = {
    "tb": [
        IgrTarget("fabG1", "ethionamide", aliases=("mabA",), region="mabA-inhA operon promoter",
                  note="inhA overexpression — a primary ETH mechanism"),
        IgrTarget("fabG1", "isoniazid", aliases=("mabA",), region="mabA-inhA operon promoter",
                  note="partial INH mechanism (katG dominates)"),
        IgrTarget("eis", "kanamycin", region="eis promoter", note="eis overexpression"),
        IgrTarget("pncA", "pyrazinamide", region="pncA promoter", note="pncA promoter mutations"),
    ],
}


@dataclass
class IgrPaths:
    """Per-species store locations for the IGR probe (Isambard ``$SCRATCHDIR`` defaults)."""

    ast_sheet: Path
    input_csv: Path  # Sample -> sr_gff_file
    baclm_dir: Path
    baclm_suffix: str = "_baclm_embeddings.pt"


def default_paths(species: str) -> IgrPaths:
    """Isambard ``$SCRATCHDIR`` defaults for a species (``tb`` → ``train_tb_ast``)."""
    scratch = os.environ.get("SCRATCHDIR", "")
    task = {"tb": "train_tb_ast", "kp": "train_kleb_ast"}[species]
    root = Path(scratch) / "processed" / task
    return IgrPaths(
        ast_sheet=root / "binary_ast_with_split.csv",
        input_csv=root / "embedding_input.csv",
        baclm_dir=root / "baclm",
    )


# ---------------------------------------------------------------------------
# GFF → per-gene locus + strand, with feature list for far-flank typing
# ---------------------------------------------------------------------------

def _parse_gff(gff_path: Path) -> tuple[dict[str, list[tuple[int, int, str, str]]], dict[str, list[tuple[int, int, str]]]]:
    """Parse one GFF into occupying features and named-gene hits.

    Returns
    -------
    feats_by_seqid
        ``seqid → sorted list of (start, end, strand, ftype_lower)`` — every occupying feature (1-based
        inclusive), for locating the promoter IGR's far-flank feature type.
    genes
        ``gene_name_lower → list of (seqid, start, end, strand)`` — CDS lines carrying ``gene=``.
    """
    feats_by_seqid: dict[str, list[tuple[int, int, str, str]]] = {}
    genes: dict[str, list[tuple[int, int, str]]] = {}
    with _open_text(gff_path) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                if line.startswith("##FASTA"):
                    break
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            ftype = parts[2].lower()
            if ftype in _NON_OCCUPYING:
                continue
            try:
                start, end = int(parts[3]), int(parts[4])
            except ValueError:
                continue
            seqid, strand, attrs = parts[0], parts[6], parts[8]
            feats_by_seqid.setdefault(seqid, []).append((start, end, strand, ftype))
            gene = None
            for tok in attrs.split(";"):
                if tok.startswith("gene="):
                    gene = tok[5:].strip().lower()
                    break
            if gene is not None:
                genes.setdefault(gene, []).append((seqid, start, end, strand))
    for seqid in feats_by_seqid:
        feats_by_seqid[seqid].sort()
    return feats_by_seqid, genes


def _far_flank_type(feats: list[tuple[int, int, str, str]], igr_start: int, igr_end: int, upstream_low: bool) -> str:
    """Feature type on the outward (non-gene) side of the promoter IGR — for the fragmentation audit.

    ``upstream_low`` True when the promoter is on the low-coordinate side of the gene (``+`` strand): the
    far flank is then the feature ending just below ``igr_start``. Otherwise (``−``) it is the feature
    starting just above ``igr_end``. Returns ``"cds"`` / an RNA type / ``"contig_end"`` / ``"none"``.
    """
    if upstream_low:
        cands = [f for f in feats if f[1] <= igr_start]  # ends at/below IGR start
        if not cands:
            return "contig_end"
        ft = max(cands, key=lambda f: f[1])[3]
    else:
        cands = [f for f in feats if f[0] >= igr_end]  # starts at/above IGR end
        if not cands:
            return "contig_end"
        ft = min(cands, key=lambda f: f[0])[3]
    return "rna" if ft in _RNA_TYPES else ("cds" if ft == "cds" else ft)


def _locate_promoter_row(
    gene_hits: list[tuple[int, int, str]],
    feats_by_seqid: dict[str, list[tuple[int, int, str, str]]],
    ig_seqid: list[str],
    ig_start: list[int],
    ig_end: list[int],
    *,
    boundary_tol: int = 3,
) -> tuple[int | None, dict, str | None]:
    """Find the ``intergenic_embeddings`` row abutting the gene's 5′ end. Returns (row, meta, skip)."""
    if len(gene_hits) == 0:
        return None, {}, "gene_absent"
    if len(gene_hits) > 1:
        return None, {}, "gene_multicopy"
    seqid, gs, ge, strand = gene_hits[0]
    upstream_low = strand == "+"  # promoter on low-coordinate side for +, high side for −
    best_row, best_dist = None, boundary_tol + 1
    for i, (sq, s, e) in enumerate(zip(ig_seqid, ig_start, ig_end, strict=True)):
        if sq != seqid:
            continue
        if upstream_low:
            if e < gs:                       # IGR fully 5′ of the gene
                dist = gs - 1 - e            # 0 when abutting (e == gs-1)
            else:
                continue
        else:
            if s > ge:                       # IGR fully 5′ (reverse strand)
                dist = s - (ge + 1)          # 0 when abutting (s == ge+1)
            else:
                continue
        if dist < best_dist:
            best_row, best_dist = i, dist
    if best_row is None:
        return None, {}, "gene_5prime_abuts_feature_or_contig_end"
    s, e = ig_start[best_row], ig_end[best_row]
    igr_len = e - s + 1
    feats = feats_by_seqid.get(seqid, [])
    meta = {
        "seqid": seqid,
        "strand": strand,
        "igr_start": int(s),
        "igr_end": int(e),
        "igr_len": int(igr_len),
        "boundary_gap": int(best_dist),
        "truncated": bool(igr_len > _BACLM_CONTEXT),
        "far_flank_type": _far_flank_type(feats, s, e, upstream_low),
    }
    return best_row, meta, None


# ---------------------------------------------------------------------------
# One sweep: locate every flank gene's promoter IGR + pull its vector
# ---------------------------------------------------------------------------

def _promoter_igr_one(sid: str, gff_path: str, pt_path: str, wanted_by_gene: dict[str, frozenset]):
    """Parse one GFF + load one baclm ``.pt`` once; return each flank gene's promoter vector + meta."""
    gpath, ppath = Path(gff_path), Path(pt_path)
    if not gpath.exists() or not ppath.exists():
        return sid, None
    try:
        feats_by_seqid, genes = _parse_gff(gpath)
    except (OSError, ValueError):
        return sid, None
    store = torch.load(ppath, map_location="cpu", mmap=True, weights_only=True)
    # 2d re-embed renamed intergenic_* -> noncoding_* (maximal non-CDS runs). Prefer the new key,
    # fall back to the legacy key so this probe reads both the old and re-embedded stores.
    if "noncoding_embeddings" in store:
        ig_emb = store["noncoding_embeddings"]
        ig_seqid, ig_start, ig_end = store["noncoding_seqid"], store["noncoding_start"], store["noncoding_end"]
    else:
        ig_emb = store["intergenic_embeddings"]
        ig_seqid, ig_start, ig_end = store["intergenic_seqid"], store["intergenic_start"], store["intergenic_end"]
    n_ig = int(ig_emb.shape[0])
    out: dict[str, dict | None] = {}
    for gene, wanted in wanted_by_gene.items():
        hits = [h for g, hlist in genes.items() if g in wanted for h in hlist]
        row, meta, skip = _locate_promoter_row(hits, feats_by_seqid, ig_seqid, ig_start, ig_end)
        if skip is not None or row is None or row >= n_ig:
            out[gene] = {"skip": skip or "row_out_of_range"}
            continue
        out[gene] = {"vector": ig_emb[row].float().clone().numpy(), **meta}
    return sid, out


def build_promoter_frames(
    sample_ids: list[str],
    sample_gff: dict[str, str],
    baclm_dir: Path,
    flank_specs: list[tuple[str, tuple[str, ...]]],
    *,
    baclm_suffix: str = "_baclm_embeddings.pt",
    pool_workers: int = 1,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """One GFF+``.pt`` sweep → per-flank-gene promoter vector frame + audit table.

    ``flank_specs`` is a list of ``(gene, aliases_tuple)``. Returns ``(frames, audits)`` each keyed by
    flank gene: ``frames[gene]`` is ``[N, 960]`` indexed by Sample; ``audits[gene]`` carries the per-sample
    ``seqid/strand/igr_len/boundary_gap/truncated/far_flank_type`` for the build-quality report.
    """
    wanted_by_gene = {g: frozenset([g.lower(), *(a.lower() for a in aliases)]) for g, aliases in flank_specs}
    baclm_dir = Path(baclm_dir)
    tasks = [
        (str(s), sample_gff.get(str(s), ""), str(baclm_dir / f"{s}{baclm_suffix}"), wanted_by_gene)
        for s in sample_ids if str(s) in sample_gff
    ]
    if pool_workers > 1:
        import multiprocessing as mp

        with mp.Pool(pool_workers) as pool:
            results = pool.starmap(_promoter_igr_one, tasks)
    else:
        results = [_promoter_igr_one(*t) for t in tasks]

    vec_rows: dict[str, list] = {g: [] for g, _ in flank_specs}
    vec_idx: dict[str, list[str]] = {g: [] for g, _ in flank_specs}
    audit_rows: dict[str, list] = {g: [] for g, _ in flank_specs}
    skips: dict[str, dict[str, int]] = {g: {} for g, _ in flank_specs}
    for sid, per_gene in results:
        if per_gene is None:
            for g in vec_rows:
                skips[g]["missing_input"] = skips[g].get("missing_input", 0) + 1
            continue
        for gene, hit in per_gene.items():
            if hit is None or "skip" in hit:
                reason = (hit or {}).get("skip", "unknown")
                skips[gene][reason] = skips[gene].get(reason, 0) + 1
                continue
            vec_rows[gene].append(hit["vector"])
            vec_idx[gene].append(sid)
            audit_rows[gene].append({"Sample": sid, **{k: v for k, v in hit.items() if k != "vector"}})

    frames, audits = {}, {}
    for gene, _ in flank_specs:
        frames[gene] = (pd.DataFrame(np.vstack(vec_rows[gene]), index=pd.Index(vec_idx[gene], name="Sample"))
                        if vec_rows[gene] else pd.DataFrame())
        audits[gene] = (pd.DataFrame(audit_rows[gene]).set_index("Sample")
                        if audit_rows[gene] else pd.DataFrame())
        logger.info("promoter IGR %-6s located in %d/%d genomes; skips=%s",
                    gene, len(frames[gene]), len(tasks), skips[gene])
    return frames, audits


def _audit_summary(audit: pd.DataFrame) -> dict:
    """Summarise a flank gene's promoter audit — CDS-flanked fraction, RNA-abutting, truncation, length."""
    if audit.empty:
        return {"n": 0}
    ff = audit["far_flank_type"].astype(str)
    n = len(audit)
    return {
        "n": int(n),
        "frac_cds_flanked": float((ff == "cds").mean()),
        "frac_rna_abutting": float(ff.isin(["rna"]).mean()),
        "frac_contig_end": float((ff == "contig_end").mean()),
        "frac_truncated": float(audit["truncated"].mean()),
        "igr_len_median": float(audit["igr_len"].median()),
        "igr_len_max": int(audit["igr_len"].max()),
        "boundary_gap_max": int(audit["boundary_gap"].max()),
    }


# ---------------------------------------------------------------------------
# One (flank gene, drug) promoter probe = learning curve over the single baclm IGR frame
# ---------------------------------------------------------------------------

def run_igr_probe(
    target: IgrTarget,
    frame: pd.DataFrame,
    audit: pd.DataFrame,
    ast_sheet: Path,
    *,
    seeds: tuple[int, ...] = (1, 2, 3),
    step: int = 500,
    fine_until: int = 6000,
) -> dict:
    """Score one flank gene's promoter IGR frame against its drug label via the learning-curve harness."""
    base = {"flank_gene": target.flank_gene, "drug": target.drug, "region": target.region,
            "note": target.note, "audit": _audit_summary(audit)}
    if frame.empty:
        return {**base, "error": "no promoter IGR located for any genome"}
    label_map, *_ = resolve_clean_splits(ast_sheet, target.drug)
    lad = ladder_over_frames({"igr": frame}, label_map, seeds=seeds, step=step, fine_until=fine_until)
    out = {**base, "n_igr": len(frame), **lad}
    if not lad.get("error") and lad["rungs"][-1].get("igr"):
        top = lad["rungs"][-1]
        logger.info("[%s promoter / %s] full pool (n=%d): AUROC %.4f  (CDS-flanked %.0f%%, trunc %.0f%%)",
                    target.flank_gene, target.drug, top["n_train"], top["igr"]["mean"],
                    100 * base["audit"].get("frac_cds_flanked", 0), 100 * base["audit"].get("frac_truncated", 0))
    return out


def plot_igr_ladder(payload: dict, png_path: Path) -> None:
    """Render each promoter's baclm-IGR learning curve (AUROC vs training size) with ±sd bands."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    probes = [p for p in payload["probes"] if p.get("rungs")]
    if not probes:
        logger.warning("no IGR probes with rungs to plot")
        return
    ncol = min(2, len(probes))
    nrow = (len(probes) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 4.0 * nrow), squeeze=False)
    for ax, p in zip(axes.flat, probes, strict=False):
        ns = [r["n_train"] for r in p["rungs"]]
        m = np.array([r["igr"]["mean"] if r["igr"] else np.nan for r in p["rungs"]])
        sd = np.array([r["igr"]["sd"] if r["igr"] else 0.0 for r in p["rungs"]])
        ax.plot(ns, m, "-o", ms=3, color="#1e8449", label="baclm promoter IGR")
        ax.fill_between(ns, m - sd, m + sd, color="#1e8449", alpha=0.15)
        ax.axhline(0.5, ls="--", lw=1, color="grey", label="chance")
        a = p["audit"]
        ax.set_title(f"{p['flank_gene']} promoter / {p['drug']}\n{p['region']} "
                     f"(CDS-flanked {100*a.get('frac_cds_flanked',0):.0f}%, trunc {100*a.get('frac_truncated',0):.0f}%)",
                     fontsize=9)
        ax.set_xlabel("training genomes")
        ax.set_ylabel("evaluate AUROC")
        ax.set_ylim(0.4, 1.0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    for ax in axes.flat[len(probes):]:
        ax.set_visible(False)
    fig.suptitle(f"Promoter IGR (baclm) AMR learning curves ({payload['species']})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    logger.info("wrote %s", png_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Locate promoter IGRs across the cohort and score each against its drug via the ladder."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", choices=["tb", "kp"], default="tb")
    ap.add_argument("--ast-sheet", type=Path)
    ap.add_argument("--input-csv", type=Path, help="Sample->GFF map (embedding_input.csv)")
    ap.add_argument("--baclm-dir", type=Path)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--ladder-step", type=int, default=500)
    ap.add_argument("--ladder-fine-until", type=int, default=6000)
    ap.add_argument("--pool-workers", type=int, default=1)
    ap.add_argument("--n", type=int, default=None, help="Stage-A smoke: cap #samples")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    paths = default_paths(args.species)
    if args.ast_sheet:
        paths.ast_sheet = args.ast_sheet
    if args.input_csv:
        paths.input_csv = args.input_csv
    if args.baclm_dir:
        paths.baclm_dir = args.baclm_dir
    seeds = tuple(int(s) for s in args.seeds.split(","))
    targets = IGR_PANEL[args.species]

    inp = pd.read_csv(paths.input_csv, usecols=["Sample", "sr_gff_file"])
    sample_gff = dict(zip(inp["Sample"].astype(str), inp["sr_gff_file"].astype(str), strict=True))
    all_ids = sorted(sample_gff)
    if args.n is not None:
        all_ids = all_ids[: args.n]

    flank_specs = list({(t.flank_gene, t.aliases) for t in targets})
    logger.info("one GFF+.pt sweep over %d genomes for %d flank genes", len(all_ids), len(flank_specs))
    frames, audits = build_promoter_frames(
        all_ids, sample_gff, paths.baclm_dir, flank_specs,
        baclm_suffix=paths.baclm_suffix, pool_workers=args.pool_workers,
    )

    probes = [
        run_igr_probe(t, frames[t.flank_gene], audits[t.flank_gene], paths.ast_sheet,
                      seeds=seeds, step=args.ladder_step, fine_until=args.ladder_fine_until)
        for t in targets
    ]

    payload = {
        "schema_version": SCHEMA_VERSION,
        "task": "pangena_predict",
        "analysis": "igr_amr_lr_baclm_promoter",
        "species": args.species,
        "seeds": list(seeds),
        "ladder_step": args.ladder_step,
        "ladder_fine_until": args.ladder_fine_until,
        "sample_limit": args.n,
        "paths": {k: str(v) for k, v in vars(paths).items()},
        "probes": probes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    logger.info("wrote %s (%d IGR probes)", args.output, len(probes))
    if args.plot:
        plot_igr_ladder(payload, args.output.with_suffix(".png"))

    print("\n=== baclm promoter IGR → AMR (learning curve; low-n → full pool) ===")
    for p in probes:
        if p.get("error"):
            print(f"  {p['flank_gene']+' prom':<16} {p['drug']:<14} ERROR: {p['error']} "
                  f"(located n={p['audit'].get('n', 0)})")
            continue
        r0, rN = p["rungs"][0], p["rungs"][-1]
        a = p["audit"]
        print(
            f"  {p['flank_gene']+' prom':<16} {p['drug']:<14} "
            f"n={r0['n_train']:<5}AUROC={r0['igr']['mean']:.3f}  →  n={rN['n_train']:<6}AUROC={rN['igr']['mean']:.3f}  "
            f"(pool={p['n_train_pool']}, CDS-flanked {100*a.get('frac_cds_flanked',0):.0f}%, "
            f"RNA-abut {100*a.get('frac_rna_abutting',0):.0f}%, trunc {100*a.get('frac_truncated',0):.0f}%)"
        )


if __name__ == "__main__":
    main()

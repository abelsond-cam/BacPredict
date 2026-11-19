r"""GPU — swap one gene's protein embedding for a wild-type one and re-predict.

The direct causal test, and the one that needs no attribution machinery at all. For a determinant
that is present in **every** genome and differs only by a point mutation — ``rpoB``, ``gyrA``,
``parC`` — the question is not "what happens if the gene is removed" (the model never saw a genome
without it) but:

    replace this genome's ``rpoB`` with a **wild-type** ``rpoB``. Does the prediction change?

The expected result is the whole claim, stated so it can fail: **no change in susceptible genomes**
(their ``rpoB`` already is wild-type, so the substitution is nearly a no-op) and **resistant ->
susceptible in genomes carrying the mutation**. One forward pass per genome. No captum, no path
integral, no baseline choice, no completeness to check.

Why this is cheap where a whole-genome wild-type baseline is not
---------------------------------------------------------------
Only **one** protein is substituted, so the only cross-genome identity needed is for that one gene —
and Bakta names ``rpoB``, ``gyrA`` and ``parC`` directly. (Using wild-type references for *every*
protein, as an integrated-gradients baseline, would need identity for all ~4,000 including the
majority Bakta leaves as genome-specific locus tags. That is a different and much larger problem.)

What the donor embedding is
---------------------------
The substituted vector is the **mean embedding of that gene across donor genomes** — by default the
cohort genomes that carry no variant in it, which makes it a wild-type consensus rather than one
arbitrary genome's copy. ``--donor-sample`` pins a single genome instead when a specific reference
is wanted, and ``--donor-variant`` builds the donor from genomes that *do* carry a named variant, so
the converse test (splice a mutant ``rpoB`` into susceptible genomes; do they flip to resistant?)
runs through the same path. The converse is the stronger of the two, because a no-op result in the
forward direction is also what a model ignoring the gene entirely would produce.

Reported per genome
-------------------
``base_logit``, ``substituted_logit``, their difference, both probabilities, whether the call
crossed 0.5, and the donor/recipient variant status — so the contingency table of (carries variant)
x (prediction flipped) falls straight out.

Examples
--------
Wild-type ``rpoB`` into every genome of the TB cohort::

    python -m bacpredict.engine.explain.substitute_wt_protein \\
        --organism tb --gene rpoB \\
        --cohort-csv "$B/train_tb_ast/attribution/attribution_cohort_rifampin_n100.csv" \\
        --checkpoint "$CKPT" --variants "$B/train_tb_ast/tbprofiler/tbprofiler_variants.parquet" \\
        --out "$B/train_tb_ast/attribution/rpoB_wt_substitution.csv"

The converse — a ``p.Ser450Leu`` ``rpoB`` into every genome::

    python -m bacpredict.engine.explain.substitute_wt_protein \\
        --organism tb --gene rpoB --donor-variant "rpoB@p.Ser450Leu" \\
        --cohort-csv ... --checkpoint "$CKPT" --variants ... \\
        --out "$B/train_tb_ast/attribution/rpoB_ser450leu_substitution.csv"
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bacpredict.engine.config import store_paths
from bacpredict.engine.explain.gene_attribution import (
    DTYPES,
    LogitForward,
    load_classifier,
    load_genome,
    verify_protein_count,
)
from bacpredict.engine.gene_lr.locate_gene import flatten_proteins

logger = logging.getLogger(__name__)


def gene_rows(sample: str, gene: str, parquet_dir: Path, suffix: str) -> list[int]:
    """Flat indices of every protein called ``gene`` in one genome.

    Parameters
    ----------
    sample : str
        Sample ID.
    gene : str
        Bakta gene symbol, matched case-insensitively.
    parquet_dir : pathlib.Path
        Directory of ``{Sample}{suffix}`` protein parquets.
    suffix : str
        Parquet filename suffix.

    Returns
    -------
    list of int
        Flat indices, which are the embedding tensor's row numbers. Empty if the gene is absent.
    """
    path = parquet_dir / f"{sample}{suffix}"
    if not path.exists():
        logger.warning("no protein parquet for %s", sample)
        return []
    records = flatten_proteins(pd.read_parquet(path))
    wanted = gene.casefold()
    return [r["flat_index"] for r in records if (r["gene_name"] or "").casefold() == wanted]


def samples_carrying(variants: pd.DataFrame, gene: str, variant_id: str | None) -> set[str]:
    """Samples carrying a variant in ``gene`` — or one specific variant.

    Parameters
    ----------
    variants : pandas.DataFrame
        TB-Profiler variant frame with ``Sample``, ``gene_name`` and ``variant_id`` columns.
    gene : str
        Gene symbol, matched case-insensitively.
    variant_id : str, optional
        A specific ``gene@p.Change`` id. ``None`` means any variant in the gene.

    Returns
    -------
    set of str
        Sample IDs.
    """
    rows = variants[variants["gene_name"].astype(str).str.casefold() == gene.casefold()]
    if variant_id is not None:
        rows = rows[rows["variant_id"].astype(str) == variant_id]
    return set(rows["Sample"].astype(str))


def donor_embedding(
    donor_samples: list[str], gene: str, esm_dir: Path, paths, device: str, dtype: torch.dtype
) -> tuple[torch.Tensor, int]:
    """Mean embedding of one gene across donor genomes.

    A consensus rather than one arbitrary genome's copy, so a single odd assembly cannot define the
    counterfactual. Accumulates in float64 for the same reason the cohort baseline does.

    Parameters
    ----------
    donor_samples : list of str
        Genomes to average over.
    gene : str
        Gene symbol.
    esm_dir : pathlib.Path
        Embedding store directory.
    paths : bacpredict.engine.config.StorePaths
        Resolved store paths, for the parquet directory and filename suffixes.
    device : str
        Device for the returned tensor.
    dtype : torch.dtype
        Dtype for the returned tensor.

    Returns
    -------
    embedding : torch.Tensor
        ``[width]`` — the mean donor embedding.
    n_donors : int
        How many genomes contributed exactly one copy.

    Raises
    ------
    ValueError
        If no donor genome yielded a single copy of the gene.
    """
    total: torch.Tensor | None = None
    n_donors = 0
    for sample in donor_samples:
        rows = gene_rows(sample, gene, Path(paths.parquet_dir), paths.parquet_suffix)
        if len(rows) != 1:
            # Multi-copy or absent: averaging a paralogue in would blur the counterfactual.
            continue
        store_path = esm_dir / f"{sample}{paths.esm_suffix}"
        if not store_path.exists():
            continue
        store = torch.load(store_path, map_location="cpu", weights_only=False)
        embeddings = store.get("prot_embeddings", store.get("protein_embeddings"))
        embeddings = embeddings.squeeze(0) if embeddings.dim() == 3 else embeddings
        vector = embeddings[rows[0]].double()
        total = vector.clone() if total is None else total + vector
        n_donors += 1
    if total is None:
        raise ValueError(f"no donor genome carried exactly one copy of {gene}")
    return (total / n_donors).to(device=device, dtype=dtype), n_donors


def main() -> None:
    """CLI entry point — substitute one gene's embedding across a cohort and record the shift."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--organism", required=True, choices=("tb", "kp"), help="Organism key.")
    parser.add_argument("--gene", required=True, help="Bakta gene symbol to substitute, e.g. rpoB.")
    parser.add_argument("--cohort-csv", type=Path, required=True, help="Cohort table; its Sample column.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Deployed fine-tune run or checkpoint dir.")
    parser.add_argument("--out", type=Path, required=True, help="Where the per-genome CSV is written.")
    parser.add_argument("--variants", type=Path, default=None,
                        help="TB-Profiler variants parquet; picks wild-type donors and labels recipients.")
    parser.add_argument("--donor-variant", default=None,
                        help="Build the donor from genomes carrying this variant_id instead of from "
                             "wild-type genomes — the converse test.")
    parser.add_argument("--donor-sample", default=None,
                        help="Use this one genome's copy as the donor instead of a consensus mean.")
    parser.add_argument("--esm-dir", type=Path, default=None, help="Override the ESM store directory.")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="fp32",
                        help="fp32 by default: a substitution can move the logit by far less than the "
                             "bf16 ULP at a confident logit, which would quantise the result to zero.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N genomes.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

    paths = store_paths(args.organism)
    esm_dir = Path(args.esm_dir or paths.esm_dir)
    dtype = DTYPES[args.dtype]

    cohort = pd.read_csv(args.cohort_csv)
    cohort["Sample"] = cohort["Sample"].astype(str)
    if args.limit:
        cohort = cohort.head(args.limit)

    carriers: set[str] = set()
    if args.variants is not None:
        variants = pd.read_parquet(args.variants)
        carriers = samples_carrying(variants, args.gene, None)
        logger.info("%d of %d cohort genomes carry a variant in %s",
                    len(carriers & set(cohort["Sample"])), len(cohort), args.gene)

    # Donors: a named genome, or genomes carrying the requested variant, or the wild-type ones.
    if args.donor_sample:
        donor_samples = [args.donor_sample]
    elif args.donor_variant:
        if args.variants is None:
            parser.error("--donor-variant needs --variants to know which genomes carry it")
        donor_samples = sorted(samples_carrying(variants, args.gene, args.donor_variant) & set(cohort["Sample"]))
    elif carriers:
        donor_samples = sorted(set(cohort["Sample"]) - carriers)
    else:
        parser.error("no way to pick wild-type donors: pass --variants, --donor-sample or --donor-variant")
    if not donor_samples:
        parser.error("the donor selection matched no genome in this cohort")
    logger.info("donor pool: %d genomes (%s)", len(donor_samples),
                args.donor_variant or (args.donor_sample or "wild-type"))

    donor, n_donors = donor_embedding(donor_samples, args.gene, esm_dir, paths, args.device, dtype)
    logger.info("donor embedding from %d genomes, L2 %.4f", n_donors, float(donor.norm()))

    model = load_classifier(args.checkpoint, args.device, dtype)
    wrapped = LogitForward(model).eval()

    label_col = "ast_label" if "ast_label" in cohort.columns else None
    records = []
    for _, row in cohort.iterrows():
        sample = row["Sample"]
        store_path = esm_dir / f"{sample}{paths.esm_suffix}"
        if not store_path.exists():
            logger.warning("no embedding for %s", sample)
            continue
        rows = gene_rows(sample, args.gene, Path(paths.parquet_dir), paths.parquet_suffix)
        if len(rows) != 1:
            # Reported, never silently dropped: "the gene is absent" and "the gene is multi-copy"
            # are different findings and both matter.
            records.append({"Sample": sample, "status": f"n_copies={len(rows)}",
                            "ast_label": row.get(label_col, -1) if label_col else -1})
            continue

        inputs = load_genome(store_path, args.device, dtype)
        n_proteins = int(inputs["protein_embeddings"].shape[1])
        verify_protein_count(sample, n_proteins, Path(paths.parquet_dir), paths.parquet_suffix)
        if rows[0] >= n_proteins:
            records.append({"Sample": sample, "status": "gene_past_embedding_truncation"})
            continue

        substituted = inputs["protein_embeddings"].clone()
        original = substituted[0, rows[0]].clone()
        substituted[0, rows[0]] = donor
        extra = (inputs["attention_mask"], inputs["contig_ids"])
        with torch.inference_mode():
            base_logit = float(wrapped(inputs["protein_embeddings"], *extra).item())
            new_logit = float(wrapped(substituted, *extra).item())

        base_p, new_p = 1 / (1 + np.exp(-base_logit)), 1 / (1 + np.exp(-new_logit))
        records.append({
            "Sample": sample,
            "status": "ok",
            "ast_label": int(row[label_col]) if label_col else -1,
            "carries_variant": sample in carriers,
            "flat_index": rows[0],
            "base_logit": base_logit,
            "substituted_logit": new_logit,
            "delta_logit": new_logit - base_logit,
            "base_prob": base_p,
            "substituted_prob": new_p,
            "delta_prob": new_p - base_p,
            "prediction_flipped": (base_p >= 0.5) != (new_p >= 0.5),
            # How far the donor sat from what it replaced — a near-zero distance means the
            # substitution was a no-op by construction, not evidence about the model.
            "donor_distance": float((donor.double() - original.double()).norm()),
        })
        if len(records) % 25 == 0:
            logger.info("%d genomes done", len(records))

    frame = pd.DataFrame(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    done = frame[frame["status"] == "ok"] if "status" in frame else frame
    logger.info("wrote %s (%d rows, %d scored)", args.out, len(frame), len(done))
    if len(done) and "carries_variant" in done:
        summary = done.groupby("carries_variant").agg(
            n=("Sample", "size"), mean_delta_logit=("delta_logit", "mean"),
            median_delta_logit=("delta_logit", "median"), n_flipped=("prediction_flipped", "sum"))
        logger.info("substituting %s (%s donor):\n%s", args.gene,
                    args.donor_variant or "wild-type", summary.to_string())


if __name__ == "__main__":
    main()

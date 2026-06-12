"""Stage 1.1 — the phenotype-ceiling ladder for rpoB / rifampicin resistance.

Three **locus-restricted** predictors of RIF resistance on the same isolates,
all confined to the rpoB locus so there is no accessory-genome / phylogeny
shortcut to guard against (hence no lineage holdout):

1. **One-hot RRDR genotype** — the observed amino acid at each RRDR codon,
   one-hot encoded → logistic regression. The information *ceiling* (expected
   AUROC >= 0.95): the causal allele in its rawest form.
2. **Masked-marginal LLR** — ESM-C's ``log P(observed) - log P(wild-type)`` at
   the panel codons → logistic regression. Tests whether the resistance signal
   is present in ESM-C *at all* (expected: well above pooled, below one-hot).
3. **Frozen pooled ESM-C rpoB vector** — the mean-pooled per-protein embedding
   pulled straight out of the existing store (no new forward) → logistic
   regression. The *suspected failure* (expected: ~= baseline).

Head-line read-out: ``AUROC(1) - AUROC(3)`` is the information lost to ESM-C's
residue mean-pool; where predictor (2) lands separates "lost at pooling" from
"absent from the model".

The pooled vector is recovered by selecting the ``PROT_EMB`` rows of the stored
tensor (``special_tokens_mask == 4``; CLS/SEP/PAD/END are 2/3/0/5) — these are
the real proteins in flat order — then indexing by the rpoB flat index from
:mod:`snp_embeddings.locate_gene`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from snp_embeddings.rpob_genotype import (
    RIFAMPIN_COLUMN,
    RRDR_PANEL,
    build_genotype_table,
    join_rifampin_label,
    load_reference,
    ref_index_for_codon,
    sample_codon_positions,
)

logger = logging.getLogger(__name__)

# bacformer SPECIAL_TOKENS_DICT — real protein rows carry PROT_EMB in the
# stored tensor's special_tokens_mask.
PROT_EMB_TOKEN_ID = 4

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def select_samples(
    ast_csv: Path, sample_column: str, label_column: str, max_samples: int | None = None
) -> list[str]:
    """Select samples in the AST CSV with a non-null binary label (no directory crawl)."""
    ast = pd.read_csv(ast_csv)
    ast = ast.dropna(subset=[label_column])
    samples = ast[sample_column].astype(str).drop_duplicates().tolist()
    if max_samples is not None:
        samples = samples[:max_samples]
    return samples


def load_pooled_rpob_vectors(
    genotype: pd.DataFrame,
    esm_store_dir: Path,
    *,
    pt_suffix: str = "_esm_embeddings.pt",
) -> tuple[np.ndarray, list[str]]:
    """Pull each sample's pooled ESM-C rpoB vector out of the embedding store.

    Returns
    -------
    tuple
        ``(matrix, sample_ids)`` — ``matrix`` is ``[n_valid, dim]`` and
        ``sample_ids`` the samples it covers (those whose ``.pt`` exists and
        whose rpoB flat index is within the stored protein rows).
    """
    vectors: list[np.ndarray] = []
    kept: list[str] = []
    n_missing_pt = 0
    n_out_of_range = 0
    for sample_id, row in genotype.iterrows():
        pt_path = esm_store_dir / f"{sample_id}{pt_suffix}"
        if not pt_path.exists():
            n_missing_pt += 1
            continue
        store = torch.load(pt_path, map_location="cpu")
        prot_emb = store["protein_embeddings"][0]
        special = store["special_tokens_mask"][0]
        protein_rows = prot_emb[special == PROT_EMB_TOKEN_ID]
        flat_index = int(row["rpob_flat_index"])
        if flat_index >= protein_rows.shape[0]:
            n_out_of_range += 1
            continue
        vectors.append(protein_rows[flat_index].float().numpy())
        kept.append(str(sample_id))
    if n_missing_pt or n_out_of_range:
        logger.warning(
            "pooled rpoB: %d samples missing .pt, %d with rpoB index past truncation",
            n_missing_pt,
            n_out_of_range,
        )
    matrix = np.vstack(vectors) if vectors else np.empty((0, 0))
    return matrix, kept


def masked_marginal_features(
    genotype: pd.DataFrame,
    reference: str,
    *,
    device: str,
    codons: list[int],
) -> tuple[np.ndarray, list[str]]:
    """Per-codon masked-marginal LLR features for predictor (2).

    For each sample, mask each panel codon in its own rpoB sequence and score
    ``log P(observed) - log P(wild-type)``. Wild-type sites score ~0; resistant
    substitutions score strongly negative if ESM-C "knows" the site.
    """
    # Imported lazily so predictors (1)/(3) run without loading the ESM model.
    from tl.embed.esm_residue_level import load_esmc_mlm, masked_marginals, substitution_llr

    model, tokenizer = load_esmc_mlm(device=device)
    wt_by_codon = {codon: reference[ref_index_for_codon(reference, codon)] for codon in codons}

    features: list[list[float]] = []
    kept: list[str] = []
    n_skipped = 0
    for sample_id, row in genotype.iterrows():
        seq = row["rpob_sequence"]
        positions = sample_codon_positions(seq, reference, codons)
        valid = {c: p for c, p in positions.items() if p is not None}
        if len(valid) != len(codons):
            n_skipped += 1
            continue
        log_probs = masked_marginals(
            model,
            tokenizer,
            seq,
            positions=list(valid.values()),
            device=device,
            expected_residues={p: seq[p] for p in valid.values()},
        )
        llrs = []
        for codon in codons:
            p = valid[codon]
            observed = seq[p]
            llrs.append(substitution_llr(log_probs[p], tokenizer, wt=wt_by_codon[codon], observed=observed))
        features.append(llrs)
        kept.append(str(sample_id))
    if n_skipped:
        logger.warning("masked-marginal: skipped %d samples with a gapped panel codon", n_skipped)
    return np.asarray(features, dtype=float), kept


def evaluate_predictor(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    test_size: float,
    standardize: bool,
) -> dict:
    """Fit logistic regression on a single stratified split and score it."""
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    if standardize:
        scaler = StandardScaler().fit(x_train)
        x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(x_train, y_train)
    proba = clf.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(y_test, proba)),
        "auprc": float(average_precision_score(y_test, proba)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features": int(features.shape[1]),
    }


def run_ladder(
    ast_csv: Path,
    parquet_dir: Path,
    esm_store_dir: Path,
    *,
    device: str,
    seed: int,
    test_size: float,
    skip_masked_marginal: bool,
    masked_marginal_codons: list[int],
    sample_column: str,
    label_column: str,
    max_samples: int | None = None,
) -> dict:
    """Run all three predictors and assemble the results dict."""
    reference = load_reference()
    sample_ids = select_samples(ast_csv, sample_column, label_column, max_samples)
    logger.info("Genotyping %d labelled samples", len(sample_ids))

    genotype = build_genotype_table(sample_ids, parquet_dir, reference)
    genotype = join_rifampin_label(genotype, ast_csv, sample_column=sample_column, label_column=label_column)
    genotype = genotype.dropna(subset=[label_column])
    # Drop ambiguous DST calls (e.g. 0.5 intermediate) — keep only clean binary labels.
    n_before = len(genotype)
    genotype = genotype[genotype[label_column].isin([0, 1])]
    if len(genotype) < n_before:
        logger.info("Dropped %d rows with ambiguous (non-0/1) %s label", n_before - len(genotype), label_column)
    genotype[label_column] = genotype[label_column].astype(int)
    logger.info("Genotyped + labelled: %d samples (%d resistant)", len(genotype), int(genotype[label_column].sum()))

    codon_cols = [c for c in genotype.columns if c.startswith("codon_")]
    results: dict = {
        "config": {
            "ast_csv": str(ast_csv),
            "parquet_dir": str(parquet_dir),
            "esm_store_dir": str(esm_store_dir),
            "seed": seed,
            "test_size": test_size,
            "label_column": label_column,
            "n_samples": int(len(genotype)),
            "n_resistant": int(genotype[label_column].sum()),
            "panel": [{"codon": c, "wt": wt, "alt": alt} for c, wt, alt in RRDR_PANEL],
        },
        "predictors": {},
    }

    # Predictor 1 — one-hot RRDR genotype (the ceiling).
    onehot = pd.get_dummies(genotype[codon_cols].astype(str))
    results["predictors"]["onehot_rrdr"] = evaluate_predictor(
        onehot.to_numpy(dtype=float),
        genotype[label_column].to_numpy(),
        seed=seed,
        test_size=test_size,
        standardize=False,
    )
    logger.info("Predictor 1 (one-hot RRDR): AUROC=%.4f", results["predictors"]["onehot_rrdr"]["auroc"])

    # Predictor 3 — frozen pooled ESM-C rpoB vector (the suspected failure).
    pooled, pooled_ids = load_pooled_rpob_vectors(genotype, esm_store_dir)
    if pooled.shape[0]:
        pooled_labels = genotype.loc[pooled_ids, label_column].to_numpy()
        results["predictors"]["pooled_esmc_rpob"] = evaluate_predictor(
            pooled, pooled_labels, seed=seed, test_size=test_size, standardize=True
        )
        logger.info("Predictor 3 (pooled ESM-C): AUROC=%.4f", results["predictors"]["pooled_esmc_rpob"]["auroc"])
    else:
        results["predictors"]["pooled_esmc_rpob"] = {"error": "no pooled vectors recovered"}
        logger.warning("Predictor 3 produced no vectors — check esm_store_dir / .pt suffix")

    # Predictor 2 — masked-marginal LLR (is the signal in ESM-C at all).
    if skip_masked_marginal:
        results["predictors"]["masked_marginal"] = {"skipped": True}
        logger.info("Predictor 2 (masked-marginal) skipped (--skip-masked-marginal)")
    else:
        feats, mm_ids = masked_marginal_features(genotype, reference, device=device, codons=masked_marginal_codons)
        if feats.shape[0]:
            mm_labels = genotype.loc[mm_ids, label_column].to_numpy()
            results["predictors"]["masked_marginal"] = evaluate_predictor(
                feats, mm_labels, seed=seed, test_size=test_size, standardize=True
            )
            results["predictors"]["masked_marginal"]["codons"] = masked_marginal_codons
            logger.info("Predictor 2 (masked-marginal): AUROC=%.4f", results["predictors"]["masked_marginal"]["auroc"])
        else:
            results["predictors"]["masked_marginal"] = {"error": "no masked-marginal features computed"}

    # Head-line: information lost to ESM-C's residue mean-pool.
    p1 = results["predictors"]["onehot_rrdr"].get("auroc")
    p3 = results["predictors"]["pooled_esmc_rpob"].get("auroc")
    if p1 is not None and p3 is not None:
        results["headline"] = {"auroc_onehot_minus_pooled": float(p1 - p3)}
        logger.info("HEAD-LINE: AUROC(one-hot) - AUROC(pooled) = %.4f", p1 - p3)

    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ast-csv", type=Path, required=True, help="TB binary_ast.csv (has Sample + rifampin).")
    parser.add_argument("--parquet-dir", type=Path, required=True, help="Dir of *_protein_sequences.parquet.")
    parser.add_argument("--esm-store-dir", type=Path, required=True, help="Dir of *_esm_embeddings.pt.")
    parser.add_argument("--output-json", type=Path, required=True, help="Where to write the results JSON.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for masked-marginal (default cpu).")
    parser.add_argument("--seed", type=int, default=0, help="Split seed.")
    parser.add_argument("--test-size", type=float, default=0.3, help="Held-out fraction (default 0.3).")
    parser.add_argument("--sample-column", type=str, default="Sample", help="Sample-id column in the AST CSV.")
    parser.add_argument(
        "--label-column", type=str, default=RIFAMPIN_COLUMN, help="Phenotype column (default rifampin)."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap the number of samples (quick login-node smoke; default: all).",
    )
    parser.add_argument(
        "--skip-masked-marginal",
        action="store_true",
        help="Run only predictors 1 and 3 (cheap; no ESM forward passes).",
    )
    parser.add_argument(
        "--masked-marginal-codons",
        type=str,
        default="panel",
        help="'panel' (the 4 canonical codons) or 'all' (full RRDR window), or a comma list of codon numbers.",
    )
    args = parser.parse_args()

    if args.masked_marginal_codons == "panel":
        codons = [codon for codon, _wt, _alt in RRDR_PANEL]
    elif args.masked_marginal_codons == "all":
        from snp_embeddings.rpob_genotype import RRDR_FIRST_CODON, RRDR_LAST_CODON

        codons = list(range(RRDR_FIRST_CODON, RRDR_LAST_CODON + 1))
    else:
        codons = [int(c) for c in args.masked_marginal_codons.split(",")]

    results = run_ladder(
        args.ast_csv,
        args.parquet_dir,
        args.esm_store_dir,
        device=args.device,
        seed=args.seed,
        test_size=args.test_size,
        skip_masked_marginal=args.skip_masked_marginal,
        masked_marginal_codons=codons,
        sample_column=args.sample_column,
        label_column=args.label_column,
        max_samples=args.max_samples,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as handle:
        json.dump(results, handle, indent=2)
    logger.info("Wrote %s", args.output_json)
    json.dump(results, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

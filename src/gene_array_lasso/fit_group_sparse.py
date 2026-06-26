r"""Step D — fit group-sparse penalised models on the pangenome-aligned frozen-ESM-C array.

Loads the Step C array (``X.npz`` + ``samples.csv`` + ``genes.csv``) and fits a **sparse-group lasso /
group elastic net** (``groupyr.LogisticSGL``) with **one 960-dim group per Panaroo gene**: the penalty
zeros whole genes it doesn't need and keeps the few that carry signal. The de-risk question is whether
that recovers the known causal family (e.g. *tet* efflux for tetracycline, mgrB/pmrB for colistin) and
beats the genome **mean-pool** baseline.

Pipeline: split by the reused kleb_ast folds (``train`` / ``validate`` / ``evaluate``); standardise on
train; grid over ``(l1_ratio, alpha)`` choosing on **validate**; report AUROC/AUPRC on the in-run
**evaluate** test. ``l1_ratio`` exposes A1 (sparse-group, higher l1) vs A2 (group elastic net, lower l1).
Selected genes = per-gene coefficient L2 norm (non-zero), ranked and annotated.

groupyr requires **dense** X (sparse is rejected), so each split is densified — run on a himem node.

Example
-------
``uv run python src/gene_array_lasso/fit_group_sparse.py --drug tetracycline \\
    --array-dir .../gene_arrays/tetracycline_p5 --out-dir .../fits/tetracycline_p5``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from groupyr import LogisticSGL
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

EMB_DIM = 960
# Causal-family name fragments per drug (lower-cased substring match against gene + annotation).
CAUSAL_HINTS = {
    "tetracycline": ["tet"],
    "colistin": ["mgrb", "pmrb", "pmra", "phoq", "phop", "arnt", "eptb"],
    "imipenem": ["kpc", "oxa", "ndm", "vim", "imp", "bla", "ompk", "carbapenem"],
    "meropenem": ["kpc", "oxa", "ndm", "vim", "imp", "bla", "ompk", "carbapenem"],
    "ciprofloxacin": ["gyra", "gyrb", "parc", "pare", "qnr"],
}


def load_array(array_dir: Path, drug: str) -> tuple[sparse.csr_matrix, pd.DataFrame, pd.DataFrame]:
    """Load ``X.npz`` + ``samples.csv`` + ``genes.csv`` and drop label-NaN samples."""
    X = sparse.load_npz(array_dir / "X.npz").tocsr()
    samples = pd.read_csv(array_dir / "samples.csv")
    genes = pd.read_csv(array_dir / "genes.csv")
    keep = samples[drug].notna().values
    return X[keep], samples[keep].reset_index(drop=True), genes


def split_indices(samples: pd.DataFrame) -> dict[str, np.ndarray]:
    """Row indices for each ``train_val_eval`` split."""
    return {s: np.where(samples["train_val_eval"].values == s)[0] for s in ("train", "validate", "evaluate")}


def _dense(X: sparse.csr_matrix, idx: np.ndarray) -> np.ndarray:
    """Densify the selected rows (groupyr needs dense X). float32 halves memory — vital at >1%."""
    return np.asarray(X[idx].todense(), dtype=np.float32)


def gene_groups(n_genes: int) -> list[np.ndarray]:
    """Contiguous 960-wide index blocks — one group per gene."""
    return [np.arange(g * EMB_DIM, (g + 1) * EMB_DIM) for g in range(n_genes)]


def mean_pool_baseline(Xtr, ytr, Xev, yev, n_genes: int) -> dict:
    """Genome mean-pool comparator: average present gene-blocks → 960-vec → plain logistic regression."""
    def pool(Xd: np.ndarray) -> np.ndarray:
        blocks = Xd.reshape(Xd.shape[0], n_genes, EMB_DIM)
        present = (np.abs(blocks).sum(axis=2) > 0).sum(axis=1, keepdims=True)
        return blocks.sum(axis=1) / np.clip(present, 1, None)
    sc = StandardScaler().fit(pool(Xtr))
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(pool(Xtr)), ytr)
    p = lr.predict_proba(sc.transform(pool(Xev)))[:, 1]
    return {"auroc": float(roc_auc_score(yev, p)), "auprc": float(average_precision_score(yev, p))}


def fit_sgl(Xtr, ytr, Xva, yva, groups, l1_ratios, alphas) -> tuple[LogisticSGL, dict, list[dict]]:
    """Grid over (l1_ratio, alpha), select on validation AUROC; return the best refit model + trace."""
    best, best_model, trace = None, None, []
    for l1 in l1_ratios:
        for a in alphas:
            m = LogisticSGL(groups=groups, l1_ratio=l1, alpha=a, max_iter=2000, tol=1e-3,
                            suppress_solver_warnings=True)
            m.fit(Xtr, ytr)
            p = m.predict_proba(Xva)[:, 1]
            auroc = float(roc_auc_score(yva, p)) if len(np.unique(yva)) > 1 else float("nan")
            coef = np.asarray(m.coef_).ravel()
            nzg = int(sum(np.linalg.norm(coef[g]) > 1e-8 for g in groups))
            rec = {"l1_ratio": l1, "alpha": a, "val_auroc": auroc, "n_selected_genes": nzg}
            trace.append(rec)
            if best is None or (auroc == auroc and auroc > best):
                best, best_model = auroc, m
    return best_model, {"best_val_auroc": best}, trace


def selected_genes(model: LogisticSGL, genes: pd.DataFrame, groups) -> pd.DataFrame:
    """Rank genes by coefficient L2 norm (non-zero only), annotated."""
    coef = np.asarray(model.coef_).ravel()
    norms = np.array([np.linalg.norm(coef[g]) for g in groups])
    out = genes.copy()
    out["coef_norm"] = norms
    out = out[out["coef_norm"] > 1e-8].sort_values("coef_norm", ascending=False).reset_index(drop=True)
    return out


def flag_causal(sel: pd.DataFrame, drug: str) -> list[dict]:
    """Find causal-family hits among the selected genes (by gene/annotation substring)."""
    hints = CAUSAL_HINTS.get(drug, [])
    if not hints or sel.empty:
        return []
    text = (sel["gene"].astype(str) + " " + sel.get("annotation", "").astype(str)).str.lower()
    hit = sel[text.apply(lambda t: any(h in t for h in hints))]
    return [{"rank": int(i), "gene": r["gene"], "annotation": r.get("annotation", ""),
             "coef_norm": float(r["coef_norm"]), "prevalence": float(r["prevalence"])}
            for i, r in hit.iterrows()]


def run(array_dir: Path, drug: str, out_dir: Path, l1_ratios: list[float], alphas: list[float]) -> None:
    """Fit the group-sparse models for one drug and write results + selected genes."""
    X, samples, genes = load_array(array_dir, drug)
    n_genes = len(genes)
    y = samples[drug].astype(int).values
    idx = split_indices(samples)
    groups = gene_groups(n_genes)
    print(f"[{drug}] X={X.shape} genes={n_genes} splits=" +
          ", ".join(f"{k}={len(v)}" for k, v in idx.items()))

    Xtr, ytr = _dense(X, idx["train"]), y[idx["train"]]
    Xva, yva = _dense(X, idx["validate"]), y[idx["validate"]]
    Xev, yev = _dense(X, idx["evaluate"]), y[idx["evaluate"]]

    scaler = StandardScaler().fit(Xtr)
    # Keep float32 through scaling (StandardScaler upcasts to float64) so the >1% dense fit fits in himem.
    Xtr_s = scaler.transform(Xtr).astype(np.float32)
    Xva_s = scaler.transform(Xva).astype(np.float32)
    Xev_s = scaler.transform(Xev).astype(np.float32)

    print(f"[{drug}] mean-pool baseline …")
    baseline = mean_pool_baseline(Xtr, ytr, Xev, yev, n_genes)

    print(f"[{drug}] SGL grid l1_ratios={l1_ratios} alphas={alphas} …")
    model, best, trace = fit_sgl(Xtr_s, ytr, Xva_s, yva, groups, l1_ratios, alphas)
    p_ev = model.predict_proba(Xev_s)[:, 1]
    sgl_eval = {"auroc": float(roc_auc_score(yev, p_ev)), "auprc": float(average_precision_score(yev, p_ev))}
    sel = selected_genes(model, genes, groups)
    causal = flag_causal(sel, drug)

    out_dir.mkdir(parents=True, exist_ok=True)
    sel.to_csv(out_dir / "selected_genes.csv", index=False)
    result = {
        "drug": drug,
        "n_genes": n_genes,
        "n_samples": int(X.shape[0]),
        "splits": {k: int(len(v)) for k, v in idx.items()},
        "best_l1_ratio": model.l1_ratio,
        "best_alpha": model.alpha,
        "best_val_auroc": best["best_val_auroc"],
        "sgl_evaluate": sgl_eval,
        "mean_pool_evaluate": baseline,
        "delta_auroc_vs_mean_pool": sgl_eval["auroc"] - baseline["auroc"],
        "n_selected_genes": int(len(sel)),
        "top_selected": sel.head(15).to_dict(orient="records"),
        "causal_family_hits": causal,
        "grid_trace": trace,
    }
    (out_dir / "results.json").write_text(json.dumps(result, indent=2, default=float))
    print(json.dumps({k: result[k] for k in
                      ("sgl_evaluate", "mean_pool_evaluate", "delta_auroc_vs_mean_pool",
                       "best_l1_ratio", "best_alpha", "n_selected_genes", "causal_family_hits")}, indent=2))
    print(f"Wrote {out_dir}/results.json + selected_genes.csv")


def main() -> None:
    """Parse CLI args and run the fit."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drug", required=True)
    parser.add_argument("--array-dir", type=Path, required=True, help="Step C output dir (X.npz + sidecars).")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--l1-ratios", type=float, nargs="+", default=[0.5, 0.9],
                        help="Sparse-group mix: ~0.9 = A1 sparse-group lasso, ~0.5 = A2 group elastic net.")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.1, 0.03, 0.01],
                        help="Penalty strengths (selected on validation).")
    args = parser.parse_args()
    run(args.array_dir, args.drug, args.out_dir, args.l1_ratios, args.alphas)


if __name__ == "__main__":
    main()

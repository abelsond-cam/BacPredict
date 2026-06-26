r"""Step C/F — sparse-group lasso on the pangenome embedding array, via skglm (sparse, no densify).

The engine, migrated from groupyr (which can't scale — dense O(n_groups²) prox): skglm's
``GeneralizedLinearEstimator(QuadraticGroup, WeightedL1GroupL2, GroupBCD)`` on **scipy CSR X** — absent-gene
zero blocks never materialise (the dominant memory lever). One 960-dim group per gene
([`group_spec.uniform_block_layout`](group_spec.py)); the group-L2 term zeros whole genes, the per-feature L1
adds within-gene sparsity. The de-risk question: does it beat the genome mean-pool and select the causal
family (*tet* / mgrB-pmrB)?

skglm 0.5 notes (verified, pinned): ``LogisticGroup`` has **no sparse path** (lacks ``gradient_g_sparse``) and
``GroupProxNewton`` rejects sparse — so the sparse first-pass datafit is **``QuadraticGroup``** (squared loss;
the handover §4 first-pass and the design doc's "fast Gaussian-on-binary" — score = Xβ → AUROC). ``GroupBCD``
needs ``ws_strategy="fixpoint"`` for ``WeightedL1GroupL2`` (no ``subdiff_distance``). Logistic is a later
upgrade (dense X, or a skglm patch adding the sparse group-gradient).

``--smoke`` runs the Stage-1 gates on the first ``--n-genomes`` samples: gap closes / float32==float64 /
sparse==dense (group indexing is the separate unit test). Full mode fits a warm-started alpha path and reports
the de-risk read.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from skglm import GeneralizedLinearEstimator
from skglm.datafits import QuadraticGroup
from skglm.penalties import WeightedL1GroupL2
from skglm.solvers import GroupBCD
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from gene_array_lasso.group_spec import EMB_DIM, GroupLayout, uniform_block_layout

CAUSAL_HINTS = {
    "tetracycline": ["tet"],
    "colistin": ["mgrb", "pmrb", "pmra", "phoq", "phop", "arnt", "eptb"],
    "imipenem": ["kpc", "oxa", "ndm", "vim", "imp", "bla", "ompk", "carbapenem"],
}


def load_array(array_dir: Path, drug: str) -> tuple[sparse.csr_matrix, pd.DataFrame, pd.DataFrame]:
    """Load CSR ``X.npz`` + ``samples.csv`` + ``genes.csv``; drop label-NaN rows. X stays sparse."""
    X = sparse.load_npz(array_dir / "X.npz").tocsr()
    samples = pd.read_csv(array_dir / "samples.csv")
    genes = pd.read_csv(array_dir / "genes.csv")
    keep = samples[drug].notna().values
    # Avoid doubling RAM: the builder writes canonical CSR, so skip sum_duplicates (allocates a full copy)
    # and skip the X[keep] copy when no rows are dropped. sort_indices is cheap (in-place) insurance.
    Xk = X if keep.all() else X[keep].tocsr()
    Xk.sort_indices()
    return Xk, samples[keep].reset_index(drop=True), genes


def build_estimator(layout: GroupLayout, alpha: float, tau: float, max_iter: int, tol: float) -> GeneralizedLinearEstimator:
    """The skglm sparse-group-lasso estimator. ``tau`` = per-feature L1 weight (small = conservative)."""
    wg = np.sqrt(layout.block if layout.block else 1.0) * np.ones(layout.n_groups)
    wf = tau * np.ones(layout.n_features)
    return GeneralizedLinearEstimator(
        datafit=QuadraticGroup(grp_ptr=layout.grp_ptr, grp_indices=layout.grp_indices),
        penalty=WeightedL1GroupL2(alpha=alpha, weights_groups=wg, weights_features=wf,
                                  grp_ptr=layout.grp_ptr, grp_indices=layout.grp_indices),
        # fit_intercept=False: skglm's QuadraticGroup sparse path mis-shapes w vs the penalty weights when an
        # intercept is added. We instead center y (below) — the offset is irrelevant to AUROC ranking anyway.
        solver=GroupBCD(max_iter=max_iter, tol=tol, ws_strategy="fixpoint", fit_intercept=False, verbose=0),
    )


def alpha_max_group(X: sparse.csr_matrix, y: np.ndarray, layout: GroupLayout) -> float:
    """Smallest alpha giving an all-zero group solution (group-L2 term, quadratic loss with intercept)."""
    r = y - y.mean()
    n = X.shape[0]
    wg = np.sqrt(layout.block if layout.block else 1.0)
    amax = 0.0
    XT = X.T.tocsr()
    for g in range(layout.n_groups):
        cols = layout.columns(g)
        gnorm = float(np.linalg.norm(XT[cols] @ r))
        amax = max(amax, gnorm / (n * wg))
    return amax


def fit_path(X, y, layout, alphas, tau, max_iter, tol) -> list[dict]:
    """Warm-started decreasing-alpha path; returns per-alpha coef, convergence flag, n selected groups."""
    out = []
    yc = np.asarray(y, dtype=float)
    yc = yc - yc.mean()  # center the target so fit_intercept=False is sound (offset irrelevant to ranking)
    # One estimator reused down the alpha path, warm-started from the previous (larger-alpha) solution.
    est = build_estimator(layout, float(alphas[0]), tau, max_iter, tol)
    est.solver.warm_start = True
    for a in alphas:
        est.penalty.alpha = float(a)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            est.fit(X, yc)
            converged = not any(issubclass(c.category, ConvergenceWarning) for c in caught)
        coef = np.asarray(est.coef_).ravel()
        gnorms = np.array([np.linalg.norm(coef[layout.columns(g)]) for g in range(layout.n_groups)])
        out.append({"alpha": float(a), "converged": bool(converged),
                    "n_selected": int((gnorms > 1e-8).sum()), "coef": coef,
                    "intercept": float(np.atleast_1d(est.intercept_)[0]) if est.intercept_ is not None else 0.0})
    return out


def score(coef, intercept, X, y) -> dict:
    """AUROC/AUPRC of the linear score Xβ+b (squared-loss fit → score is a ranking)."""
    s = X @ coef + intercept
    return {"auroc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s))}


def mean_pool_baseline(Xtr, ytr, Xev, yev, layout) -> dict:
    """Genome mean-pool comparator: mean of present gene-blocks → 960-vec → logistic regression."""
    def pool(X):
        n_genes = layout.n_groups
        B = np.asarray(X.todense()).reshape(X.shape[0], n_genes, EMB_DIM)
        present = (np.abs(B).sum(2) > 0).sum(1, keepdims=True)
        return B.sum(1) / np.clip(present, 1, None)
    sc = StandardScaler().fit(pool(Xtr))
    lr = LogisticRegression(max_iter=2000).fit(sc.transform(pool(Xtr)), ytr)
    p = lr.predict_proba(sc.transform(pool(Xev)))[:, 1]
    return {"auroc": float(roc_auc_score(yev, p)), "auprc": float(average_precision_score(yev, p))}


def selected_genes(coef, genes, layout) -> pd.DataFrame:
    """Genes ranked by coefficient L2 norm (non-zero), annotated."""
    norms = np.array([np.linalg.norm(coef[layout.columns(g)]) for g in range(layout.n_groups)])
    out = genes.copy()
    out["coef_norm"] = norms
    return out[out["coef_norm"] > 1e-8].sort_values("coef_norm", ascending=False).reset_index(drop=True)


def causal_hits(sel: pd.DataFrame, drug: str) -> list[dict]:
    """Causal-family hits among selected genes (substring match on gene + annotation)."""
    hints = CAUSAL_HINTS.get(drug, [])
    if not hints or sel.empty:
        return []
    txt = (sel["gene"].astype(str) + " " + sel.get("annotation", "").astype(str)).str.lower()
    hit = sel[txt.apply(lambda t: any(h in t for h in hints))]
    return [{"rank": int(i), "gene": r["gene"], "annotation": r.get("annotation", ""),
             "coef_norm": float(r["coef_norm"]), "prevalence": float(r["prevalence"])} for i, r in hit.iterrows()]


def run_smoke(array_dir: Path, drug: str, n_genomes: int, tau: float, max_iter: int, tol: float) -> None:
    """Stage-1 gates on the first ``n_genomes`` samples: gap-closes / float32==float64 / sparse==dense."""
    X, samples, genes = load_array(array_dir, drug)
    X, y = X[:n_genomes], samples[drug].astype(int).values[:n_genomes]
    layout = uniform_block_layout(len(genes), EMB_DIM)
    nnz = int(X.nnz)
    print(f"[smoke {drug}] X={X.shape} genes={layout.n_groups} nnz={nnz:,}")
    amax = alpha_max_group(X, y, layout)
    alpha = 0.3 * amax
    print(f"alpha_max={amax:.4g}  fitting at alpha={alpha:.4g}")

    res = fit_path(X.astype(np.float64), y, layout, [alpha], tau, max_iter, tol)[0]
    gate1 = res["converged"]
    coef64 = res["coef"]
    cf = fit_path(X.astype(np.float32), y, layout, [alpha], tau, max_iter, tol)[0]["coef"]
    cd = fit_path(np.asarray(X.todense(), dtype=np.float64), y, layout, [alpha], tau, max_iter, tol)[0]["coef"]

    def support(c):  # which genes are selected (group-norm > 0)
        return frozenset(g for g in range(layout.n_groups) if np.linalg.norm(c[layout.columns(g)]) > 1e-8)
    scale = max(np.abs(coef64).max(), 1e-12)
    # "doesn't move the solution" = identical selected support AND coefficients close in RELATIVE terms.
    f32_rel = float(np.abs(coef64 - cf).max() / scale)
    gate3 = bool(support(coef64) == support(cf) and f32_rel < 1e-2)
    gate4 = bool(np.allclose(coef64, cd, atol=1e-6))
    print(f"GATE1 duality gap closes (not max_iter): {gate1}")
    print("GATE2 group indexing: see tests/gene_array_lasso/test_group_indexing.py (run pytest)")
    print(f"GATE3 float32 doesn't move solution: {gate3}  (same support={support(coef64)==support(cf)}, "
          f"max rel|diff|={f32_rel:.2e})")
    print(f"GATE4 sparse==dense coef: {gate4}  (max|diff| {np.abs(coef64 - cd).max():.2e})")
    sel = selected_genes(coef64, genes, layout)
    print(f"selected {len(sel)} genes; top: {sel['gene'].head(8).tolist()}")
    print(f"causal hits: {causal_hits(sel, drug)}")
    print(f"ALL GATES PASS (1,3,4; 2 via pytest): {gate1 and gate3 and gate4}")


def run_fit(array_dir: Path, drug: str, out_dir: Path, n_alphas: int, tau: float, max_iter: int, tol: float) -> None:
    """Full fit: warm-started alpha path, select on validate, report on the in-run evaluate test."""
    X, samples, genes = load_array(array_dir, drug)
    y = samples[drug].astype(int).values
    sp = samples["train_val_eval"].values
    tr, va, ev = (np.where(sp == s)[0] for s in ("train", "validate", "evaluate"))
    layout = uniform_block_layout(len(genes), EMB_DIM)
    Xtr = X[tr].astype(np.float32)
    print(f"[{drug}] X={X.shape} genes={layout.n_groups} nnz={X.nnz:,} splits tr/va/ev={len(tr)}/{len(va)}/{len(ev)}")

    amax = alpha_max_group(Xtr, y[tr], layout)
    alphas = np.geomspace(amax, amax * 1e-2, n_alphas)
    path = fit_path(Xtr, y[tr], layout, alphas, tau, max_iter, tol)
    # pick alpha by validation AUROC
    best = max(path, key=lambda r: score(r["coef"], r["intercept"], X[va], y[va])["auroc"])
    sgl_ev = score(best["coef"], best["intercept"], X[ev], y[ev])
    baseline = mean_pool_baseline(X[tr], y[tr], X[ev], y[ev], layout)
    sel = selected_genes(best["coef"], genes, layout)
    causal = causal_hits(sel, drug)

    out_dir.mkdir(parents=True, exist_ok=True)
    sel.to_csv(out_dir / "selected_genes.csv", index=False)
    result = {
        "drug": drug, "engine": "skglm-QuadraticGroup-WeightedL1GroupL2-GroupBCD", "n_genes": layout.n_groups,
        "nnz": int(X.nnz), "splits": {"train": len(tr), "validate": len(va), "evaluate": len(ev)},
        "best_alpha": best["alpha"], "best_alpha_converged": best["converged"], "tau": tau,
        "n_selected_genes": int(len(sel)), "sgl_evaluate": sgl_ev, "mean_pool_evaluate": baseline,
        "delta_auroc_vs_mean_pool": sgl_ev["auroc"] - baseline["auroc"],
        "top_selected": sel.head(15).to_dict(orient="records"), "causal_family_hits": causal,
        "path": [{k: r[k] for k in ("alpha", "converged", "n_selected")} for r in path],
    }
    (out_dir / "results.json").write_text(json.dumps(result, indent=2, default=float))
    print(json.dumps({k: result[k] for k in ("sgl_evaluate", "mean_pool_evaluate", "delta_auroc_vs_mean_pool",
                                              "best_alpha", "n_selected_genes", "causal_family_hits")}, indent=2))
    print(f"Wrote {out_dir}/results.json")


def main() -> None:
    """CLI: ``--smoke`` for the Stage-1 gates, else the full warm-started fit."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--drug", required=True)
    p.add_argument("--array-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--smoke", action="store_true", help="Run Stage-1 gates on --n-genomes samples.")
    p.add_argument("--n-genomes", type=int, default=100)
    p.add_argument("--n-alphas", type=int, default=20)
    p.add_argument("--tau", type=float, default=0.05, help="Per-feature L1 weight (small = conservative; "
                   "embeddings have correlated dims, aggressive within-group L1 over-prunes).")
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--tol", type=float, default=1e-8)
    args = p.parse_args()
    if args.smoke:
        run_smoke(args.array_dir, args.drug, args.n_genomes, args.tau, args.max_iter, args.tol)
    else:
        if args.out_dir is None:
            p.error("--out-dir required for a full fit")
        run_fit(args.array_dir, args.drug, args.out_dir, args.n_alphas, args.tau, args.max_iter, args.tol)


if __name__ == "__main__":
    main()

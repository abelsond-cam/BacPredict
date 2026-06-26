"""Step A — verify the zero-block absence-encoding premise.

The sparse-group-lasso design encodes an *absent* gene as an all-zero 960-dim block. That is only
defensible if real gene embeddings are never near zero — i.e. a zero block is an out-of-distribution
*fabricated* point that the group-L2 penalty ignores for selection (zero group norm), **not** a plausible
"no signal" vector the model could confuse with a real gene.

So this measures **the minimum L2 norm of any real protein embedding** across a sample of the ESM-C store
(plus the distribution). The one number that matters — the global minimum — goes in the methods text. If it
is ≫ 0 (orders of magnitude above 0), the zero-block encoding is justified.

Run anywhere (loads a handful of small ``.pt`` files; login node is fine).

Example
-------
``uv run python src/gene_array_lasso/verify_embedding_norms.py --n-samples 50``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

RDS_ROOT = Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david")
ESM_DIR_DEFAULT = RDS_ROOT / "processed" / "klebsiella_esm_embeddings"
SPLITS_DEFAULT = RDS_ROOT / "processed" / "gene_array_lasso" / "panaroo_input_tsv" / "colistin_splits.csv"


def real_protein_norms(store: dict) -> np.ndarray:
    """L2 norm of each *real* (non-padded) protein embedding in one sample's store."""
    emb = store.get("prot_embeddings", store.get("protein_embeddings"))
    emb = emb[0].to(torch.float32)
    mask = store.get("attention_mask")
    if mask is not None:
        m = mask.reshape(-1).bool()
        if m.numel() == emb.shape[0]:
            emb = emb[m]
    return torch.linalg.vector_norm(emb, dim=1).numpy()


def main() -> None:
    """Sample the store, report min / percentile / max real-protein embedding L2 norm."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--esm-dir", type=Path, default=ESM_DIR_DEFAULT)
    parser.add_argument("--splits-csv", type=Path, default=SPLITS_DEFAULT,
                        help="Sample IDs to draw from (Sample column); falls back to globbing the store.")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.splits_csv.exists():
        ids = pd.read_csv(args.splits_csv)["Sample"].astype(str).tolist()
    else:
        ids = [p.name.replace("_esm_embeddings.pt", "") for p in args.esm_dir.glob("*_esm_embeddings.pt")]
    if not ids:
        raise SystemExit(f"No samples found via {args.splits_csv} or {args.esm_dir}")
    pick = rng.choice(ids, size=min(args.n_samples, len(ids)), replace=False)

    all_norms: list[np.ndarray] = []
    used = 0
    for sid in pick:
        pt = args.esm_dir / f"{sid}_esm_embeddings.pt"
        if not pt.exists():
            continue
        store = torch.load(pt, map_location="cpu", weights_only=False)
        all_norms.append(real_protein_norms(store))
        used += 1
    norms = np.concatenate(all_norms)

    gmin = float(norms.min())
    print(f"samples used: {used}   real proteins: {len(norms):,}")
    print(f"L2 norm  min={gmin:.4f}  p0.1={np.percentile(norms, 0.1):.4f}  "
          f"p1={np.percentile(norms, 1):.4f}  median={np.median(norms):.4f}  max={norms.max():.4f}")
    print(f"proteins with norm < 1.0: {(norms < 1.0).sum()}   < 0.1: {(norms < 0.1).sum()}")
    verdict = "JUSTIFIED — min norm >> 0; a zero block is out-of-distribution" if gmin > 1.0 else \
              "CHECK — some real embeddings are small; zero-block encoding may be ambiguous"
    print(f"zero-block absence encoding: {verdict}")


if __name__ == "__main__":
    main()

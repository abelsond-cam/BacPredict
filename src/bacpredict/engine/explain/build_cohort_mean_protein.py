r"""CPU — average every protein embedding in a cohort into one baseline vector.

Integrated gradients needs a baseline: the input the attribution is measured *against*. Attributions
sum to ``f(genome) - f(baseline)``, so the baseline defines what every score means.

Why not the zero vector
-----------------------
⛔ The obvious baseline is zeros, and on this architecture it measures nothing. The checkpoints'
bundled ``modeling_bacformer.py`` runs ``nn.LayerNorm`` as the **first** operation on the raw protein
embeddings (``:642``, ``:666``), so ``f(alpha * x) == f(x)`` exactly above ``alpha ~ 5e-5``. The model
is degree-0 homogeneous in its input: ``input_x_gradient`` is identically zero by Euler's theorem,
and integrated gradients' integrand goes as ``1/alpha``, concentrating essentially the whole integral
below the smallest quadrature node any affordable ``n_steps`` reaches. The answer then swings with
``n_steps`` instead of converging — measured ``-0.0105`` at ``n_steps=50`` and ``-10.51`` at
``n_steps=200`` against a target gap of ``+4.5516`` (job 41324596).

What this builds instead
------------------------
**One vector**: the arithmetic mean of every protein embedding in the cohort — all proteins, all
genomes, a single ``[width]`` tensor. It is used as a *baseline genome*, by copying it into every
token position, so the path integrated along is

    a genome of T identical average proteins  ->  this actual genome

and a protein's attribution is its share of that difference. Nothing is normalised *by* the mean;
it is the origin, not a divisor.

Measured on the 100-genome TB cohort (407,819 proteins): ``L2 = 0.6544``, ``std = 0.02113``, and the
model's logit at that baseline is **genome-independent** — ``-2.7569``, ``-2.7571``, ``-2.7586``,
``-2.7607`` over genomes of 4,056 to 4,097 proteins spanning both AST labels and different lineages
(job 41326249). Two consequences:

- Every genome's attributions sum to ``base_logit - baseline_logit``, i.e. to its logit plus ~2.758,
  against a shared origin. Scores are therefore already comparable **across** genomes, and a
  per-genome z-normalisation would destroy that comparability rather than create it.
- The baseline sits mildly susceptible (``-2.76`` is ``P(R) = 0.060``), so "a generic genome" is a
  weak non-resistant prior, which is the sensible reference for a resistance call.

What this baseline is NOT
-------------------------
The counterfactual per gene is "replace this protein with a **generic average protein**", not
"replace mutant ``rpoB`` with **wild-type** ``rpoB``". The latter is the validation substitution, and
it needs a per-gene baseline built from genomes carrying no variant in that gene. For one named
determinant that is easy; for a whole genome it needs cross-genome identity for every protein,
including the majority that Bakta leaves as genome-specific locus tags — which is what the nuna gene
families supply.

Examples
--------
Build the TB baseline over the attribution cohort::

    python -m bacpredict.engine.explain.build_cohort_mean_protein \\
        --organism tb --cohort-csv attribution_cohort_rifampin_n100.csv \\
        --out "$B/train_tb_ast/attribution/cohort_mean_protein.pt"
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch

from bacpredict.engine.config import store_paths

logger = logging.getLogger(__name__)


def protein_embeddings_of(store: dict) -> torch.Tensor:
    """Pull the ``[n_proteins, width]`` protein matrix out of one loaded ``.pt`` store.

    The two embedding stores disagree on the key — ``prot_embeddings`` in the per-task AST stores,
    ``protein_embeddings`` in some older files — and on whether a leading batch dimension is present.
    This normalises both.

    Parameters
    ----------
    store : dict
        A loaded ``{Sample}_esm_embeddings.pt``.

    Returns
    -------
    torch.Tensor
        ``[n_proteins, width]``.

    Raises
    ------
    KeyError
        If neither embedding key is present.
    """
    for key in ("prot_embeddings", "protein_embeddings"):
        if key in store:
            embeddings = store[key]
            return embeddings.squeeze(0) if embeddings.dim() == 3 else embeddings
    raise KeyError(f"no protein embeddings in store; keys were {sorted(store)}")


def cohort_mean_protein(esm_paths: list[Path]) -> tuple[torch.Tensor, int]:
    """Average every protein embedding across a list of genome stores.

    Accumulates in float64 — a float32 running sum over ~400k vectors loses low-order bits, and this
    vector is the origin every attribution is measured from, so it is worth the exactness.

    Parameters
    ----------
    esm_paths : list of pathlib.Path
        One ``{Sample}_esm_embeddings.pt`` per genome.

    Returns
    -------
    mean : torch.Tensor
        ``[width]`` float32 — the mean protein embedding.
    n_proteins : int
        How many proteins were averaged.

    Raises
    ------
    ValueError
        If no genome store could be read.
    """
    total: torch.Tensor | None = None
    n_proteins = 0
    for index, path in enumerate(esm_paths):
        if not path.exists():
            logger.warning("missing store, skipping: %s", path.name)
            continue
        embeddings = protein_embeddings_of(torch.load(path, map_location="cpu", weights_only=False)).double()
        total = embeddings.sum(0) if total is None else total + embeddings.sum(0)
        n_proteins += embeddings.shape[0]
        if index % 25 == 0:
            logger.info("%d/%d genomes, %d proteins", index, len(esm_paths), n_proteins)
    if total is None:
        raise ValueError("no genome stores could be read; the baseline would be undefined")
    return (total / n_proteins).float(), n_proteins


def main() -> None:
    """CLI entry point — build one cohort-mean protein vector and write it beside a JSON manifest."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--organism", required=True, choices=("tb", "kp"), help="Organism key.")
    parser.add_argument("--cohort-csv", type=Path, required=True,
                        help="Cohort table from select_attribution_cohort; its Sample column picks the genomes.")
    parser.add_argument("--out", type=Path, required=True, help="Where to write the [width] baseline tensor.")
    parser.add_argument("--esm-dir", type=Path, default=None, help="Override the ESM store directory.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

    paths = store_paths(args.organism)
    esm_dir = Path(args.esm_dir or paths.esm_dir)
    samples = pd.read_csv(args.cohort_csv)["Sample"].astype(str).tolist()
    logger.info("averaging proteins over %d genomes from %s", len(samples), esm_dir)

    mean, n_proteins = cohort_mean_protein([esm_dir / f"{sample}{paths.esm_suffix}" for sample in samples])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mean, args.out)

    manifest = {
        "organism": args.organism,
        "cohort_csv": str(args.cohort_csv),
        "esm_dir": str(esm_dir),
        "n_genomes": len(samples),
        "n_proteins": n_proteins,
        "width": int(mean.shape[0]),
        "l2_norm": float(mean.norm()),
        "std": float(mean.std()),
    }
    args.out.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    logger.info("%d proteins over %d genomes -> %s  L2=%.4f std=%.5f",
                n_proteins, len(samples), args.out, manifest["l2_norm"], manifest["std"])


if __name__ == "__main__":
    main()

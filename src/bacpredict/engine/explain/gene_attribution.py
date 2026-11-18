r"""GPU — ablate one protein at a time and record the shift in the model's logit.

Each genome is a sequence of protein embeddings. Replacing one protein's embedding with a zero
vector and re-running the model gives that gene's attribution: how much the model's output depended
on it. Over a genome this is one forward pass per protein, which is the whole cost of the method.

What is stored, and why it is the logit
---------------------------------------
captum returns ``f(x) - f(x with the gene ablated)``. Running that on the model's **probability**
gives a signed score in ``[-1, +1]``, which reads naturally — but it saturates. A model that is
confident at ``P = 1.000`` does not move when one gene of four thousand is removed, and the shift
vanishes below representable precision: the published Bacformer run found **1,895 of 4,242**
Salmonella genomes had zero attribution variance for exactly this reason, and a genome with zero
variance contributes nothing to any statistic.

So the target here is the **logit**, which never saturates, and each genome's base logit is stored
beside it. That loses nothing, because the probability-scale score comes back exactly:

.. code-block:: python

    delta_p = sigmoid(base_logit) - sigmoid(base_logit - attribution_logit)

One run, both scales, no saturation in the stored quantity.

Precision
---------
``--dtype bf16`` runs the weights as deployed. bf16 carries roughly three significant digits, so if a
base logit is ~8 and one gene shifts it by ~0.001, the difference may not be representable and
attributions come back quantised or flatly zero. ``--dtype fp32`` runs the same weights in higher-
precision arithmetic as a check. This is **not** the ``dtype="auto"`` trap: that was about loading
weights as fp32 for training and evaluation, where it measurably hurt. Here the weights are the
deployed ones either way, and only the arithmetic of an inference-only diagnostic changes.

Examples
--------
Smoke one genome end to end (a short GPU sbatch, never the login node)::

    python -m bacpredict.engine.explain.gene_attribution \\
        --organism tb --cohort-csv attribution_cohort_rifampin.csv \\
        --checkpoint "$B/train_tb_ast/checkpoints/mycobacterium_tuberculosis_rifampin_.../" \\
        --out-dir "$B/train_tb_ast/attribution/rifampin" --limit 1

One shard of an array job::

    python -m bacpredict.engine.explain.gene_attribution \\
        --organism tb --cohort-csv attribution_cohort_rifampin.csv \\
        --checkpoint "$CKPT" --out-dir "$OUT" \\
        --shard-index "$SLURM_ARRAY_TASK_ID" --n-shards 40
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from bacpredict.engine.concat.bacformer_genome_vectors import forward_inputs
from bacpredict.engine.config import store_paths
from bacpredict.engine.finetune.evaluate import resolve_checkpoint_dir
from bacpredict.engine.gene_lr.locate_gene import flatten_proteins

logger = logging.getLogger(__name__)

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
GRADIENT_METHODS = ("integrated_gradients", "input_x_gradient")
METHODS = ("ablation", *GRADIENT_METHODS)


class LogitForward(torch.nn.Module):
    """Wrap a genome-classification model so its forward returns one logit per genome.

    captum needs a plain tensor of shape ``[batch]`` from the forward it perturbs. The deployed model
    returns a model-output object carrying ``logits`` of shape ``[batch, 1]``; this flattens that and
    casts to float32 so the subtraction captum performs is not itself done in bf16.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        protein_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        contig_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``[batch]`` float32 logits for the given genome tensors."""
        out = self.model(
            protein_embeddings=protein_embeddings,
            attention_mask=attention_mask,
            contig_ids=contig_ids,
            return_dict=True,
        )
        return out.logits.reshape(-1).float()


def load_classifier(checkpoint: Path, device: str, dtype: torch.dtype) -> torch.nn.Module:
    """Load a deployed AMR fine-tune as a classification model, in eval mode.

    Loads the way :mod:`bacpredict.engine.finetune.evaluate` does — ``trust_remote_code`` so the
    checkpoint's own bundled ``modeling_bacformer.py`` runs, ``num_labels=1`` and
    ``problem_type="binary_classification"`` to match how it was trained. ``problem_type`` is safe
    here: the value only breaks HuggingFace's generic config reader on the ``Trainer`` resume path,
    not ``from_pretrained``.

    Parameters
    ----------
    checkpoint : pathlib.Path
        A run dir or a ``checkpoint-*`` dir; resolved by
        :func:`bacpredict.engine.finetune.evaluate.resolve_checkpoint_dir`.
    device : str
        ``"cuda"`` or ``"cpu"``.
    dtype : torch.dtype
        ``torch.bfloat16`` (as deployed) or ``torch.float32`` (precision check).

    Returns
    -------
    torch.nn.Module
        The model on ``device``, in eval mode, with gradients disabled.
    """
    from transformers import AutoModelForSequenceClassification

    model_dir = resolve_checkpoint_dir(Path(checkpoint))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), num_labels=1, problem_type="binary_classification",
        return_dict=True, trust_remote_code=True, torch_dtype=dtype,
    )
    model = model.to(device=device, dtype=dtype).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    logger.info("Loaded %s from %s as %s", type(model).__name__, model_dir, dtype)
    return model


def load_genome(esm_path: Path, device: str, model_dtype: torch.dtype) -> dict:
    """Read one genome's stored ``.pt`` and build the model's forward kwargs.

    Asserts the **plain per-protein layout** — one row per protein, no interleaved CLS/SEP rows.
    Two docstrings in this repo describe the Kp store as interleaved and the files on disk are not,
    so the layout is checked rather than assumed: an interleaved store would silently shift every
    flat index by one and quietly mis-attribute every gene.

    Parameters
    ----------
    esm_path : pathlib.Path
        The genome's ``{Sample}_esm_embeddings.pt``.
    device : str
        Device to place the tensors on.
    model_dtype : torch.dtype
        Dtype for the float tensors.

    Returns
    -------
    dict
        ``protein_embeddings`` ``[1, T, D]``, ``attention_mask`` ``[1, T]``, ``contig_ids`` ``[1, T]``.

    Raises
    ------
    ValueError
        If the store is the interleaved bundle layout, or its attention mask is not all ones.
    """
    store = torch.load(esm_path, map_location="cpu", weights_only=False)
    if "special_tokens_mask" in store:
        raise ValueError(
            f"{esm_path} is the interleaved Bacformer-input bundle (has special_tokens_mask). This "
            "module assumes the plain per-protein layout so that row i is protein i; ablating the "
            "bundle layout would attribute special tokens and shift every flat index."
        )
    inputs = forward_inputs(store, device, model_dtype)
    mask = inputs["attention_mask"]
    if not bool((mask == 1).all()):
        raise ValueError(f"{esm_path} has a padded attention_mask ({int((mask == 1).sum())}/{mask.numel()} real).")
    return inputs


def verify_protein_count(sample: str, n_rows: int, parquet_dir: Path, suffix: str) -> int | None:
    """Check the ``.pt`` row count against the sample's protein parquet; return the parquet count.

    The flat index only means anything if the embedding tensor and the annotation parquet agree on
    how many proteins the genome has. Embedding-time truncation keeps the first 6,000 proteins, so a
    long genome can disagree — and a disagreement silently shifts gene identity for everything after
    the first missing protein.

    Parameters
    ----------
    sample : str
        Sample ID.
    n_rows : int
        Row count of the embedding tensor.
    parquet_dir : pathlib.Path
        Directory of ``{Sample}{suffix}`` parquets.
    suffix : str
        Parquet filename suffix.

    Returns
    -------
    int or None
        The parquet's flattened protein count, or ``None`` if the parquet is absent.

    Raises
    ------
    ValueError
        If the counts disagree.
    """
    path = parquet_dir / f"{sample}{suffix}"
    if not path.exists():
        logger.warning("No protein parquet for %s at %s; skipping the flat-index guard", sample, path)
        return None
    n_parquet = len(flatten_proteins(pd.read_parquet(path)))
    if n_parquet != n_rows:
        raise ValueError(
            f"{sample}: embedding has {n_rows} proteins but the parquet has {n_parquet}. The flat "
            "index is not trustworthy for this genome (embedding truncates at 6,000 proteins)."
        )
    return n_parquet


def attribute_genome(
    wrapped: LogitForward,
    inputs: dict,
    *,
    perturbations_per_eval: int = 1,
    check_group_constant: bool = False,
) -> tuple[np.ndarray, float]:
    """Ablate every protein of one genome and return per-protein logit shifts plus the base logit.

    Parameters
    ----------
    wrapped : LogitForward
        The model, wrapped to return ``[batch]`` logits.
    inputs : dict
        Forward kwargs from :func:`load_genome`.
    perturbations_per_eval : int, default 1
        How many ablated copies to evaluate per forward pass. Larger values use the batch dimension
        that would otherwise sit idle; attention is memory-efficient (SDPA) so this scales.
    check_group_constant : bool, default False
        Verify captum gave every dimension of a group the same value before the sum is divided by
        the embedding width. Cheap, and run on the first genome to catch a captum behaviour change.

    Returns
    -------
    attributions : numpy.ndarray
        ``[n_proteins]`` float64 shift in the logit, one per protein, in flat order. Positive means
        removing the protein lowered the logit (pushed the model towards susceptible).
    base_logit : float
        The model's logit for the unablated genome.
    """
    from captum.attr import FeatureAblation

    embeddings = inputs["protein_embeddings"]
    _batch, n_proteins, width = embeddings.shape
    extra = (inputs["attention_mask"], inputs["contig_ids"])

    with torch.inference_mode():
        base_logit = float(wrapped(embeddings, *extra).item())

    # One group per protein: all `width` dims of token t share the group id t, so captum ablates a
    # whole protein at once rather than one embedding dimension at a time.
    feature_mask = torch.arange(n_proteins, device=embeddings.device).unsqueeze(0).unsqueeze(-1).expand(1, -1, width)

    ablator = FeatureAblation(wrapped)
    with torch.inference_mode():
        attributions = ablator.attribute(
            inputs=embeddings,
            baselines=torch.zeros_like(embeddings),
            additional_forward_args=extra,
            feature_mask=feature_mask,
            perturbations_per_eval=perturbations_per_eval,
        )

    attr = attributions[0].double()
    if check_group_constant:
        spread = (attr.max(dim=1).values - attr.min(dim=1).values).abs().max().item()
        if spread > 1e-6:
            raise ValueError(
                f"captum gave a protein's {width} dimensions differing attributions (max spread "
                f"{spread:.3g}); the sum-and-divide-by-{width} convention no longer holds."
            )
    # captum broadcasts a group's whole delta to each of its members, so the row sum is `width` times
    # the quantity we want.
    return (attr.sum(dim=1) / width).cpu().numpy(), base_logit


def attribute_genome_gradients(
    wrapped: LogitForward,
    inputs: dict,
    *,
    method: str = "integrated_gradients",
    n_steps: int = 50,
    internal_batch_size: int | None = 4,
) -> tuple[np.ndarray, float]:
    """Attribute every protein of one genome by gradients, returning per-protein scores and the base logit.

    Why this exists beside the ablation path: ablation deletes a gene and asks whether the model
    still predicts resistance. For a determinant that is **present in every genome and differs only
    by a point mutation** — ``gyrA``, ``parC``, ``rpoB`` — deletion is the wrong counterfactual,
    because the model never learned "``gyrA`` present → resistant". A gradient reads how the output
    responds to the protein's *embedding*, which is where a point mutation actually lives.

    It is also far cheaper. Ablation costs one forward per protein (5,390 for a Kp genome, measured
    at 2,492 s in fp32); integrated gradients costs ``n_steps`` forward-backward passes for the
    **whole genome at once**.

    ⚠ The per-protein aggregation is **not** the one :func:`attribute_genome` uses. captum's
    ablation broadcasts a feature group's whole delta to every member, so that path sums and divides
    by the embedding width. A gradient method returns a genuine per-element attribution, and its
    completeness property is stated over their **sum** — so here the embedding dimension is summed
    and *not* divided. Dividing would silently rescale every score by 960.

    Parameters
    ----------
    wrapped : LogitForward
        The model, wrapped to return ``[batch]`` logits.
    inputs : dict
        Forward kwargs from :func:`load_genome`.
    method : {"integrated_gradients", "input_x_gradient"}, default "integrated_gradients"
        ``integrated_gradients`` integrates along a path from the zero baseline and satisfies
        completeness; ``input_x_gradient`` is a single-pass local approximation.
    n_steps : int, default 50
        Path steps for integrated gradients. Ignored by ``input_x_gradient``.
    internal_batch_size : int, optional
        How many path steps captum evaluates per forward. Lower it if a long genome runs out of
        memory — the backward pass holds activations for 30 layers.

    Returns
    -------
    attributions : numpy.ndarray
        ``[n_proteins]`` float64, one score per protein in flat order.
    base_logit : float
        The model's logit for the unperturbed genome.

    Raises
    ------
    ValueError
        If ``method`` is not a known gradient method.
    """
    from captum.attr import InputXGradient, IntegratedGradients

    if method not in GRADIENT_METHODS:
        raise ValueError(f"Unknown gradient method {method!r}; expected one of {sorted(GRADIENT_METHODS)}")

    embeddings = inputs["protein_embeddings"]
    extra = (inputs["attention_mask"], inputs["contig_ids"])

    # No inference_mode anywhere in this function: it marks tensors in a way that permanently
    # excludes them from autograd, so the backward pass would fail. The model's parameters stay
    # requires_grad=False (set at load) — the gradient wanted here is with respect to the input.
    with torch.no_grad():
        base_logit = float(wrapped(embeddings, *extra).item())

    grad_input = embeddings.clone().detach().requires_grad_(True)
    if method == "integrated_gradients":
        attributions = IntegratedGradients(wrapped).attribute(
            inputs=grad_input,
            baselines=torch.zeros_like(grad_input),
            additional_forward_args=extra,
            n_steps=n_steps,
            internal_batch_size=internal_batch_size,
        )
    else:
        attributions = InputXGradient(wrapped).attribute(
            inputs=grad_input, additional_forward_args=extra,
        )

    # Sum, never divide — see the warning above.
    return attributions[0].detach().double().sum(dim=1).cpu().numpy(), base_logit


def delta_probability(base_logit, attribution_logit):
    """Convert a stored logit shift into the probability-scale attribution, exactly.

    The run stores ``attribution_logit`` and each genome's ``base_logit``; this recovers the signed
    probability shift that attributing ``sigmoid(logit)`` directly would have produced, without the
    saturation that makes that quantity degenerate on a confident model.

    Parameters
    ----------
    base_logit : float or numpy.ndarray
        The genome's logit with nothing ablated.
    attribution_logit : float or numpy.ndarray
        The stored per-gene shift, ``f(x) - f(x ablated)``.

    Returns
    -------
    numpy.ndarray
        ``sigmoid(base) - sigmoid(base - attribution)``, in ``[-1, +1]``. Positive means removing the
        gene lowered the model's probability of resistance.
    """
    def _sigmoid(x):
        return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=np.float64)))

    base = np.asarray(base_logit, dtype=np.float64)
    attr = np.asarray(attribution_logit, dtype=np.float64)
    return _sigmoid(base) - _sigmoid(base - attr)


def shard(frame: pd.DataFrame, shard_index: int | None, n_shards: int | None) -> pd.DataFrame:
    """Return this task's contiguous slice of the cohort, or the whole frame when unsharded."""
    if shard_index is None or n_shards is None:
        return frame
    if not 0 <= shard_index < n_shards:
        raise ValueError(f"shard-index {shard_index} out of range for n-shards {n_shards}")
    bounds = np.linspace(0, len(frame), n_shards + 1).astype(int)
    return frame.iloc[bounds[shard_index]: bounds[shard_index + 1]]


def main() -> None:
    """CLI entry point — attribute every genome in this shard of the cohort."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--organism", required=True, choices=("tb", "kp"), help="Organism key.")
    parser.add_argument("--cohort-csv", type=Path, required=True, help="Cohort table from select_attribution_cohort.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Deployed fine-tune run or checkpoint dir.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Where per-genome parquets are written.")
    parser.add_argument("--esm-dir", type=Path, default=None, help="Override the ESM store directory.")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bf16",
                        help="bf16 matches deployment; fp32 is the precision check.")
    parser.add_argument("--method", choices=METHODS, default="ablation",
                        help="ablation deletes each protein; the gradient methods read the response to "
                             "its embedding, which is where a point mutation lives.")
    parser.add_argument("--perturbations-per-eval", type=int, default=1,
                        help="Ablation only — ablated copies per forward pass.")
    parser.add_argument("--n-steps", type=int, default=50,
                        help="integrated_gradients only — path steps.")
    parser.add_argument("--internal-batch-size", type=int, default=4,
                        help="integrated_gradients only — path steps per forward; lower it on OOM.")
    parser.add_argument("--shard-index", type=int, default=None, help="This array task's index.")
    parser.add_argument("--n-shards", type=int, default=None, help="Total array tasks.")
    parser.add_argument("--limit", type=int, default=None, help="Stop after this many genomes (smoke tests).")
    parser.add_argument("--skip-protein-count-check", action="store_true",
                        help="Skip the .pt-vs-parquet flat-index guard (not recommended).")
    parser.add_argument("--overwrite", action="store_true", help="Recompute genomes that already have output.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("No GPU visible — this will be far too slow for anything but a unit test.")
    dtype = DTYPES[args.dtype]

    cohort = pd.read_csv(args.cohort_csv, low_memory=False)
    cohort["Sample"] = cohort["Sample"].astype(str)
    cohort = shard(cohort, args.shard_index, args.n_shards)
    if args.limit is not None:
        cohort = cohort.head(args.limit)
    logger.info("This task will attribute %d genomes by %s (%s)", len(cohort), args.method, args.dtype)

    paths = store_paths(args.organism)
    esm_dir = args.esm_dir or paths.esm_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = load_classifier(args.checkpoint, device, dtype)
    wrapped = LogitForward(model)

    manifest: list[dict] = []
    failures = 0
    first = True
    for row in cohort.itertuples(index=False):
        sample = str(row.Sample)
        out_path = args.out_dir / f"{sample}_attribution.parquet"
        if out_path.exists() and not args.overwrite:
            # A resume must not mix precisions. bf16 and fp32 attributions are different numbers
            # (bf16 quantises most genes to exactly zero), so a finished bf16 genome must not be
            # accepted as satisfying an fp32 run. A parquet that cannot be read is treated as a
            # half-written casualty of a killed job and recomputed.
            try:
                existing = pq.read_schema(out_path).metadata or {}
                done_dtype = existing.get(b"bacpredict_dtype", b"").decode() or "unknown"
                done_method = existing.get(b"bacpredict_method", b"").decode() or "ablation"
                if done_method != args.method:
                    # An ablation score and a gradient score are different quantities on different
                    # scales; one must never be accepted as satisfying a run of the other.
                    logger.warning("%s exists but was computed by %s, not %s — recomputing",
                                   sample, done_method, args.method)
                    done_dtype = "<other-method>"
            except Exception:  # noqa: BLE001 - any unreadable file is simply redone
                logger.warning("%s exists but is unreadable; recomputing", out_path.name)
                done_dtype = None
            if done_dtype == args.dtype:
                logger.info("%s already done in %s; skipping", sample, args.dtype)
                continue
            if done_dtype is not None:
                logger.warning("%s exists but was computed in %s, not %s — recomputing",
                               sample, done_dtype, args.dtype)

        esm_path = Path(esm_dir) / f"{sample}{paths.esm_suffix}"
        if not esm_path.exists():
            logger.error("No embedding for %s at %s; skipping", sample, esm_path)
            manifest.append({"Sample": sample, "status": "missing_embedding", "path": str(esm_path)})
            continue

        started = time.perf_counter()
        try:
            inputs = load_genome(esm_path, device, dtype)
            n_proteins = int(inputs["protein_embeddings"].shape[1])
            if not args.skip_protein_count_check:
                verify_protein_count(sample, n_proteins, Path(paths.parquet_dir), paths.parquet_suffix)

            if args.method == "ablation":
                attributions, base_logit = attribute_genome(
                    wrapped, inputs,
                    perturbations_per_eval=args.perturbations_per_eval,
                    check_group_constant=first,
                )
            else:
                attributions, base_logit = attribute_genome_gradients(
                    wrapped, inputs,
                    method=args.method,
                    n_steps=args.n_steps,
                    internal_batch_size=args.internal_batch_size,
                )
            first = False
            elapsed = time.perf_counter() - started

            table = pa.Table.from_pydict({
                "Sample": [sample] * n_proteins,
                "flat_index": np.arange(n_proteins, dtype=np.int32),
                "attribution_logit": attributions,
                "base_logit": np.full(n_proteins, base_logit, dtype=np.float64),
            })
            # dtype travels with the data, so a later reader never has to guess which precision
            # produced a score, and the resume check above can tell them apart.
            table = table.replace_schema_metadata({b"bacpredict_dtype": args.dtype.encode(),
                                                   b"bacpredict_method": args.method.encode()})
            # Write beside the target then rename: a killed job leaves no half-written parquet that
            # the next run would mistake for finished work.
            tmp_path = out_path.with_suffix(".parquet.partial")
            pq.write_table(table, tmp_path)
            tmp_path.replace(out_path)
        except Exception as exc:  # one bad genome must not kill the rest of the shard
            logger.exception("%s failed; continuing with the rest of the shard", sample)
            failures += 1
            manifest.append({"Sample": sample, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                             "dtype": args.dtype, "method": args.method})
            continue

        manifest.append({
            "Sample": sample, "status": "ok", "n_proteins": n_proteins, "base_logit": base_logit,
            "ast_label": int(getattr(row, "ast_label", -1)), "seconds": round(elapsed, 1),
            "dtype": args.dtype, "method": args.method,
        })
        logger.info("%s: %d proteins, base logit %+.4f, %.1fs (%.1f ms/protein)",
                    sample, n_proteins, base_logit, elapsed, 1000 * elapsed / max(n_proteins, 1))

    # Always written, even when empty: a shard that legitimately had nothing to do and a shard that
    # died before its first genome must not look the same to whoever audits the fan-out.
    tag = "all" if args.shard_index is None else f"shard{args.shard_index:04d}"
    manifest_path = args.out_dir / f"attribution_manifest_{tag}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Wrote %s (%d genomes, %d failed)", manifest_path, len(manifest), failures)
    if failures:
        raise SystemExit(f"{failures} genome(s) failed in this shard; see {manifest_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()

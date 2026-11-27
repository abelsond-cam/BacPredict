r"""Training Bacformer on MIC rather than on R/S: the objectives, the drug's ``sigma``, and the MIC metrics.

Three objectives share one network (``num_labels=1``) and differ only in what its scalar output ``mu`` is taken
to mean and what loss is put on it:

====================  ==========================================  ==============================================
``--mic-objective``   loss                                        ``mu`` is
====================  ==========================================  ==============================================
``bce``               binary cross-entropy (the incumbent)        a logit; this module is not used at all
``mse``               ``(mu - point)^2``                          the log2 MIC, against a hand-imputed point
``censored``          interval-censored Gaussian NLL              the mean latent log2 MIC
====================  ==========================================  ==============================================

all in doubling dilutions from the drug's EUCAST R breakpoint, so ``mu > 0`` is the resistant call in every arm
and the R/S metrics are computed the same way throughout.

**``sigma`` — what it is and when it moves.** ``sigma`` is the residual standard deviation of log2 MIC around
``mu``, in doubling dilutions: one scalar per drug, not per genome. **During training it is held FIXED** at
:data:`INIT_LOG_SIGMA` — 0.4 dilutions, the PsA MIC assay's measured analytical reproducibility
(David's decision, 2026-11-27) — and it is still saved into every checkpoint as :data:`SIGMA_FILE` so a resumed
run continues from the same value.

It is fixed because ``mu`` and ``sigma`` are **jointly non-identified** by the cells that carry no MIC. A bare
R/S call is the interval ``(0, inf)`` or ``(-inf, 0]``, so its likelihood is ``Phi(+-mu/sigma)`` — a function of
the *ratio* alone, identical at ``(mu, sigma)``, ``(2mu, 2sigma)``, ``(50mu, 50sigma)``. 20-40% of cells are
bare calls, and the only cells that resist joint scaling are the width-1 exact ones, whose cost grows merely as
``log sigma``. AdamW then normalises gradient magnitude away, so that logarithmically weak restoring force buys
no protection at all and the pair marches the flat valley at ~lr per step: measured on PsA amikacin, sigma went
1.18 -> 48.6 -> 2.5e5 while the NLL sat flat at 2.54 and ``mu``'s SD reached 17.3 dilutions. Freezing ``sigma``
cuts that valley; it is also the standard identification constraint for an ordered probit, where the scale is
conventionally pinned for exactly this reason. Set ``sigma_lr`` above 0 to learn it anyway, and read the two
paragraphs above first.

It is then **re-fitted once, on validate, at the selected checkpoint** (David's decision, 2026-11-26). Fitting
it on train would read the scale of residuals the weights have already been fitted to, which at the measured
train/validate loss gaps on this cohort is badly too small; fitting it on holdout would be leakage. Validate is
the split that is neither, and re-fitting one scale parameter on held-out data is exactly temperature scaling.
The re-fit is safe because ``sigma`` provably cannot move a ranking metric — ``P(R) = Phi(mu/sigma)`` is monotone
in ``mu`` and crosses 0.5 at ``mu = 0`` for every ``sigma`` — so it changes the calibration numbers and nothing
else (:mod:`bacpredict.engine.finetune.censored_mic_loss`).

The ``mse`` arm has no ``sigma`` in its loss. It is given one the same way at the end, by the same validate
re-fit, so that its probabilities are as well calibrated as they can be and the arms are compared on their best
behaviour rather than on an accident of parameterisation.

**Checkpoint selection stays ``eval_auroc``** for every arm, so that what is compared is the objective and not
the stopping rule. The censored NLL and ``sigma`` are logged at every evaluation as diagnostics.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from bacformer.modeling.trainer import BacformerLargeTrainer

from bacpredict.engine.finetune.censored_mic_loss import (
    ASSAY_SIGMA_DILUTIONS,
    censored_log_likelihood,
    interval_censored_gaussian_nll,
    prob_resistant,
)
from bacpredict.engine.finetune.metrics import compute_full_metrics

logger = logging.getLogger(__name__)

#: ``--mic-objective`` values. ``bce`` is the incumbent and never reaches this module.
MIC_OBJECTIVES = ("bce", "mse", "censored")
MIC_OBJECTIVES_HERE = ("mse", "censored")

#: ``log sigma`` during training, held fixed: ``log 0.4`` — the measured analytical reproducibility of the PsA
#: MIC assay, derived in :data:`ASSAY_SIGMA_DILUTIONS` from published essential-agreement rates rather than
#: from a rule of thumb. Its practical meaning: with ``mu`` in the right dilution cell, the latent value is in
#: that cell 78.9% of the time, in an adjacent one 10.6%, and two or more cells away 0.02%. A prediction one
#: dilution off the observed cell therefore costs 2.01 nats — a real penalty, not a verdict — and the cliff
#: between one and two dilutions is the assay's own resolution.
INIT_LOG_SIGMA = math.log(ASSAY_SIGMA_DILUTIONS)

#: Where a checkpoint's ``sigma`` lives. Not in the safetensors: a key the base model does not have is exactly
#: what ``from_pretrained``/``load_state_dict`` drop with a warning nobody reads.
SIGMA_FILE = "mic_sigma.json"

#: Columns of ``<drug>_mic_intervals.csv`` this module needs.
INTERVAL_COLUMNS = ("Sample", "ast_label", "lo", "hi", "point", "censor", "basis")


############################################################## the interval table ##############################################################


def load_interval_table(path: str | Path) -> pd.DataFrame:
    """Read a drug's ``<drug>_mic_intervals.csv`` (:mod:`bacpredict.engine.ast_labels.mic_intervals`)."""
    df = pd.read_csv(path)
    missing = [c for c in INTERVAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing column(s) {missing}; has {list(df.columns)}")
    df["Sample"] = df["Sample"].astype(str)
    return df.drop_duplicates(subset="Sample", keep="first")


def interval_map(table: pd.DataFrame, objective: str) -> dict[str, tuple[float, float, float, float]]:
    """``Sample -> (ast_label, lo, hi, point)``, keeping only rows the objective can actually use.

    ``mse`` needs a point and so drops the bare-call cells; ``censored`` uses every row, a bare call being the
    open interval on one side of the breakpoint. Both require a clean 0/1 ``ast_label``, because every arm is
    scored against the split table's binary phenotype.
    """
    if objective not in MIC_OBJECTIVES_HERE:
        raise ValueError(f"objective must be one of {MIC_OBJECTIVES_HERE}, got {objective!r}")
    ok = table["ast_label"].isin([0, 1]) & table["lo"].notna() & table["hi"].notna()
    if objective == "mse":
        ok &= table["point"].notna()
    t = table.loc[ok]
    return {
        s: (float(y), float(lo), float(hi), float(p) if pd.notna(p) else float("nan"))
        for s, y, lo, hi, p in zip(t["Sample"], t["ast_label"], t["lo"], t["hi"], t["point"], strict=True)
    }


def censor_map(table: pd.DataFrame) -> dict[str, str]:
    """``Sample -> censor`` (``exact`` / ``interval`` / ``left`` / ``right`` / ``bare``), for the MIC read-outs."""
    return dict(zip(table["Sample"], table["censor"].astype(str), strict=True))


############################################################## the trainer ##############################################################


class MicBacformerTrainer(BacformerLargeTrainer):
    """:class:`BacformerLargeTrainer` with a MIC objective and, for ``censored``, the drug's learned ``sigma``.

    The model is run with ``labels=None`` so its own built-in loss never fires: the label here is a 4-vector,
    not the scalar ``problem_type`` would hand to ``binary_classification`` or ``regression``.

    ``log_sigma`` is held by the trainer rather than registered on the model on purpose. As a model parameter it
    would be written into the checkpoint's safetensors under a key the pretrained Bacformer does not have, and
    every ``from_pretrained`` / ``load_state_dict`` on the resume and ``load_best_model_at_end`` paths would drop
    it with a warning — silently restarting ``sigma`` mid-run. Here it is saved and read explicitly, and a
    failure to find it is loud.

    Parameters
    ----------
    objective : str
        ``"censored"`` or ``"mse"``.
    sigma_lr : float
        Learning rate for ``log_sigma`` alone (``censored`` only). The network's own LR (1.5e-4) would move one
        scalar by about 0.5 in log-space over a whole run, so ``sigma`` would never reach its optimum; this
        parameter group gets its own, and still rides the warmup and linear decay.
    """

    def __init__(self, *args, objective: str = "censored", sigma_lr: float = 0.0, **kwargs):
        if objective not in MIC_OBJECTIVES_HERE:
            raise ValueError(f"objective must be one of {MIC_OBJECTIVES_HERE}, got {objective!r}")
        super().__init__(*args, **kwargs)
        self.objective = objective
        self.sigma_lr = sigma_lr
        self.log_sigma = torch.nn.Parameter(
            torch.tensor(float(INIT_LOG_SIGMA)), requires_grad=objective == "censored" and sigma_lr > 0)

    # ---- sigma ----------------------------------------------------------------------------------------

    @property
    def sigma(self) -> float:
        """The current ``sigma`` in doubling dilutions (1.0 for the ``mse`` arm, which does not learn one)."""
        return float(self.log_sigma.detach().float().exp()) if self.objective == "censored" else 1.0

    def save_sigma(self, directory: str | Path) -> None:
        """Write ``mic_sigma.json`` beside a checkpoint."""
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / SIGMA_FILE).write_text(json.dumps(
            {"objective": self.objective, "log_sigma": float(self.log_sigma.detach().float()),
             "sigma": self.sigma}, indent=2))

    def load_sigma(self, directory: str | Path, *, required: bool = True) -> float | None:
        """Read ``mic_sigma.json`` from a checkpoint into ``log_sigma``; return the ``sigma`` loaded."""
        path = Path(directory) / SIGMA_FILE
        if not path.is_file():
            if required and self.objective == "censored":
                raise FileNotFoundError(
                    f"no {path}: that checkpoint carries no sigma, so resuming would silently restart it at "
                    f"{math.exp(INIT_LOG_SIGMA)}")
            return None
        value = float(json.loads(path.read_text())["log_sigma"])
        with torch.no_grad():
            self.log_sigma.fill_(value)
        logger.info("loaded sigma %.4f from %s", math.exp(value), path)
        return math.exp(value)

    # ---- HF hooks -------------------------------------------------------------------------------------

    def create_optimizer(self):  # noqa: D102 — HF hook
        # Place sigma on the training device BEFORE the optimizer captures it. Moving a parameter after the
        # optimizer holds it leaves its Adam state on the old device, which fails on the first step.
        self.log_sigma.data = self.log_sigma.data.to(self.args.device)
        optimizer = super().create_optimizer()
        if self.objective == "censored" and self.sigma_lr > 0:
            optimizer.add_param_group({"params": [self.log_sigma], "lr": self.sigma_lr, "weight_decay": 0.0})
            logger.info("sigma is parameter group %d, lr %g", len(optimizer.param_groups) - 1, self.sigma_lr)
        else:
            logger.info("sigma held FIXED at %.4f dilutions (sigma_lr=%g): mu and sigma are jointly "
                        "non-identified by the bare R/S cells", self.sigma, self.sigma_lr)
        return optimizer

    def _save(self, output_dir: str | None = None, state_dict=None):
        super()._save(output_dir, state_dict)
        self.save_sigma(output_dir or self.args.output_dir)

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):  # noqa: D102
        labels = inputs.pop("labels")
        outputs = model(
            protein_embeddings=inputs.pop("protein_embeddings"),
            labels=None,
            attention_mask=inputs.pop("attention_mask", None),
            contig_ids=inputs.pop("contig_ids", inputs.pop("token_type_ids", None)),
        )
        logits = outputs["logits"] if isinstance(outputs, dict) else getattr(outputs, "logits", outputs[-1])
        mu = logits.reshape(-1).float()
        labels = labels.reshape(-1, 4)
        if self.objective == "censored":
            # .to() is differentiable, so on the inference path (where create_optimizer never ran and sigma is
            # still on the CPU) the gradient would still reach the leaf; here it is simply a no-op.
            loss = interval_censored_gaussian_nll(mu, self.log_sigma.to(mu.device), labels[:, 1], labels[:, 2])
        else:
            point = labels[:, 3].float()
            if torch.isnan(point).any():
                raise ValueError("the mse arm was given a cell with no imputed point MIC: build its dataset "
                                 "with interval_map(..., 'mse'), which drops the bare-call cells")
            loss = F.mse_loss(mu, point)
        return (loss, {"logits": logits}) if return_outputs else loss


############################################################## metrics ##############################################################


class MicEvalMetrics:
    """``compute_metrics`` for the MIC arms: the R/S block the BCE arm reports, plus ``sigma`` and the NLL.

    ``metric_for_best_model`` stays ``eval_auroc``, which is why this must emit the same keys as
    :func:`bacpredict.engine.finetune.metrics.compute_metrics_binary_genome_pred`. The probability it scores is
    ``Phi(mu/sigma)``, whose ranking and whose 0.5 threshold are both identical to ``mu``'s — so ``eval_auroc``
    and ``eval_balanced_accuracy`` are properties of the network alone and ``sigma`` cannot flatter them.

    ``eval_censored_nll`` is the held-out interval-censored NLL at the *current* ``sigma``. For the ``mse`` arm
    that ``sigma`` is the fixed 1.0 it never learns, so its NLL is a like-for-like trace rather than its best.
    """

    def __init__(self, trainer_ref: list):
        self.trainer_ref = trainer_ref  # one-element list, filled once the trainer exists

    def __call__(self, preds, prefix: str = "eval") -> dict[str, float]:
        """Metrics for one evaluation pass."""
        trainer = self.trainer_ref[0]
        sigma = trainer.sigma if trainer is not None else 1.0
        mu = np.asarray(preds.predictions, dtype=float).reshape(-1)
        labels = np.asarray(preds.label_ids, dtype=float).reshape(-1, 4)
        y_true, lo, hi = labels[:, 0].astype(int), labels[:, 1], labels[:, 2]
        y_prob = prob_resistant(mu, sigma)
        full = compute_full_metrics(y_true, y_prob)
        nll = -float(np.mean(censored_log_likelihood(mu, sigma, lo, hi)))
        return {
            f"{prefix}_accuracy": float(np.mean((y_prob >= 0.5).astype(int) == y_true)) if y_true.size else float("nan"),
            f"{prefix}_auroc": full["auroc"],
            f"{prefix}_auprc": full["auprc"],
            f"{prefix}_f1": full["f1"],
            f"{prefix}_balanced_accuracy": full["balanced_accuracy"],
            f"{prefix}_nr_samples": full["n_samples"],
            f"{prefix}_sigma": sigma,
            f"{prefix}_censored_nll": nll,
            f"{prefix}_mu_mean": float(np.mean(mu)) if mu.size else float("nan"),
            f"{prefix}_mu_sd": float(np.std(mu)) if mu.size else float("nan"),
        }

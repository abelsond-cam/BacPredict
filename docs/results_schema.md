# BacPredict results JSON schema (v1.1)

Canonical results file written by [engine/finetune/metrics.py](../src/bacpredict/engine/finetune/metrics.py) (`write_results_json`). Written next to a checkpoint as `results.json` (training-time, threshold 0.5 only) or `eval_results.json` (the shared evaluator [engine/finetune/evaluate.py](../src/bacpredict/engine/finetune/evaluate.py), which also adds the optional `operating_point` block). Shared by every AST/phenotype task (`apps/kleb`, `apps/tb`, `kleb_iso_source`).

## Example

```json
{
  "schema_version": "1.0",
  "task": "kleb_ast",
  "drug": "ceftriaxone",
  "model": {
    "name_or_path": "macwiatrak/bacformer-large-masked-complete-genomes",
    "git_sha": "d4cda1a...",
    "checkpoint_dir": "/.../klebsiella_pneumoniae_ceftriaxone_lr_0.00015_finetuned_fold00_seed1"
  },
  "split": {
    "source": "kfold",
    "evaluate_seed": 1,
    "n_folds": 5,
    "fold": 0,
    "n_evaluate": 1234
  },
  "metrics": {
    "auroc": 0.97,
    "auprc": 0.94,
    "sensitivity": 0.92,
    "specificity": 0.95,
    "balanced_accuracy": 0.93,
    "f1": 0.91,
    "prevalence": 0.41,
    "n_samples": 1234,
    "confusion_matrix": [[tn, fp], [fn, tp]],
    "threshold": 0.5,
    "calibration": {"prob_true": [...], "prob_pred": [...], "n_bins": 10}
  },
  "timestamp": "2026-05-25T12:34:56.789012+00:00",
  "host": "gpu-q-49.hpc.cam.ac.uk"
}
```

## Field reference

### Top level

| Field | Type | Notes |
|---|---|---|
| `schema_version` | string | Currently `"1.0"`. |
| `task` | string | Task package slug — e.g. `kleb_ast`, `tb_ast`, `kleb_iso_source`. |
| `drug` | string | Drug or label column the model predicts (NaN-filtered before splitting). |
| `model.name_or_path` | string | HF model ID or local checkpoint dir passed to `--model-name-or-path`. |
| `model.git_sha` | string \| null | `git rev-parse HEAD` of the working tree at write time. |
| `model.checkpoint_dir` | string | Absolute path of the trainer `output_dir`. |
| `split.source` | string | One of `"csv"` (full mode), `"kfold"`, `"smoke"`. |
| `split.evaluate_seed` | int \| null | Pinned holdout seed (only meaningful when `source == "kfold"`). |
| `split.n_folds` / `split.fold` | int \| null | Set when `source == "kfold"`. |
| `split.n_evaluate` | int | Number of samples scored to produce `metrics`. |
| `metrics` | object | See below. |
| `timestamp` | string | ISO-8601 UTC. |
| `host` | string | `socket.gethostname()` at write time. |

### Metrics block

All required, all computed on the evaluate holdout (see §0.4 of root [CLAUDE.md](../CLAUDE.md)).

| Field | Type | Notes |
|---|---|---|
| `auroc`, `auprc` | float | Standard ranking metrics. `NaN` if labels are single-class. |
| `sensitivity`, `specificity` | float | At `threshold`. `NaN` if denominator is 0. |
| `balanced_accuracy`, `f1` | float | At `threshold`. |
| `prevalence` | float | `mean(y_true)`; `NaN` if `n_samples == 0`. |
| `n_samples` | int | Post-`ignore_index` filtering. |
| `confusion_matrix` | int[2][2] | `[[tn, fp], [fn, tp]]`. |
| `threshold` | float | Decision threshold (default 0.5). |
| `calibration.prob_true`, `prob_pred` | float[] | `sklearn.calibration.calibration_curve` outputs. Empty arrays if calibration failed (e.g. all probs in one bin). |
| `calibration.n_bins` | int | Bin count (default 10). |

### `operating_point` block (optional, schema v1.1)

Written by the **evaluator** (`evaluate.py`), not by training. Holds metrics at a tuned threshold instead of the fixed 0.5. The threshold is chosen by **Youden's J** (max sensitivity+specificity). ⚠ **The convention changed on 2026-08-13 to Youden on the HOLDOUT**, one convention for both arms of any comparison — which means sens/spec/balanced accuracy are **optimistically biased** and must be reported as "at the optimal operating point". AUROC/AUPRC are unaffected. Older files selected on validation; `operating_point.selected_on` records which, and the two pilot GWAS drugs differ. Absent in training-time `results.json` (which is 0.5-only) — readers must treat it as optional.

| Field | Type | Notes |
|---|---|---|
| `objective` | string | `"youden_j"`. |
| `selected_on` | string | `"validation"` — where the threshold was chosen. |
| `threshold` | float | The tuned cut applied to the evaluate set. |
| `sensitivity`, `specificity`, `balanced_accuracy`, `f1` | float | Evaluate-set metrics at `threshold`. |
| `confusion_matrix` | int[2][2] | `[[tn, fp], [fn, tp]]` at `threshold`. |

`auroc`/`auprc` are threshold-independent and live only in `metrics` (they are unchanged by tuning).

### Split-table runs (`finetune_amr --split-table`)

A run trained from a `<drug>_split.csv` (`engine/splits/build_amr_holdout_splits`, read by `load_splits`) adds to
the fields above. `metrics` is still the holdout's, so v1 readers keep working.

| Field | Type | Notes |
|---|---|---|
| `split.source` | string | `"split_table"`, or `"smoke"` for the 10-genome run. |
| `split.table`, `split.table_md5` | string | The split table read, and its md5. |
| `split.n_train`, `split.n_validate` | int | Genomes trained and validated on (0/1 labels only). |
| `split.extra_holdouts` | string[] | The splits scored beside the holdout (default `["holdout_st"]`). |
| `species` | string[] | Present when the table has a `species` column. |
| `holdouts.<split>.n_samples` | int | Genomes scored, `holdout` first, then each extra holdout; 0 when the drug has none. |
| `holdouts.<split>.metrics` | object | The metrics block above, plus `per_class`. `holdouts.holdout.metrics` equals `metrics`. |
| `holdouts.<split>.metrics.per_class` | object | `"0"` (S) and `"1"` (R): `n`, `n_predicted`, `recall`, `precision`, `f1` at 0.5. |
| `holdouts.<split>.by_species` | object | Only when the table holds more than one species: `{species: metrics block with per_class}`. |
| `holdouts.<split>.scores` | string | The per-sample archive beside `results.json`: `eval_scores.npz` for `holdout`, `eval_scores_<split>.npz` otherwise. |

Each archive holds `sample_ids`, `y_true`, `y_prob`, `drug`, and per row `split` and `species` (`""` when the table
has none); `operating_threshold` is NaN (no threshold is tuned at training time).

## Forward compatibility

v1 readers must tolerate (ignore) unknown top-level keys. The HGT-vs-vertical mechanism stratification milestone will add a `metrics_by_mechanism` block:

```json
"metrics_by_mechanism": {
  "hgt":          { ...same shape as metrics... },
  "chromosomal":  { ...same shape... },
  "mixed":        { ...same shape... }
}
```

That addition is non-breaking — `metrics` stays the all-isolates view.

## Producing the file

The shared trainer `bacpredict.engine.finetune.finetune_amr` writes the JSON automatically after `trainer.train()` completes. Sub-step 3 (MAG contrast) just re-runs training with `--model-name-or-path macwiatrak/bacformer-large-masked-MAG`; the JSON lands next to the resulting checkpoint dir.

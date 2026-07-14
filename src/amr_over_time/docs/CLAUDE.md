# amr_over_time — predict AST across the Klebsiella set → resistance over time

A **separate top-level package** (`src/amr_over_time/`), extracted from `bacpredict` in 2026-07 to
keep the core a purpose-built AMR-*accuracy* module. It answers a different question: *given an
already-fine-tuned Kp AST model, what is the predicted resistance rate across the whole Klebsiella
collection, and how does it move over time?*

> **Status: ran; results were not great.** Kept for reference and possible revival, deliberately
> out of the accuracy pipeline. This is the main historical consumer of `metadata_v2`.

## What it does

Two stages, one drug at a time:

1. **`predict_amr_for_metadata.py`** — reads `metadata_v2`, filters to `kpsc_final_list == True`
   (the ~80k whole-Klebsiella set), drops samples with no ESM-C embedding on disk, runs Bacformer
   inference with the drug's fine-tuned checkpoint via
   `bacpredict.engine.finetune.predict.predict_proba`, applies the per-drug **Youden's J** threshold
   (from the checkpoint's `eval_results.json`), and writes
   `Sample, predicted_<drug>_AST_prob, predicted_<drug>_AST`. Driven by the 22-drug SLURM array
   `scripts/predict_amr_panel_on_slurm.sh`.
2. **`plot_resistance_over_time.py`** — per drug × stratum, a two-stage model: an LMM confounder
   denoise (`poly(year) + (1|study) + (1|country)`) to remove study/geography batch structure, then
   a Kalman/ARIMA time smooth. Writes PNGs to `visualisations/` (checked-in) and, at runtime, to the
   data-root dir `KP.data_root()/predicting_AST_over_time/`.

## Inputs / dependencies

- The **flat whole-Klebsiella ESM-C store** `processed/klebsiella_esm_embeddings/` (~80k;
  **Cambridge/CSD3 only** — not the per-task Isambard AST store; see the core CLAUDE.md
  "Training data architecture"). Also used by `kleb_iso_source`.
- `metadata_v2_all_samples_and_columns.tsv` — `kpsc_final_list`, `collection_date_parsed`, and the
  `EBI_<drug>_AST` / `predicted_<drug>_AST_prob` overlay columns.
- A fine-tuned Kp AST checkpoint (produced by the core `bacpredict.engine.finetune`).

**One-way dependency on `bacpredict.engine`** (`engine.finetune.predict.predict_proba`,
`engine.config.KP`) — exactly like `kleb_iso_source`. Nothing in `bacpredict` imports this package.

Run with the shared aarch64 venv + `PYTHONPATH=$HOME/BacPredict/src` (`amr_over_time` is on it).

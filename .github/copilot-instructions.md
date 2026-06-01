# Copilot Instructions

## Build, test, type-check, and lint

Use the same dependency install patterns as CI and release workflows:

```bash
pip install ".[test]" -f https://download.pytorch.org/whl/cpu/torch_stable.html
pytest
pytest test/test_df_utils.py::test_validate_df_inputs_returns_correct_metadata_for_valid_inputs
```

```bash
pip install ".[typecheck]" -f https://download.pytorch.org/whl/cpu/torch_stable.html
mypy src test
```

```bash
python -m pip install -U pip setuptools wheel build
python -m build
```

```bash
ruff check .
```

## High-level architecture

- `src/chronos/base.py` is the dispatch layer. `BaseChronosPipeline.from_pretrained()` inspects model config and routes loading to the correct pipeline class (`ChronosPipeline`, `ChronosBoltPipeline`, or `Chronos2Pipeline`). It also handles `s3://...` model URIs by caching them locally through `src/chronos/boto_utils.py`.
- `src/chronos/chronos.py` implements the original Chronos models. The pipeline wraps a Hugging Face causal or seq2seq LM plus a tokenizer (`MeanScaleUniformBins`) that scales real-valued series and bucketizes them into tokens. Longer horizons are generated autoregressively by feeding the median forecast back into the context.
- `src/chronos/chronos_bolt.py` implements Chronos-Bolt. It is patch-based rather than token-based: the model normalizes the context, patches it, runs a T5-style encoder/decoder, and predicts quantiles directly. For horizons beyond the model default, it unrolls quantile forecasts instead of sampling tokens.
- `src/chronos/chronos2/` contains the newest architecture and the only fine-tuning path in this repo. `model.py` defines a custom encoder with time attention and group attention, `dataset.py` prepares grouped target/covariate inputs, and `pipeline.py` handles inference, optional cross-learning across tasks, long-horizon unrolling, `predict_df`, `predict_fev`, and fine-tuning.
- `src/chronos/df_utils.py` is the shared bridge from long-format pandas dataframes to the list-of-dicts input format used by dataframe-based prediction. It is where timestamp regularity, per-series frequency consistency, dtype casting, series ordering, and `future_df` alignment are enforced.
- `test/` is organized by pipeline family plus dataframe/util helpers. The dummy model folders under `test/` are the normal way to exercise loading and inference behavior without depending on remote model downloads.

## Key conventions

- Prefer the public package surface from `src/chronos/__init__.py` when writing examples or tests; the repo treats the exported pipeline/config classes there as the main API.
- Variable-length series are aligned with **left-padding using `torch.nan`**, not zeros. This shows up in `BaseChronosPipeline`, `chronos.utils.left_pad_and_stack_1D`, and the Chronos-2 dataset utilities.
- Public forecast outputs are normalized to **CPU `float32`** even when inference runs in bf16/fp32 on device. Do not preserve internal model dtype in user-facing outputs.
- The three pipeline families intentionally return different forecast layouts: original Chronos returns samples, while Chronos-Bolt and Chronos-2 return quantile-oriented forecasts.
- Dataframe prediction paths expect regular timestamps with one shared frequency across all series. When `future_df` is provided, it must contain the same IDs as `df`, exactly `prediction_length` rows per ID, and only covariate columns.
- `validate_inputs=False` in dataframe helpers is an advanced fast path, not a relaxed parser. Callers must already have data sorted by `(id_column, timestamp_column)` and must guarantee timestamp/future alignment themselves.
- Chronos-2 dictionary inputs have stronger structure than the simpler pipelines: `future_covariates` keys must be a subset of `past_covariates`, and known-future covariates are deliberately ordered after past-only covariates to match downstream grouping logic.
- Chronos-2 supports categorical covariates only through numpy-backed arrays during input preparation; string-valued torch tensors are not part of the supported path.
- In Chronos-2, `batch_size` counts the total number of target and covariate series presented to the model, not just the number of top-level tasks.
- Longer-than-default prediction lengths are allowed by default across pipelines but are treated as a warned path; code that must enforce model-native horizons uses `limit_prediction_length=True`.

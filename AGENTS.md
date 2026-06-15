# AGENTS.md

Guidance for coding agents working in this repository.

## Project Overview

This repository is based on `amazon-science/chronos-forecasting` and contains local extensions for Chronos-2 pretraining.

Key areas:

- `src/chronos/`: Chronos package source code.
- `src/chronos/chronos2/`: Chronos-2 model and pipeline implementation.
- `scripts/training/`: Training entry points.
- `scripts/training/train-2.py`: Local Chronos-2 pretraining script.
- `scripts/training/configs/chronos-2-small.yaml`: Local Chronos-2 pretraining config.
- `datasets/pretrained/`: Local pretraining datasets and download helpers.

## Environment

Use `uv` for the development environment.

Recommended setup:

```powershell
uv sync --extra dev
```

Run commands through the project environment:

```powershell
uv run python -m pytest
uv run python scripts/training/train-2.py --help
```

If installing packages manually into the existing environment is necessary:

```powershell
uv pip install -e ".[dev]"
```

## Data Directories

Large local datasets live under:

- `datasets/pretrained/chronos_datasets/`
- `datasets/pretrained/GiftEvalPretrain/`
- `datasets/pretrained/kernel_synth/`

These are local training assets. Do not commit downloaded dataset shards, Hugging Face caches, checkpoints, TensorBoard runs, or model weights.

Important local helpers:

- `datasets/pretrained/chronos_datasets/download_chronos_datasets.py`
- `datasets/pretrained/GiftEvalPretrain/download_gifteval_pretrain.py`
- `datasets/pretrained/*/dataset_list.txt`

`Salesforce/GiftEvalPretrain` can also be downloaded directly with `huggingface-cli download` using `--include "<dataset_name>/*"` when a single dataset is large or needs resumable file-level download.

## Training

Chronos-2 pretraining is configured primarily through:

```text
scripts/training/configs/chronos-2-small.yaml
```

The config currently supports multiple data sources through:

- `training_data_paths`
- `probability`
- `validation_data_paths`
- `validation_num_samples`
- `validation_seed`
- `seed`

The probability values are root-level sampling weights. When a root expands into multiple concrete datasets, the root weight is split across the complete datasets found under that root.

For reproducibility, set both:

```yaml
seed: 42
validation_seed: 42
```

`tf32` and `bf16` are forwarded to Hugging Face `TrainingArguments`. If both are disabled, training is generally FP32 unless code explicitly changes tensor or model dtypes elsewhere.

## Git Hygiene

Do not add local experiment artifacts to Git.

Ignored local paths include:

- `A100&B150S/`
- `output/`
- `.venv/`
- `chronos-2-finetuned`

If a large local artifact was accidentally tracked, remove it from the index without deleting the local file:

```powershell
git rm -r --cached -- "path/to/artifact"
```

Then ensure the path is listed in `.gitignore`.

## Coding Guidelines

- Preserve the existing repository style.
- Keep changes scoped to the requested feature or bug fix.
- Avoid unrelated refactors.
- Use structured dataset APIs instead of ad hoc file parsing when possible.
- Be careful with Windows paths and directories containing non-ASCII characters.
- Prefer `rg` for searching.
- Use `apply_patch` for manual source edits.

## Verification

For Python syntax checks:

```powershell
uv run python -m py_compile scripts/training/train-2.py
```

For package tests:

```powershell
uv run python -m pytest
```

When changing data loading code, verify at least one small dataset from each configured source can be opened and iterated without loading large datasets fully into memory.

## Notes For Future Agents

- This workspace may contain large partially downloaded datasets. Do not delete data unless explicitly requested.
- Some Hugging Face downloads may be incomplete due to network interruptions. Prefer resumable file-level downloads for very large datasets.
- `datasets/pretrained/*/.hf_cache` directories are disposable only after the corresponding datasets have been fully saved locally.
- Local training results may be important for comparison even when they are ignored by Git.

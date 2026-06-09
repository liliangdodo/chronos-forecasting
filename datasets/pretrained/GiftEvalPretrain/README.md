# GiftEvalPretrain Datasets

This directory stores the subset of `Salesforce/GiftEvalPretrain` used by the Chronos-2 training data table but not available from `autogluon/chronos_datasets`.

Download the selected sub-datasets listed in `dataset_list.txt`:

```bash
uv run python datasets/pretrained/GiftEvalPretrain/download_gifteval_pretrain.py
```

For faster large-file downloads:

```bash
uv pip install hf_transfer
uv run python datasets/pretrained/GiftEvalPretrain/download_gifteval_pretrain.py --enable-hf-transfer --retries 5 --retry-delay 30
```

If Hugging Face access is slow from your network:

```bash
uv run python datasets/pretrained/GiftEvalPretrain/download_gifteval_pretrain.py --hf-endpoint https://hf-mirror.com --enable-hf-transfer --retries 5 --retry-delay 30
```

Retry only previous failures:

```bash
uv run python datasets/pretrained/GiftEvalPretrain/download_gifteval_pretrain.py --dataset-list datasets/pretrained/GiftEvalPretrain/failed_downloads.txt --enable-hf-transfer --retries 5 --retry-delay 30
```

Each dataset is saved as a Hugging Face `save_to_disk` directory:

```text
datasets/pretrained/GiftEvalPretrain/<dataset_name>
```

用huggingface cli下载大数据集

```powershell
$env:HF_HUB_ENABLE_HF_TRANSFER = "1"

hf download Salesforce/GiftEvalPretrain --repo-type dataset --include "buildings_900k/*" --local-dir "datasets/pretrained/GiftEvalPretrain" 
```

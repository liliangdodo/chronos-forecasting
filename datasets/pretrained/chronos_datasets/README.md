# Chronos Pretrained Datasets

This directory stores local copies of train splits from `autogluon/chronos_datasets`.

Download the official Chronos-2 training dataset configs listed in `dataset_list.txt`:

```bash
python datasets/pretrained/download_chronos_datasets.py
```

Download all dataset configs from the repo:

```bash
python datasets/pretrained/download_chronos_datasets.py --all-configs
```

Download selected configs for a quick smoke test:

```bash
python datasets/pretrained/download_chronos_datasets.py --configs monash_m1_yearly monash_nn5_weekly
```

Large datasets are loaded with `keep_in_memory=False` and saved with `Dataset.save_to_disk`.
Each split is written to:

```text
datasets/pretrained/<config_name>/train
```

Reload a saved split:

```python
from datasets import load_from_disk

ds = load_from_disk("datasets/pretrained/monash_m1_yearly/train")
```

On Windows, the default Hugging Face cache is `C:\tmp\chronos_datasets_hf_cache`.
This short path avoids file-lock path length issues when downloading large configs such as Weatherbench.
If a large download is interrupted, rerun the same command; completed datasets are skipped, incomplete
dataset directories are cleaned up, and each config is retried.

Retry only once, with a longer delay between attempts:

```bash
python datasets/pretrained/download_chronos_datasets.py --retries 5 --retry-delay 30
```

For faster large-file downloads, install `hf_transfer` and enable it:

```bash
uv pip install hf_transfer
uv run python datasets/pretrained/download_chronos_datasets.py --enable-hf-transfer --retries 5 --retry-delay 30
```

If access to Hugging Face is slow from your network, you can also override the endpoint:

```bash
uv run python datasets/pretrained/download_chronos_datasets.py --hf-endpoint https://hf-mirror.com --enable-hf-transfer --retries 5 --retry-delay 30
```

To retry only the configs that failed in a previous run:

```bash
uv run python datasets/pretrained/download_chronos_datasets.py --dataset-list datasets/pretrained/failed_downloads.txt --enable-hf-transfer --retries 5 --retry-delay 30
```

## 大数据集忽略，下载失败

1、Weatherbench 相关的数据集
- weatherbench_daily 19G
- weatherbench_hourly* 780G
    weatherbench_hourly_10m_u_component_of_wind
    weatherbench_hourly_10m_v_component_of_wind
    weatherbench_hourly_2m_temperature
    weatherbench_hourly_geopotential
    weatherbench_hourly_potential_vorticity
    weatherbench_hourly_relative_humidity
    weatherbench_hourly_specific_humidity
    weatherbench_hourly_temperature
    weatherbench_hourly_toa_incident_solar_radiation
    weatherbench_hourly_total_cloud_cover
    weatherbench_hourly_total_precipitation
    weatherbench_hourly_u_component_of_wind
    weatherbench_hourly_v_component_of_wind
    weatherbench_hourly_vorticity
    weatherbench_weekly

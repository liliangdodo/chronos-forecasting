"""Download train splits from autogluon/chronos_datasets to local disk.

Examples
--------
Download all dataset configs:
    python datasets/pretrained/download_chronos_datasets.py --all-configs

Download official Chronos-2 training dataset configs listed in dataset_list.txt:
    python datasets/pretrained/download_chronos_datasets.py
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable


DEFAULT_REPO_ID = "autogluon/chronos_datasets"
DEFAULT_SPLIT = "train"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUTPUT_DIR = SCRIPT_DIR
DEFAULT_CACHE_DIR = (
    Path("C:/tmp/chronos_datasets_hf_cache")
    if os.name == "nt"
    else SCRIPT_DIR / ".hf_cache"
)
DEFAULT_DATASET_LIST = SCRIPT_DIR / "dataset_list.txt"


def import_hf_datasets() -> Any:
    """Import Hugging Face datasets even though this repo has a top-level datasets/ dir."""
    original_sys_path = sys.path.copy()
    blocked_paths = {str(REPO_ROOT), str(SCRIPT_DIR), str(SCRIPT_DIR.parent)}
    sys.path = [
        path
        for path in sys.path
        if path not in ("", ".") and str(Path(path).resolve()) not in blocked_paths
    ]
    try:
        module = importlib.import_module("datasets")
    finally:
        sys.path = original_sys_path

    if not hasattr(module, "load_dataset"):
        raise ImportError(
            "Could not import the Hugging Face `datasets` package. "
            "Install it with `pip install datasets` or run `pip install -r requirements.txt`."
        )
    return module


HF_DATASETS: Any | None = None


def get_hf_datasets() -> Any:
    global HF_DATASETS
    if HF_DATASETS is None:
        HF_DATASETS = import_hf_datasets()
    return HF_DATASETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Hugging Face dataset configs from autogluon/chronos_datasets "
            "and save their train splits to disk."
        )
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repo id. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help=f"Dataset split to download. Default: {DEFAULT_SPLIT}",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Dataset config names to download. Overrides --dataset-list and --all-configs.",
    )
    parser.add_argument(
        "--dataset-list",
        type=Path,
        default=DEFAULT_DATASET_LIST,
        help=(
            "Text file with one dataset config name per line. Blank lines and lines starting with # are ignored. "
            f"Default: {DEFAULT_DATASET_LIST}"
        ),
    )
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Discover and download all configs from the repo instead of reading --dataset-list.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where datasets are saved. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=(
            "HF datasets cache directory. On Windows the default is intentionally short to avoid "
            f"path-length issues with file locks. Default: {DEFAULT_CACHE_DIR}"
        ),
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of attempts for each dataset config. Default: 3",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=10.0,
        help="Seconds to wait between retries. Default: 10",
    )
    parser.add_argument(
        "--enable-hf-transfer",
        action="store_true",
        help=(
            "Enable Hugging Face's hf_transfer backend for faster large-file downloads. "
            "Requires `pip install hf_transfer` in the active Python environment."
        ),
    )
    parser.add_argument(
        "--hf-endpoint",
        default=None,
        help=(
            "Override the Hugging Face endpoint, for example https://hf-mirror.com. "
            "This sets HF_ENDPOINT before importing datasets/huggingface_hub."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing saved dataset directory.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward trust_remote_code=True to datasets.load_dataset/get_dataset_config_names.",
    )
    return parser.parse_args()


def get_config_names(repo_id: str, cache_dir: Path, trust_remote_code: bool) -> list[str]:
    hf_datasets = get_hf_datasets()
    return hf_datasets.get_dataset_config_names(
        repo_id,
        cache_dir=str(cache_dir),
        trust_remote_code=trust_remote_code,
    )


def read_dataset_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset list file not found: {path}")

    config_names: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        config_name = line.split("#", maxsplit=1)[0].strip()
        if not config_name:
            continue
        if config_name in config_names:
            raise ValueError(f"Duplicate config `{config_name}` in {path} at line {line_number}")
        config_names.append(config_name)

    if not config_names:
        raise ValueError(f"Dataset list is empty: {path}")
    return config_names


def validate_config_names(
    *,
    requested_config_names: list[str],
    repo_id: str,
    cache_dir: Path,
    trust_remote_code: bool,
) -> None:
    available_config_names = set(
        get_config_names(repo_id=repo_id, cache_dir=cache_dir, trust_remote_code=trust_remote_code)
    )
    missing = [name for name in requested_config_names if name not in available_config_names]
    if missing:
        missing_text = "\n".join(f"- {name}" for name in missing)
        raise ValueError(
            f"The following config(s) were not found in {repo_id}:\n{missing_text}"
        )


def write_metadata(
    dataset_dir: Path,
    *,
    repo_id: str,
    config_name: str,
    split: str,
    dataset: Any,
) -> None:
    metadata = {
        "repo_id": repo_id,
        "config_name": config_name,
        "split": split,
        "num_rows": dataset.num_rows,
        "features": dataset.features.to_dict(),
    }
    metadata_path = dataset_dir / "download_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def is_saved_dataset_complete(dataset_dir: Path) -> bool:
    return (
        dataset_dir.is_dir()
        and (dataset_dir / "dataset_info.json").is_file()
        and (dataset_dir / "state.json").is_file()
    )


def download_one(
    *,
    repo_id: str,
    config_name: str,
    split: str,
    output_dir: Path,
    cache_dir: Path,
    overwrite: bool,
    trust_remote_code: bool,
) -> Path:
    dataset_dir = output_dir / config_name / split
    if is_saved_dataset_complete(dataset_dir):
        if overwrite:
            shutil.rmtree(dataset_dir)
        else:
            print(f"[skip] {config_name}: already exists at {dataset_dir}")
            return dataset_dir
    elif dataset_dir.exists():
        print(f"[cleanup] {config_name}: removing incomplete dataset directory {dataset_dir}")
        shutil.rmtree(dataset_dir)

    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[download] {repo_id}/{config_name} split={split}")
    hf_datasets = get_hf_datasets()
    dataset = hf_datasets.load_dataset(
        repo_id,
        config_name,
        split=split,
        cache_dir=str(cache_dir),
        keep_in_memory=False,
        trust_remote_code=trust_remote_code,
    )
    dataset.save_to_disk(str(dataset_dir))
    write_metadata(
        dataset_dir,
        repo_id=repo_id,
        config_name=config_name,
        split=split,
        dataset=dataset,
    )
    print(f"[done] {config_name}: rows={dataset.num_rows}, saved={dataset_dir}")
    return dataset_dir


def download_one_with_retries(
    *,
    repo_id: str,
    config_name: str,
    split: str,
    output_dir: Path,
    cache_dir: Path,
    overwrite: bool,
    trust_remote_code: bool,
    retries: int,
    retry_delay: float,
) -> Path:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            if retries > 1:
                print(f"[attempt] {config_name}: {attempt}/{retries}")
            return download_one(
                repo_id=repo_id,
                config_name=config_name,
                split=split,
                output_dir=output_dir,
                cache_dir=cache_dir,
                overwrite=overwrite,
                trust_remote_code=trust_remote_code,
            )
        except Exception as exc:
            last_error = exc
            print(f"[retryable-failed] {config_name}: attempt {attempt}/{retries}: {exc}")
            if attempt < retries:
                time.sleep(retry_delay)

    assert last_error is not None
    raise last_error


def download_many(
    *,
    repo_id: str,
    config_names: Iterable[str],
    split: str,
    output_dir: Path,
    cache_dir: Path,
    overwrite: bool,
    trust_remote_code: bool,
    retries: int,
    retry_delay: float,
) -> None:
    failures: list[tuple[str, str]] = []
    for config_name in config_names:
        try:
            download_one_with_retries(
                repo_id=repo_id,
                config_name=config_name,
                split=split,
                output_dir=output_dir,
                cache_dir=cache_dir,
                overwrite=overwrite,
                trust_remote_code=trust_remote_code,
                retries=retries,
                retry_delay=retry_delay,
            )
        except Exception as exc:
            failures.append((config_name, str(exc)))
            print(f"[failed] {config_name}: {exc}")

    if failures:
        output_dir.mkdir(parents=True, exist_ok=True)
        failed_path = output_dir / "failed_downloads.txt"
        failed_path.write_text(
            "\n".join(name for name, _ in failures) + "\n",
            encoding="utf-8",
        )
        summary = "\n".join(f"- {name}: {error}" for name, error in failures)
        raise SystemExit(
            f"Failed to download {len(failures)} dataset(s). "
            f"Config names were written to {failed_path}.\n{summary}"
        )


def main() -> None:
    args = parse_args()
    if args.enable_hf_transfer:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    output_dir = args.output_dir.resolve()
    cache_dir = args.cache_dir.resolve()

    if args.configs is not None:
        config_names = args.configs
    elif args.all_configs:
        print(f"[configs] discovering configs for {args.repo_id}")
        config_names = get_config_names(
            repo_id=args.repo_id,
            cache_dir=cache_dir,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        dataset_list_path = args.dataset_list.resolve()
        print(f"[configs] reading dataset list from {dataset_list_path}")
        config_names = read_dataset_list(dataset_list_path)
        validate_config_names(
            requested_config_names=config_names,
            repo_id=args.repo_id,
            cache_dir=cache_dir,
            trust_remote_code=args.trust_remote_code,
        )

    print(f"[start] downloading {len(config_names)} config(s) to {output_dir}")
    print(f"[cache] using HF datasets cache at {cache_dir}")
    if args.enable_hf_transfer:
        print("[transfer] HF_HUB_ENABLE_HF_TRANSFER=1")
    if args.hf_endpoint:
        print(f"[endpoint] HF_ENDPOINT={args.hf_endpoint}")
    download_many(
        repo_id=args.repo_id,
        config_names=config_names,
        split=args.split,
        output_dir=output_dir,
        cache_dir=cache_dir,
        overwrite=args.overwrite,
        trust_remote_code=args.trust_remote_code,
        retries=args.retries,
        retry_delay=args.retry_delay,
    )
    print("[complete] all requested datasets are available locally")


if __name__ == "__main__":
    main()

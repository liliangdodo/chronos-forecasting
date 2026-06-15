"""Download selected Salesforce/GiftEvalPretrain sub-datasets.

The Salesforce/GiftEvalPretrain repository stores each dataset as a
Hugging Face `Dataset.save_to_disk` directory. This script downloads only
the subdirectories listed in `dataset_list.txt`.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path
from typing import Iterable


DEFAULT_REPO_ID = "Salesforce/GiftEvalPretrain"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR
DEFAULT_DATASET_LIST = SCRIPT_DIR / "dataset_list.txt"
DEFAULT_CACHE_DIR = (
    Path("C:/tmp/gifteval_pretrain_hf_cache")
    if os.name == "nt"
    else SCRIPT_DIR / ".hf_cache"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download selected save_to_disk subdirectories from Salesforce/GiftEvalPretrain."
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repo id. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--dataset-list",
        type=Path,
        default=DEFAULT_DATASET_LIST,
        help=f"Text file with one GiftEvalPretrain subdirectory per line. Default: {DEFAULT_DATASET_LIST}",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset subdirectories to download. Overrides --dataset-list.",
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
            "Hugging Face cache directory. On Windows the default is intentionally short to avoid "
            f"path-length issues. Default: {DEFAULT_CACHE_DIR}"
        ),
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of attempts for each dataset. Default: 3",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=10.0,
        help="Seconds to wait between retries. Default: 10",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help=(
            "Maximum number of parallel download workers used by huggingface_hub.snapshot_download. "
            "Set to 1 to download files one by one. Default: 8"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing local dataset directory before downloading it again.",
    )
    parser.add_argument(
        "--enable-hf-transfer",
        action="store_true",
        help="Enable hf_transfer for faster large-file downloads. Requires `pip install hf_transfer`.",
    )
    parser.add_argument(
        "--hf-endpoint",
        default=None,
        help="Override the Hugging Face endpoint, for example https://hf-mirror.com.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the datasets and allow patterns without downloading.",
    )
    return parser.parse_args()


def read_dataset_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset list file not found: {path}")

    names: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        name = line.split("#", maxsplit=1)[0].strip()
        if not name:
            continue
        if name in names:
            raise ValueError(f"Duplicate dataset `{name}` in {path} at line {line_number}")
        names.append(name)

    if not names:
        raise ValueError(f"Dataset list is empty: {path}")
    return names


def is_saved_dataset_complete(path: Path, expected_files: set[str] | None = None) -> bool:
    if not (
        path.is_dir()
        and (path / "state.json").is_file()
        and (path / "dataset_info.json").is_file()
    ):
        return False

    if expected_files is None:
        return True

    local_files = {
        file.relative_to(path).as_posix()
        for file in path.rglob("*")
        if file.is_file()
    }
    return expected_files.issubset(local_files)


def list_local_dataset_files(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        file.relative_to(path).as_posix()
        for file in path.rglob("*")
        if file.is_file()
    }


def get_remote_dataset_files(repo_id: str, dataset_names: Iterable[str]) -> dict[str, set[str]]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    remote_dirs = {file_name.split("/", maxsplit=1)[0] for file_name in files if "/" in file_name}
    missing = [name for name in dataset_names if name not in remote_dirs]
    if missing:
        missing_text = "\n".join(f"- {name}" for name in missing)
        raise ValueError(f"The following dataset directories were not found in {repo_id}:\n{missing_text}")

    remote_files_by_dataset: dict[str, set[str]] = {}
    for dataset_name in dataset_names:
        prefix = f"{dataset_name}/"
        remote_files_by_dataset[dataset_name] = {
            file_name.removeprefix(prefix)
            for file_name in files
            if file_name.startswith(prefix) and not file_name.endswith("/")
        }
    return remote_files_by_dataset


def download_one(
    *,
    repo_id: str,
    dataset_name: str,
    output_dir: Path,
    cache_dir: Path,
    overwrite: bool,
    expected_files: set[str],
    max_workers: int,
) -> Path:
    from huggingface_hub import snapshot_download

    dataset_dir = output_dir / dataset_name
    if is_saved_dataset_complete(dataset_dir, expected_files=expected_files):
        if overwrite:
            print(f"[overwrite] {dataset_name}: removing complete dataset directory {dataset_dir}")
            shutil.rmtree(dataset_dir)
        else:
            print(f"[skip] {dataset_name}: complete dataset already exists at {dataset_dir}")
            return dataset_dir
    elif dataset_dir.exists():
        if overwrite:
            print(f"[overwrite] {dataset_name}: removing incomplete dataset directory {dataset_dir}")
            shutil.rmtree(dataset_dir)
        else:
            local_files = list_local_dataset_files(dataset_dir)
            missing_count = len(expected_files - local_files)
            print(
                f"[resume] {dataset_name}: keeping {len(local_files)} existing file(s), "
                f"downloading {missing_count} missing file(s)"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    allow_patterns = [f"{dataset_name}/*"]
    print(f"[download] {repo_id}/{dataset_name}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(output_dir),
        cache_dir=str(cache_dir),
        allow_patterns=allow_patterns,
        local_dir_use_symlinks=False,
        max_workers=max_workers,
    )

    if not is_saved_dataset_complete(dataset_dir, expected_files=expected_files):
        local_files = list_local_dataset_files(dataset_dir)
        missing_files = sorted(expected_files - local_files)
        preview = ", ".join(missing_files[:10])
        suffix = "" if len(missing_files) <= 10 else f", ... ({len(missing_files)} total)"
        raise RuntimeError(f"Downloaded directory is incomplete: {dataset_dir}. Missing: {preview}{suffix}")

    print(f"[done] {dataset_name}: saved={dataset_dir}")
    return dataset_dir


def download_one_with_retries(
    *,
    repo_id: str,
    dataset_name: str,
    output_dir: Path,
    cache_dir: Path,
    overwrite: bool,
    expected_files: set[str],
    max_workers: int,
    retries: int,
    retry_delay: float,
) -> Path:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            if retries > 1:
                print(f"[attempt] {dataset_name}: {attempt}/{retries}")
            return download_one(
                repo_id=repo_id,
                dataset_name=dataset_name,
                output_dir=output_dir,
                cache_dir=cache_dir,
                overwrite=overwrite,
                expected_files=expected_files,
                max_workers=max_workers,
            )
        except Exception as exc:
            last_error = exc
            print(f"[retryable-failed] {dataset_name}: attempt {attempt}/{retries}: {exc}")
            if attempt < retries:
                time.sleep(retry_delay)

    assert last_error is not None
    raise last_error


def download_many(
    *,
    repo_id: str,
    dataset_names: list[str],
    output_dir: Path,
    cache_dir: Path,
    overwrite: bool,
    remote_files_by_dataset: dict[str, set[str]],
    max_workers: int,
    retries: int,
    retry_delay: float,
) -> None:
    failures: list[tuple[str, str]] = []
    for dataset_name in dataset_names:
        try:
            download_one_with_retries(
                repo_id=repo_id,
                dataset_name=dataset_name,
                output_dir=output_dir,
                cache_dir=cache_dir,
                overwrite=overwrite,
                expected_files=remote_files_by_dataset[dataset_name],
                max_workers=max_workers,
                retries=retries,
                retry_delay=retry_delay,
            )
        except Exception as exc:
            failures.append((dataset_name, str(exc)))
            print(f"[failed] {dataset_name}: {exc}")

    if failures:
        output_dir.mkdir(parents=True, exist_ok=True)
        failed_path = output_dir / "failed_downloads.txt"
        failed_path.write_text("\n".join(name for name, _ in failures) + "\n", encoding="utf-8")
        summary = "\n".join(f"- {name}: {error}" for name, error in failures)
        raise SystemExit(
            f"Failed to download {len(failures)} dataset(s). "
            f"Dataset names were written to {failed_path}.\n{summary}"
        )


def main() -> None:
    args = parse_args()
    if args.enable_hf_transfer:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    if args.max_workers < 1:
        raise ValueError(f"--max-workers must be at least 1, got {args.max_workers}")

    dataset_names = args.datasets if args.datasets is not None else read_dataset_list(args.dataset_list.resolve())
    output_dir = args.output_dir.resolve()
    cache_dir = args.cache_dir.resolve()

    print(f"[start] selected {len(dataset_names)} dataset(s)")
    for dataset_name in dataset_names:
        print(f"[selected] {dataset_name}")
    print(f"[output] {output_dir}")
    print(f"[cache] {cache_dir}")
    print(f"[max-workers] {args.max_workers}")
    if args.enable_hf_transfer:
        print("[transfer] HF_HUB_ENABLE_HF_TRANSFER=1")
    if args.hf_endpoint:
        print(f"[endpoint] HF_ENDPOINT={args.hf_endpoint}")

    if args.dry_run:
        for dataset_name in dataset_names:
            print(f"[dry-run] allow_patterns={dataset_name}/*")
        return

    remote_files_by_dataset = get_remote_dataset_files(args.repo_id, dataset_names)
    download_many(
        repo_id=args.repo_id,
        dataset_names=dataset_names,
        output_dir=output_dir,
        cache_dir=cache_dir,
        overwrite=args.overwrite,
        remote_files_by_dataset=remote_files_by_dataset,
        max_workers=args.max_workers,
        retries=args.retries,
        retry_delay=args.retry_delay,
    )
    print("[complete] all requested GiftEvalPretrain datasets are available locally")


if __name__ == "__main__":
    main()

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import csv
import json
import logging
import math
import random
import time
from copy import deepcopy
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

import numpy as np
import torch
import typer
import transformers
from gluonts.itertools import Cyclic, Filter, Map
from torch.utils.data import IterableDataset, get_worker_info
from transformers import TrainingArguments
from transformers.trainer_callback import TrainerCallback
from typer_config import use_yaml_config

from chronos import Chronos2Pipeline
from chronos.chronos2 import Chronos2Model
from chronos.chronos2.config import Chronos2CoreConfig
from chronos.chronos2.dataset import DatasetMode, left_pad_and_cat_2D, prepare_inputs
from chronos.chronos2.trainer import Chronos2Trainer, EvaluateAndSaveFinalStepCallback
from train import get_next_path, is_main_process, log_on_main, save_training_info


app = typer.Typer(pretty_exceptions_enable=False)
logger = logging.getLogger(__file__)


def configure_logger(log_dir: Optional[Path] = None) -> None:
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None and is_main_process():
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "train.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def _parse_collection_arg(value, name: str, expected_type: type):
    if isinstance(value, str):
        value = ast.literal_eval(value)

    if not isinstance(value, expected_type):
        raise TypeError(f"Expected `{name}` to be of type {expected_type.__name__}, got {type(value).__name__}.")

    return value


def _parse_optional_list_arg(value, name: str) -> Optional[list]:
    if value is None:
        return None
    return _parse_collection_arg(value, name=name, expected_type=list)


def _to_numpy_sequence(value: Any, np_dtype=None):
    if value is None:
        return None

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    if isinstance(value, np.ndarray):
        array = value
    elif isinstance(value, (list, tuple)):
        if value and all(isinstance(item, (list, tuple, np.ndarray)) for item in value):
            array = np.stack(
                [
                    np.asarray(item, dtype=np_dtype)
                    if np_dtype is not None
                    else np.asarray(item)
                    for item in value
                ]
            )
        else:
            array = np.asarray(value, dtype=np_dtype) if np_dtype is not None else np.asarray(value)
    else:
        array = np.asarray(value, dtype=np_dtype) if np_dtype is not None else np.asarray(value)

    if array.dtype == object:
        flat_items = array.tolist()
        if flat_items and all(isinstance(item, (list, tuple, np.ndarray)) for item in flat_items):
            array = np.stack(
                [
                    np.asarray(item, dtype=np_dtype)
                    if np_dtype is not None
                    else np.asarray(item)
                    for item in flat_items
                ]
            )
        elif np_dtype is not None:
            array = array.astype(np_dtype)
    elif np_dtype is not None and np.issubdtype(array.dtype, np.number):
        array = array.astype(np_dtype)

    return array


def _infer_target_from_sequence_columns(entry: Mapping[str, Any], np_dtype=np.float32) -> np.ndarray:
    """Infer target variates from numeric sequence columns in long-format HF datasets."""
    excluded_columns = {
        "id",
        "item_id",
        "start",
        "timestamp",
        "freq",
        "past_covariates",
        "future_covariates",
    }
    expected_length = len(entry["timestamp"]) if "timestamp" in entry else None
    target_columns: list[str] = []
    target_arrays: list[np.ndarray] = []

    for column_name, value in entry.items():
        if column_name in excluded_columns or value is None:
            continue
        try:
            array = _to_numpy_sequence(value, np_dtype=np_dtype)
        except (TypeError, ValueError):
            continue
        if array.ndim != 1 or not np.issubdtype(array.dtype, np.number):
            continue
        if expected_length is not None and len(array) != expected_length:
            continue
        target_columns.append(column_name)
        target_arrays.append(array)

    if not target_arrays:
        raise ValueError(
            "Chronos-2 training data entry does not contain `target`, and no numeric sequence columns "
            f"could be inferred as targets. Available columns: {sorted(entry.keys())}"
        )

    logger.debug(f"Inferred target columns: {target_columns}")
    return np.stack(target_arrays)


def _normalize_target(entry: Mapping[str, Any], np_dtype=np.float32) -> np.ndarray:
    if "target" in entry:
        target = _to_numpy_sequence(entry["target"], np_dtype=np_dtype)
    else:
        target = _infer_target_from_sequence_columns(entry, np_dtype=np_dtype)

    if target.ndim == 1:
        target = target[None, :]
    elif target.ndim != 2:
        raise ValueError(
            "Chronos-2 training data expects each `target` to be either 1-d "
            f"or 2-d, but found shape {target.shape}."
        )

    return target


def _normalize_covariates(
    raw_covariates: Any,
    *,
    field_name: str,
) -> dict[str, np.ndarray]:
    if raw_covariates is None:
        return {}
    if not isinstance(raw_covariates, Mapping):
        raise TypeError(f"Expected `{field_name}` to be a mapping, got {type(raw_covariates).__name__}.")

    normalized_covariates: dict[str, np.ndarray] = {}
    for cov_name, cov_value in raw_covariates.items():
        if cov_value is None:
            continue
        if hasattr(cov_value, "__len__") and len(cov_value) == 0:
            continue
        normalized_covariates[cov_name] = _to_numpy_sequence(cov_value)

    return normalized_covariates


def _normalize_arrow_entry(
    entry: Mapping[str, Any],
    np_dtype=np.float32,
) -> dict[str, Any]:
    normalized: dict[str, Any] = {"target": _normalize_target(dict(entry), np_dtype=np_dtype)}

    past_covariates = _normalize_covariates(entry.get("past_covariates"), field_name="past_covariates")
    future_covariates = _normalize_covariates(entry.get("future_covariates"), field_name="future_covariates")

    invalid_future_keys = set(future_covariates) - set(past_covariates)
    if invalid_future_keys:
        raise ValueError(
            "`future_covariates` keys must be a subset of `past_covariates` keys, "
            f"but found missing history for: {sorted(invalid_future_keys)}."
        )

    if past_covariates:
        normalized["past_covariates"] = past_covariates
    if future_covariates:
        normalized["future_covariates"] = future_covariates

    return normalized


def _to_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric_value):
        return None
    return numeric_value


def _format_metric_value(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.10g}"


def _get_world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def _write_training_benchmark_csv(
    path: Path,
    *,
    output_dir: Path,
    launch_time: datetime,
    end_time: datetime,
    training_duration_seconds: float,
    train_metrics: Mapping[str, Any],
    trainer_n_gpu: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    context_length: int,
    prediction_length: int,
    memory_metrics: Optional[Mapping[str, Optional[float]]] = None,
) -> None:
    cuda_device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    cuda_device_names = (
        " | ".join(torch.cuda.get_device_name(idx) for idx in range(cuda_device_count))
        if cuda_device_count > 0
        else ""
    )
    world_size = _get_world_size()
    effective_global_batch_size = (
        per_device_train_batch_size
        * max(trainer_n_gpu, 1)
        * max(world_size, 1)
        * gradient_accumulation_steps
    )

    row = {
        "run_dir": str(output_dir),
        "launch_time": launch_time.isoformat(timespec="seconds"),
        "end_time": end_time.isoformat(timespec="seconds"),
        "cuda_device_count": cuda_device_count,
        "cuda_device_names": cuda_device_names,
        "trainer_n_gpu": trainer_n_gpu,
        "world_size": world_size,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_global_batch_size": effective_global_batch_size,
        "context_length": context_length,
        "prediction_length": prediction_length,
        "training_duration_seconds": _format_metric_value(training_duration_seconds),
        "train_runtime": _format_metric_value(_to_optional_float(train_metrics.get("train_runtime"))),
        "train_samples_per_second": _format_metric_value(_to_optional_float(train_metrics.get("train_samples_per_second"))),
        "train_steps_per_second": _format_metric_value(_to_optional_float(train_metrics.get("train_steps_per_second"))),
        "train_loss": _format_metric_value(_to_optional_float(train_metrics.get("train_loss"))),
        "epoch": _format_metric_value(_to_optional_float(train_metrics.get("epoch"))),
        "global_step": int(train_metrics.get("global_step", 0)),
        "max_steps": int(train_metrics.get("max_steps", 0)),
        "total_flos": _format_metric_value(_to_optional_float(train_metrics.get("total_flos"))),
    }
    if memory_metrics is not None:
        memory_device_index = memory_metrics.get("memory_device_index")
        row.update(
            {
                "memory_device_index": -1 if memory_device_index is None else int(memory_device_index),
                "max_memory_allocated_mb": _format_metric_value(memory_metrics.get("max_memory_allocated_mb")),
                "max_memory_reserved_mb": _format_metric_value(memory_metrics.get("max_memory_reserved_mb")),
                "avg_memory_allocated_mb": _format_metric_value(memory_metrics.get("avg_memory_allocated_mb")),
                "avg_memory_reserved_mb": _format_metric_value(memory_metrics.get("avg_memory_reserved_mb")),
            }
        )

    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _write_loss_plot_svg(
    path: Path,
    *,
    title: str,
    x_label: str,
    x_key: str,
    loss_points: list[tuple[float, float]],
    eval_points: list[tuple[float, float]],
    train_summary_points: list[tuple[float, float]],
) -> None:
    width = 960
    height = 540
    left = 80
    right = 30
    top = 40
    bottom = 60
    plot_width = width - left - right
    plot_height = height - top - bottom
    series = [
        ("loss", "#1f77b4", loss_points),
        ("eval_loss", "#d62728", eval_points),
        ("train_loss", "#2ca02c", train_summary_points),
    ]
    available_series = [(label, color, points) for label, color, points in series if points]

    if not available_series:
        path.write_text(
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
                f'<rect width="100%" height="100%" fill="white"/>'
                f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle" '
                f'font-family="Arial" font-size="18">{title}: no data</text>'
                "</svg>"
            ),
            encoding="utf-8",
        )
        return

    all_x = [point[0] for _, _, points in available_series for point in points]
    all_y = [point[1] for _, _, points in available_series for point in points]
    x_min = min(all_x)
    x_max = max(all_x)
    y_min = min(all_y)
    y_max = max(all_y)

    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    if y_min == y_max:
        y_padding = max(abs(y_min) * 0.05, 0.05)
        y_min -= y_padding
        y_max += y_padding
    else:
        y_padding = (y_max - y_min) * 0.08
        y_min -= y_padding
        y_max += y_padding

    def scale_x(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def scale_y(value: float) -> float:
        return top + plot_height - (value - y_min) / (y_max - y_min) * plot_height

    def polyline(points: list[tuple[float, float]]) -> str:
        return " ".join(f"{scale_x(x):.2f},{scale_y(y):.2f}" for x, y in points)

    y_ticks = 5
    x_ticks = 5
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="24" text-anchor="middle" font-family="Arial" font-size="20">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333"/>',
    ]

    for idx in range(y_ticks + 1):
        value = y_min + (y_max - y_min) * idx / y_ticks
        y = scale_y(value)
        svg_parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e0e0e0"/>'
        )
        svg_parts.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="12">{value:.4g}</text>'
        )

    for idx in range(x_ticks + 1):
        value = x_min + (x_max - x_min) * idx / x_ticks
        x = scale_x(value)
        svg_parts.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#f0f0f0"/>'
        )
        svg_parts.append(
            f'<text x="{x:.2f}" y="{top + plot_height + 22}" text-anchor="middle" font-family="Arial" font-size="12">{value:.4g}</text>'
        )

    for label, color, points in available_series:
        svg_parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{polyline(points)}"/>'
        )
        last_x, last_y = points[-1]
        svg_parts.append(
            f'<circle cx="{scale_x(last_x):.2f}" cy="{scale_y(last_y):.2f}" r="3.5" fill="{color}"/>'
        )

    legend_x = left + plot_width - 150
    legend_y = top + 10
    for idx, (label, color, _) in enumerate(available_series):
        y = legend_y + idx * 22
        svg_parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 20}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        svg_parts.append(
            f'<text x="{legend_x + 28}" y="{y + 4}" font-family="Arial" font-size="12">{label}</text>'
        )

    svg_parts.append(
        f'<text x="{left + plot_width / 2}" y="{height - 15}" text-anchor="middle" font-family="Arial" font-size="14">{x_label}</text>'
    )
    svg_parts.append(
        f'<text x="22" y="{top + plot_height / 2}" text-anchor="middle" font-family="Arial" font-size="14" '
        f'transform="rotate(-90 22 {top + plot_height / 2})">loss</text>'
    )
    svg_parts.append(f"<!-- x_key: {x_key} -->")
    svg_parts.append("</svg>")
    path.write_text("\n".join(svg_parts), encoding="utf-8")


class LossHistoryCallback(TrainerCallback):
    """Persist loss logs to CSV and generate training curves."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self.output_dir: Optional[Path] = None
        self.csv_path: Optional[Path] = None
        self.records: list[dict[str, Optional[float] | str | int]] = []

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "loss_history.csv"
        with self.csv_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=["event", "global_step", "epoch", "loss", "eval_loss", "train_loss", "learning_rate"],
            )
            writer.writeheader()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs or self.csv_path is None:
            return

        row = {
            "event": "log",
            "global_step": int(state.global_step),
            "epoch": _to_optional_float(logs.get("epoch", state.epoch)),
            "loss": _to_optional_float(logs.get("loss")),
            "eval_loss": _to_optional_float(logs.get("eval_loss")),
            "train_loss": _to_optional_float(logs.get("train_loss")),
            "learning_rate": _to_optional_float(logs.get("learning_rate")),
        }

        if row["loss"] is None and row["eval_loss"] is None and row["train_loss"] is None:
            return

        self.records.append(row)
        with self.csv_path.open("a", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=["event", "global_step", "epoch", "loss", "eval_loss", "train_loss", "learning_rate"],
            )
            writer.writerow(
                {
                    "event": row["event"],
                    "global_step": row["global_step"],
                    "epoch": _format_metric_value(row["epoch"]),
                    "loss": _format_metric_value(row["loss"]),
                    "eval_loss": _format_metric_value(row["eval_loss"]),
                    "train_loss": _format_metric_value(row["train_loss"]),
                    "learning_rate": _format_metric_value(row["learning_rate"]),
                }
            )

        metrics_to_log = []
        if row["loss"] is not None:
            metrics_to_log.append(f"loss={row['loss']:.6f}")
        if row["eval_loss"] is not None:
            metrics_to_log.append(f"eval_loss={row['eval_loss']:.6f}")
        if row["train_loss"] is not None:
            metrics_to_log.append(f"train_loss={row['train_loss']:.6f}")
        epoch_text = "" if row["epoch"] is None else f", epoch={row['epoch']:.4f}"
        log_on_main(
            f"Loss update: step={row['global_step']}{epoch_text}, " + ", ".join(metrics_to_log),
            self.logger,
        )

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or self.output_dir is None:
            return

        loss_step_points = [
            (float(record["global_step"]), float(record["loss"]))
            for record in self.records
            if record["loss"] is not None
        ]
        eval_step_points = [
            (float(record["global_step"]), float(record["eval_loss"]))
            for record in self.records
            if record["eval_loss"] is not None
        ]
        train_summary_step_points = [
            (float(record["global_step"]), float(record["train_loss"]))
            for record in self.records
            if record["train_loss"] is not None
        ]
        loss_epoch_points = [
            (float(record["epoch"]), float(record["loss"]))
            for record in self.records
            if record["epoch"] is not None and record["loss"] is not None
        ]
        eval_epoch_points = [
            (float(record["epoch"]), float(record["eval_loss"]))
            for record in self.records
            if record["epoch"] is not None and record["eval_loss"] is not None
        ]
        train_summary_epoch_points = [
            (float(record["epoch"]), float(record["train_loss"]))
            for record in self.records
            if record["epoch"] is not None and record["train_loss"] is not None
        ]

        _write_loss_plot_svg(
            self.output_dir / "loss_vs_step.svg",
            title="Loss vs Step",
            x_label="step",
            x_key="global_step",
            loss_points=loss_step_points,
            eval_points=eval_step_points,
            train_summary_points=train_summary_step_points,
        )
        _write_loss_plot_svg(
            self.output_dir / "loss_vs_epoch.svg",
            title="Loss vs Epoch",
            x_label="epoch",
            x_key="epoch",
            loss_points=loss_epoch_points,
            eval_points=eval_epoch_points,
            train_summary_points=train_summary_epoch_points,
        )


class MemoryTrackingCallback(TrainerCallback):
    """Track peak and average CUDA memory usage for the local training process."""

    def __init__(self) -> None:
        self.device_index: Optional[int] = None
        self.allocated_sum_mb = 0.0
        self.reserved_sum_mb = 0.0
        self.sample_count = 0

    def _can_track(self, state) -> bool:
        return state.is_world_process_zero and torch.cuda.is_available()

    def on_train_begin(self, args, state, control, **kwargs):
        if not self._can_track(state):
            return
        self.device_index = torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats(self.device_index)
        self.allocated_sum_mb = 0.0
        self.reserved_sum_mb = 0.0
        self.sample_count = 0
        self._sample_memory()

    def on_step_end(self, args, state, control, **kwargs):
        if not self._can_track(state):
            return
        self._sample_memory()

    def on_substep_end(self, args, state, control, **kwargs):
        if not self._can_track(state):
            return
        self._sample_memory()

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if not self._can_track(state):
            return
        self._sample_memory()

    def on_train_end(self, args, state, control, **kwargs):
        if not self._can_track(state):
            return
        self._sample_memory()

    def _sample_memory(self) -> None:
        assert self.device_index is not None
        allocated_mb = torch.cuda.memory_allocated(self.device_index) / (1024**2)
        reserved_mb = torch.cuda.memory_reserved(self.device_index) / (1024**2)
        self.allocated_sum_mb += allocated_mb
        self.reserved_sum_mb += reserved_mb
        self.sample_count += 1

    def get_metrics(self) -> dict[str, Optional[float]]:
        if self.device_index is None or not torch.cuda.is_available():
            return {
                "memory_device_index": None,
                "max_memory_allocated_mb": None,
                "max_memory_reserved_mb": None,
                "avg_memory_allocated_mb": None,
                "avg_memory_reserved_mb": None,
            }

        average_allocated_mb = None
        average_reserved_mb = None
        if self.sample_count > 0:
            average_allocated_mb = self.allocated_sum_mb / self.sample_count
            average_reserved_mb = self.reserved_sum_mb / self.sample_count

        return {
            "memory_device_index": float(self.device_index),
            "max_memory_allocated_mb": torch.cuda.max_memory_allocated(self.device_index) / (1024**2),
            "max_memory_reserved_mb": torch.cuda.max_memory_reserved(self.device_index) / (1024**2),
            "avg_memory_allocated_mb": average_allocated_mb,
            "avg_memory_reserved_mb": average_reserved_mb,
        }


class ArrowFileDataset:
    """
    Minimal Arrow dataset reader that preserves multivariate `target` arrays.

    We intentionally avoid `gluonts.dataset.common.FileDataset` here because it
    validates `target` as 1-dimensional, while Chronos-2 pretraining can use
    multivariate records with shape `(n_variates, time_length)`.
    """

    def __init__(self, path: Path, np_dtype=np.float32) -> None:
        self.path = Path(path)
        self.np_dtype = np_dtype
        self._num_rows: Optional[int] = None
        self._batch_offsets: Optional[list[tuple[int, int, int]]] = None

    @staticmethod
    def _entry_from_batch(batch, row_idx: int) -> dict[str, Any]:
        entry = {}
        for col_idx, name in enumerate(batch.schema.names):
            value = batch.column(col_idx)[row_idx].as_py()
            if name in {"past_covariates", "future_covariates"} and isinstance(value, list):
                value = {item["name"]: item["values"] for item in value}
            entry[name] = value
        return entry

    def _open_reader(self):
        import pyarrow as pa
        import pyarrow.ipc as ipc

        source = pa.memory_map(str(self.path), "rb")
        try:
            reader = ipc.open_file(source)
        except pa.ArrowInvalid:
            source.seek(0)
            reader = ipc.open_stream(source)
        return source, reader

    def _ensure_index(self) -> None:
        if self._num_rows is not None and self._batch_offsets is not None:
            return

        source, reader = self._open_reader()
        try:
            batch_offsets = []
            offset = 0
            for batch_idx in range(reader.num_record_batches):
                batch = reader.get_batch(batch_idx)
                batch_offsets.append((offset, offset + batch.num_rows, batch_idx))
                offset += batch.num_rows
            self._num_rows = offset
            self._batch_offsets = batch_offsets
        finally:
            source.close()

    def __len__(self) -> int:
        self._ensure_index()
        assert self._num_rows is not None
        return self._num_rows

    def __getitem__(self, row_index: int) -> dict[str, Any]:
        self._ensure_index()
        assert self._num_rows is not None and self._batch_offsets is not None
        if row_index < 0:
            row_index += self._num_rows
        if row_index < 0 or row_index >= self._num_rows:
            raise IndexError(row_index)

        source, reader = self._open_reader()
        try:
            for start, end, batch_idx in self._batch_offsets:
                if start <= row_index < end:
                    return self._entry_from_batch(reader.get_batch(batch_idx), row_index - start)
        finally:
            source.close()

        raise IndexError(row_index)

    def __iter__(self):
        source, reader = self._open_reader()
        try:
            for batch_idx in range(reader.num_record_batches):
                batch = reader.get_batch(batch_idx)
                for row_idx in range(batch.num_rows):
                    yield self._entry_from_batch(batch, row_idx)
        finally:
            source.close()


class HFDiskDataset:
    """Read a Hugging Face Dataset saved with `Dataset.save_to_disk`."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._dataset = None

    def _load_dataset(self):
        if self._dataset is not None:
            return self._dataset

        try:
            from datasets import load_from_disk
        except ImportError as exc:
            raise ImportError(
                "Reading chronos_datasets from disk requires the Hugging Face `datasets` package. "
                "Install it with `uv sync --extra dev` or `uv pip install datasets`."
            ) from exc

        self._dataset = load_from_disk(str(self.path))
        return self._dataset

    def __len__(self) -> int:
        return len(self._load_dataset())

    def __getitem__(self, row_index: int) -> dict[str, Any]:
        dataset = self._load_dataset()
        if row_index < 0:
            row_index += len(dataset)
        if row_index < 0 or row_index >= len(dataset):
            raise IndexError(row_index)
        return dict(dataset[int(row_index)])

    def __iter__(self):
        dataset = self._load_dataset()
        for entry in dataset:
            yield dict(entry)


def is_hf_saved_dataset(path: Path) -> bool:
    state_path = path / "state.json"
    if not (path.is_dir() and state_path.is_file() and (path / "dataset_info.json").is_file()):
        return False

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        data_files = state["_data_files"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return False

    return bool(data_files) and all((path / data_file["filename"]).is_file() for data_file in data_files)


def is_hf_config_dataset_dir(path: Path, split: str = "train") -> bool:
    return is_hf_saved_dataset(path / split)


def read_dataset_list(path: Path) -> list[str]:
    config_names: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        config_name = line.split("#", maxsplit=1)[0].strip()
        if config_name:
            config_names.append(config_name)
    return config_names


def expand_training_data_path(path: Path, split: str = "train") -> list[Path]:
    """Expand a configured data path into concrete Arrow/HF saved-dataset paths."""
    path = Path(path)

    if path.is_file():
        if path.suffix == ".arrow":
            return [path]
        if path.name == "dataset_list.txt":
            root = path.parent
            return [
                root / config_name / split
                for config_name in read_dataset_list(path)
                if is_hf_saved_dataset(root / config_name / split)
            ]
        return []

    if is_hf_saved_dataset(path):
        return [path]

    if is_hf_config_dataset_dir(path, split=split):
        return [path / split]

    dataset_list_path = path / "dataset_list.txt"
    if dataset_list_path.is_file():
        dataset_paths = []
        for config_name in read_dataset_list(dataset_list_path):
            direct_path = path / config_name
            split_path = direct_path / split
            if is_hf_saved_dataset(direct_path):
                dataset_paths.append(direct_path)
            elif is_hf_saved_dataset(split_path):
                dataset_paths.append(split_path)
        return dataset_paths

    arrow_paths = sorted(path.glob("*.arrow"))
    if arrow_paths:
        return arrow_paths

    hf_dataset_paths = []
    for child in sorted(path.iterdir()):
        if is_hf_saved_dataset(child):
            hf_dataset_paths.append(child)
        elif is_hf_config_dataset_dir(child, split=split):
            hf_dataset_paths.append(child / split)
    return hf_dataset_paths


def expand_training_data_paths(paths: Sequence[Path | str], split: str = "train") -> tuple[list[Path], list[str]]:
    expanded_paths: list[Path] = []
    missing_paths: list[str] = []

    for raw_path in paths:
        path = Path(raw_path)
        concrete_paths = expand_training_data_path(path, split=split)
        if concrete_paths:
            expanded_paths.extend(concrete_paths)
        else:
            missing_paths.append(str(path))

    return expanded_paths, missing_paths


def expand_probabilities_for_paths(
    paths: Sequence[Path | str],
    probabilities: Sequence[float],
    split: str = "train",
) -> tuple[list[Path], list[float], list[str]]:
    expanded_paths: list[Path] = []
    expanded_probabilities: list[float] = []
    missing_paths: list[str] = []

    for raw_path, probability in zip(paths, probabilities):
        concrete_paths = expand_training_data_path(Path(raw_path), split=split)
        if not concrete_paths:
            missing_paths.append(str(raw_path))
            continue
        probability_per_concrete_path = float(probability) / len(concrete_paths)
        expanded_paths.extend(concrete_paths)
        expanded_probabilities.extend([probability_per_concrete_path] * len(concrete_paths))

    if expanded_probabilities:
        probability_sum = sum(expanded_probabilities)
        expanded_probabilities = [prob / probability_sum for prob in expanded_probabilities]

    return expanded_paths, expanded_probabilities, missing_paths


ValidationRecordId = tuple[str, int]


class IdentifiedEntry(dict):
    """Dataset entry carrying a stable source path and row index."""

    def __init__(self, entry: Mapping[str, Any], record_id: ValidationRecordId) -> None:
        super().__init__(entry)
        self.record_id = record_id


class IdentifiedDataset:
    """Attach stable IDs without adding fields to the model input."""

    def __init__(self, dataset, source_path: Path) -> None:
        self.dataset = dataset
        self.source_path = str(source_path.resolve())

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, row_index: int) -> IdentifiedEntry:
        return IdentifiedEntry(self.dataset[row_index], record_id=(self.source_path, int(row_index)))

    def __iter__(self):
        for row_index, entry in enumerate(self.dataset):
            yield IdentifiedEntry(entry, record_id=(self.source_path, row_index))


class ExcludeValidationEntriesDataset:
    """Filter validation records out of the training stream."""

    def __init__(self, dataset, excluded_record_ids: set[ValidationRecordId]) -> None:
        self.dataset = dataset
        self.excluded_record_ids = excluded_record_ids

    def __iter__(self):
        for entry in self.dataset:
            if entry.record_id not in self.excluded_record_ids:
                yield entry


class FilteredRandomAccessDataset:
    """Apply a predicate while preserving source row IDs for random validation sampling."""

    def __init__(self, dataset, predicate) -> None:
        self.dataset = dataset
        self.predicate = predicate

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, row_index: int) -> IdentifiedEntry:
        return self.dataset[row_index]

    def __iter__(self):
        for entry in self.dataset:
            if self.predicate(entry):
                yield entry


def build_raw_dataset(path: Path):
    if path.is_file() and path.suffix == ".arrow":
        dataset = ArrowFileDataset(path=path)
    elif is_hf_saved_dataset(path):
        dataset = HFDiskDataset(path=path)
    else:
        raise ValueError(f"Unsupported training data path: {path}")

    return IdentifiedDataset(dataset=dataset, source_path=path)


def has_enough_chronos2_observations(
    entry: dict,
    min_length: int = 0,
    max_missing_prop: float = 1.0,
) -> bool:
    target = _normalize_target(entry)
    return (
        target.shape[-1] >= min_length
        and np.isnan(target).mean() <= max_missing_prop
        and np.isfinite(target).any()
    )


def _validate_sampling_probabilities(
    datasets: Sequence,
    probabilities: Sequence[float],
) -> tuple[list, list[float]]:
    if len(datasets) != len(probabilities):
        raise ValueError("`datasets` and `probabilities` must have the same length.")
    if not datasets:
        raise ValueError("Cannot sample validation entries from an empty dataset list.")

    active_datasets = []
    active_probabilities = []
    for dataset, probability in zip(datasets, probabilities):
        probability = float(probability)
        if not math.isfinite(probability) or probability < 0:
            raise ValueError(f"Validation sampling probabilities must be finite and non-negative, got {probability}.")
        if probability > 0:
            active_datasets.append(dataset)
            active_probabilities.append(probability)
    if not active_datasets:
        raise ValueError("At least one validation sampling probability must be positive.")

    return active_datasets, active_probabilities


def _allocate_stratified_sample_counts(
    probabilities: Sequence[float],
    num_samples: int,
    rng: np.random.Generator,
) -> list[int]:
    if num_samples <= 0:
        return [0] * len(probabilities)

    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    probabilities_array /= probabilities_array.sum()
    num_strata = len(probabilities_array)

    if num_samples < num_strata:
        selected = rng.choice(num_strata, size=num_samples, replace=False, p=probabilities_array)
        allocations = [0] * num_strata
        for idx in selected:
            allocations[int(idx)] = 1
        return allocations

    allocations = [1] * num_strata
    remaining = num_samples - num_strata
    expected = probabilities_array * remaining
    extra = np.floor(expected).astype(int)
    allocations = [allocation + int(extra_count) for allocation, extra_count in zip(allocations, extra)]

    shortfall = num_samples - sum(allocations)
    if shortfall > 0:
        remainders = expected - extra
        # Add seed-controlled jitter only to break exact ties deterministically.
        tie_breaker = rng.random(num_strata) * 1e-12
        for idx in np.argsort(-(remainders + tie_breaker))[:shortfall]:
            allocations[int(idx)] += 1

    return allocations


def _reservoir_sample_dataset_entries(
    dataset,
    num_samples: int,
    rng: np.random.Generator,
    max_entries_to_scan: int,
) -> list[dict]:
    if num_samples <= 0:
        return []

    samples: list[dict] = []
    seen = 0
    for entry in dataset:
        seen += 1
        if len(samples) < num_samples:
            samples.append(entry)
        else:
            replace_idx = int(rng.integers(seen))
            if replace_idx < num_samples:
                samples[replace_idx] = entry

        if seen >= max_entries_to_scan:
            break

    return samples


def _has_random_access(dataset) -> bool:
    return hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__")


def _random_sample_dataset_entries(
    dataset,
    num_samples: int,
    rng: np.random.Generator,
    max_attempts_multiplier: int,
) -> list[dict]:
    if num_samples <= 0:
        return []
    if not _has_random_access(dataset):
        return []

    dataset_length = len(dataset)
    if dataset_length <= 0:
        return []

    target_num_samples = min(num_samples, dataset_length)
    max_attempts = min(dataset_length, max(target_num_samples * max_attempts_multiplier, target_num_samples))
    candidate_indices = rng.choice(dataset_length, size=max_attempts, replace=False)
    samples = []

    for row_index in candidate_indices:
        entry = dataset[int(row_index)]
        if isinstance(dataset, FilteredRandomAccessDataset) and not dataset.predicate(entry):
            continue

        samples.append(entry)
        if len(samples) >= target_num_samples:
            break

    return samples


def sample_validation_entries(
    datasets: Sequence,
    probabilities: Sequence[float],
    num_samples: int,
    seed: int,
    *,
    stratified: bool = True,
    scan_multiplier: int = 5,
    random_access_attempt_multiplier: int = 5,
) -> list[dict]:
    """Sample a fixed, lightweight validation set from one or more iterable datasets."""
    if num_samples <= 0:
        return []
    if scan_multiplier <= 0:
        raise ValueError(f"`scan_multiplier` must be positive, got {scan_multiplier}.")
    if random_access_attempt_multiplier <= 0:
        raise ValueError(
            f"`random_access_attempt_multiplier` must be positive, got {random_access_attempt_multiplier}."
        )

    active_datasets, active_probabilities = _validate_sampling_probabilities(datasets, probabilities)
    rng = np.random.default_rng(seed)

    if stratified:
        allocations = _allocate_stratified_sample_counts(active_probabilities, num_samples, rng)
        sampled_entries = []
        for dataset_idx, (dataset, allocation) in enumerate(zip(active_datasets, allocations)):
            stratum_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            stratum_entries = _random_sample_dataset_entries(
                dataset=dataset,
                num_samples=allocation,
                rng=stratum_rng,
                max_attempts_multiplier=random_access_attempt_multiplier,
            )
            if len(stratum_entries) < allocation:
                remaining = allocation - len(stratum_entries)
                selected_record_ids = {
                    entry.record_id for entry in stratum_entries if isinstance(entry, IdentifiedEntry)
                }
                fallback_entries = _reservoir_sample_dataset_entries(
                    dataset=ExcludeValidationEntriesDataset(dataset, selected_record_ids)
                    if selected_record_ids
                    else dataset,
                    num_samples=remaining,
                    rng=stratum_rng,
                    max_entries_to_scan=max(remaining * scan_multiplier, remaining),
                )
                stratum_entries.extend(fallback_entries)
            if len(stratum_entries) < allocation:
                raise ValueError(
                    f"Only {len(stratum_entries)} usable entries were found in validation stratum {dataset_idx}, "
                    f"but {allocation} were requested."
                )
            sampled_entries.extend(stratum_entries)

        rng.shuffle(sampled_entries)
        return sampled_entries

    iterators = [iter(dataset) for dataset in active_datasets]
    sampled_entries: list[dict] = []

    while len(sampled_entries) < num_samples and active_datasets:
        normalized_probabilities = np.asarray(active_probabilities, dtype=np.float64)
        normalized_probabilities /= normalized_probabilities.sum()
        dataset_idx = int(rng.choice(len(iterators), p=normalized_probabilities))

        try:
            entry = next(iterators[dataset_idx])
        except StopIteration:
            del active_datasets[dataset_idx]
            del active_probabilities[dataset_idx]
            del iterators[dataset_idx]
            continue

        sampled_entries.append(entry)

    if len(sampled_entries) < num_samples:
        raise ValueError(
            f"Only {len(sampled_entries)} unique usable entries were found while sampling "
            f"{num_samples} lightweight validation entries."
        )

    return sampled_entries


def get_validation_record_ids(entries: Sequence[dict]) -> set[ValidationRecordId]:
    record_ids = set()
    for entry in entries:
        if not isinstance(entry, IdentifiedEntry):
            raise TypeError("Validation entries must carry stable source IDs.")
        record_ids.add(entry.record_id)
    return record_ids


def _manifest_path_parts(source_path: str) -> tuple[str, ...]:
    normalized = str(source_path).replace("\\", "/").rstrip("/")
    return tuple(part.lower() for part in normalized.split("/") if part)


def _common_suffix_length(left: Sequence[str], right: Sequence[str]) -> int:
    count = 0
    for left_part, right_part in zip(reversed(left), reversed(right)):
        if left_part != right_part:
            break
        count += 1
    return count


def _get_identified_source_path(dataset) -> Optional[str]:
    current = dataset
    while current is not None:
        source_path = getattr(current, "source_path", None)
        if source_path is not None:
            return str(source_path)
        current = getattr(current, "dataset", None)
    return None


def _build_manifest_source_lookup(datasets: Sequence) -> dict[str, Any]:
    source_lookup = {}
    for dataset in datasets:
        source_path = _get_identified_source_path(dataset)
        if source_path is None:
            raise TypeError("Validation manifest reuse requires datasets with stable source paths.")
        source_lookup[source_path] = dataset
    return source_lookup


def _resolve_manifest_source_dataset(
    manifest_source_path: str,
    source_lookup: Mapping[str, Any],
):
    if manifest_source_path in source_lookup:
        return manifest_source_path, source_lookup[manifest_source_path]

    manifest_parts = _manifest_path_parts(manifest_source_path)
    candidates = []
    for source_path, dataset in source_lookup.items():
        suffix_length = _common_suffix_length(manifest_parts, _manifest_path_parts(source_path))
        if suffix_length > 0:
            candidates.append((suffix_length, source_path, dataset))

    if not candidates:
        raise ValueError(
            f"Validation manifest source path does not match any configured dataset: {manifest_source_path}"
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_suffix_length = candidates[0][0]
    best_candidates = [candidate for candidate in candidates if candidate[0] == best_suffix_length]
    if len(best_candidates) > 1:
        matching_paths = [source_path for _, source_path, _ in best_candidates]
        raise ValueError(
            "Validation manifest source path is ambiguous after suffix matching: "
            f"{manifest_source_path}. Candidates: {matching_paths}"
        )

    _, source_path, dataset = best_candidates[0]
    return source_path, dataset


def load_validation_entries_from_manifest(
    path: Path,
    datasets: Sequence,
    *,
    expected_num_samples: Optional[int] = None,
) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Validation manifest {path} does not contain a `records` list.")

    if expected_num_samples is not None and expected_num_samples > 0 and len(records) != expected_num_samples:
        raise ValueError(
            f"Validation manifest {path} contains {len(records)} records, "
            f"but `validation_num_samples` is {expected_num_samples}."
        )

    source_lookup = _build_manifest_source_lookup(datasets)
    resolved_sources: dict[str, tuple[str, Any]] = {}
    validation_entries = []

    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(f"Invalid validation manifest record: {record}")
        manifest_source_path = record.get("source_path")
        row_index = record.get("row_index")
        if not isinstance(manifest_source_path, str) or not isinstance(row_index, int):
            raise ValueError(f"Invalid validation manifest record: {record}")

        if manifest_source_path not in resolved_sources:
            resolved_sources[manifest_source_path] = _resolve_manifest_source_dataset(
                manifest_source_path,
                source_lookup,
            )
        resolved_source_path, dataset = resolved_sources[manifest_source_path]
        entry = dataset[row_index]
        if isinstance(dataset, FilteredRandomAccessDataset) and not dataset.predicate(entry):
            raise ValueError(
                f"Validation manifest record no longer passes the dataset filter: "
                f"source_path={manifest_source_path}, row_index={row_index}"
            )
        validation_entries.append(IdentifiedEntry(entry, record_id=(resolved_source_path, row_index)))

    return validation_entries


def write_validation_manifest(
    path: Path,
    *,
    validation_seed: int,
    validation_entries: Sequence[dict],
) -> None:
    record_ids = get_validation_record_ids(validation_entries)
    source_counts: dict[str, int] = {}
    for source_path, _ in record_ids:
        source_counts[source_path] = source_counts.get(source_path, 0) + 1
    payload = {
        "validation_seed": validation_seed,
        "num_validation_entries": len(validation_entries),
        "num_excluded_training_entries": len(record_ids),
        "num_sources": len(source_counts),
        "source_counts": [
            {"source_path": source_path, "num_entries": count}
            for source_path, count in sorted(source_counts.items())
        ],
        "records": [
            {"source_path": source_path, "row_index": row_index}
            for source_path, row_index in sorted(record_ids)
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


class Chronos2ArrowDataset(IterableDataset):
    """
    Iterable dataset for Chronos-2 pretraining/fine-tuning on GluonTS Arrow files.

    The expected Arrow schema mirrors `scripts/training/train.py`: each entry should
    contain at least a `target` field. `target` may be univariate `(T,)` or
    multivariate `(n_variates, T)`.
    """

    def __init__(
        self,
        datasets: Sequence,
        probabilities: Sequence[float],
        context_length: int,
        prediction_length: int,
        batch_size: int,
        output_patch_size: int,
        min_past: int,
        mode: str = "training",
        np_dtype=np.float32,
    ) -> None:
        super().__init__()

        assert len(datasets) == len(probabilities)
        assert mode in ("training", "validation")

        self.datasets = datasets
        self.probabilities = list(probabilities)
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.num_output_patches = math.ceil(prediction_length / output_patch_size)
        self.min_past = min_past
        self.mode = mode
        self.np_dtype = np_dtype

    def preprocess_entry(self, entry: Mapping[str, Any]) -> dict[str, Any]:
        normalized_entry = _normalize_arrow_entry(entry, np_dtype=self.np_dtype)
        dataset_mode = DatasetMode.TRAIN if self.mode == "training" else DatasetMode.VALIDATION
        prepared = prepare_inputs(
            [normalized_entry],
            prediction_length=self.prediction_length,
            min_past=self.min_past,
            mode=dataset_mode,
        )
        return prepared[0]

    def _construct_slice(self, prepared: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        past_tensor = prepared["context"].clone()
        n_targets = int(prepared["n_targets"])
        n_covariates = int(prepared["n_covariates"])
        n_future_covariates = int(prepared["n_future_covariates"])
        n_past_only_covariates = n_covariates - n_future_covariates
        full_length = past_tensor.shape[-1]

        if self.mode == "training":
            slice_idx = np.random.randint(self.min_past, full_length - self.prediction_length + 1)
        else:
            slice_idx = full_length - self.prediction_length

        if slice_idx >= self.context_length:
            context = past_tensor[:, slice_idx - self.context_length : slice_idx]
        else:
            context = past_tensor[:, :slice_idx]

        future_target = past_tensor[:, slice_idx : slice_idx + self.prediction_length].clone()
        future_target[n_targets:] = torch.nan

        if n_future_covariates > 0:
            future_covariates = past_tensor[-n_future_covariates:, slice_idx : slice_idx + self.prediction_length]
        else:
            future_covariates = torch.zeros((0, self.prediction_length), dtype=torch.float32)

        future_covariates_padding = torch.full(
            (n_targets + n_past_only_covariates, self.prediction_length),
            fill_value=torch.nan,
            dtype=torch.float32,
        )
        future_covariates = torch.cat([future_covariates_padding, future_covariates], dim=0)

        return context, future_target, future_covariates

    def _build_batch(self, entries: Sequence[dict]) -> dict[str, torch.Tensor | int]:
        batch_context_list = []
        batch_future_target_list = []
        batch_future_covariates_list = []
        batch_group_ids_list = []

        for group_id, entry in enumerate(entries):
            context, future_target, future_covariates = self._construct_slice(entry)
            group_size = context.shape[0]

            batch_context_list.append(context)
            batch_future_target_list.append(future_target)
            batch_future_covariates_list.append(future_covariates)
            batch_group_ids_list.append(torch.full((group_size,), fill_value=group_id, dtype=torch.long))

        return {
            "context": left_pad_and_cat_2D(batch_context_list),
            "future_target": torch.cat(batch_future_target_list, dim=0),
            "future_covariates": torch.cat(batch_future_covariates_list, dim=0),
            "group_ids": torch.cat(batch_group_ids_list, dim=0),
            "num_output_patches": self.num_output_patches,
        }

    def _split_for_workers(self) -> tuple[list, list[float]]:
        datasets = list(self.datasets)
        probabilities = list(self.probabilities)

        worker_info = get_worker_info()
        if worker_info is None:
            return datasets, probabilities

        datasets = list(datasets[worker_info.id :: worker_info.num_workers])
        probabilities = list(probabilities[worker_info.id :: worker_info.num_workers])

        if len(datasets) == 0:
            return [], []

        probabilities = [prob / sum(probabilities) for prob in probabilities]
        return datasets, probabilities

    def _iter_training(self) -> Iterator[dict[str, torch.Tensor | int]]:
        datasets, probabilities = self._split_for_workers()
        if len(datasets) == 0:
            return

        iterators = [iter(Cyclic(Map(self.preprocess_entry, dataset))) for dataset in datasets]

        while True:
            current_batch_size = 0
            entries = []

            while current_batch_size < self.batch_size:
                dataset_idx = np.random.choice(range(len(iterators)), p=probabilities)
                entry = next(iterators[dataset_idx])
                entries.append(entry)
                current_batch_size += entry["context"].shape[0]

            yield self._build_batch(entries)

    def _iter_validation(self) -> Iterator[dict[str, torch.Tensor | int]]:
        datasets, _ = self._split_for_workers()
        if len(datasets) == 0:
            return

        current_batch_size = 0
        entries = []

        for dataset in datasets:
            for raw_entry in dataset:
                entry = self.preprocess_entry(raw_entry)
                entries.append(entry)
                current_batch_size += entry["context"].shape[0]

                if current_batch_size >= self.batch_size:
                    yield self._build_batch(entries)
                    current_batch_size = 0
                    entries = []

        if entries:
            yield self._build_batch(entries)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor | int]]:
        if self.mode == "training":
            yield from self._iter_training()
        else:
            yield from self._iter_validation()


def load_chronos2_model(
    *,
    model_id: Optional[str],
    random_init: bool,
    context_length: int,
    prediction_length: int,
    input_patch_size: int,
    input_patch_stride: int,
    output_patch_size: int,
    max_output_patches: int,
    quantiles: Sequence[float],
    use_reg_token: bool,
    use_arcsinh: bool,
    time_encoding_scale: Optional[int],
    d_model: int,
    d_kv: int,
    d_ff: int,
    num_layers: int,
    num_heads: int,
    dropout_rate: float,
    layer_norm_epsilon: float,
    initializer_factor: float,
    feed_forward_proj: str,
    pad_token_id: int,
    rope_theta: float,
    attn_implementation: Optional[str],
) -> Chronos2Model:
    if random_init:
        log_on_main("Using random initialization", logger)

        chronos_config = {
            "context_length": context_length,
            "input_patch_size": input_patch_size,
            "input_patch_stride": input_patch_stride,
            "output_patch_size": output_patch_size,
            "quantiles": list(quantiles),
            "use_reg_token": use_reg_token,
            "use_arcsinh": use_arcsinh,
            "max_output_patches": max(max_output_patches, math.ceil(prediction_length / output_patch_size)),
            "time_encoding_scale": time_encoding_scale or context_length,
        }
        config = Chronos2CoreConfig(
            d_model=d_model,
            d_kv=d_kv,
            d_ff=d_ff,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_factor=initializer_factor,
            feed_forward_proj=feed_forward_proj,
            pad_token_id=pad_token_id,
            rope_theta=rope_theta,
            attn_implementation=attn_implementation,
            chronos_config=chronos_config,
            architectures=["Chronos2Model"],
            chronos_pipeline_class="Chronos2Pipeline",
        )
        return Chronos2Model(config)

    if model_id is None:
        raise ValueError("`model_id` must be provided when `random_init` is False.")

    log_on_main(f"Using pretrained initialization from {model_id}", logger)
    log_on_main(
        "When `random_init` is False, Chronos-2 architecture parameters from the YAML are loaded from the checkpoint "
        "config instead of being rebuilt from the script arguments. The training script only expands context/output horizon "
        "constraints when needed for the requested prediction setup.",
        logger,
    )
    pipeline = Chronos2Pipeline.from_pretrained(model_id, device_map="cpu")
    model = pipeline.model

    model.chronos_config.context_length = max(model.chronos_config.context_length, context_length)
    model.chronos_config.max_output_patches = max(
        model.chronos_config.max_output_patches,
        math.ceil(prediction_length / model.chronos_config.output_patch_size),
    )
    model.config.chronos_config = model.chronos_config.__dict__
    model.config.chronos_pipeline_class = "Chronos2Pipeline"
    model.config.architectures = ["Chronos2Model"]

    return model


@app.command()
@use_yaml_config(param_name="config")
def main(
    training_data_paths: str,
    probability: Optional[str] = None,
    validation_data_paths: Optional[str] = None,
    validation_num_samples: int = 256,
    validation_seed: int = 0,
    validation_manifest_path: Optional[str] = None,
    context_length: int = 2048,
    prediction_length: int = 64,
    min_past: int = 64,
    max_steps: int = 200_000,
    save_steps: int = 50_000,
    log_steps: int = 500,
    per_device_train_batch_size: int = 256,
    per_device_eval_batch_size: Optional[int] = None,
    learning_rate: float = 1e-3,
    optim: str = "adamw_torch_fused",
    gradient_accumulation_steps: int = 1,
    model_id: Optional[str] = None,
    random_init: bool = True,
    output_dir: str = "./output/",
    tf32: bool = True,
    bf16: bool = True,
    torch_compile: bool = False,
    dataloader_num_workers: int = 0,
    max_missing_prop: float = 0.9,
    disable_data_parallel: bool = True,
    lr_scheduler_type: str = "linear",
    warmup_ratio: float = 0.0,
    seed: Optional[int] = None,
    input_patch_size: int = 16,
    input_patch_stride: int = 16,
    output_patch_size: int = 16,
    max_output_patches: int = 64,
    quantiles: str = "[0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]",
    use_reg_token: bool = True,
    use_arcsinh: bool = True,
    time_encoding_scale: Optional[int] = None,
    d_model: int = 512,
    d_kv: int = 64,
    d_ff: int = 2048,
    num_layers: int = 6,
    num_heads: int = 8,
    dropout_rate: float = 0.1,
    layer_norm_epsilon: float = 1e-6,
    initializer_factor: float = 0.05,
    feed_forward_proj: str = "relu",
    pad_token_id: int = 0,
    rope_theta: float = 10000.0,
    attn_implementation: Optional[str] = "sdpa",
):
    launch_time = datetime.now().astimezone()
    if seed is None:
        seed = random.randint(0, 2**32)

    transformers.set_seed(seed=seed)
    log_on_main(f"Using SEED: {seed}", logger)

    training_data_paths = _parse_collection_arg(training_data_paths, "training_data_paths", list)
    validation_data_paths = _parse_optional_list_arg(validation_data_paths, "validation_data_paths")
    quantiles = _parse_collection_arg(quantiles, "quantiles", list)
    validation_manifest_path = Path(validation_manifest_path) if validation_manifest_path else None

    if probability is None:
        probability = [1.0 / len(training_data_paths)] * len(training_data_paths)
    else:
        probability = _parse_collection_arg(probability, "probability", list)

    if len(training_data_paths) != len(probability):
        raise ValueError(
            f"`training_data_paths` and `probability` must have the same length, got {len(training_data_paths)} "
            f"and {len(probability)}."
        )
    probability = [float(prob) for prob in probability]
    if any(not math.isfinite(prob) or prob < 0 for prob in probability) or sum(probability) <= 0:
        raise ValueError("`probability` values must be finite, non-negative, and include at least one positive value.")
    probability = [prob / sum(probability) for prob in probability]
    if validation_num_samples < 0:
        raise ValueError(f"`validation_num_samples` must be non-negative, got {validation_num_samples}.")

    expanded_training_data_paths, expanded_probability, missing_training_paths = expand_probabilities_for_paths(
        training_data_paths,
        probability,
        split="train",
    )
    if missing_training_paths:
        raise ValueError(
            "The following training data path(s) did not resolve to Arrow files or Hugging Face saved datasets: "
            f"{missing_training_paths}"
        )
    if not expanded_training_data_paths:
        raise ValueError("No training datasets were found after expanding `training_data_paths`.")

    expanded_validation_data_paths: Optional[list[Path]] = None
    if validation_data_paths:
        expanded_validation_data_paths, missing_validation_paths = expand_training_data_paths(
            validation_data_paths,
            split="train",
        )
        if missing_validation_paths:
            raise ValueError(
                "The following validation data path(s) did not resolve to Arrow files or Hugging Face saved datasets: "
                f"{missing_validation_paths}"
            )
        if not expanded_validation_data_paths:
            raise ValueError("No validation datasets were found after expanding `validation_data_paths`.")

    if input_patch_size != output_patch_size:
        raise ValueError(
            "Chronos-2 requires `input_patch_size == output_patch_size`, "
            f"but found {input_patch_size} and {output_patch_size}."
        )

    tf32_supported = torch.cuda.is_available() and torch.cuda.is_tf32_supported()
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if tf32 and not tf32_supported:
        log_on_main(
            "TF32 format is not supported by the current CUDA device. Setting tf32 to False.",
            logger,
        )
        tf32 = False

    if bf16 and not bf16_supported:
        log_on_main(
            "bf16 is not supported by the current CUDA device. Setting bf16 to False.",
            logger,
        )
        bf16 = False

    if dataloader_num_workers > len(expanded_training_data_paths):
        log_on_main(
            f"Setting the number of data loader workers to {len(expanded_training_data_paths)}, instead of {dataloader_num_workers}.",
            logger,
        )
        dataloader_num_workers = len(expanded_training_data_paths)

    raw_training_config = deepcopy(locals())
    output_dir = get_next_path("run", base_dir=Path(output_dir), file_type="")
    configure_logger(output_dir)

    log_on_main(f"Logging dir: {output_dir}", logger)
    log_on_main(
        f"Loading and filtering {len(training_data_paths)} datasets for training: {training_data_paths}",
        logger,
    )
    log_on_main(
        f"Expanded to {len(expanded_training_data_paths)} concrete training datasets: "
        f"{[str(path) for path in expanded_training_data_paths]}",
        logger,
    )
    log_on_main(f"Configured mixing probabilities: {probability}", logger)
    log_on_main(f"Expanded mixing probabilities: {expanded_probability}", logger)
    log_on_main(
        "The default Chronos-2 training config is intended to work directly with multivariate Arrow datasets "
        "such as `kernelsynth-cotemporaneous-multivariate-data.arrow`, where each record stores "
        "`target` with shape (n_variates, time_length). Univariate Arrow inputs are still accepted, and records "
        "may additionally provide `past_covariates` / `future_covariates` using the same organization rules as "
        "`src/chronos/chronos2/dataset.py`.",
        logger,
    )
    log_on_main(
        "Training loss history will be written to `loss_history.csv`, and loss curves will be written to "
        "`loss_vs_step.svg` and `loss_vs_epoch.svg` inside the run output directory.",
        logger,
    )
    log_on_main(
        "Training benchmark summary will include peak and average CUDA memory usage for the main training process.",
        logger,
    )

    train_datasets = [
        FilteredRandomAccessDataset(
            build_raw_dataset(Path(data_path)),
            partial(
                has_enough_chronos2_observations,
                min_length=min_past + prediction_length,
                max_missing_prop=max_missing_prop,
            ),
        )
        for data_path in expanded_training_data_paths
    ]

    eval_dataset = None
    excluded_training_record_ids: set[ValidationRecordId] = set()
    memory_callback = MemoryTrackingCallback()
    callbacks = [LossHistoryCallback(logger), memory_callback]
    if validation_num_samples > 0:
        validation_source = "configured validation datasets"
        validation_probabilities = [1.0] * len(expanded_validation_data_paths or [])
        if expanded_validation_data_paths:
            log_on_main(
                f"Using {len(expanded_validation_data_paths)} configured validation datasets: "
                f"{[str(path) for path in expanded_validation_data_paths]}",
                logger,
            )
            validation_datasets = [
                FilteredRandomAccessDataset(
                    build_raw_dataset(Path(data_path)),
                    partial(
                        has_enough_chronos2_observations,
                        min_length=min_past + prediction_length,
                        max_missing_prop=max_missing_prop,
                    ),
                )
                for data_path in expanded_validation_data_paths
            ]
        else:
            validation_source = "training datasets"
            validation_datasets = train_datasets
            validation_probabilities = expanded_probability

        if validation_manifest_path is not None:
            log_on_main(
                f"Loading fixed lightweight validation entries from manifest: {validation_manifest_path}",
                logger,
            )
            validation_entries = load_validation_entries_from_manifest(
                validation_manifest_path,
                validation_datasets,
                expected_num_samples=validation_num_samples,
            )
            validation_source = f"manifest {validation_manifest_path}"
        else:
            log_on_main(
                f"Sampling {validation_num_samples} fixed entries for lightweight validation.",
                logger,
            )
            validation_entries = sample_validation_entries(
                datasets=validation_datasets,
                probabilities=validation_probabilities,
                num_samples=validation_num_samples,
                seed=validation_seed,
            )
        excluded_training_record_ids = get_validation_record_ids(validation_entries)
        log_on_main(
            f"Cached {len(validation_entries)} lightweight validation entries from {validation_source}; "
            f"excluding {len(excluded_training_record_ids)} matching records from training.",
            logger,
        )
        if is_main_process():
            validation_manifest_path = output_dir / "validation_manifest.json"
            write_validation_manifest(
                validation_manifest_path,
                validation_seed=validation_seed,
                validation_entries=validation_entries,
            )
            log_on_main(f"Validation manifest written to {validation_manifest_path}", logger)
        eval_dataset = Chronos2ArrowDataset(
            datasets=[validation_entries],
            probabilities=[1.0],
            context_length=context_length,
            prediction_length=prediction_length,
            batch_size=per_device_eval_batch_size or per_device_train_batch_size,
            output_patch_size=output_patch_size,
            min_past=min_past,
            mode="validation",
        )
        callbacks.append(EvaluateAndSaveFinalStepCallback())

    if excluded_training_record_ids:
        train_datasets = [
            ExcludeValidationEntriesDataset(dataset, excluded_training_record_ids)
            for dataset in train_datasets
        ]

    log_on_main("Initializing model", logger)
    model = load_chronos2_model(
        model_id=model_id,
        random_init=random_init,
        context_length=context_length,
        prediction_length=prediction_length,
        input_patch_size=input_patch_size,
        input_patch_stride=input_patch_stride,
        output_patch_size=output_patch_size,
        max_output_patches=max_output_patches,
        quantiles=quantiles,
        use_reg_token=use_reg_token,
        use_arcsinh=use_arcsinh,
        time_encoding_scale=time_encoding_scale,
        d_model=d_model,
        d_kv=d_kv,
        d_ff=d_ff,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        layer_norm_epsilon=layer_norm_epsilon,
        initializer_factor=initializer_factor,
        feed_forward_proj=feed_forward_proj,
        pad_token_id=pad_token_id,
        rope_theta=rope_theta,
        attn_implementation=attn_implementation,
    )

    train_dataset = Chronos2ArrowDataset(
        datasets=train_datasets,
        probabilities=expanded_probability,
        context_length=context_length,
        prediction_length=prediction_length,
        batch_size=per_device_train_batch_size,
        output_patch_size=output_patch_size,
        min_past=min_past,
        mode="training",
    )

    training_args_kwargs = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size or per_device_train_batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        optim=optim,
        logging_strategy="steps",
        logging_steps=log_steps,
        save_strategy="steps",
        save_steps=save_steps,
        report_to=["tensorboard"],
        max_steps=max_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_num_workers=dataloader_num_workers,
        tf32=tf32,
        bf16=bf16,
        torch_compile=torch_compile,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        save_only_model=True,
    )

    if eval_dataset is None:
        training_args_kwargs.update(
            eval_strategy="no",
            load_best_model_at_end=False,
        )
    else:
        training_args_kwargs.update(
            eval_strategy="steps",
            eval_steps=save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            label_names=["future_target"],
        )

    training_args = TrainingArguments(**training_args_kwargs)

    if disable_data_parallel and not dist_is_launched() and torch.cuda.is_available():
        training_args._n_gpu = 1
        assert training_args.n_gpu == 1

    trainer = Chronos2Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
    )
    log_on_main("Training", logger)
    train_start_time = time.perf_counter()
    train_result = trainer.train()
    training_duration_seconds = time.perf_counter() - train_start_time
    end_time = datetime.now().astimezone()

    if is_main_process():
        train_metrics = dict(train_result.metrics)
        train_metrics["global_step"] = trainer.state.global_step
        train_metrics["max_steps"] = trainer.state.max_steps
        benchmark_csv_path = output_dir / "training_benchmark.csv"
        _write_training_benchmark_csv(
            benchmark_csv_path,
            output_dir=output_dir,
            launch_time=launch_time,
            end_time=end_time,
            training_duration_seconds=training_duration_seconds,
            train_metrics=train_metrics,
            trainer_n_gpu=training_args.n_gpu,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            context_length=context_length,
            prediction_length=prediction_length,
            memory_metrics=memory_callback.get_metrics(),
        )
        log_on_main(f"Training benchmark summary written to {benchmark_csv_path}", logger)
        model.chronos_config.context_length = max(model.chronos_config.context_length, context_length)
        model.chronos_config.max_output_patches = max(
            model.chronos_config.max_output_patches,
            math.ceil(prediction_length / model.chronos_config.output_patch_size),
        )
        model.config.chronos_config = model.chronos_config.__dict__
        model.config.chronos_pipeline_class = "Chronos2Pipeline"
        model.config.architectures = ["Chronos2Model"]
        model.save_pretrained(output_dir / "checkpoint-final")
        save_training_info(
            output_dir / "checkpoint-final",
            training_config=raw_training_config,
            training_duration_seconds=training_duration_seconds,
        )


def dist_is_launched() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_torchelastic_launched()


if __name__ == "__main__":
    configure_logger()
    app()

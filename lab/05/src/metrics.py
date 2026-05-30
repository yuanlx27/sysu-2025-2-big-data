from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ForecastResult:
    model: str
    mae: float
    rmse: float
    train_time_sec: float
    predict_time_sec: float
    predictions: np.ndarray


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def write_results(results: list[ForecastResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "mae", "rmse", "train_time_sec", "predict_time_sec"])
        for item in results:
            writer.writerow(
                [
                    item.model,
                    f"{item.mae:.4f}",
                    f"{item.rmse:.4f}",
                    f"{item.train_time_sec:.4f}",
                    f"{item.predict_time_sec:.4f}",
                ]
            )

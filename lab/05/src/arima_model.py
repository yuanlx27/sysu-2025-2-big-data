from __future__ import annotations

import time

import numpy as np

from metrics import ForecastResult, mean_absolute_error, root_mean_squared_error


def forecast_arima(train: np.ndarray, test: np.ndarray, order: tuple[int, int, int] = (5, 1, 0)) -> ForecastResult:
    if order[1] != 1 or order[2] != 0:
        raise ValueError("This lightweight implementation supports ARIMA(p, 1, 0)")
    p = order[0]

    train_start = time.perf_counter()
    intercept, coef = fit_differenced_ar(train, p)
    train_time = time.perf_counter() - train_start

    predict_start = time.perf_counter()
    predictions = forecast_differenced_ar(train, len(test), p, intercept, coef)
    predict_time = time.perf_counter() - predict_start

    return ForecastResult(
        model=f"ARIMA{order}",
        mae=mean_absolute_error(test, predictions),
        rmse=root_mean_squared_error(test, predictions),
        train_time_sec=train_time,
        predict_time_sec=predict_time,
        predictions=predictions,
    )


def fit_differenced_ar(series: np.ndarray, p: int) -> tuple[float, np.ndarray]:
    diff = np.diff(series)
    if len(diff) <= p:
        raise ValueError(f"Need more than {p + 1} training prices for ARIMA({p},1,0)")

    x_rows = []
    y = []
    for i in range(p, len(diff)):
        x_rows.append(diff[i - p : i][::-1])
        y.append(diff[i])
    x = np.asarray(x_rows, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    design = np.column_stack([np.ones(len(x)), x])
    params, *_ = np.linalg.lstsq(design, y_arr, rcond=None)
    return float(params[0]), params[1:]


def forecast_differenced_ar(
    series: np.ndarray,
    steps: int,
    p: int,
    intercept: float,
    coef: np.ndarray,
) -> np.ndarray:
    diff_history = np.diff(series).astype(np.float64).tolist()
    price = float(series[-1])
    predictions: list[float] = []
    for _ in range(steps):
        recent = np.asarray(diff_history[-p:][::-1], dtype=np.float64)
        next_diff = float(intercept + recent @ coef)
        price += next_diff
        predictions.append(price)
        diff_history.append(next_diff)
    return np.asarray(predictions, dtype=np.float64)

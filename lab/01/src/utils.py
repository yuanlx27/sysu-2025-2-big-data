from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, TypeVar

import numpy as np
from sklearn.datasets import fetch_openml, load_digits
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
SAMPLE_DIR = DATA_DIR / "sample"
MODELS_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "report"
FIGURES_DIR = REPORT_DIR / "figures"


@dataclass
class DatasetBundle:
    name: str
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    image_shape: tuple[int, int]


T = TypeVar("T")


def ensure_dirs() -> None:
    for path in (RAW_DIR, SAMPLE_DIR, MODELS_DIR, REPORT_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def time_call(fn: Callable[[], T]) -> tuple[T, float]:
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def load_digit_dataset(
    train_limit: Optional[int] = None,
    test_limit: Optional[int] = None,
    random_state: int = 42,
) -> DatasetBundle:
    """Load MNIST when available, with an offline fallback for reproducibility."""
    ensure_dirs()

    try:
        mnist = fetch_openml(
            "mnist_784",
            version=1,
            as_frame=False,
            data_home=str(RAW_DIR),
            parser="auto",
        )
        x = mnist.data.astype("float32") / 255.0
        y = mnist.target.astype("int64")
        x_train, x_test = x[:60000], x[60000:]
        y_train, y_test = y[:60000], y[60000:]
        name = "MNIST"
        image_shape = (28, 28)
    except Exception:
        digits = load_digits()
        x = digits.data.astype("float32") / 16.0
        y = digits.target.astype("int64")
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.25,
            random_state=random_state,
            stratify=y,
        )
        name = "sklearn-digits"
        image_shape = (8, 8)

    rng = np.random.default_rng(random_state)
    if train_limit is not None and train_limit < len(x_train):
        idx = rng.choice(len(x_train), size=train_limit, replace=False)
        x_train = x_train[idx]
        y_train = y_train[idx]
    if test_limit is not None and test_limit < len(x_test):
        idx = rng.choice(len(x_test), size=test_limit, replace=False)
        x_test = x_test[idx]
        y_test = y_test[idx]

    return DatasetBundle(
        name=name,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        image_shape=image_shape,
    )


def get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default

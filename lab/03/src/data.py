from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris


def load_iris_data() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    iris = load_iris()
    x = iris.data.astype(float)
    y = iris.target.astype(int)
    x = standardize(x)
    return x, y, list(iris.feature_names), list(iris.target_names)


def standardize(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    return (x - mean) / std

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import calinski_harabasz_score, silhouette_score


@dataclass
class MetricResult:
    algorithm: str
    param: str
    clusters: int
    noise_points: int
    accuracy: float
    silhouette: float | None
    calinski_harabasz: float | None


def evaluate_clustering(
    x: np.ndarray,
    y_true: np.ndarray,
    labels: np.ndarray,
    algorithm: str,
    param: str,
) -> MetricResult:
    non_noise = labels != -1
    cluster_ids = sorted(set(labels[non_noise].tolist()))
    mapped = np.full_like(y_true, fill_value=-1)

    for cluster_id in cluster_ids:
        mask = labels == cluster_id
        majority_label = Counter(y_true[mask].tolist()).most_common(1)[0][0]
        mapped[mask] = majority_label

    accuracy = float((mapped == y_true).mean())
    silhouette, calinski = _internal_scores(x, labels)
    return MetricResult(
        algorithm=algorithm,
        param=param,
        clusters=len(cluster_ids),
        noise_points=int((labels == -1).sum()),
        accuracy=accuracy,
        silhouette=silhouette,
        calinski_harabasz=calinski,
    )


def _internal_scores(
    x: np.ndarray,
    labels: np.ndarray,
) -> tuple[float | None, float | None]:
    mask = labels != -1
    labels_without_noise = labels[mask]
    cluster_count = len(set(labels_without_noise.tolist()))

    if cluster_count < 2 or len(labels_without_noise) <= cluster_count:
        return None, None

    x_without_noise = x[mask]
    return (
        float(silhouette_score(x_without_noise, labels_without_noise)),
        float(calinski_harabasz_score(x_without_noise, labels_without_noise)),
    )

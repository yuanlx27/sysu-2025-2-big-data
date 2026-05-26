from __future__ import annotations

import numpy as np


class KMeans:
    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: int = 42,
    ) -> None:
        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.n_iter_: int = 0

    def fit(self, x: np.ndarray) -> "KMeans":
        if len(x) < self.n_clusters:
            raise ValueError("n_clusters cannot exceed number of samples")

        rng = np.random.default_rng(self.random_state)
        initial_indices = rng.choice(len(x), size=self.n_clusters, replace=False)
        centers = x[initial_indices].copy()

        for iteration in range(1, self.max_iter + 1):
            labels = self._assign_labels(x, centers)
            new_centers = self._update_centers(x, labels, centers)
            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers

            if shift <= self.tol:
                self.n_iter_ = iteration
                break
        else:
            self.n_iter_ = self.max_iter

        self.centers_ = centers
        self.labels_ = self._assign_labels(x, centers)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.centers_ is None:
            raise RuntimeError("model has not been fitted")
        return self._assign_labels(x, self.centers_)

    def _assign_labels(self, x: np.ndarray, centers: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
        return distances.argmin(axis=1)

    def _update_centers(
        self,
        x: np.ndarray,
        labels: np.ndarray,
        old_centers: np.ndarray,
    ) -> np.ndarray:
        centers = old_centers.copy()
        for cluster_id in range(self.n_clusters):
            members = x[labels == cluster_id]
            if len(members) > 0:
                centers[cluster_id] = members.mean(axis=0)
            else:
                centers[cluster_id] = self._farthest_point(x, old_centers)
        return centers

    def _farthest_point(self, x: np.ndarray, centers: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
        nearest_distance = distances.min(axis=1)
        return x[nearest_distance.argmax()].copy()

from __future__ import annotations

import numpy as np


NOISE = -1
UNASSIGNED = -99


class DBSCAN:
    def __init__(self, eps: float, min_samples: int) -> None:
        if eps <= 0:
            raise ValueError("eps must be positive")
        if min_samples <= 0:
            raise ValueError("min_samples must be positive")
        self.eps = eps
        self.min_samples = min_samples
        self.labels_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "DBSCAN":
        labels = np.full(len(x), UNASSIGNED, dtype=int)
        visited = np.zeros(len(x), dtype=bool)
        cluster_id = 0

        for point_idx in range(len(x)):
            if visited[point_idx]:
                continue

            visited[point_idx] = True
            neighbors = self._region_query(x, point_idx)
            if len(neighbors) < self.min_samples:
                labels[point_idx] = NOISE
                continue

            self._expand_cluster(x, labels, visited, point_idx, neighbors, cluster_id)
            cluster_id += 1

        labels[labels == UNASSIGNED] = NOISE
        self.labels_ = labels
        return self

    def _expand_cluster(
        self,
        x: np.ndarray,
        labels: np.ndarray,
        visited: np.ndarray,
        point_idx: int,
        neighbors: list[int],
        cluster_id: int,
    ) -> None:
        labels[point_idx] = cluster_id
        queue = list(neighbors)
        cursor = 0

        while cursor < len(queue):
            neighbor_idx = queue[cursor]
            cursor += 1

            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                neighbor_neighbors = self._region_query(x, neighbor_idx)
                if len(neighbor_neighbors) >= self.min_samples:
                    for candidate in neighbor_neighbors:
                        if candidate not in queue:
                            queue.append(candidate)

            if labels[neighbor_idx] in (UNASSIGNED, NOISE):
                labels[neighbor_idx] = cluster_id

    def _region_query(self, x: np.ndarray, point_idx: int) -> list[int]:
        distances = np.linalg.norm(x - x[point_idx], axis=1)
        return np.flatnonzero(distances <= self.eps).tolist()

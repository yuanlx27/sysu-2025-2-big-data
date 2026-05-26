from __future__ import annotations

import csv
from pathlib import Path

from data import load_iris_data
from dbscan import DBSCAN
from kmeans import KMeans
from metrics import MetricResult, evaluate_clustering
from visualize import plot_clusters, plot_metric_comparison, project_to_2d


ROOT_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT_DIR / "report"
FIGURE_DIR = REPORT_DIR / "figures"
METRICS_PATH = REPORT_DIR / "metrics.csv"


def main() -> None:
    x, y_true, _, _ = load_iris_data()
    x_2d = project_to_2d(x)
    results: list[MetricResult] = []

    for k in (2, 3, 4):
        model = KMeans(n_clusters=k, random_state=42).fit(x)
        labels = model.labels_
        if labels is None:
            raise RuntimeError("K-means did not produce labels")

        results.append(evaluate_clustering(x, y_true, labels, "kmeans", f"k={k}"))
        plot_clusters(
            x_2d,
            labels,
            f"K-means clustering (k={k})",
            FIGURE_DIR / f"kmeans_k{k}.svg",
        )

    for eps, min_samples in ((0.4, 5), (0.6, 5), (0.8, 5)):
        model = DBSCAN(eps=eps, min_samples=min_samples).fit(x)
        labels = model.labels_
        if labels is None:
            raise RuntimeError("DBSCAN did not produce labels")

        param = f"eps={eps}, min_samples={min_samples}"
        results.append(evaluate_clustering(x, y_true, labels, "dbscan", param))
        eps_tag = str(eps).replace(".", "_")
        plot_clusters(
            x_2d,
            labels,
            f"DBSCAN clustering (eps={eps}, min_samples={min_samples})",
            FIGURE_DIR / f"dbscan_eps_{eps_tag}_min_{min_samples}.svg",
        )

    write_metrics(results, METRICS_PATH)
    plot_metric_comparison(results, FIGURE_DIR / "metrics_comparison.svg")

    print(f"Wrote {METRICS_PATH.relative_to(ROOT_DIR)}")
    print(f"Wrote figures to {FIGURE_DIR.relative_to(ROOT_DIR)}")


def write_metrics(results: list[MetricResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "algorithm",
                "param",
                "clusters",
                "noise_points",
                "accuracy",
                "silhouette",
                "calinski_harabasz",
            ]
        )
        for item in results:
            writer.writerow(
                [
                    item.algorithm,
                    item.param,
                    item.clusters,
                    item.noise_points,
                    _format_float(item.accuracy),
                    _format_optional_float(item.silhouette),
                    _format_optional_float(item.calinski_harabasz),
                ]
            )


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


if __name__ == "__main__":
    main()

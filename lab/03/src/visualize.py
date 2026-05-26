from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from sklearn.decomposition import PCA

from metrics import MetricResult

COLORS = [
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#e45756",
    "#72b7b2",
    "#b279a2",
]


def project_to_2d(x: np.ndarray) -> np.ndarray:
    return PCA(n_components=2, random_state=42).fit_transform(x)


def plot_clusters(
    x_2d: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 760, 560
    margin_left, margin_right, margin_top, margin_bottom = 72, 150, 72, 72
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    xs, ys = _scale_points(x_2d, margin_left, margin_top, plot_w, plot_h)
    unique_labels = sorted(set(labels.tolist()))
    color_by_label = {
        label: "#8a8f98" if label == -1 else COLORS[index % len(COLORS)]
        for index, label in enumerate(unique_labels)
    }

    parts = [
        _svg_open(width, height),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="42" y="42" font-family="Arial" font-size="24" font-weight="700" fill="#111827">{escape(title)}</text>',
        _axes(margin_left, margin_top, plot_w, plot_h, "PCA 1", "PCA 2"),
    ]

    for label in unique_labels:
        for x_pos, y_pos in zip(xs[labels == label], ys[labels == label]):
            if label == -1:
                parts.append(
                    f'<path d="M{x_pos - 5:.1f},{y_pos - 5:.1f} L{x_pos + 5:.1f},{y_pos + 5:.1f} M{x_pos + 5:.1f},{y_pos - 5:.1f} L{x_pos - 5:.1f},{y_pos + 5:.1f}" stroke="{color_by_label[label]}" stroke-width="2"/>'
                )
            else:
                parts.append(
                    f'<circle cx="{x_pos:.1f}" cy="{y_pos:.1f}" r="5.2" fill="{color_by_label[label]}" opacity="0.86"/>'
                )

    legend_x = width - margin_right + 24
    legend_y = margin_top + 20
    for index, label in enumerate(unique_labels):
        y = legend_y + index * 28
        label_text = "noise" if label == -1 else f"cluster {label}"
        parts.append(f'<circle cx="{legend_x}" cy="{y}" r="6" fill="{color_by_label[label]}"/>')
        parts.append(
            f'<text x="{legend_x + 14}" y="{y + 5}" font-family="Arial" font-size="14" fill="#334155">{escape(label_text)}</text>'
        )

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def plot_metric_comparison(results: list[MetricResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metric_specs = [
        ("accuracy", "Accuracy", [item.accuracy for item in results]),
        (
            "silhouette",
            "Silhouette",
            [np.nan if item.silhouette is None else item.silhouette for item in results],
        ),
        (
            "calinski_harabasz",
            "Calinski-Harabasz",
            [
                np.nan if item.calinski_harabasz is None else item.calinski_harabasz
                for item in results
            ],
        ),
    ]
    width, height = 1180, 520
    chart_w, chart_h = 315, 300
    starts = [70, 430, 790]
    parts = [
        _svg_open(width, height),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="44" y="42" font-family="Arial" font-size="24" font-weight="700" fill="#111827">Metric Comparison</text>',
    ]

    labels = [f"{item.algorithm} {item.param}" for item in results]
    for chart_index, (_, title, values) in enumerate(metric_specs):
        left = starts[chart_index]
        top = 82
        numeric_values = [value for value in values if not np.isnan(value)]
        max_value = max(numeric_values) if numeric_values else 1.0
        if title != "Accuracy":
            max_value *= 1.08
        else:
            max_value = 1.0
        bar_gap = 8
        bar_w = (chart_w - bar_gap * (len(values) - 1)) / len(values)

        parts.append(f'<text x="{left}" y="{top - 22}" font-family="Arial" font-size="18" font-weight="700" fill="#111827">{title}</text>')
        parts.append(_chart_frame(left, top, chart_w, chart_h))
        for idx, value in enumerate(values):
            x_pos = left + idx * (bar_w + bar_gap)
            if np.isnan(value):
                bar_h = 0
                value_label = "N/A"
            else:
                bar_h = chart_h * value / max_value
                value_label = f"{value:.2f}" if abs(value) < 100 else f"{value:.0f}"
            y_pos = top + chart_h - bar_h
            parts.append(
                f'<rect x="{x_pos:.1f}" y="{y_pos:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="#4c78a8" rx="3"/>'
            )
            parts.append(
                f'<text x="{x_pos + bar_w / 2:.1f}" y="{y_pos - 7:.1f}" text-anchor="middle" font-family="Arial" font-size="11" fill="#111827">{value_label}</text>'
            )
            parts.append(_rotated_label(labels[idx], x_pos + bar_w / 2, top + chart_h + 24))

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def _scale_points(
    points: np.ndarray,
    left: float,
    top: float,
    width: float,
    height: float,
) -> tuple[np.ndarray, np.ndarray]:
    min_x, max_x = points[:, 0].min(), points[:, 0].max()
    min_y, max_y = points[:, 1].min(), points[:, 1].max()
    pad_x = (max_x - min_x) * 0.08 or 1.0
    pad_y = (max_y - min_y) * 0.08 or 1.0
    min_x, max_x = min_x - pad_x, max_x + pad_x
    min_y, max_y = min_y - pad_y, max_y + pad_y
    scaled_x = left + (points[:, 0] - min_x) / (max_x - min_x) * width
    scaled_y = top + height - (points[:, 1] - min_y) / (max_y - min_y) * height
    return scaled_x, scaled_y


def _svg_open(width: int, height: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'


def _axes(left: float, top: float, width: float, height: float, x_label: str, y_label: str) -> str:
    bottom = top + height
    right = left + width
    parts = [
        f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="#f8fafc" stroke="#cbd5e1"/>',
    ]
    for idx in range(1, 5):
        y = top + height * idx / 5
        x = left + width * idx / 5
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="#e2e8f0"/>')
    parts.extend(
        [
            f'<text x="{left + width / 2}" y="{bottom + 44}" text-anchor="middle" font-family="Arial" font-size="15" fill="#334155">{escape(x_label)}</text>',
            f'<text x="{left - 46}" y="{top + height / 2}" text-anchor="middle" transform="rotate(-90 {left - 46} {top + height / 2})" font-family="Arial" font-size="15" fill="#334155">{escape(y_label)}</text>',
        ]
    )
    return "\n".join(parts)


def _chart_frame(left: float, top: float, width: float, height: float) -> str:
    bottom = top + height
    right = left + width
    parts = [
        f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="#f8fafc" stroke="#cbd5e1"/>',
    ]
    for idx in range(1, 5):
        y = top + height * idx / 5
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#e2e8f0"/>')
    parts.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#334155"/>')
    return "\n".join(parts)


def _rotated_label(label: str, x: float, y: float) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="end" transform="rotate(-42 {x:.1f} {y:.1f})" '
        f'font-family="Arial" font-size="10" fill="#334155">{escape(label)}</text>'
    )

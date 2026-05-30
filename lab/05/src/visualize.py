from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np

from metrics import ForecastResult


COLORS = {
    "actual": "#111827",
    "ARIMA": "#2563eb",
    "LSTM": "#dc2626",
    "loss": "#16a34a",
}


def plot_forecast(
    dates: list[str],
    train: np.ndarray,
    test: np.ndarray,
    results: list[ForecastResult],
    output_path: Path,
    tail: int = 60,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history = train[-tail:]
    actual = np.concatenate([history, test])
    labels = dates[-len(actual) :]
    series = [actual] + [np.concatenate([np.full(len(history), np.nan), item.predictions]) for item in results]
    y_min = min(float(np.nanmin(item)) for item in series)
    y_max = max(float(np.nanmax(item)) for item in series)

    width, height = 900, 480
    left, top, chart_w, chart_h = 72, 36, 780, 340
    parts = _svg_start(width, height)
    parts.append(_frame(left, top, chart_w, chart_h, "Close Price"))
    parts.append(_polyline(actual, left, top, chart_w, chart_h, y_min, y_max, COLORS["actual"]))
    for item in results:
        key = "ARIMA" if item.model.startswith("ARIMA") else item.model
        forecast_line = np.concatenate([np.full(len(history), np.nan), item.predictions])
        parts.append(_polyline(forecast_line, left, top, chart_w, chart_h, y_min, y_max, COLORS.get(key, "#7c3aed")))
    parts.append(_x_label(labels[0], left, top + chart_h + 30))
    parts.append(_x_label(labels[-1], left + chart_w, top + chart_h + 30, anchor="end"))
    parts.append(_legend(left + 20, height - 64, [("Actual", COLORS["actual"])] + [(item.model, COLORS["ARIMA"] if item.model.startswith("ARIMA") else COLORS["LSTM"]) for item in results]))
    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def plot_loss(losses: list[float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 760, 420
    left, top, chart_w, chart_h = 70, 34, 620, 290
    values = np.asarray(losses, dtype=np.float64)
    y_min = float(values.min())
    y_max = float(values.max())
    parts = _svg_start(width, height)
    parts.append(_frame(left, top, chart_w, chart_h, "MSE Loss"))
    parts.append(_polyline(values, left, top, chart_w, chart_h, y_min, y_max, COLORS["loss"]))
    parts.append(_x_label("epoch 1", left, top + chart_h + 30))
    parts.append(_x_label(f"epoch {len(values)}", left + chart_w, top + chart_h + 30, anchor="end"))
    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def _svg_start(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]


def _frame(left: float, top: float, width: float, height: float, y_label: str) -> str:
    return "\n".join(
        [
            f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="#f8fafc" stroke="#cbd5e1"/>',
            f'<line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" stroke="#475569"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" stroke="#475569"/>',
            f'<text x="{left - 48}" y="{top + height / 2}" text-anchor="middle" transform="rotate(-90 {left - 48} {top + height / 2})" font-family="Arial" font-size="14" fill="#334155">{escape(y_label)}</text>',
        ]
    )


def _polyline(values: np.ndarray, left: float, top: float, width: float, height: float, y_min: float, y_max: float, color: str) -> str:
    span = y_max - y_min or 1.0
    points = []
    denom = max(len(values) - 1, 1)
    for i, value in enumerate(values):
        if np.isnan(value):
            continue
        x = left + width * i / denom
        y = top + height - height * (float(value) - y_min) / span
        points.append(f"{x:.1f},{y:.1f}")
    return f'<polyline fill="none" stroke="{color}" stroke-width="2.4" points="{" ".join(points)}"/>'


def _x_label(text: str, x: float, y: float, anchor: str = "start") -> str:
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Arial" font-size="12" fill="#475569">{escape(text)}</text>'


def _legend(x: float, y: float, items: list[tuple[str, str]]) -> str:
    parts = []
    offset = 0
    for label, color in items:
        parts.append(f'<line x1="{x + offset}" y1="{y}" x2="{x + offset + 28}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{x + offset + 36}" y="{y + 5}" font-family="Arial" font-size="13" fill="#334155">{escape(label)}</text>')
        offset += max(120, len(label) * 9 + 54)
    return "\n".join(parts)

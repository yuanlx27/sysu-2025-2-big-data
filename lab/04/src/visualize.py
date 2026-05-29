from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from metrics import ClassificationResult


COLORS = ["#2563eb", "#16a34a", "#dc2626", "#9333ea"]


def plot_accuracy(results: list[ClassificationResult], output_path: Path) -> None:
    _plot_bars(
        results,
        [item.accuracy for item in results],
        "Accuracy Comparison",
        "Accuracy",
        output_path,
        max_value=1.0,
        value_format="{:.3f}",
    )


def plot_time(results: list[ClassificationResult], output_path: Path) -> None:
    _plot_grouped_time(results, output_path)


def _plot_bars(
    results: list[ClassificationResult],
    values: list[float],
    title: str,
    y_label: str,
    output_path: Path,
    max_value: float | None = None,
    value_format: str = "{:.2f}",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 820, 520
    left, top, chart_w, chart_h = 86, 78, 650, 320
    max_v = max_value or max(values) * 1.15 or 1.0
    bar_gap = 46
    bar_w = (chart_w - bar_gap * (len(values) + 1)) / len(values)
    parts = [_svg_open(width, height), '<rect width="100%" height="100%" fill="#ffffff"/>']
    parts.append(f'<text x="40" y="42" font-family="Arial" font-size="24" font-weight="700" fill="#111827">{escape(title)}</text>')
    parts.append(_chart_frame(left, top, chart_w, chart_h, y_label))

    for idx, (item, value) in enumerate(zip(results, values)):
        x = left + bar_gap + idx * (bar_w + bar_gap)
        bar_h = chart_h * value / max_v
        y = top + chart_h - bar_h
        color = COLORS[idx % len(COLORS)]
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" rx="4"/>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-family="Arial" font-size="13" fill="#111827">{value_format.format(value)}</text>')
        parts.append(_rotated_label(item.model, x + bar_w / 2, top + chart_h + 34))

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def _plot_grouped_time(results: list[ClassificationResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 900, 540
    left, top, chart_w, chart_h = 86, 82, 720, 320
    max_v = max(max(item.train_time_sec, item.test_time_sec) for item in results) * 1.2 or 1.0
    group_w = chart_w / len(results)
    bar_w = min(42, group_w * 0.25)
    parts = [_svg_open(width, height), '<rect width="100%" height="100%" fill="#ffffff"/>']
    parts.append('<text x="40" y="42" font-family="Arial" font-size="24" font-weight="700" fill="#111827">Time Comparison</text>')
    parts.append(_chart_frame(left, top, chart_w, chart_h, "Seconds"))

    for idx, item in enumerate(results):
        center = left + group_w * idx + group_w / 2
        for offset, value, color, label in (
            (-bar_w / 1.8, item.train_time_sec, "#2563eb", "train"),
            (bar_w / 1.8, item.test_time_sec, "#f97316", "test"),
        ):
            x = center + offset - bar_w / 2
            bar_h = chart_h * value / max_v
            y = top + chart_h - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" rx="4"/>')
            parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 7:.1f}" text-anchor="middle" font-family="Arial" font-size="11" fill="#111827">{value:.1f}</text>')
        parts.append(_rotated_label(item.model, center, top + chart_h + 36))

    parts.append('<rect x="650" y="30" width="14" height="14" fill="#2563eb" rx="2"/>')
    parts.append('<text x="672" y="42" font-family="Arial" font-size="14" fill="#334155">train</text>')
    parts.append('<rect x="730" y="30" width="14" height="14" fill="#f97316" rx="2"/>')
    parts.append('<text x="752" y="42" font-family="Arial" font-size="14" fill="#334155">test</text>')
    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def _svg_open(width: int, height: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'


def _chart_frame(left: float, top: float, width: float, height: float, y_label: str) -> str:
    bottom = top + height
    right = left + width
    parts = [
        f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="#f8fafc" stroke="#cbd5e1"/>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#334155"/>',
        f'<text x="{left - 54}" y="{top + height / 2}" text-anchor="middle" transform="rotate(-90 {left - 54} {top + height / 2})" font-family="Arial" font-size="15" fill="#334155">{escape(y_label)}</text>',
    ]
    for idx in range(1, 5):
        y = top + height * idx / 5
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#e2e8f0"/>')
    return "\n".join(parts)


def _rotated_label(label: str, x: float, y: float) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="end" transform="rotate(-35 {x:.1f} {y:.1f})" '
        f'font-family="Arial" font-size="12" fill="#334155">{escape(label)}</text>'
    )

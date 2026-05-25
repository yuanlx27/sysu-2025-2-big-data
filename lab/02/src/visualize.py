from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "report" / "figures"

COLORS = {
    "Person": "#3b82f6",
    "Paper": "#f97316",
    "Institution": "#10b981",
    "Concept": "#8b5cf6",
    "Year": "#64748b",
}


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA_DIR / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def graph_overview(nodes: list[dict[str, str]], rels: list[dict[str, str]]) -> str:
    width, height = 1120, 760
    cx, cy = width / 2, height / 2
    radius = 280
    important = [
        "geoffrey_hinton",
        "yann_lecun",
        "yoshua_bengio",
        "attention_is_all_you_need",
        "bert",
        "alexnet",
        "transformer",
        "deep_learning",
        "natural_language_processing",
        "google_brain",
        "openai",
        "university_of_toronto",
        "convolutional_neural_networks",
        "self_attention",
    ]
    node_map = {node["id"]: node for node in nodes if node["id"] in important}
    selected = list(node_map.values())
    positions: dict[str, tuple[float, float]] = {}
    for index, node in enumerate(selected):
        angle = 2 * math.pi * index / len(selected)
        positions[node["id"]] = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))

    selected_rels = [rel for rel in rels if rel["source"] in positions and rel["target"] in positions]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#64748b"/></marker></defs>',
        '<text x="40" y="48" font-family="Arial" font-size="28" font-weight="700" fill="#111827">AI Knowledge Graph Overview</text>',
    ]
    for rel in selected_rels:
        x1, y1 = positions[rel["source"]]
        x2, y2 = positions[rel["target"]]
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#94a3b8" stroke-width="2" marker-end="url(#arrow)"/>'
        )
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        parts.append(
            f'<text x="{mx:.1f}" y="{my:.1f}" font-family="Arial" font-size="12" fill="#475569">{escape(rel["type"])}</text>'
        )
    for node in selected:
        x, y = positions[node["id"]]
        color = COLORS.get(node["label"], "#334155")
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="34" fill="{color}" opacity="0.92"/>')
        label = escape(node["name"])
        parts.append(
            f'<text x="{x:.1f}" y="{y + 54:.1f}" text-anchor="middle" font-family="Arial" font-size="14" fill="#111827">{label}</text>'
        )
    legend_x, legend_y = 42, 88
    for offset, (label, color) in enumerate(COLORS.items()):
        y = legend_y + offset * 28
        parts.append(f'<circle cx="{legend_x}" cy="{y}" r="8" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 18}" y="{y + 5}" font-family="Arial" font-size="15" fill="#334155">{label}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def entity_distribution(nodes: list[dict[str, str]]) -> str:
    counts = Counter(node["label"] for node in nodes)
    width, height = 900, 500
    margin_left, margin_bottom = 100, 80
    chart_w, chart_h = 700, 320
    max_count = max(counts.values())
    labels = sorted(counts.keys())
    bar_w = chart_w / len(labels) * 0.58
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="48" y="52" font-family="Arial" font-size="26" font-weight="700" fill="#111827">Entity Type Distribution</text>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{margin_left + chart_w}" y2="{height - margin_bottom}" stroke="#334155"/>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{margin_left}" y2="{height - margin_bottom - chart_h}" stroke="#334155"/>',
    ]
    for index, label in enumerate(labels):
        count = counts[label]
        x = margin_left + index * chart_w / len(labels) + 34
        bar_h = chart_h * count / max_count
        y = height - margin_bottom - bar_h
        color = COLORS.get(label, "#334155")
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" rx="4"/>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 12:.1f}" text-anchor="middle" font-family="Arial" font-size="16" fill="#111827">{count}</text>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{height - margin_bottom + 30}" text-anchor="middle" font-family="Arial" font-size="15" fill="#334155">{label}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    nodes = read_csv("nodes.csv")
    rels = read_csv("relationships.csv")
    (FIG_DIR / "graph_overview.svg").write_text(graph_overview(nodes, rels), encoding="utf-8")
    (FIG_DIR / "entity_distribution.svg").write_text(entity_distribution(nodes), encoding="utf-8")
    print("wrote report/figures/graph_overview.svg and entity_distribution.svg")


if __name__ == "__main__":
    main()

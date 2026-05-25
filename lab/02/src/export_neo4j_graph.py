from __future__ import annotations

import math
import os
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "report" / "figures"

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "sysu-big-data")
FOCUS_NAME = os.getenv("NEO4J_FOCUS", "Geoffrey Hinton")
OUTPUT_PATH = Path(os.getenv("NEO4J_GRAPH_OUT", FIG_DIR / "neo4j_query_graph.svg"))

COLORS = {
    "Person": "#3b82f6",
    "Paper": "#f97316",
    "Institution": "#10b981",
    "Concept": "#8b5cf6",
    "Year": "#64748b",
}


def node_label(node) -> str:
    if "label" in node:
        return str(node["label"])
    labels = [label for label in node.labels if label != "Entity"]
    return labels[0] if labels else "Entity"


def node_name(node) -> str:
    return str(node.get("name", node.element_id))


def relation_endpoints(rel) -> tuple[str, str]:
    return rel.start_node.element_id, rel.end_node.element_id


def wrap_text(text: str, max_len: int = 18) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_len:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
    if current:
        lines.append(current)
    return lines or [text]


def render_svg(nodes: dict[str, object], relationships: dict[str, object]) -> str:
    width, height = 1120, 760
    cx, cy = width / 2, height / 2 + 20
    radius = 270
    ordered_nodes = sorted(nodes.values(), key=lambda node: (node_label(node), node_name(node)))
    positions: dict[str, tuple[float, float]] = {}

    for index, node in enumerate(ordered_nodes):
        if node_name(node) == FOCUS_NAME:
            positions[node.element_id] = (cx, cy)
            continue
        angle = 2 * math.pi * index / max(1, len(ordered_nodes) - 1)
        positions[node.element_id] = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#64748b"/></marker></defs>',
        f'<text x="42" y="48" font-family="Arial" font-size="28" font-weight="700" fill="#111827">Neo4j Query Graph: {escape(FOCUS_NAME)}</text>',
    ]

    for rel in relationships.values():
        source_id, target_id = relation_endpoints(rel)
        if source_id not in positions or target_id not in positions:
            continue
        x1, y1 = positions[source_id]
        x2, y2 = positions[target_id]
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#94a3b8" stroke-width="2" marker-end="url(#arrow)"/>'
        )
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        parts.append(
            f'<text x="{mx:.1f}" y="{my:.1f}" text-anchor="middle" font-family="Arial" font-size="12" fill="#475569">{escape(rel.type)}</text>'
        )

    for node in ordered_nodes:
        x, y = positions[node.element_id]
        label = node_label(node)
        color = COLORS.get(label, "#334155")
        radius_px = 42 if node_name(node) == FOCUS_NAME else 32
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius_px}" fill="{color}" opacity="0.94"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{y + 5:.1f}" text-anchor="middle" font-family="Arial" font-size="12" font-weight="700" fill="#ffffff">{escape(label)}</text>'
        )
        for line_index, line in enumerate(wrap_text(node_name(node))[:3]):
            parts.append(
                f'<text x="{x:.1f}" y="{y + radius_px + 22 + line_index * 16:.1f}" text-anchor="middle" font-family="Arial" font-size="13" fill="#111827">{escape(line)}</text>'
            )

    legend_x, legend_y = 42, 90
    for offset, (label, color) in enumerate(COLORS.items()):
        y = legend_y + offset * 28
        parts.append(f'<circle cx="{legend_x}" cy="{y}" r="8" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 18}" y="{y + 5}" font-family="Arial" font-size="15" fill="#334155">{label}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise SystemExit("Please install dependencies first: python3 -m pip install -r requirements.txt") from exc

    query = """
    MATCH path = (p:Entity {name: $focus})-[*1..2]-(x)
    RETURN path
    LIMIT 50
    """
    nodes: dict[str, object] = {}
    relationships: dict[str, object] = {}

    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    with driver.session() as session:
        for record in session.run(query, focus=FOCUS_NAME):
            path = record["path"]
            for node in path.nodes:
                nodes[node.element_id] = node
            for relationship in path.relationships:
                relationships[relationship.element_id] = relationship
    driver.close()

    if not nodes:
        raise SystemExit(f"No graph data returned for focus node: {FOCUS_NAME}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(render_svg(nodes, relationships), encoding="utf-8")
    print(f"wrote {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

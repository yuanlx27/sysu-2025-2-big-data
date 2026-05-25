from __future__ import annotations

import csv
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
NODE_LABELS = {"Person", "Paper", "Institution", "Concept", "Year"}
REL_TYPES = {"AUTHORED", "AFFILIATED_WITH", "PROPOSED", "RELATED_TO", "PUBLISHED_IN"}

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "sysu-big-data")


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA_DIR / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise SystemExit("Please install dependencies first: python3 -m pip install -r requirements.txt") from exc

    nodes = read_csv("nodes.csv")
    relationships = read_csv("relationships.csv")

    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
        for node in nodes:
            label = node["label"]
            if label not in NODE_LABELS:
                raise ValueError(f"Unsupported node label: {label}")
            session.run(
                f"""
                MERGE (e:Entity:{label} {{id: $id}})
                SET e.name = $name,
                    e.label = $label,
                    e.source_doc = $source_doc
                """,
                **node,
            )
        for rel in relationships:
            rel_type = rel["type"]
            if rel_type not in REL_TYPES:
                raise ValueError(f"Unsupported relationship type: {rel_type}")
            session.run(
                f"""
                MATCH (a:Entity {{id: $source}}), (b:Entity {{id: $target}})
                MERGE (a)-[r:{rel_type}]->(b)
                SET r.evidence = $evidence,
                    r.source_doc = $source_doc
                """,
                **rel,
            )
    driver.close()
    print(f"imported nodes={len(nodes)} relationships={len(relationships)}")


if __name__ == "__main__":
    main()

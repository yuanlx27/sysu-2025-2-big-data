from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "report"


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA_DIR / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    nodes = read_csv("nodes.csv")
    relationships = read_csv("relationships.csv")
    labels = Counter(row["label"] for row in nodes)
    rel_types = Counter(row["type"] for row in relationships)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with (REPORT_DIR / "result.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["nodes", len(nodes)])
        writer.writerow(["relationships", len(relationships)])
        for label, count in sorted(labels.items()):
            writer.writerow([f"node_{label}", count])
        for rel_type, count in sorted(rel_types.items()):
            writer.writerow([f"relation_{rel_type}", count])

    print(f"nodes={len(nodes)} relationships={len(relationships)}")


if __name__ == "__main__":
    main()

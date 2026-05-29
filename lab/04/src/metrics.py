from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClassificationResult:
    model: str
    feature: str
    accuracy: float
    train_time_sec: float
    test_time_sec: float


def write_results(results: list[ClassificationResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["model", "feature", "accuracy", "train_time_sec", "test_time_sec"])
        for item in results:
            writer.writerow(
                [
                    item.model,
                    item.feature,
                    f"{item.accuracy:.4f}",
                    f"{item.train_time_sec:.2f}",
                    f"{item.test_time_sec:.2f}",
                ]
            )

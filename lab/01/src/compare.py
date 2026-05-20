from __future__ import annotations

import pandas as pd

from utils import REPORT_DIR


def read_result(path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")
    return pd.read_csv(path)


def run() -> None:
    result_path = REPORT_DIR / "result.csv"
    result = read_result(result_path)
    required = {"method", "dataset", "accuracy", "train_time_sec", "test_time_sec"}
    missing = required.difference(result.columns)
    if missing:
        raise ValueError(f"Missing columns in {result_path}: {sorted(missing)}")
    if len(result) < 2:
        raise ValueError(f"Expected at least two result rows in {result_path}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    run()

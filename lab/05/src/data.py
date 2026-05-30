from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode
from urllib.error import URLError
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from pandas.errors import ParserError


@dataclass(frozen=True)
class StockSeries:
    symbol: str
    source_path: Path
    dates: list[str]
    close: np.ndarray
    used_demo_data: bool


def load_stock_series(root_dir: Path, horizon: int = 7) -> StockSeries:
    csv_env = os.environ.get("LAB05_CSV")
    symbol = os.environ.get("LAB05_SYMBOL", "AAPL").upper()

    used_demo_data = False
    if csv_env:
        csv_path = (root_dir / csv_env).resolve() if not Path(csv_env).is_absolute() else Path(csv_env)
    else:
        csv_path = root_dir / "data" / "raw" / f"{symbol.lower()}_alpha_vantage.csv"
        if not csv_path.exists() or not is_valid_stock_csv(csv_path):
            try:
                download_alpha_vantage_csv(symbol, csv_path)
            except (OSError, URLError, ValueError) as exc:
                print(f"Could not download {symbol} from Alpha Vantage: {exc}")
                csv_path = root_dir / "data" / "raw" / "demo_stock.csv"
                create_demo_csv(csv_path)
                used_demo_data = True

    try:
        dates, close = read_close_prices(csv_path)
    except (ParserError, ValueError) as exc:
        if csv_env:
            raise
        print(f"Could not read {csv_path.relative_to(root_dir)} as stock CSV: {exc}")
        csv_path = root_dir / "data" / "raw" / "demo_stock.csv"
        create_demo_csv(csv_path)
        used_demo_data = True
        dates, close = read_close_prices(csv_path)
    if len(close) < horizon + 40:
        raise ValueError(f"Need at least {horizon + 40} rows, got {len(close)} from {csv_path}")
    return StockSeries(symbol=symbol, source_path=csv_path, dates=dates, close=close, used_demo_data=used_demo_data)


def download_alpha_vantage_csv(symbol: str, output_path: Path) -> None:
    api_key = os.environ.get("LAB05_ALPHA_VANTAGE_APIKEY")
    if not api_key:
        raise ValueError("set LAB05_ALPHA_VANTAGE_APIKEY or use LAB05_CSV")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    query = urlencode(
        {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": os.environ.get("LAB05_OUTPUTSIZE", "compact"),
            "datatype": "csv",
            "apikey": api_key,
        }
    )
    url = f"https://www.alphavantage.co/query?{query}"
    urlretrieve(url, output_path)
    if output_path.stat().st_size < 100:
        raise ValueError("downloaded file is too small")
    if not is_valid_stock_csv(output_path):
        raise ValueError("downloaded file does not contain Date and Close columns")


def is_valid_stock_csv(csv_path: Path) -> bool:
    try:
        header = pd.read_csv(csv_path, nrows=0)
    except (OSError, ParserError, UnicodeDecodeError):
        return False
    columns = {col.lower() for col in header.columns}
    return ("date" in columns or "timestamp" in columns or "日期" in columns) and (
        "close" in columns or "收盘" in columns or "收盘价" in columns
    )


def create_demo_csv(output_path: Path, n_days: int = 260) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    rows = []
    price = 120.0
    start_day = pd.Timestamp("2025-01-02")
    for i in range(n_days):
        seasonal = math.sin(i / 11.0) * 0.7
        drift = 0.04
        shock = rng.normal(0.0, 1.2)
        price = max(10.0, price + drift + seasonal + shock)
        day = start_day + pd.offsets.BDay(i)
        rows.append({"Date": day.strftime("%Y-%m-%d"), "Close": round(price, 2)})
    pd.DataFrame(rows).to_csv(output_path, index=False)


def read_close_prices(csv_path: Path) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(csv_path)
    columns = {col.lower(): col for col in df.columns}
    date_col = columns.get("date") or columns.get("timestamp") or columns.get("日期")
    close_col = columns.get("close") or columns.get("收盘") or columns.get("收盘价")
    if date_col is None or close_col is None:
        raise ValueError("CSV must contain Date and Close columns")

    clean = df[[date_col, close_col]].dropna().copy()
    clean[date_col] = pd.to_datetime(clean[date_col])
    clean[close_col] = pd.to_numeric(clean[close_col], errors="coerce")
    clean = clean.dropna().sort_values(date_col)

    dates = clean[date_col].dt.strftime("%Y-%m-%d").tolist()
    close = clean[close_col].to_numpy(dtype=np.float64)
    return dates, close


def train_test_split(series: np.ndarray, horizon: int = 7) -> tuple[np.ndarray, np.ndarray]:
    return series[:-horizon], series[-horizon:]

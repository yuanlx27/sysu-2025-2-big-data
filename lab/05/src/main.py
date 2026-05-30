from __future__ import annotations

import os
from pathlib import Path

from arima_model import forecast_arima
from data import load_stock_series, train_test_split
from lstm_model import forecast_lstm
from metrics import write_results
from visualize import plot_forecast, plot_loss


ROOT_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT_DIR / "report"
FIGURE_DIR = REPORT_DIR / "figures"
RESULT_PATH = REPORT_DIR / "result.csv"


def main() -> None:
    horizon = _env_int("LAB05_HORIZON", 7)
    lstm_epochs = _env_int("LAB05_LSTM_EPOCHS", 60)
    window = _env_int("LAB05_WINDOW", 20)

    print("Stock price forecasting experiment")
    print(f"Config: horizon={horizon}, lstm_window={window}, lstm_epochs={lstm_epochs}")

    print("Load data")
    stock = load_stock_series(ROOT_DIR, horizon=horizon)
    train, test = train_test_split(stock.close, horizon=horizon)
    print(f"  symbol: {stock.symbol}")
    print(f"  rows: {len(stock.close)}")
    print(f"  source: {stock.source_path.relative_to(ROOT_DIR)}")
    if stock.used_demo_data:
        print("  note: using generated demo data because online data was unavailable")

    print("Fit ARIMA model")
    arima_result = forecast_arima(train, test)
    print(f"  MAE={arima_result.mae:.4f}, RMSE={arima_result.rmse:.4f}")

    print("Train LSTM model")
    lstm_result, losses = forecast_lstm(train, test, window=window, epochs=lstm_epochs)
    print(f"  MAE={lstm_result.mae:.4f}, RMSE={lstm_result.rmse:.4f}")

    results = [arima_result, lstm_result]
    write_results(results, RESULT_PATH)
    plot_forecast(stock.dates, train, test, results, FIGURE_DIR / "forecast.svg")
    plot_loss(losses, FIGURE_DIR / "loss.svg")

    print("Write outputs")
    print(f"  metrics: {RESULT_PATH.relative_to(ROOT_DIR)}")
    print(f"  forecast figure: {(FIGURE_DIR / 'forecast.svg').relative_to(ROOT_DIR)}")
    print(f"  loss figure: {(FIGURE_DIR / 'loss.svg').relative_to(ROOT_DIR)}")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


if __name__ == "__main__":
    main()

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from metrics import ForecastResult, mean_absolute_error, root_mean_squared_error


class PriceLSTM(nn.Module):
    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(inputs)
        return self.linear(output[:, -1, :])


def forecast_lstm(
    train: np.ndarray,
    test: np.ndarray,
    window: int = 20,
    epochs: int = 60,
    batch_size: int = 16,
) -> tuple[ForecastResult, list[float]]:
    torch.manual_seed(2026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean = float(train.mean())
    std = float(train.std()) or 1.0
    train_scaled = (train - mean) / std

    x_train, y_train = make_windows(train_scaled, window)
    dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32).unsqueeze(-1),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = PriceLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    losses: list[float] = []

    train_start = time.perf_counter()
    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(inputs), targets)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(inputs)
        losses.append(epoch_loss / len(dataset))
    train_time = time.perf_counter() - train_start

    predict_start = time.perf_counter()
    predictions = recursive_forecast(model, train_scaled, len(test), window, device)
    predictions = predictions * std + mean
    predict_time = time.perf_counter() - predict_start

    result = ForecastResult(
        model="LSTM",
        mae=mean_absolute_error(test, predictions),
        rmse=root_mean_squared_error(test, predictions),
        train_time_sec=train_time,
        predict_time_sec=predict_time,
        predictions=predictions,
    )
    return result, losses


def make_windows(series: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(window, len(series)):
        xs.append(series[i - window : i])
        ys.append(series[i])
    return np.asarray(xs), np.asarray(ys)


def recursive_forecast(
    model: PriceLSTM,
    train_scaled: np.ndarray,
    steps: int,
    window: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    history = train_scaled.astype(np.float64).tolist()
    predictions: list[float] = []
    with torch.no_grad():
        for _ in range(steps):
            inputs = torch.tensor(history[-window:], dtype=torch.float32, device=device).view(1, window, 1)
            pred = float(model(inputs).cpu().item())
            predictions.append(pred)
            history.append(pred)
    return np.asarray(predictions, dtype=np.float64)

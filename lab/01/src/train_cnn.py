from __future__ import annotations

import csv
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils import MODELS_DIR, get_env_int, load_digit_dataset, time_call


class SmallCNN(nn.Module):
    def __init__(self, image_shape: tuple[int, int]) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, image_shape[0], image_shape[1])
            flattened = int(np.prod(self.features(dummy).shape[1:]))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def make_loader(x: np.ndarray, y: np.ndarray, image_shape: tuple[int, int], batch_size: int, shuffle: bool) -> DataLoader:
    images = torch.tensor(x.reshape(-1, 1, image_shape[0], image_shape[1]), dtype=torch.float32)
    labels = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=shuffle)


def train_model(model: SmallCNN, loader: DataLoader, epochs: int, device: torch.device) -> None:
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
        print(f"CNN epoch {epoch + 1}/{epochs}, loss={total_loss / len(loader.dataset):.4f}", file=sys.stderr)


def predict(model: SmallCNN, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for images, _ in loader:
            logits = model(images.to(device))
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions)


def run() -> None:
    torch.manual_seed(42)
    train_limit = get_env_int("CNN_TRAIN_LIMIT", 5000)
    test_limit = get_env_int("TEST_LIMIT", 1000)
    epochs = get_env_int("CNN_EPOCHS", 3)
    batch_size = get_env_int("CNN_BATCH_SIZE", 64)

    data = load_digit_dataset(train_limit=train_limit, test_limit=test_limit)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN(data.image_shape).to(device)

    train_loader = make_loader(data.x_train, data.y_train, data.image_shape, batch_size, shuffle=True)
    test_loader = make_loader(data.x_test, data.y_test, data.image_shape, batch_size, shuffle=False)

    _, train_time = time_call(lambda: train_model(model, train_loader, epochs, device))
    predictions, test_time = time_call(lambda: predict(model, test_loader, device))
    accuracy = accuracy_score(data.y_test, predictions)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / "cnn.pt")
    csv.writer(sys.stdout).writerow(
        [
            "CNN",
            data.name,
            f"{accuracy:.4f}",
            f"{train_time:.4f}",
            f"{test_time:.4f}",
        ]
    )
    print(f"CNN accuracy={accuracy:.4f}, train={train_time:.2f}s, test={test_time:.2f}s", file=sys.stderr)


if __name__ == "__main__":
    run()

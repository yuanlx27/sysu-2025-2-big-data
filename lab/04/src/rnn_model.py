from __future__ import annotations

import re
import time
from collections import Counter
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

from metrics import ClassificationResult


TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+")


class Vocabulary:
    def __init__(self, texts: list[str], max_size: int = 20000, min_freq: int = 2) -> None:
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(tokenize(text))

        self.token_to_id = {"<pad>": 0, "<unk>": 1}
        for token, freq in counter.most_common(max_size - len(self.token_to_id)):
            if freq < min_freq:
                break
            self.token_to_id[token] = len(self.token_to_id)

    def encode(self, text: str, max_len: int) -> tuple[list[int], int]:
        ids = [self.token_to_id.get(token, 1) for token in tokenize(text)[:max_len]]
        length = max(1, len(ids))
        if len(ids) < max_len:
            ids.extend([0] * (max_len - len(ids)))
        return ids, length

    def __len__(self) -> int:
        return len(self.token_to_id)


class ReviewDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, texts: list[str], labels: list[int], vocab: Vocabulary, max_len: int) -> None:
        encoded = [vocab.encode(text, max_len) for text in texts]
        self.inputs = torch.tensor([ids for ids, _ in encoded], dtype=torch.long)
        self.lengths = torch.tensor([length for _, length in encoded], dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.lengths[index], self.labels[index]


class SentimentGRU(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 64, hidden_dim: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(inputs)
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        logits = self.classifier(self.dropout(hidden[-1])).squeeze(1)
        return logits


def train_rnn(
    train_texts: list[str],
    train_labels: list[int],
    test_texts: list[str],
    test_labels: list[int],
    model_dir: Path,
    epochs: int = 2,
    batch_size: int = 64,
    max_len: int = 200,
    vocab_size: int = 20000,
) -> ClassificationResult:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocabulary(train_texts, max_size=vocab_size)
    train_dataset = ReviewDataset(train_texts, train_labels, vocab, max_len)
    test_dataset = ReviewDataset(test_texts, test_labels, vocab, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = SentimentGRU(len(vocab)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_start = time.perf_counter()
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, lengths, labels in train_loader:
            inputs = inputs.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * inputs.size(0)
        avg_loss = total_loss / len(train_dataset)
        print(f"RNN epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}")
    train_time = time.perf_counter() - train_start

    test_start = time.perf_counter()
    predictions = predict(model, test_loader, device)
    test_time = time.perf_counter() - test_start

    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "token_to_id": vocab.token_to_id,
            "max_len": max_len,
        },
        model_dir / "rnn.pt",
    )
    return ClassificationResult(
        model="GRU RNN",
        feature="Word Embedding",
        accuracy=float(accuracy_score(test_labels, predictions)),
        train_time_sec=train_time,
        test_time_sec=test_time,
    )


def predict(model: SentimentGRU, loader: DataLoader, device: torch.device) -> list[int]:
    model.eval()
    predictions: list[int] = []
    with torch.no_grad():
        for inputs, lengths, _ in loader:
            logits = model(inputs.to(device), lengths.to(device))
            probs = torch.sigmoid(logits)
            predictions.extend((probs >= 0.5).long().cpu().tolist())
    return predictions


def tokenize(text: str) -> list[str]:
    return [item.lower() for item in TOKEN_RE.findall(text)]

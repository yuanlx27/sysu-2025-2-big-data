from __future__ import annotations

import csv
import html
import random
import re
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path


IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


@dataclass(frozen=True)
class TextDataset:
    train_texts: list[str]
    train_labels: list[int]
    test_texts: list[str]
    test_labels: list[int]


def load_imdb_dataset(root_dir: Path, max_train: int | None = None, max_test: int | None = None) -> TextDataset:
    raw_dir = root_dir / "data" / "raw"
    processed_dir = root_dir / "data" / "processed"
    dataset_dir = raw_dir / "aclImdb"
    if not dataset_dir.exists():
        download_imdb(raw_dir)

    train_csv = processed_dir / "train.csv"
    test_csv = processed_dir / "test.csv"
    if not train_csv.exists() or not test_csv.exists():
        processed_dir.mkdir(parents=True, exist_ok=True)
        write_split_csv(dataset_dir, "train", train_csv)
        write_split_csv(dataset_dir, "test", test_csv)

    train_texts, train_labels = read_split_csv(train_csv, max_train)
    test_texts, test_labels = read_split_csv(test_csv, max_test)
    return TextDataset(train_texts, train_labels, test_texts, test_labels)


def download_imdb(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / "aclImdb_v1.tar.gz"
    if not archive_path.exists():
        print(f"Downloading IMDB dataset from {IMDB_URL}")
        urllib.request.urlretrieve(IMDB_URL, archive_path)

    print(f"Extracting {archive_path}")
    with tarfile.open(archive_path, "r:gz") as tar:
        safe_extract(tar, raw_dir)


def safe_extract(tar: tarfile.TarFile, output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    for member in tar.getmembers():
        member_path = (output_dir / member.name).resolve()
        if output_dir not in member_path.parents and member_path != output_dir:
            raise RuntimeError(f"Unsafe path in archive: {member.name}")
    tar.extractall(output_dir)


def write_split_csv(dataset_dir: Path, split: str, output_path: Path) -> None:
    rows: list[tuple[str, int]] = []
    for label_name, label_value in (("neg", 0), ("pos", 1)):
        split_dir = dataset_dir / split / label_name
        for path in sorted(split_dir.glob("*.txt")):
            text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
            rows.append((text, label_value))

    random.Random(42).shuffle(rows)
    with output_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["text", "label"])
        writer.writerows(rows)


def read_split_csv(path: Path, max_items: int | None = None) -> tuple[list[str], list[int]]:
    rows: list[tuple[str, int]] = []
    with path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows.append((row["text"], int(row["label"])))

    if max_items is not None and max_items < len(rows):
        rows = balanced_sample(rows, max_items)

    texts = [text for text, _ in rows]
    labels = [label for _, label in rows]
    return texts, labels


def balanced_sample(rows: list[tuple[str, int]], max_items: int) -> list[tuple[str, int]]:
    rng = random.Random(42)
    by_label = {
        0: [row for row in rows if row[1] == 0],
        1: [row for row in rows if row[1] == 1],
    }
    per_label = max_items // 2
    sampled = by_label[0][:per_label] + by_label[1][: max_items - per_label]
    rng.shuffle(sampled)
    return sampled


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

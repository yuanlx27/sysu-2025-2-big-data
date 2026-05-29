from __future__ import annotations

import os
from pathlib import Path

from data import load_imdb_dataset
from metrics import ClassificationResult, write_results
from rnn_model import train_rnn
from tfidf_models import train_logistic_regression, train_svm
from visualize import plot_accuracy, plot_time


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "report"
FIGURE_DIR = REPORT_DIR / "figures"
RESULT_PATH = REPORT_DIR / "result.csv"


def main() -> None:
    max_train = _env_int("LAB04_MAX_TRAIN")
    max_test = _env_int("LAB04_MAX_TEST")
    rnn_epochs = _env_int("LAB04_RNN_EPOCHS", default=2)
    rnn_batch_size = _env_int("LAB04_RNN_BATCH_SIZE", default=64)

    dataset = load_imdb_dataset(ROOT_DIR, max_train=max_train, max_test=max_test)
    print(
        "Loaded IMDB reviews: "
        f"train={len(dataset.train_texts)}, test={len(dataset.test_texts)}"
    )

    results: list[ClassificationResult] = []
    print("[1/3] Train TF-IDF + Logistic Regression")
    results.append(
        train_logistic_regression(
            dataset.train_texts,
            dataset.train_labels,
            dataset.test_texts,
            dataset.test_labels,
            MODEL_DIR,
        )
    )

    print("[2/3] Train TF-IDF + Linear SVM")
    results.append(
        train_svm(
            dataset.train_texts,
            dataset.train_labels,
            dataset.test_texts,
            dataset.test_labels,
            MODEL_DIR,
        )
    )

    print("[3/3] Train GRU RNN")
    results.append(
        train_rnn(
            dataset.train_texts,
            dataset.train_labels,
            dataset.test_texts,
            dataset.test_labels,
            MODEL_DIR,
            epochs=rnn_epochs,
            batch_size=rnn_batch_size,
        )
    )

    write_results(results, RESULT_PATH)
    plot_accuracy(results, FIGURE_DIR / "accuracy_comparison.svg")
    plot_time(results, FIGURE_DIR / "time_comparison.svg")

    print(f"Wrote {RESULT_PATH.relative_to(ROOT_DIR)}")
    print(f"Wrote figures to {FIGURE_DIR.relative_to(ROOT_DIR)}")


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


if __name__ == "__main__":
    main()

from __future__ import annotations

import time
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from metrics import ClassificationResult


def train_logistic_regression(
    train_texts: list[str],
    train_labels: list[int],
    test_texts: list[str],
    test_labels: list[int],
    model_dir: Path,
) -> ClassificationResult:
    pipeline = Pipeline(
        [
            ("tfidf", build_vectorizer()),
            ("classifier", LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)),
        ]
    )
    return _fit_evaluate_save(
        pipeline,
        "Logistic Regression",
        train_texts,
        train_labels,
        test_texts,
        test_labels,
        model_dir / "logistic_regression.joblib",
    )


def train_svm(
    train_texts: list[str],
    train_labels: list[int],
    test_texts: list[str],
    test_labels: list[int],
    model_dir: Path,
) -> ClassificationResult:
    pipeline = Pipeline(
        [
            ("tfidf", build_vectorizer()),
            ("classifier", LinearSVC(random_state=42)),
        ]
    )
    return _fit_evaluate_save(
        pipeline,
        "Linear SVM",
        train_texts,
        train_labels,
        test_texts,
        test_labels,
        model_dir / "svm.joblib",
    )


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=20000,
        min_df=2,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )


def _fit_evaluate_save(
    pipeline: Pipeline,
    model_name: str,
    train_texts: list[str],
    train_labels: list[int],
    test_texts: list[str],
    test_labels: list[int],
    output_path: Path,
) -> ClassificationResult:
    train_start = time.perf_counter()
    pipeline.fit(train_texts, train_labels)
    train_time = time.perf_counter() - train_start

    test_start = time.perf_counter()
    predictions = pipeline.predict(test_texts)
    test_time = time.perf_counter() - test_start

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fp:
        pickle.dump(pipeline, fp)
    return ClassificationResult(
        model=model_name,
        feature="TF-IDF",
        accuracy=float(accuracy_score(test_labels, predictions)),
        train_time_sec=train_time,
        test_time_sec=test_time,
    )

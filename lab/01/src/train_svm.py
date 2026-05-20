from __future__ import annotations

import csv
import sys

import joblib
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from utils import MODELS_DIR, get_env_int, load_digit_dataset, time_call


def run() -> None:
    train_limit = get_env_int("SVM_TRAIN_LIMIT", 3000)
    test_limit = get_env_int("TEST_LIMIT", 1000)
    data = load_digit_dataset(train_limit=train_limit, test_limit=test_limit)

    model = SVC(kernel="rbf", gamma="scale", C=5.0)

    _, train_time = time_call(lambda: model.fit(data.x_train, data.y_train))
    predictions, test_time = time_call(lambda: model.predict(data.x_test))
    accuracy = accuracy_score(data.y_test, predictions)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "svm.joblib")
    csv.writer(sys.stdout).writerow(
        [
            "SVM",
            data.name,
            f"{accuracy:.4f}",
            f"{train_time:.4f}",
            f"{test_time:.4f}",
        ]
    )
    print(f"SVM accuracy={accuracy:.4f}, train={train_time:.2f}s, test={test_time:.2f}s", file=sys.stderr)


if __name__ == "__main__":
    run()

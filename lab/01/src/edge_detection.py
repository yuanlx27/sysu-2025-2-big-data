from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from utils import FIGURES_DIR, SAMPLE_DIR, ensure_dirs


SAMPLE_IMAGE = SAMPLE_DIR / "sample.png"


def create_sample_image(path: Path) -> None:
    image = np.full((360, 540, 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (45, 50), (245, 250), (40, 120, 220), -1)
    cv2.circle(image, (365, 150), 92, (60, 170, 90), -1)
    cv2.line(image, (80, 305), (470, 305), (30, 30, 30), 8)
    cv2.putText(
        image,
        "Sobel",
        (300, 320),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (20, 20, 20),
        4,
        cv2.LINE_AA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def run() -> None:
    ensure_dirs()
    if not SAMPLE_IMAGE.exists():
        create_sample_image(SAMPLE_IMAGE)

    image = cv2.imread(str(SAMPLE_IMAGE), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read sample image: {SAMPLE_IMAGE}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_x = cv2.convertScaleAbs(grad_x)
    abs_y = cv2.convertScaleAbs(grad_y)
    sobel = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)

    cv2.imwrite(str(FIGURES_DIR / "original.png"), image)
    cv2.imwrite(str(FIGURES_DIR / "gray.png"), gray)
    cv2.imwrite(str(FIGURES_DIR / "sobel.png"), sobel)
    print(f"Edge figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    run()

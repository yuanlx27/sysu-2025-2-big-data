#!/usr/bin/env bash
set -eu

ROOT_DIR="$(dirname "$0")"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-../../.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
  PYTHON="python"
fi

mkdir -p data/samples report/figures

if [ ! -f data/samples/cat_dog.jpg ]; then
  echo "Missing data/samples/cat_dog.jpg. See data/README.md for the source image URL."
  exit 1
fi

echo "[1/1] Run ViLT VQA cases"
"$PYTHON" src/run_cases.py --cases data/cases.json --output report/result.csv --figures report/figures

#!/usr/bin/env bash
set -eu

ROOT_DIR="$(dirname "$0")"
cd "$ROOT_DIR"

mkdir -p data/raw data/processed models report/figures

echo "[1/1] Run IMDB sentiment classification experiments"
python3 src/experiment.py

#!/usr/bin/env bash
set -eu

ROOT_DIR="$(dirname "$0")"
cd "$ROOT_DIR"

mkdir -p report/figures

echo "[1/1] Run Iris clustering experiments"
python3 src/experiment.py

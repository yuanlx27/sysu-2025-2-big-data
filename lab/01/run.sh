#!/usr/bin/env bash
set -eu

ROOT_DIR="$(dirname $0)"
cd "$ROOT_DIR"

mkdir -p data/raw data/sample models report/figures
printf "method,dataset,accuracy,train_time_sec,test_time_sec\n" > report/result.csv

echo "[1/4] Sobel edge detection"
python3 src/edge_detection.py

echo "[2/4] Train SVM classifier"
python3 src/train_svm.py >> report/result.csv

echo "[3/4] Train CNN classifier"
python3 src/train_cnn.py >> report/result.csv

echo "[4/4] Check comparison CSV"
python3 src/compare.py

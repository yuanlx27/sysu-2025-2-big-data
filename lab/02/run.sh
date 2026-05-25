#!/usr/bin/env bash
set -eu

ROOT_DIR="$(dirname "$0")"
cd "$ROOT_DIR"

mkdir -p data/processed report/figures

echo "[1/4] Extract entities and relations"
python3 src/extract.py

echo "[2/4] Build graph summary"
python3 src/build_graph.py

echo "[3/4] Generate local visualizations"
python3 src/visualize.py

echo "[4/4] Start Neo4j"
if ! docker compose --version &> /dev/null; then
  echo "docker compose is required to start Neo4j." >&2
  exit 1
fi

docker compose up -d

cat <<MSG

Neo4j import:
  1. Wait for Neo4j to finish starting.
  2. python3 src/import_neo4j.py
  3. Open http://localhost:7474 and run src/neo4j/queries.cypher

MSG

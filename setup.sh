#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

RAW_DIR="${SCRIPT_DIR}/data/raw"
RAW_FILE="${RAW_DIR}/yellow_tripdata_2023-01.parquet"
VENV_DIR="${SCRIPT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"

echo "==> Creating virtual environment with ${PYTHON_BIN}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python 3.10 not found. Install python3.10 or set PYTHON_BIN."
  exit 1
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "==> Installing Python dependencies"
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "==> Downloading NYC taxi parquet dataset"
mkdir -p "${RAW_DIR}"
python pipeline/download_raw_data.py

echo "==> Running PySpark feature engineering"
python pipeline/spark_features.py

echo "==> Starting Redis for Feast online materialization"
if command -v docker >/dev/null 2>&1; then
  docker compose up -d redis
  echo "Waiting for Redis to become ready..."
  sleep 3
else
  echo "Docker not found. Ensure Redis is running on localhost:6379 before materialization."
fi

echo "==> Applying Feast feature definitions"
if [[ "$(uname -s)" == "MINGW"* || "$(uname -s)" == "CYGWIN"* || "$(uname -s)" == "MSYS"* ]]; then
  echo "Detected Windows shell. Running Feast apply inside Docker..."
  docker compose run --rm --no-deps api bash -c "cd feature_store/feature_repo && feast apply"
else
  cd feature_store/feature_repo
  feast apply
  cd "${SCRIPT_DIR}"
fi

echo "==> Materializing features to Redis"
bash feature_store/materialize.sh

echo "==> Training model and logging feature metadata"
python training/train.py

echo "Setup complete. Run docker-compose up to start the API."

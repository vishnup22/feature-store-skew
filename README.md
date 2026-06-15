# Feature Store Skew Demo

[![CI](https://github.com/your-org/feature-store-skew/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/feature-store-skew/actions/workflows/ci.yml)

End-to-end ML engineering demo that proves **train-serve feature parity** using Feast, Redis, PySpark, and FastAPI. Every component runs locally against real NYC Taxi trip data — no mocks, no placeholders.

## Architecture

```
┌─────────────────────────┐
│ Raw Parquet             │
│ yellow_tripdata_2023-01 │
└────────────┬────────────┘
             │
             v
┌─────────────────────────┐
│ PySpark Pipeline        │
│ pipeline/spark_features │
└────────────┬────────────┘
             │
             v
┌─────────────────────────┐
│ Processed Parquet       │
│ vendor_features.parquet │
└────────────┬────────────┘
             │
             v
┌─────────────────────────┐       Feast FileSource
│ Feast Offline Store     │◄──────────────────────────┐
│ (local file provider)   │                           │
└────────────┬────────────┘                           │
             │                                         │
             v                                         │
┌─────────────────────────┐                           │
│ Training Script         │                           │
│ training/train.py       │                           │
│ - get_historical_features                          │
│ - XGBoost + feature hashes                         │
└────────────┬────────────┘                           │
             │                                         │
             │ feast materialize                       │
             v                                         │
┌─────────────────────────┐                           │
│ Redis Online Store      │◄──────────────────────────┘
│ localhost:6379          │
└────────────┬────────────┘
             │
             v
┌─────────────────────────┐
│ FastAPI Serving         │
│ serving/api.py          │
│ - get_online_features   │
└────────────┬────────────┘
             │
             v
┌─────────────────────────┐
│ Client / pytest / notebook
└─────────────────────────┘
```

## What is Train-Serve Skew?

Train-serve skew happens when the features used during model training differ from the features served at inference time. Common causes include:

- Different transformation code in training vs serving pipelines
- Schema or dtype drift between offline and online stores
- Point-in-time correctness bugs in historical joins
- Column reordering that silently changes model inputs

This project prevents skew by:

1. Computing features once in PySpark and registering the same parquet in Feast
2. Training with `get_historical_features()` (offline store)
3. Serving with `get_online_features()` (Redis online store)
4. Logging SHA256 hashes of the offline feature matrix during training (global + per-vendor)
5. Failing fast if serving column order differs from `data/training_meta.json`
6. Proving parity with pytest and a comparison notebook

Shared helpers in `feature_store/feature_hash.py` and `feature_store/feast_client.py` keep hashing and column ordering consistent across training, serving, and tests.

## Quickstart

### Prerequisites

- Python 3.10 (Feast 0.36 is tested on 3.10; 3.11+ may hit dependency conflicts)
- Java 17+ (required by PySpark)
- Docker + Docker Compose (Redis, Feast apply on Windows, API)

### 1. Clone and bootstrap

**Linux / macOS / WSL**

```bash
git clone https://github.com/your-org/feature-store-skew.git
cd feature-store-skew
chmod +x setup.sh feature_store/materialize.sh
./setup.sh
```

**Windows (PowerShell)**

```powershell
git clone https://github.com/your-org/feature-store-skew.git
cd feature-store-skew
.\setup.ps1
```

Both setup scripts will:

1. Create a `.venv` virtual environment and install dependencies
2. Download `yellow_tripdata_2023-01.parquet`
3. Run PySpark feature engineering
4. Apply Feast definitions (`feast apply` — via Docker on native Windows)
5. Materialize January 2023 features to Redis (`feature_store/materialize.py`)
6. Train XGBoost and write `data/training_meta.json` and `data/model.json`

### 2. Start Redis + API

```bash
docker compose up --build
```

API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

Health check:

```bash
curl http://localhost:8000/health
```

Example prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"vendor_id": "1"}'
```

Example feature lookup:

```bash
curl http://localhost:8000/features/1
```

The `/predict` response includes per-vendor hash parity:

```json
{
  "vendor_id": "1",
  "prediction": 0,
  "prediction_label": "not_high_tip",
  "probability": 0.42,
  "online_feature_hash": "...",
  "training_hash": "...",
  "hashes_match": true,
  "feature_values": { "...": 0.0 }
}
```

### 3. Run parity tests

Redis must be running and features materialized before tests pass.

**Linux / macOS / WSL**

```bash
source .venv/bin/activate
pytest tests/test_skew.py -v
```

**Windows (PowerShell)**

```powershell
.\.venv\Scripts\Activate.ps1
pytest tests/test_skew.py -v
```

Expected result: all tests pass, with tabulated PASS/FAIL output per test.

## Continuous Integration

GitHub Actions runs the full parity pipeline on every push and pull request to `main`/`master`:

1. Download NYC taxi parquet
2. PySpark feature engineering
3. Feast apply + Redis materialization (`feature_store/materialize.py`)
4. Offline training + metadata logging
5. `pytest tests/test_skew.py -v`
6. FastAPI health check + Docker image build

Workflow file: [`.github/workflows/ci.yml`](.github/workflows/ci.yml)

Replace `your-org/feature-store-skew` in the README badge URL after publishing the repository.

## Project Layout

```
feature-store-skew/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── setup.sh
├── setup.ps1
├── pytest.ini
├── data/
│   ├── raw/                         # downloaded parquet
│   ├── processed/                   # Spark output
│   ├── training_meta.json           # feature contract (generated)
│   └── model.json                   # XGBoost model (generated)
├── pipeline/
│   └── spark_features.py
├── feature_store/
│   ├── feast_client.py              # shared Feast init + column-order patch
│   ├── feature_hash.py              # SHA256 hashing + Float32 enforcement
│   ├── materialize.py               # materialize to Redis (used by CI + setup.ps1)
│   ├── materialize.sh               # bash wrapper for setup.sh
│   └── feature_repo/
│       ├── feature_store.yaml
│       ├── entities.py
│       ├── data_sources.py
│       └── features.py
├── training/
│   └── train.py
├── serving/
│   └── api.py
├── tests/
│   └── test_skew.py
└── notebooks/
    └── skew_analysis.ipynb
```

## Features

All features belong to the `vendor_stats` Feast feature view and are enforced as Float32 end-to-end.

| Feature | Description |
|---|---|
| `trip_count_last_7d` | Rolling 7-day trip count per vendor |
| `avg_fare_last_7d` | Rolling 7-day average fare per vendor |
| `avg_trip_distance_last_7d` | Rolling 7-day average trip distance |
| `avg_passenger_count_last_7d` | Rolling 7-day average passenger count |
| `peak_hour_ratio_last_7d` | Share of trips during 7–9am and 5–7pm |

Training uses timestamp `2023-01-31 23:59:59` so offline point-in-time features align with the latest values materialized into Redis.

## Training Contract

`data/training_meta.json` is the serving contract. It is written during training and validated at API startup.

| Field | Purpose |
|---|---|
| `feature_columns` | Canonical column order for online lookups |
| `feature_dtypes` | Expected dtypes per feature |
| `feature_hash` | SHA256 over the full offline matrix (sorted by `vendor_id`) |
| `vendor_hashes` | Per-vendor SHA256 hashes used by `/predict` |
| `sample_vendor_id` | Vendor used for startup column-order validation |
| `training_timestamp` | Point-in-time cutoff for offline features |

Feature values are rounded to 4 decimal places before hashing.

## Notebook Analysis

Open `notebooks/skew_analysis.ipynb` after setup to compare offline vs online vectors side-by-side, inspect dtype parity, recompute the SHA256 hash, and visualize feature distributions.

## Dependencies

`requirements.txt` pins Feast 0.36-compatible versions:

| Package | Version | Notes |
|---|---|---|
| feast | 0.36.0 | Requires `pandas<2`, `numpy<1.25` |
| pandas | 1.5.3 | Feast 0.36 constraint |
| numpy | 1.24.4 | Feast 0.36 constraint |
| dask / distributed | 2023.5.1 | Pinned for Feast + pandas 1.5 compatibility |
| scikit-learn | 1.4.1.post1 | `1.4.1` is unavailable on PyPI |

Use **Python 3.10** locally. Feast CLI is not supported on native Windows; use WSL, Linux, macOS, or Docker for `feast apply` (the Windows setup script runs it inside Docker automatically).

## Guardrails

- FastAPI refuses to start if online feature column order differs from `data/training_meta.json`
- FastAPI refuses to start if the sample vendor has null online features (re-run materialization)
- Pytest fails on dtype mismatch, null online features, value drift, or hash mismatch
- `/predict` compares per-vendor online hashes against `vendor_hashes` in training metadata
- Feature values are rounded to 4 decimal places before hashing

## License

MIT

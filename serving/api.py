from __future__ import annotations

import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from feast import FeatureStore
from pydantic import BaseModel, Field

from feature_store.feast_client import get_feature_store as init_feature_store
from feature_store.feature_hash import (
    compute_feature_hash,
    enforce_float32_features,
    extract_feature_dtypes,
)

FEATURE_REPO_PATH = PROJECT_ROOT / "feature_store" / "feature_repo"
META_PATH = PROJECT_ROOT / "data" / "training_meta.json"
MODEL_PATH = PROJECT_ROOT / "data" / "model.json"
PROCESSED_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "vendor_features.parquet"

FEATURE_REFS: list[str] = [
    "vendor_stats:trip_count_last_7d",
    "vendor_stats:avg_fare_last_7d",
    "vendor_stats:avg_trip_distance_last_7d",
    "vendor_stats:avg_passenger_count_last_7d",
    "vendor_stats:peak_hour_ratio_last_7d",
]

store: FeatureStore | None = None
model: xgb.XGBClassifier | None = None
training_meta: dict[str, Any] = {}
feature_columns: list[str] = []


class PredictRequest(BaseModel):
    vendor_id: str = Field(..., description="Vendor identifier used for online feature lookup.")


def get_feature_store() -> FeatureStore:
    if store is None:
        raise RuntimeError("Feature store has not been initialized.")
    return store


def create_feature_store() -> FeatureStore:
    return init_feature_store(FEATURE_REPO_PATH)


def load_training_metadata() -> dict[str, Any]:
    if not META_PATH.exists():
        raise RuntimeError(f"Training metadata not found at {META_PATH}. Run training first.")
    with META_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_feature_column_order(
    expected_columns: list[str],
    online_columns: list[str],
) -> None:
    if online_columns != expected_columns:
        raise RuntimeError(
            "Online feature column order does not match offline training order. "
            f"expected={expected_columns}, actual={online_columns}"
        )


def fetch_online_feature_frame(vendor_ids: list[str]) -> pd.DataFrame:
    entity_rows = [{"vendor_id": vendor_id} for vendor_id in vendor_ids]
    online_df = get_feature_store().get_online_features(
        features=FEATURE_REFS,
        entity_rows=entity_rows,
    ).to_df()

    online_df = enforce_float32_features(online_df, feature_columns)
    online_df = online_df.sort_values("vendor_id").reset_index(drop=True)

    null_counts = online_df[feature_columns].isna().sum()
    null_features = null_counts[null_counts > 0].index.tolist()
    if null_features:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Online store returned null values for {null_features} "
                f"(vendor_id={vendor_ids}) — Redis entries may have expired, "
                "re-run feast materialize."
            ),
        )

    return online_df


def list_all_vendor_ids() -> list[str]:
    table = pq.read_table(PROCESSED_FEATURES_PATH, columns=["vendor_id"])
    return (
        table.to_pandas()["vendor_id"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store, model, training_meta, feature_columns

    try:
        logger.info("Loading training metadata from %s", META_PATH)
        training_meta = load_training_metadata()
        expected_columns = list(training_meta["feature_columns"])

        logger.info("Connecting to feature store at %s", FEATURE_REPO_PATH)
        store = create_feature_store()
        sample_vendor_id = str(training_meta["sample_vendor_id"])
        sample_online = store.get_online_features(
            features=FEATURE_REFS,
            entity_rows=[{"vendor_id": sample_vendor_id}],
        ).to_df()

        null_features = [col for col in expected_columns if sample_online[col].isna().any()]
        if null_features:
            raise RuntimeError(
                f"Online store returned null features for vendor_id {sample_vendor_id} — "
                "run feast materialize before starting the server."
            )

        actual_columns = [col for col in sample_online.columns if col in set(expected_columns)]
        validate_feature_column_order(expected_columns, actual_columns)

        feature_columns = expected_columns

        if not MODEL_PATH.exists():
            raise RuntimeError(f"Model file not found: {MODEL_PATH} — run training first.")
        logger.info("Loading model from %s", MODEL_PATH)
        model = xgb.XGBClassifier()
        model.load_model(str(MODEL_PATH))

        logger.info("Startup complete — feature_columns=%s", feature_columns)
    except Exception:
        logger.exception("Startup failed — server will not accept requests")
        raise

    yield


app = FastAPI(title="Taxi Feature Store Serving API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/features/{vendor_id}")
def get_features(vendor_id: str) -> dict[str, Any]:
    online_df = fetch_online_feature_frame([vendor_id])
    if online_df.empty:
        raise HTTPException(status_code=404, detail=f"No online features found for vendor_id={vendor_id}")

    row = online_df.iloc[0]
    values = row[feature_columns].astype(float).tolist()
    return {
        "vendor_id": vendor_id,
        "feature_columns": feature_columns,
        "feature_values": values,
        "feature_dtypes": extract_feature_dtypes(online_df, feature_columns),
        "vector_shape": [1, len(feature_columns)],
        "feature_hash": compute_feature_hash(online_df, feature_columns),
    }


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    online_df = fetch_online_feature_frame([request.vendor_id])
    if online_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No online features found for vendor_id={request.vendor_id}",
        )

    vendor_hashes = training_meta.get("vendor_hashes", {})
    training_hash = vendor_hashes.get(request.vendor_id)
    if training_hash is None:
        raise HTTPException(
            status_code=404,
            detail=f"No training hash for vendor_id={request.vendor_id} — re-run training.",
        )

    feature_frame = online_df[feature_columns].astype(np.float32)
    online_feature_hash = compute_feature_hash(online_df, feature_columns)
    prediction = int(model.predict(feature_frame)[0])
    probability = float(model.predict_proba(feature_frame)[0][1])
    row = feature_frame.iloc[0]

    return {
        "vendor_id": request.vendor_id,
        "prediction": prediction,
        "prediction_label": "high_tip" if prediction == 1 else "not_high_tip",
        "probability": probability,
        "online_feature_hash": online_feature_hash,
        "training_hash": training_hash,
        "hashes_match": online_feature_hash == training_hash,
        "feature_values": {col: float(row[col]) for col in feature_columns},
    }

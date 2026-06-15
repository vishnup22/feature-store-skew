from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from feast import FeatureStore
from tabulate import tabulate

from feature_store.feast_client import get_feature_store as init_feature_store
from feature_store.feature_hash import (
    compute_feature_hash,
    enforce_float32_features,
    extract_feature_dtypes,
)

FEATURE_REPO_PATH = PROJECT_ROOT / "feature_store" / "feature_repo"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "yellow_tripdata_2023-01.parquet"
META_PATH = PROJECT_ROOT / "data" / "training_meta.json"
MODEL_PATH = PROJECT_ROOT / "data" / "model.json"
TRAINING_TIMESTAMP = pd.Timestamp("2023-01-31 23:59:59")

FEATURE_REFS: list[str] = [
    "vendor_stats:trip_count_last_7d",
    "vendor_stats:avg_fare_last_7d",
    "vendor_stats:avg_trip_distance_last_7d",
    "vendor_stats:avg_passenger_count_last_7d",
    "vendor_stats:peak_hour_ratio_last_7d",
]

FEATURE_COLUMNS: list[str] = [ref.split(":")[1] for ref in FEATURE_REFS]


def get_feature_store() -> FeatureStore:
    return init_feature_store(FEATURE_REPO_PATH)


def build_entity_dataframe(processed_features_path: Path) -> pd.DataFrame:
    table = pq.read_table(processed_features_path)
    vendors = (
        table.to_pandas()["vendor_id"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    return pd.DataFrame(
        {
            "vendor_id": vendors,
            "event_timestamp": TRAINING_TIMESTAMP,
        }
    )


def load_vendor_labels(raw_data_path: Path, as_of_timestamp: pd.Timestamp) -> pd.DataFrame:
    raw = pq.read_table(raw_data_path, columns=["VendorID", "tpep_pickup_datetime", "fare_amount"])
    trips = raw.to_pandas()
    trips["vendor_id"] = trips["VendorID"].astype(str)
    trips["event_timestamp"] = pd.to_datetime(trips["tpep_pickup_datetime"], utc=True).dt.tz_localize(None)
    trips = trips[trips["event_timestamp"] <= as_of_timestamp]

    tip_rate = (
        trips.groupby("vendor_id")["fare_amount"]
        .apply(lambda fares: float((fares > 10).mean()))
        .rename("tip_rate")
    )
    threshold = tip_rate.mean()
    labels = tip_rate.apply(lambda rate: int(rate >= threshold)).rename("high_tip").reset_index()
    return labels.astype({"high_tip": int})


def pull_offline_features(
    store: FeatureStore,
    entity_df: pd.DataFrame,
) -> pd.DataFrame:
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=FEATURE_REFS,
    ).to_df()

    training_df = training_df.dropna(subset=FEATURE_COLUMNS).copy()
    training_df = enforce_float32_features(training_df, FEATURE_COLUMNS)
    return training_df.sort_values("vendor_id").reset_index(drop=True)


def build_training_metadata(training_df: pd.DataFrame) -> dict[str, Any]:
    feature_hash = compute_feature_hash(training_df, FEATURE_COLUMNS)
    sample_vendor_id = str(training_df.iloc[0]["vendor_id"])
    sample_vector = training_df.iloc[0][FEATURE_COLUMNS].astype(float).tolist()

    vendor_hashes = {
        str(vendor_id): compute_feature_hash(group.reset_index(drop=True), FEATURE_COLUMNS)
        for vendor_id, group in training_df.groupby("vendor_id")
    }

    return {
        "feature_vector_shape": [int(training_df.shape[0]), int(len(FEATURE_COLUMNS))],
        "feature_dtypes": extract_feature_dtypes(training_df, FEATURE_COLUMNS),
        "feature_hash": feature_hash,
        "vendor_hashes": vendor_hashes,
        "feature_columns": FEATURE_COLUMNS,
        "sample_vendor_id": sample_vendor_id,
        "sample_feature_vector": sample_vector,
        "training_timestamp": TRAINING_TIMESTAMP.isoformat(),
    }


def train_model(training_df: pd.DataFrame, labels: pd.DataFrame) -> xgb.XGBClassifier:
    merged = training_df.merge(labels, on="vendor_id", how="inner")
    if merged.empty:
        raise RuntimeError("No labeled training rows available after joining features and labels.")

    features = merged[FEATURE_COLUMNS].astype(np.float32)
    target = merged["high_tip"].astype(int)

    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(features, target)
    return model


def save_artifacts(metadata: dict[str, Any], model: xgb.XGBClassifier) -> None:
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with META_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    model.save_model(str(MODEL_PATH))


def print_summary(metadata: dict[str, Any]) -> None:
    rows = [
        ["feature_vector_shape", metadata["feature_vector_shape"]],
        ["feature_hash", metadata["feature_hash"]],
        ["sample_vendor_id", metadata["sample_vendor_id"]],
        ["sample_feature_vector", metadata["sample_feature_vector"]],
        ["feature_columns", metadata["feature_columns"]],
    ]
    for column, dtype in metadata["feature_dtypes"].items():
        rows.append([f"dtype:{column}", dtype])

    print(tabulate(rows, headers=["field", "value"], tablefmt="github"))


def main() -> None:
    store = get_feature_store()
    processed_features_path = PROJECT_ROOT / "data" / "processed" / "vendor_features.parquet"
    entity_df = build_entity_dataframe(processed_features_path)
    training_df = pull_offline_features(store, entity_df)
    labels = load_vendor_labels(RAW_DATA_PATH, TRAINING_TIMESTAMP)
    metadata = build_training_metadata(training_df)
    model = train_model(training_df, labels)
    save_artifacts(metadata, model)
    print_summary(metadata)


if __name__ == "__main__":
    main()

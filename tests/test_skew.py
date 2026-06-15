from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import xgboost as xgb
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from feast import FeatureStore
from tabulate import tabulate

from feature_store.feast_client import get_feature_store as init_feature_store
from feature_store.feature_hash import (
    compute_feature_hash,
    enforce_float32_features,
    extract_feature_dtypes,
    round_feature_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_REPO_PATH = PROJECT_ROOT / "feature_store" / "feature_repo"
META_PATH = PROJECT_ROOT / "data" / "training_meta.json"
PROCESSED_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "vendor_features.parquet"

FEATURE_REFS: list[str] = [
    "vendor_stats:trip_count_last_7d",
    "vendor_stats:avg_fare_last_7d",
    "vendor_stats:avg_trip_distance_last_7d",
    "vendor_stats:avg_passenger_count_last_7d",
    "vendor_stats:peak_hour_ratio_last_7d",
]


@pytest.fixture(scope="session")
def training_meta() -> dict[str, Any]:
    with META_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="session")
def feature_store() -> FeatureStore:
    return init_feature_store(FEATURE_REPO_PATH)


@pytest.fixture(scope="session")
def vendor_ids() -> list[str]:
    table = pq.read_table(PROCESSED_FEATURES_PATH, columns=["vendor_id"])
    return (
        table.to_pandas()["vendor_id"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )


@pytest.fixture(scope="session")
def offline_features(
    feature_store: FeatureStore,
    training_meta: dict[str, Any],
    vendor_ids: list[str],
) -> pd.DataFrame:
    entity_df = pd.DataFrame(
        {
            "vendor_id": vendor_ids,
            "event_timestamp": pd.Timestamp(training_meta["training_timestamp"]),
        }
    )
    offline_df = feature_store.get_historical_features(
        entity_df=entity_df,
        features=FEATURE_REFS,
    ).to_df()
    feature_columns = training_meta["feature_columns"]
    offline_df = offline_df.dropna(subset=feature_columns).copy()
    return enforce_float32_features(offline_df, feature_columns).sort_values("vendor_id").reset_index(drop=True)


@pytest.fixture(scope="session")
def online_features(
    feature_store: FeatureStore,
    training_meta: dict[str, Any],
    vendor_ids: list[str],
) -> pd.DataFrame:
    entity_rows = [{"vendor_id": vendor_id} for vendor_id in vendor_ids]
    online_df = feature_store.get_online_features(
        features=FEATURE_REFS,
        entity_rows=entity_rows,
    ).to_df()
    feature_columns = training_meta["feature_columns"]
    online_df = online_df[feature_columns + [col for col in online_df.columns if col not in feature_columns]]
    return enforce_float32_features(online_df, feature_columns).sort_values("vendor_id").reset_index(drop=True)


def print_result_table(test_name: str, passed: bool, details: list[list[Any]]) -> None:
    status = "PASS" if passed else "FAIL"
    rows = [["test", test_name], ["status", status], *details]
    print(tabulate(rows, headers=["metric", "value"], tablefmt="github"))


def test_feature_column_order_matches(
    training_meta: dict[str, Any],
    feature_store: FeatureStore,
    online_features: pd.DataFrame,
) -> None:
    expected_columns = training_meta["feature_columns"]
    sample_vendor_id = training_meta["sample_vendor_id"]
    sample_online = feature_store.get_online_features(
        features=FEATURE_REFS,
        entity_rows=[{"vendor_id": sample_vendor_id}],
    ).to_df()
    actual_columns = [col for col in sample_online.columns if col in set(expected_columns)]

    passed = actual_columns == expected_columns
    print_result_table(
        "test_feature_column_order_matches",
        passed,
        [
            ["expected_columns", expected_columns],
            ["actual_columns", actual_columns],
            ["sample_vendor_id", sample_vendor_id],
        ],
    )
    assert actual_columns == expected_columns


def test_feature_dtypes_match(
    training_meta: dict[str, Any],
    online_features: pd.DataFrame,
) -> None:
    expected_dtypes = training_meta["feature_dtypes"]
    actual_dtypes = extract_feature_dtypes(online_features, training_meta["feature_columns"])
    diff_rows = []
    passed = True

    for column, expected_dtype in expected_dtypes.items():
        actual_dtype = actual_dtypes[column]
        match = actual_dtype == expected_dtype
        passed = passed and match
        diff_rows.append([column, expected_dtype, actual_dtype, "PASS" if match else "FAIL"])

    print(tabulate(diff_rows, headers=["feature", "expected", "actual", "status"], tablefmt="github"))
    print_result_table("test_feature_dtypes_match", passed, [["mismatched_features", len(diff_rows) - sum(row[-1] == "PASS" for row in diff_rows)]])
    assert passed, f"Dtype mismatch detected: expected={expected_dtypes}, actual={actual_dtypes}"


def test_feature_vector_values_match(
    training_meta: dict[str, Any],
    offline_features: pd.DataFrame,
    online_features: pd.DataFrame,
) -> None:
    feature_columns = training_meta["feature_columns"]
    sample_vendor_id = training_meta["sample_vendor_id"]

    offline_row = offline_features[offline_features["vendor_id"] == sample_vendor_id].iloc[0]
    online_row = online_features[online_features["vendor_id"] == sample_vendor_id].iloc[0]

    offline_values = round_feature_matrix(
        offline_row.to_frame().T,
        feature_columns,
    ).iloc[0]
    online_values = round_feature_matrix(
        online_row.to_frame().T,
        feature_columns,
    ).iloc[0]

    diff_rows = []
    passed = True
    for column in feature_columns:
        delta = abs(float(offline_values[column]) - float(online_values[column]))
        match = delta <= 1e-4
        passed = passed and match
        diff_rows.append(
            [
                column,
                float(offline_values[column]),
                float(online_values[column]),
                delta,
                "PASS" if match else "FAIL",
            ]
        )

    print(tabulate(diff_rows, headers=["feature", "offline", "online", "abs_delta", "status"], tablefmt="github"))
    print_result_table(
        "test_feature_vector_values_match",
        passed,
        [["sample_vendor_id", sample_vendor_id], ["tolerance", "1e-4"]],
    )
    assert passed


def test_feature_hash_matches(
    training_meta: dict[str, Any],
    online_features: pd.DataFrame,
) -> None:
    feature_columns = training_meta["feature_columns"]
    expected_hash = training_meta["feature_hash"]
    actual_hash = compute_feature_hash(online_features, feature_columns)
    passed = actual_hash == expected_hash

    print_result_table(
        "test_feature_hash_matches",
        passed,
        [
            ["expected_hash", expected_hash],
            ["actual_hash", actual_hash],
        ],
    )
    assert actual_hash == expected_hash


def test_no_null_features_in_online_store(
    training_meta: dict[str, Any],
    online_features: pd.DataFrame,
) -> None:
    feature_columns = training_meta["feature_columns"]
    null_counts = online_features[feature_columns].isna().sum()
    total_nulls = int(null_counts.sum())
    passed = total_nulls == 0

    rows = [[column, int(null_counts[column])] for column in feature_columns]
    print(tabulate(rows, headers=["feature", "null_count"], tablefmt="github"))
    print_result_table("test_no_null_features_in_online_store", passed, [["total_nulls", total_nulls]])
    assert total_nulls == 0


def test_vector_shape_matches(
    training_meta: dict[str, Any],
    online_features: pd.DataFrame,
) -> None:
    expected_width = len(training_meta["feature_columns"])
    sample_vendor_id = training_meta["sample_vendor_id"]
    sample_online = online_features[online_features["vendor_id"] == sample_vendor_id]
    actual_shape = [1, sample_online[training_meta["feature_columns"]].shape[1]]
    passed = actual_shape[1] == expected_width

    print_result_table(
        "test_vector_shape_matches",
        passed,
        [
            ["expected_width", expected_width],
            ["actual_shape", actual_shape],
            ["sample_vendor_id", sample_vendor_id],
        ],
    )
    assert actual_shape[1] == expected_width


def test_training_contract_valid() -> None:
    assert META_PATH.exists(), f"training_meta.json not found at {META_PATH}"
    with META_PATH.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)

    required_keys = {"feature_columns", "feature_dtypes", "feature_hash", "sample_vendor_id", "training_timestamp"}
    missing = required_keys - meta.keys()
    assert not missing, f"training_meta.json missing keys: {missing}"

    expected_columns = [ref.split(":")[1] for ref in FEATURE_REFS]
    assert meta["feature_columns"] == expected_columns, (
        f"feature_columns mismatch: expected={expected_columns}, actual={meta['feature_columns']}"
    )


def test_model_file_exists_and_loadable() -> None:
    model_path = PROJECT_ROOT / "data" / "model.json"
    assert model_path.exists(), f"model.json not found at {model_path}"
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    assert model.n_features_in_ == len(FEATURE_REFS), (
        f"Model expects {model.n_features_in_} features, but FEATURE_REFS has {len(FEATURE_REFS)}"
    )


def test_predict_endpoint_returns_hashes_match() -> None:
    from serving.api import app

    with TestClient(app) as client:
        for vendor_id in ["1", "2"]:
            response = client.post("/predict", json={"vendor_id": vendor_id})
            assert response.status_code == 200, f"vendor_id={vendor_id}: status {response.status_code}"
            body = response.json()

            assert "online_feature_hash" in body, f"vendor_id={vendor_id}: missing online_feature_hash"
            assert "training_hash" in body, f"vendor_id={vendor_id}: missing training_hash"
            assert body["online_feature_hash"] == body["training_hash"], (
                f"vendor_id={vendor_id}: per-vendor hash mismatch — "
                f"online={body['online_feature_hash']}, training={body['training_hash']}"
            )
            assert body["hashes_match"] is True, (
                f"vendor_id={vendor_id}: hashes_match=False"
            )

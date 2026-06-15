from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from feast import FeatureStore
from feast.repo_config import load_repo_config


def _patch_online_feature_column_order(store: FeatureStore) -> FeatureStore:
    original_get_online_features = store.get_online_features

    def get_online_features(
        features: list[str],
        entity_rows: list[dict[str, Any]],
        full_feature_names: bool = False,
    ) -> Any:
        response = original_get_online_features(
            features=features,
            entity_rows=entity_rows,
            full_feature_names=full_feature_names,
        )
        feature_columns = [
            ref.rsplit(":", maxsplit=1)[-1] for ref in features
        ]
        original_to_df = response.to_df

        def ordered_to_df() -> Any:
            frame = original_to_df()
            leading_columns = [
                column for column in frame.columns if column not in feature_columns
            ]
            return frame[leading_columns + feature_columns]

        response.to_df = ordered_to_df
        return response

    store.get_online_features = get_online_features
    return store


def get_feature_store(repo_path: Path) -> FeatureStore:
    connection_string = os.getenv("FEAST_REDIS_CONNECTION_STRING", "localhost:6379")
    config = load_repo_config(repo_path, repo_path / "feature_store.yaml")
    config.online_store.connection_string = connection_string
    store = FeatureStore(repo_path=str(repo_path), config=config)
    return _patch_online_feature_column_order(store)

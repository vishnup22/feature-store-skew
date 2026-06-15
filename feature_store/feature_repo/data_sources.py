from pathlib import Path

from feast.infra.offline_stores.file_source import FileSource

PROCESSED_FEATURES_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "vendor_features.parquet"
)

vendor_features_source = FileSource(
    name="vendor_features_source",
    path=str(PROCESSED_FEATURES_PATH),
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

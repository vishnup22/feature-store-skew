from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def materialize_features() -> None:
    from feature_store.feast_client import get_feature_store

    repo_path = Path(__file__).resolve().parent / "feature_repo"
    store = get_feature_store(repo_path)
    store.materialize(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 1, 31, 23, 59, 59),
    )
    print("Materialized vendor_stats features to Redis.")


if __name__ == "__main__":
    materialize_features()

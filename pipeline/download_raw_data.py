from __future__ import annotations

import urllib.request
from pathlib import Path

DEFAULT_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"
DEFAULT_PATH = Path("data/raw/yellow_tripdata_2023-01.parquet")
USER_AGENT = "feature-store-skew/1.0 (https://github.com/vishnup22/feature-store-skew)"


def download_raw_data(
    raw_path: Path = DEFAULT_PATH,
    url: str = DEFAULT_URL,
) -> Path:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=300) as response, raw_path.open("wb") as handle:
        handle.write(response.read())
    return raw_path


if __name__ == "__main__":
    output_path = download_raw_data()
    print(f"Downloaded {output_path.stat().st_size} bytes to {output_path}")

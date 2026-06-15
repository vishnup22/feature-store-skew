from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import pandas as pd


def normalize_feature_columns(feature_columns: Iterable[str]) -> list[str]:
    return list(feature_columns)


def round_feature_matrix(
    df: pd.DataFrame,
    feature_columns: list[str],
    decimals: int = 4,
) -> pd.DataFrame:
    rounded = df[feature_columns].copy()
    rounded = rounded.apply(pd.to_numeric, errors="coerce")
    return rounded.round(decimals)


def compute_feature_hash(
    df: pd.DataFrame,
    feature_columns: list[str],
    sort_key: str = "vendor_id",
    decimals: int = 4,
) -> str:
    if sort_key not in df.columns:
        raise ValueError(f"Sort key '{sort_key}' not found in dataframe columns.")

    sorted_df = df.sort_values(sort_key).reset_index(drop=True)
    canonical = (
        sorted_df[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .round(decimals)
    )
    payload = canonical.to_numpy(dtype=np.float64).tobytes()
    return hashlib.sha256(payload).hexdigest()


def extract_feature_dtypes(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, str]:
    return {column: str(df[column].dtype) for column in feature_columns}


def enforce_float32_features(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    casted = df.copy()
    for column in feature_columns:
        casted[column] = casted[column].astype(np.float32)
    return casted

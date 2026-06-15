from __future__ import annotations

import argparse
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, StringType


FEATURE_COLUMNS: list[str] = [
    "trip_count_last_7d",
    "avg_fare_last_7d",
    "avg_trip_distance_last_7d",
    "avg_passenger_count_last_7d",
    "peak_hour_ratio_last_7d",
]


def create_spark_session(app_name: str = "vendor-feature-engineering") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def load_trips(spark: SparkSession, raw_parquet_path: Path) -> DataFrame:
    trips = spark.read.parquet(str(raw_parquet_path))

    is_peak_hour = (
        ((F.hour("tpep_pickup_datetime") >= F.lit(7)) & (F.hour("tpep_pickup_datetime") <= F.lit(9)))
        | ((F.hour("tpep_pickup_datetime") >= F.lit(17)) & (F.hour("tpep_pickup_datetime") <= F.lit(19)))
    )

    return (
        trips.withColumn("vendor_id", F.col("VendorID").cast(StringType()))
        .withColumn("event_timestamp", F.col("tpep_pickup_datetime"))
        .withColumn("trip_date", F.to_date("event_timestamp"))
        .withColumn("is_peak_hour", F.when(is_peak_hour, F.lit(1)).otherwise(F.lit(0)))
        .filter(F.col("vendor_id").isNotNull())
        .filter(F.col("trip_date").isNotNull())
    )


def build_daily_vendor_aggregates(trips: DataFrame) -> DataFrame:
    return trips.groupBy("vendor_id", "trip_date").agg(
        F.count(F.lit(1)).alias("daily_trips"),
        F.sum("fare_amount").alias("daily_fare_sum"),
        F.sum("trip_distance").alias("daily_distance_sum"),
        F.sum("passenger_count").alias("daily_passenger_sum"),
        F.sum("is_peak_hour").alias("daily_peak_trips"),
    )


def add_rolling_and_overall_features(daily: DataFrame) -> DataFrame:
    daily = daily.withColumn(
        "date_epoch",
        F.datediff(F.col("trip_date"), F.lit("1970-01-01")),
    )

    rolling_window = (
        Window.partitionBy("vendor_id")
        .orderBy("date_epoch")
        .rangeBetween(-6, 0)
    )

    rolling_trip_count = F.sum("daily_trips").over(rolling_window)
    rolling_fare_sum = F.sum("daily_fare_sum").over(rolling_window)
    rolling_distance_sum = F.sum("daily_distance_sum").over(rolling_window)
    rolling_passenger_sum = F.sum("daily_passenger_sum").over(rolling_window)
    rolling_peak_trips = F.sum("daily_peak_trips").over(rolling_window)

    features = (
        daily.withColumn("trip_count_last_7d", rolling_trip_count.cast(FloatType()))
        .withColumn(
            "avg_fare_last_7d",
            F.when(rolling_trip_count > 0, rolling_fare_sum / rolling_trip_count)
            .otherwise(F.lit(0.0))
            .cast(FloatType()),
        )
        .withColumn(
            "avg_trip_distance_last_7d",
            F.when(rolling_trip_count > 0, rolling_distance_sum / rolling_trip_count)
            .otherwise(F.lit(0.0))
            .cast(FloatType()),
        )
        .withColumn(
            "avg_passenger_count_last_7d",
            F.when(rolling_trip_count > 0, rolling_passenger_sum / rolling_trip_count)
            .otherwise(F.lit(0.0))
            .cast(FloatType()),
        )
        .withColumn(
            "peak_hour_ratio_last_7d",
            F.when(rolling_trip_count > 0, rolling_peak_trips / rolling_trip_count)
            .otherwise(F.lit(0.0))
            .cast(FloatType()),
        )
        .withColumn("event_timestamp", F.to_timestamp(F.col("trip_date")))
        .withColumn("created", F.current_timestamp())
        .withColumn("date", F.date_format("trip_date", "yyyy-MM-dd"))
    )

    for column in FEATURE_COLUMNS:
        features = features.withColumn(column, F.col(column).cast(FloatType()))

    return features.select(
        "vendor_id",
        "event_timestamp",
        "created",
        "date",
        *FEATURE_COLUMNS,
    )


def write_processed_features(features: DataFrame, output_path: Path) -> None:
    import shutil

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f"_{output_path.name}.tmp"

    if tmp_path.exists():
        shutil.rmtree(tmp_path)

    features.write.mode("overwrite").partitionBy("date").parquet(str(tmp_path))

    if output_path.exists():
        shutil.rmtree(output_path)
    shutil.move(str(tmp_path), str(output_path))


def run_feature_pipeline(raw_parquet_path: Path, output_path: Path) -> None:
    spark = create_spark_session()
    try:
        trips = load_trips(spark, raw_parquet_path)
        daily = build_daily_vendor_aggregates(trips)
        features = add_rolling_and_overall_features(daily)
        write_processed_features(features, output_path)
        row_count = features.count()
        vendor_count = features.select("vendor_id").distinct().count()
        print(f"Wrote {row_count} feature rows for {vendor_count} vendors to {output_path}")
    finally:
        spark.stop()


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build vendor-level taxi features with PySpark.")
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=project_root / "data" / "raw" / "yellow_tripdata_2023-01.parquet",
        help="Path to the raw NYC taxi parquet file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=project_root / "data" / "processed" / "vendor_features.parquet",
        help="Directory for processed vendor feature parquet output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_feature_pipeline(args.raw_path, args.output_path)


if __name__ == "__main__":
    main()

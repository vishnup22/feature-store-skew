from datetime import timedelta

from feast import FeatureView, Field
from feast.types import Float32

from data_sources import vendor_features_source
from entities import vendor

vendor_stats = FeatureView(
    name="vendor_stats",
    entities=[vendor],
    ttl=timedelta(days=7),
    schema=[
        Field(name="trip_count_last_7d", dtype=Float32),
        Field(name="avg_fare_last_7d", dtype=Float32),
        Field(name="avg_trip_distance_last_7d", dtype=Float32),
        Field(name="avg_passenger_count_last_7d", dtype=Float32),
        Field(name="peak_hour_ratio_last_7d", dtype=Float32),
    ],
    source=vendor_features_source,
    online=True,
)

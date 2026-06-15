from feast import Entity
from feast.value_type import ValueType

vendor = Entity(
    name="vendor",
    join_keys=["vendor_id"],
    value_type=ValueType.STRING,
    description="NYC taxi vendor identifier",
)

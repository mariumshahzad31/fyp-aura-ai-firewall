#!/usr/bin/env python3
"""Structured Streaming job: consumes Kafka aura.events -> micro-batch anomaly score -> Kafka aura.threats."""

from __future__ import annotations

import os
import sys

from pyspark.sql import SparkSession  # noqa: PLC0415
from pyspark.sql import functions as F  # noqa: PLC0415
from pyspark.sql.types import (  # noqa: PLC0415
    DoubleType,
    StringType,
    StructField,
    StructType,
)


def main() -> None:
    bootstrap = os.environ.get("AURA_KAFKA_BOOTSTRAP", "localhost:9092")
    topic_in = os.environ.get("AURA_KAFKA_TOPIC_IN", "aura.events")
    topic_out = os.environ.get("AURA_KAFKA_TOPIC_OUT", "aura.threats")
    checkpoint = os.environ.get("AURA_SPARK_CHECKPOINT", "/tmp/aura-stream-checkpoint")

    spark = SparkSession.builder.appName("aura-threat-stream").getOrCreate()
    spark.sparkContext.setLogLevel(os.environ.get("AURA_SPARK_LOG_LEVEL", "WARN"))

    schema = StructType(
        [
            StructField("ts_wall", DoubleType(), True),
            StructField("risk_class", StringType(), True),
            StructField("cve_id", StringType(), True),
            StructField("tenant_id", StringType(), True),
            StructField("source_ip", StringType(), True),
        ]
    )

    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", bootstrap)
        .option("subscribe", topic_in)
        .option("startingOffsets", "latest")
        .load()
        .selectExpr("CAST(value AS STRING) AS json_text")
        .withColumn(
            "data",
            F.from_json(F.col("json_text"), schema),
        )
        .select(F.col("data.*"))
    )

    scored = raw.withColumn(
        "threat_score",
        F.when(F.col("risk_class").isin("Malicious", "Critical"), F.lit(1.0))
        .otherwise(F.when(F.col("risk_class") == "Suspicious", F.lit(0.55)).otherwise(F.lit(0.12))),
    ).filter(F.col("threat_score") >= float(os.environ.get("AURA_GRAPH_EDGE_THRESHOLD", "0.35")))

    def write_batch(batch_df, _epoch_id):  # noqa: ANN001
        kafka_df = batch_df.select(
            F.struct(
                F.col("risk_class"),
                F.col("cve_id"),
                F.col("tenant_id"),
                F.col("source_ip"),
                F.col("threat_score"),
            ).alias("value_struct")
        ).select(F.to_json(F.col("value_struct")).alias("value"))
        (
            kafka_df.write.format("kafka")
            .option("kafka.bootstrap.servers", bootstrap)
            .option("topic", topic_out)
            .save()
        )

    q = scored.writeStream.foreachBatch(write_batch).option("checkpointLocation", checkpoint).outputMode(
        "update"
    ).start()
    q.awaitTermination()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

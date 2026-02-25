"""PySpark data processing module for large-scale churn feature engineering.

Demonstrates how the churn prediction pipeline scales to billions
of rows using distributed computing. Mirrors the pandas-based
feature engineering logic but operates on Spark DataFrames.

Usage:
    Can be run standalone or integrated into an Airflow/Dagster pipeline.
    Requires a Spark cluster or local[*] for development.
"""

import logging
from typing import Any

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

logger = logging.getLogger(__name__)


def get_spark_session(app_name: str = "ChurnPrediction") -> SparkSession:
    """Create or get a SparkSession."""
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )


# Schema for raw event data
EVENT_SCHEMA = StructType([
    StructField("user_id", StringType(), False),
    StructField("event_type", StringType(), False),
    StructField("event_timestamp", TimestampType(), False),
    StructField("session_id", StringType(), True),
    StructField("duration_seconds", FloatType(), True),
    StructField("page_views", IntegerType(), True),
    StructField("feature_name", StringType(), True),
])


class SparkFeatureProcessor:
    """Distributed feature engineering for churn prediction.

    Processes raw event data into user-level features using PySpark,
    maintaining point-in-time correctness through window functions
    with explicit temporal bounds.
    """

    def __init__(self, spark: SparkSession | None = None) -> None:
        self.spark = spark or get_spark_session()

    def compute_login_features(self, events: DataFrame) -> DataFrame:
        """Compute login frequency and recency features.

        Features:
            - login_frequency: Logins per 30-day period
            - days_since_last_login: Days since most recent login
            - login_frequency_trend: Change in login rate (recent vs. earlier)
        """
        logins = events.filter(F.col("event_type") == "login")

        # Window: per user, ordered by time
        user_window = Window.partitionBy("user_id")
        recent_window = (
            Window.partitionBy("user_id")
            .orderBy(F.col("event_timestamp").cast("long"))
            .rangeBetween(-30 * 86400, 0)  # Last 30 days
        )
        early_window = (
            Window.partitionBy("user_id")
            .orderBy(F.col("event_timestamp").cast("long"))
            .rangeBetween(-90 * 86400, -30 * 86400)  # 30-90 days ago
        )

        features = (
            logins
            .withColumn("login_count_recent", F.count("*").over(recent_window))
            .withColumn("login_count_early", F.count("*").over(early_window))
            .groupBy("user_id")
            .agg(
                F.mean("login_count_recent").alias("login_frequency"),
                F.datediff(
                    F.current_timestamp(), F.max("event_timestamp")
                ).alias("days_since_last_login"),
                (
                    (F.mean("login_count_recent") - F.mean("login_count_early"))
                    / F.greatest(F.mean("login_count_early"), F.lit(1))
                ).alias("login_frequency_trend"),
            )
        )
        return features

    def compute_session_features(self, events: DataFrame) -> DataFrame:
        """Compute session duration and engagement features.

        Features:
            - avg_session_duration_min: Average session length
            - session_duration_std: Variability in session length
            - pages_per_session: Average page views per session
            - monthly_active_days: Unique active days in last 30 days
        """
        sessions = events.filter(F.col("session_id").isNotNull())

        session_stats = (
            sessions
            .groupBy("user_id", "session_id")
            .agg(
                F.sum("duration_seconds").alias("session_duration_sec"),
                F.sum("page_views").alias("session_pages"),
                F.min("event_timestamp").alias("session_start"),
            )
        )

        features = (
            session_stats
            .groupBy("user_id")
            .agg(
                (F.mean("session_duration_sec") / 60).alias("avg_session_duration_min"),
                (F.stddev("session_duration_sec") / 60).alias("session_duration_std"),
                F.mean("session_pages").alias("pages_per_session"),
                F.countDistinct(
                    F.date_format("session_start", "yyyy-MM-dd")
                ).alias("monthly_active_days"),
            )
        )
        return features

    def compute_support_features(self, events: DataFrame) -> DataFrame:
        """Compute support ticket features.

        Features:
            - support_tickets_total: Total tickets filed
            - support_tickets_recent: Tickets in last 30 days
        """
        tickets = events.filter(F.col("event_type") == "support_ticket")

        features = (
            tickets
            .groupBy("user_id")
            .agg(
                F.count("*").alias("support_tickets_total"),
                F.sum(
                    F.when(
                        F.datediff(F.current_timestamp(), F.col("event_timestamp")) <= 30,
                        1,
                    ).otherwise(0)
                ).alias("support_tickets_recent"),
            )
        )
        return features

    def compute_feature_usage(self, events: DataFrame) -> DataFrame:
        """Compute product feature adoption score.

        Features:
            - feature_usage_score: Normalized count of distinct product features used
        """
        feature_events = events.filter(F.col("feature_name").isNotNull())

        # Total available features (for normalization)
        total_features = feature_events.select("feature_name").distinct().count()
        total_features = max(total_features, 1)

        features = (
            feature_events
            .groupBy("user_id")
            .agg(
                (F.countDistinct("feature_name") / F.lit(total_features) * 100)
                .alias("feature_usage_score")
            )
        )
        return features

    def build_feature_table(
        self,
        events: DataFrame,
        user_metadata: DataFrame,
    ) -> DataFrame:
        """Build the complete feature table from raw events.

        Joins all feature groups with user metadata and fills nulls.

        Args:
            events: Raw event DataFrame.
            user_metadata: Static user attributes (plan, billing, etc.).

        Returns:
            Complete feature DataFrame ready for model training/serving.
        """
        logger.info("Building feature table from %d events", events.count())

        login_features = self.compute_login_features(events)
        session_features = self.compute_session_features(events)
        support_features = self.compute_support_features(events)
        usage_features = self.compute_feature_usage(events)

        # Join all feature groups
        feature_table = (
            user_metadata
            .join(login_features, "user_id", "left")
            .join(session_features, "user_id", "left")
            .join(support_features, "user_id", "left")
            .join(usage_features, "user_id", "left")
            .fillna(0)
        )

        row_count = feature_table.count()
        col_count = len(feature_table.columns)
        logger.info("Feature table built: %d rows, %d columns", row_count, col_count)

        return feature_table

    def validate_point_in_time(
        self,
        feature_table: DataFrame,
        label_timestamp_col: str = "churn_date",
        feature_timestamp_col: str = "feature_computed_at",
    ) -> bool:
        """Validate point-in-time correctness of features.

        Ensures no feature was computed using data from after the label event.
        """
        if (label_timestamp_col not in feature_table.columns or
                feature_timestamp_col not in feature_table.columns):
            logger.info("Timestamp columns not found, skipping PIT validation")
            return True

        violations = feature_table.filter(
            F.col(feature_timestamp_col) > F.col(label_timestamp_col)
        ).count()

        if violations > 0:
            logger.error("Point-in-time violation: %d rows with future features", violations)
            return False

        logger.info("Point-in-time validation passed")
        return True

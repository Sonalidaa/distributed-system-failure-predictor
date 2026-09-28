import pandas as pd
import numpy as np


INPUT_PATH = "data/telemetry.csv"
OUTPUT_PATH = "data/ml_dataset.csv"


FEATURE_COLUMNS = [
    "cpu_usage",
    "memory_usage",
    "request_rate",
    "latency_ms",
    "error_rate",
    "db_latency_ms",
]


def add_time_features(df):
    """
    Create rolling statistics and trends
    using the previous 5 observations.
    """

    df = df.sort_values(
        ["service", "timestamp"]
    ).copy()

    grouped = df.groupby("service")

    for column in FEATURE_COLUMNS:

        # Average value over previous 5 observations
        df[f"{column}_mean_5"] = (
            grouped[column]
            .transform(
                lambda x: x.shift(1).rolling(5).mean()
            )
        )

        # Maximum value over previous 5 observations
        df[f"{column}_max_5"] = (
            grouped[column]
            .transform(
                lambda x: x.shift(1).rolling(5).max()
            )
        )

        # Change from 5 observations ago
        df[f"{column}_change_5"] = (
            grouped[column]
            .transform(
                lambda x: x.shift(1) - x.shift(6)
            )
        )

    return df


def create_future_failure_target(df):
    """
    Label a row as 1 if the service experiences
    a failure within the next 5 observations.
    """

    df = df.sort_values(
        ["service", "timestamp"]
    ).copy()

    grouped = df.groupby("service")

    future_failure = pd.Series(
        0,
        index=df.index,
        dtype=int
    )

    for shift in range(1, 6):

        future_failure = np.maximum(
            future_failure,
            grouped["failure"]
            .shift(-shift)
            .fillna(0)
            .astype(int)
        )

    df["failure_in_next_5"] = future_failure

    return df


def main():

    print("Loading telemetry data...")

    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["timestamp"]
    )

    print(f"Original rows: {len(df)}")

    # Sort chronologically
    df = df.sort_values(
        ["service", "timestamp"]
    ).copy()

    # Create time-based features
    df = add_time_features(df)

    # Create prediction target
    df = create_future_failure_target(df)

    # Remove rows where we don't have
    # enough historical observations
    df = df.dropna().copy()

    # Encode service as numeric category
    df["service_encoded"] = (
        df["service"].astype("category").cat.codes
    )

    # Save
    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"ML dataset created: {len(df)} rows"
    )

    print(
        f"Saved to: {OUTPUT_PATH}"
    )

    print("\nTarget distribution:")

    print(
        df["failure_in_next_5"]
        .value_counts()
    )

    print("\nColumns:")

    print(
        df.columns.tolist()
    )


if __name__ == "__main__":
    main()
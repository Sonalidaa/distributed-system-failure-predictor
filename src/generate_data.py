import numpy as np
import pandas as pd


np.random.seed(42)


SERVICES = [
    "api_gateway",
    "order_service",
    "payment_service",
    "database"
]


def generate_normal_metrics():
    """Generate metrics for a healthy service."""

    cpu = np.random.normal(45, 8)
    memory = np.random.normal(55, 7)
    request_rate = np.random.normal(300, 50)
    latency = np.random.normal(60, 10)
    error_rate = np.random.normal(0.01, 0.005)
    db_latency = np.random.normal(30, 5)

    return {
        "cpu_usage": np.clip(cpu, 0, 100),
        "memory_usage": np.clip(memory, 0, 100),
        "request_rate": max(0, request_rate),
        "latency_ms": max(1, latency),
        "error_rate": np.clip(error_rate, 0, 1),
        "db_latency_ms": max(1, db_latency),
    }


def generate_failure_sequence(failure_type, length=10):
    """
    Generate a gradual and noisy degradation sequence
    before a system failure.
    """

    sequence = []

    for step in range(length):

        progress = step / (length - 1)

        metrics = generate_normal_metrics()

        # Gradual degradation with realistic noise

        if failure_type == "cpu_overload":

            metrics["cpu_usage"] = (
                45
                + progress * 25
                + np.random.normal(0, 6)
            )

            metrics["latency_ms"] += (
                progress * 120
                + np.random.normal(0, 15)
            )

            metrics["error_rate"] += (
                progress * 0.06
                + np.random.normal(0, 0.01)
            )

        elif failure_type == "memory_leak":

            metrics["memory_usage"] = (
                52
                + progress * 28
                + np.random.normal(0, 5)
            )

            metrics["latency_ms"] += (
                progress * 100
                + np.random.normal(0, 15)
            )

            metrics["error_rate"] += (
                progress * 0.05
                + np.random.normal(0, 0.01)
            )

        elif failure_type == "database_slowdown":

            metrics["db_latency_ms"] = (
                35
                + progress * 180
                + np.random.normal(0, 15)
            )

            metrics["latency_ms"] += (
                progress * 150
                + np.random.normal(0, 20)
            )

            metrics["error_rate"] += (
                progress * 0.07
                + np.random.normal(0, 0.01)
            )

        elif failure_type == "network_degradation":

            metrics["latency_ms"] = (
                60
                + progress * 250
                + np.random.normal(0, 25)
            )

            metrics["error_rate"] += (
                progress * 0.08
                + np.random.normal(0, 0.015)
            )

            metrics["request_rate"] *= (
                1 - progress * 0.25
            )

        # Keep values realistic

        metrics["cpu_usage"] = np.clip(
            metrics["cpu_usage"],
            0,
            100
        )

        metrics["memory_usage"] = np.clip(
            metrics["memory_usage"],
            0,
            100
        )

        metrics["request_rate"] = max(
            0,
            metrics["request_rate"]
        )

        metrics["latency_ms"] = max(
            1,
            metrics["latency_ms"]
        )

        metrics["db_latency_ms"] = max(
            1,
            metrics["db_latency_ms"]
        )

        metrics["error_rate"] = np.clip(
            metrics["error_rate"],
            0,
            1
        )

        sequence.append(metrics)

    return sequence

def generate_dataset(
    num_normal=8000,
    num_failures=2000
):

    data = []

    start_time = pd.Timestamp("2026-01-01")

    # ---------------------------------------
    # 1. Generate normal system behavior
    # ---------------------------------------

    for i in range(num_normal):

        timestamp = start_time + pd.Timedelta(minutes=i)

        service = np.random.choice(SERVICES)

        metrics = generate_normal_metrics()

        row = {
            "timestamp": timestamp,
            "service": service,
            **metrics,
            "failure_type": "normal",
            "failure": 0
        }

        data.append(row)

    # ---------------------------------------
    # 2. Generate failure scenarios
    # ---------------------------------------

    failure_types = [
        "cpu_overload",
        "memory_leak",
        "database_slowdown",
        "network_degradation"
    ]

    current_time = (
        start_time + pd.Timedelta(minutes=num_normal)
    )

    for _ in range(num_failures // 10):

        failure_type = np.random.choice(failure_types)

        service = np.random.choice(SERVICES)

        sequence = generate_failure_sequence(
            failure_type,
            length=10
        )

        for step, metrics in enumerate(sequence):

            timestamp = (
                current_time +
                pd.Timedelta(minutes=step)
            )

            # Last step represents the actual failure
            failure = 1 if step >= 8 else 0

            row = {
                "timestamp": timestamp,
                "service": service,
                **metrics,
                "failure_type": failure_type,
                "failure": failure
            }

            data.append(row)

        current_time += pd.Timedelta(minutes=10)

    df = pd.DataFrame(data)

    # Shuffle the data
    df = df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    return df


if __name__ == "__main__":

    df = generate_dataset()

    output_path = "data/telemetry.csv"

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Generated {len(df)} telemetry records."
    )

    print(
        f"Saved to {output_path}"
    )

    print("\nFailure distribution:")

    print(
        df["failure_type"].value_counts()
    )

    print("\nSample data:")

    print(
        df.head()
    )
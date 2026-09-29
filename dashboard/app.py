import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import requests
import os


# ----------------------------------
# Configuration
# ----------------------------------

st.set_page_config(
    page_title="Distributed System Failure Predictor",
    page_icon="⚡",
    layout="wide"
)


DATA_PATH = "data/telemetry.csv"
MODEL_PATH = "models/xgboost_failure_predictor.json"


FEATURE_COLUMNS = [
    "cpu_usage_mean_5",
    "cpu_usage_max_5",
    "cpu_usage_change_5",

    "memory_usage_mean_5",
    "memory_usage_max_5",
    "memory_usage_change_5",

    "request_rate_mean_5",
    "request_rate_max_5",
    "request_rate_change_5",

    "latency_ms_mean_5",
    "latency_ms_max_5",
    "latency_ms_change_5",

    "error_rate_mean_5",
    "error_rate_max_5",
    "error_rate_change_5",

    "db_latency_ms_mean_5",
    "db_latency_ms_max_5",
    "db_latency_ms_change_5",

    "service_encoded",
]


# ----------------------------------
# Load data
# ----------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["timestamp"]
    )

    return df


# ----------------------------------
# Feature engineering
# ----------------------------------

def prepare_features(df):

    df = df.sort_values(
        ["service", "timestamp"]
    ).copy()

    grouped = df.groupby("service")

    base_features = [
        "cpu_usage",
        "memory_usage",
        "request_rate",
        "latency_ms",
        "error_rate",
        "db_latency_ms",
    ]

    for column in base_features:

        df[f"{column}_mean_5"] = (
            grouped[column]
            .transform(
                lambda x: x.shift(1).rolling(5).mean()
            )
        )

        df[f"{column}_max_5"] = (
            grouped[column]
            .transform(
                lambda x: x.shift(1).rolling(5).max()
            )
        )

        df[f"{column}_change_5"] = (
            grouped[column]
            .transform(
                lambda x: x.shift(1) - x.shift(6)
            )
        )

    df["service_encoded"] = (
        df["service"]
        .astype("category")
        .cat.codes
    )

    return df.dropna()


# ----------------------------------
# Load model
# ----------------------------------

@st.cache_resource
def load_model():

    model = xgb.XGBClassifier()

    model.load_model(
        MODEL_PATH
    )

    return model


# ----------------------------------
# Dashboard
# ----------------------------------

st.title(
    "⚡ Distributed System Failure Predictor"
)

st.markdown(
    "ML-powered early warning system for distributed services."
)


df = load_data()



ml_data = prepare_features(df)


# ----------------------------------
# Sidebar
# ----------------------------------

st.sidebar.header("System Controls")

services = sorted(
    ml_data["service"].unique()
)

selected_service = st.sidebar.selectbox(
    "Select service",
    services
)


service_data = ml_data[
    ml_data["service"] == selected_service
].copy()


# ----------------------------------
# Latest observation
# ----------------------------------

latest = service_data.iloc[-1]

X_latest = latest[
    FEATURE_COLUMNS
].to_frame().T.astype(float)

# ----------------------------------
# FastAPI prediction
# ----------------------------------


API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000/predict"
)

response = requests.post(
    API_URL,
    json=X_latest.iloc[0].to_dict(),
    timeout=5
)

response.raise_for_status()

prediction_result = response.json()

probability = prediction_result["failure_probability"]
prediction = prediction_result["prediction"].upper()
top_factors = prediction_result["top_factors"]


# ----------------------------------
# Metrics
# ----------------------------------

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Failure Risk",
        f"{probability * 100:.1f}%"
    )


with col2:

    st.metric(
        "Status",
        prediction
    )


with col3:

    st.metric(
        "CPU",
        f"{latest['cpu_usage']:.1f}%"
    )


with col4:

    st.metric(
        "Latency",
        f"{latest['latency_ms']:.1f} ms"
    )


# ----------------------------------
# Telemetry charts
# ----------------------------------

st.subheader(
    f"{selected_service} Telemetry"
)

chart_data = service_data.tail(100)

st.line_chart(
    chart_data.set_index("timestamp")[
        [
            "cpu_usage",
            "memory_usage",
            "latency_ms",
        ]
    ]
)


st.subheader("Error Rate")

st.line_chart(
    chart_data.set_index("timestamp")[
        ["error_rate"]
    ]
)


# ----------------------------------
# SHAP explanation
# ----------------------------------



# explainer = shap.TreeExplainer(model)

# shap_values = explainer(
#     X_latest
# )

# importance = pd.DataFrame({
#     "Feature": X_latest.columns,
#     "Impact": np.abs(
#         shap_values.values[0]
#     )
# })

# importance = importance.sort_values(
#     "Impact",
#     ascending=False
# ).head(8)

# st.bar_chart(
#     importance.set_index("Feature")
# )

# ---------------------------------------------------------
# Human-readable SHAP explanation
# ---------------------------------------------------------

st.subheader("Why is the model predicting this?")

shap_data = pd.DataFrame(top_factors)

# Convert feature names into human-readable names
feature_names = {
    "latency_ms_max_5": "Maximum latency",
    "latency_ms_mean_5": "Average latency",
    "latency_ms_change_5": "Latency change",

    "error_rate_max_5": "Maximum error rate",
    "error_rate_mean_5": "Average error rate",
    "error_rate_change_5": "Error rate change",

    "memory_usage_max_5": "Maximum memory usage",
    "memory_usage_mean_5": "Average memory usage",
    "memory_usage_change_5": "Memory usage change",

    "cpu_usage_max_5": "Maximum CPU usage",
    "cpu_usage_mean_5": "Average CPU usage",
    "cpu_usage_change_5": "CPU usage change",

    "request_rate_max_5": "Maximum request rate",
    "request_rate_mean_5": "Average request rate",
    "request_rate_change_5": "Request rate change",

    "db_latency_ms_max_5": "Maximum DB latency",
    "db_latency_ms_mean_5": "Average DB latency",
    "db_latency_ms_change_5": "DB latency change",

    "service_encoded": "Service type"
}


def format_feature_value(feature, value):
    """Convert model feature values into human-readable values."""

    if "error_rate" in feature:
        return f"{value * 100:.2f}%"

    if "latency_ms" in feature:
        return f"{value:.1f} ms"

    if "memory_usage" in feature or "cpu_usage" in feature:
        return f"{value:.1f}%"

    if "request_rate" in feature:
        return f"{value:.1f} req/s"

    if "db_latency" in feature:
        return f"{value:.1f} ms"

    return f"{value:.2f}"


# Sort by absolute SHAP impact
shap_data["abs_impact"] = shap_data["impact"].abs()
shap_data = shap_data.sort_values(
    "abs_impact",
    ascending=False
)

max_impact = shap_data["abs_impact"].max()


# ---------------------------------------------------------
# Human-readable explanations
# ---------------------------------------------------------

for _, row in shap_data.iterrows():

    feature = row["feature"]
    impact = row["impact"]

    readable_name = feature_names.get(feature, feature)

    # Get the exact feature value used by the model
    feature_value = float(X_latest.iloc[0][feature])

    formatted_value = format_feature_value(
        feature,
        feature_value
    )

    # Determine contribution direction
    if impact > 0:
        direction = "Increases failure risk"
        icon = "🔴"
    else:
        direction = "Decreases failure risk"
        icon = "🟢"

    # Relative importance
    relative_impact = abs(impact) / max_impact

    if relative_impact >= 0.7:
        strength = "High impact"
    elif relative_impact >= 0.3:
        strength = "Moderate impact"
    else:
        strength = "Low impact"

    with st.container():

        st.markdown(
            f"<div style='font-size:18px; font-weight:600; margin-bottom:8px;'>"
            f"{icon} {readable_name}"
            f"</div>",
            unsafe_allow_html=True
)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Observed value",
                formatted_value
            )

        with col2:
            st.metric(
                "Model impact",
                f"{impact:+.2f}"
            )

        with col3:
            st.write("**Effect**")
            st.write(direction)

        st.markdown(
            f"<span style='font-size:13px; color:#888;'>{strength}</span>",
            unsafe_allow_html=True
)

        st.markdown(
            "<hr style='margin:8px 0 12px 0;'>",
            unsafe_allow_html=True
)


# ---------------------------------------------------------
# Technical SHAP section
# ---------------------------------------------------------

with st.expander("Technical model explanation"):

    technical_data = shap_data.copy()

    technical_data["Feature"] = technical_data["feature"].map(
        lambda x: feature_names.get(x, x)
    )

    technical_data["Direction"] = technical_data["impact"].apply(
        lambda x: "↑ Failure risk" if x > 0
        else "↓ Failure risk"
    )

    technical_data["SHAP impact"] = technical_data["impact"].round(3)

    st.dataframe(
        technical_data[
            ["Feature", "SHAP impact", "Direction"]
        ],
        hide_index=True,
        use_container_width=True
    )
# ----------------------------------
# Model Evaluation
# ----------------------------------

st.subheader("Model Evaluation")

st.markdown(
    "Performance of the failure prediction model "
    "on the chronological test set."
)

col1, col2 = st.columns(2)

with col1:

    st.image(
        "models/plots/confusion_matrix.png",
        caption="XGBoost Confusion Matrix",
        use_container_width=True
    )

with col2:

    st.image(
        "models/plots/precision_recall_curve.png",
        caption="Precision-Recall Curve",
        use_container_width=True
    )

st.image(
    "models/plots/model_comparison.png",
    caption="Model Performance Comparison",
    use_container_width=True
)
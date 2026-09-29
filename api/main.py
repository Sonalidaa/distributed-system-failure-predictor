from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import shap

app = FastAPI(
    title="Distributed System Failure Predictor",
    description="ML inference API for predicting upcoming service failures.",
    version="1.0.0"
)


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


model = xgb.XGBClassifier()

model.load_model(
    MODEL_PATH
)

explainer = shap.TreeExplainer(model)

class TelemetryInput(BaseModel):

    cpu_usage_mean_5: float
    cpu_usage_max_5: float
    cpu_usage_change_5: float

    memory_usage_mean_5: float
    memory_usage_max_5: float
    memory_usage_change_5: float

    request_rate_mean_5: float
    request_rate_max_5: float
    request_rate_change_5: float

    latency_ms_mean_5: float
    latency_ms_max_5: float
    latency_ms_change_5: float

    error_rate_mean_5: float
    error_rate_max_5: float
    error_rate_change_5: float

    db_latency_ms_mean_5: float
    db_latency_ms_max_5: float
    db_latency_ms_change_5: float

    service_encoded: float


@app.get("/")
def root():

    return {
        "service": "Distributed System Failure Predictor",
        "status": "running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "xgboost"
    }


@app.post("/predict")
def predict(
    telemetry: TelemetryInput
):

    data = pd.DataFrame(
        [telemetry.model_dump()]
    )

    data = data[
        FEATURE_COLUMNS
    ]

    # -----------------------------
    # XGBoost prediction
    # -----------------------------

    probability = model.predict_proba(
        data
    )[0][1]

    prediction = (
        "failure"
        if probability >= 0.5
        else "healthy"
    )

    # -----------------------------
    # SHAP explanation
    # -----------------------------

    shap_values = explainer(
        data
    )

    contributions = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "impact": shap_values.values[0]
    })

    contributions["abs_impact"] = (
        contributions["impact"].abs()
    )

    contributions = contributions.sort_values(
        "abs_impact",
        ascending=False
    ).head(5)

    explanations = []

    for _, row in contributions.iterrows():

        direction = (
            "increased"
            if row["impact"] > 0
            else "decreased"
        )

        explanations.append({
            "feature": row["feature"],
            "impact": round(
                float(row["impact"]),
                4
            ),
            "direction": direction
        })

    return {
        "prediction": prediction,
        "failure_probability": round(
            float(probability),
            4
        ),
        "top_factors": explanations
    }

    data = pd.DataFrame(
        [telemetry.model_dump()]
    )

    data = data[
        FEATURE_COLUMNS
    ]

    probability = model.predict_proba(
        data
    )[0][1]

    prediction = (
        "failure"
        if probability >= 0.5
        else "healthy"
    )

    return {
        "prediction": prediction,
        "failure_probability": round(
            float(probability),
            4
        )
    }
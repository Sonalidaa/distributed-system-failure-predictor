# Distributed System Failure Predictor

ML-powered early warning system for predicting upcoming failures in distributed services from telemetry data.

The system combines temporal feature engineering, XGBoost classification, SHAP explainability, FastAPI inference, Streamlit visualization, and Docker Compose to provide an end-to-end failure prediction pipeline.

## 🚀 Features

- Synthetic distributed-service telemetry generation
- Temporal feature engineering using rolling statistics and metric changes
- Forward-looking failure prediction
- Logistic Regression, Random Forest, and XGBoost comparison
- Chronological train/test evaluation
- SHAP-based model explainability
- Human-readable failure explanations
- FastAPI inference service
- Interactive Streamlit monitoring dashboard
- Dockerized multi-container architecture
- Model evaluation visualizations

## 🏗️ Architecture

```text
                  Telemetry Simulator
                         │
                         ▼
                Feature Engineering
                         │
                         ▼
              ┌─────────────────────┐
              │    ML Models        │
              │                     │
              │ Logistic Regression │
              │ Random Forest       │
              │ XGBoost             │
              └──────────┬──────────┘
                         │
                         ▼
                 Failure Prediction
                         │
                         ▼
                  SHAP Explainability
                         │
                         ▼
                  FastAPI Inference
                         │
                         ▼
                 Streamlit Dashboard
                         │
                         ▼
                    Docker Compose
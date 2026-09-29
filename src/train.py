import pandas as pd
import shap
import os
import matplotlib.pyplot as plt



from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)

from xgboost import XGBClassifier


DATA_PATH = "data/ml_dataset.csv"


def load_data():

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["timestamp"]
    )

    return df


def prepare_features(df):

    feature_columns = [
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

    X = df[feature_columns]

    y = df["failure_in_next_5"]

    return X, y


def evaluate_model(name, model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    print(f"\n{name}")
    print("-" * 40)

    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")

    return {
        "model": name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
    }


def main():

    print("Loading dataset...")

    df = load_data()

    X, y = prepare_features(df)

    print(f"Features: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")

    # ----------------------------------
    # Train / test split
    # ----------------------------------

    # ----------------------------------
# Chronological train/test split
# ----------------------------------

    # ----------------------------------
    # Chronological train/test split
    # ----------------------------------

    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print("\nChronological split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    # ----------------------------------
    # Scale data for Logistic Regression
    # ----------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ----------------------------------
    # 1. Logistic Regression
    # ----------------------------------

    logistic_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    logistic_model.fit(
        X_train_scaled,
        y_train
    )

    results = []

    results.append(
        evaluate_model(
            "Logistic Regression",
            logistic_model,
            X_test_scaled,
            y_test
        )
    )

   

    # ----------------------------------
    # 2. Random Forest
    # ----------------------------------

    random_forest = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    random_forest.fit(
        X_train,
        y_train
    )

    results.append(
        evaluate_model(
            "Random Forest",
            random_forest,
            X_test,
            y_test
        )
    )

    # ----------------------------------
    # 3. XGBoost
    # ----------------------------------

    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    xgb_model.fit(
        X_train,
        y_train
    )
        # Save trained XGBoost model
    os.makedirs("models", exist_ok=True)

    xgb_model.save_model(
        "models/xgboost_failure_predictor.json"
    )

    print(
        "\nXGBoost model saved to "
        "models/xgboost_failure_predictor.json"
    )

    results.append(
        evaluate_model(
            "XGBoost",
            xgb_model,
            X_test,
            y_test
        )
    )

        # ----------------------------------
    # SHAP Explainability
    # ----------------------------------

    print("\nGenerating SHAP explanations...")

    # Use XGBoost's underlying Booster
    booster = xgb_model.get_booster()

    explainer = shap.TreeExplainer(
        booster
    )

    shap_values = explainer.shap_values(
        X_test
    )

    feature_importance = pd.DataFrame({
        "feature": X_test.columns,
        "importance": abs(
            shap_values
        ).mean(axis=0)
    })

    feature_importance = feature_importance.sort_values(
        "importance",
        ascending=False
    )

    print("\nTop features influencing failure prediction:")

    print(
        feature_importance.head(10).to_string(
            index=False
        )
    )

    # ----------------------------------
    # Compare models
    # ----------------------------------

    results_df = pd.DataFrame(results)

        # ----------------------------------
    # Evaluation Visualizations
    # ----------------------------------

    os.makedirs("models/plots", exist_ok=True)

    # ----------------------------------
    # Confusion Matrix
    # ----------------------------------

    xgb_predictions = xgb_model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        xgb_predictions
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Healthy", "Failure"]
    ).plot(
        ax=ax
    )

    ax.set_title(
        "XGBoost Confusion Matrix"
    )

    fig.tight_layout()

    fig.savefig(
        "models/plots/confusion_matrix.png",
        dpi=150
    )

    plt.close(fig)


    # ----------------------------------
    # Precision-Recall Curve
    # ----------------------------------

    xgb_probabilities = (
        xgb_model.predict_proba(X_test)[:, 1]
    )

    precision, recall, _ = (
        precision_recall_curve(
            y_test,
            xgb_probabilities
        )
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.plot(
        recall,
        precision,
        label="XGBoost"
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    ax.set_title(
        "XGBoost Precision-Recall Curve"
    )

    ax.legend()

    ax.grid(True)

    fig.tight_layout()

    fig.savefig(
        "models/plots/precision_recall_curve.png",
        dpi=150
    )

    plt.close(fig)


    # ----------------------------------
    # Model Comparison
    # ----------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    results_df.set_index("model")[
        ["precision", "recall", "f1"]
    ].plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylim(0, 1)

    ax.set_ylabel("Score")

    ax.set_title(
        "Model Performance Comparison"
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        "models/plots/model_comparison.png",
        dpi=150
    )

    plt.close(fig)

    print(
        "\nEvaluation plots saved to:"
        " models/plots/"
    )

    print("\n\nMODEL COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
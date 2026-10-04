import os
import sys

import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# Project root
# --------------------------------------------------

ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


# --------------------------------------------------
# Paths
# --------------------------------------------------

TRAIN_PATH = os.path.join(
    ROOT_DIR,
    "data",
    "processed",
    "ml_train.csv"
)

TEST_PATH = os.path.join(
    ROOT_DIR,
    "data",
    "processed",
    "ml_test.csv"
)

MODEL_DIR = os.path.join(
    ROOT_DIR,
    "models"
)

OUTPUT_DIR = os.path.join(
    ROOT_DIR,
    "data",
    "processed"
)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 70)
    print("UDYAMPULSE AI - PHASE 2.7")
    print("ANOMALY DETECTION")
    print("=" * 70)

    # --------------------------------------------------
    # Load datasets
    # --------------------------------------------------

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    test_df = pd.read_csv(
        TEST_PATH
    )

    print("\nTraining records:", len(train_df))
    print("Testing records:", len(test_df))

    # --------------------------------------------------
    # Features
    #
    # IMPORTANT:
    # No future or risk-target columns.
    # --------------------------------------------------

    anomaly_features = [
        "employee_count",
        "revenue",
        "expenses",
        "cash_balance",
        "accounts_receivable",
        "accounts_payable",
        "inventory_value",
        "loan_emi",
        "loan_outstanding",
        "profit",
        "profit_margin",
        "working_capital",
        "cash_ratio",
        "receivable_ratio",
        "payable_ratio",
        "debt_service_ratio",
        "inventory_ratio",
        "revenue_growth",
        "expense_growth",
        "cash_growth",
        "receivable_growth"
    ]

    # --------------------------------------------------
    # Check features
    # --------------------------------------------------

    missing_features = [
        feature
        for feature in anomaly_features
        if feature not in train_df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing anomaly features:\n"
            + "\n".join(missing_features)
        )

    print(
        "\nNumber of anomaly features:",
        len(anomaly_features)
    )

    # --------------------------------------------------
    # Prepare data
    # --------------------------------------------------

    X_train = train_df[
        anomaly_features
    ].copy()

    X_test = test_df[
        anomaly_features
    ].copy()

    # --------------------------------------------------
    # Handle missing/infinite values
    # --------------------------------------------------

    X_train = X_train.replace(
        [float("inf"), float("-inf")],
        pd.NA
    )

    X_test = X_test.replace(
        [float("inf"), float("-inf")],
        pd.NA
    )

    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # --------------------------------------------------
    # Isolation Forest
    #
    # contamination=0.05 means we assume roughly
    # 5% of observations may be unusual.
    # --------------------------------------------------

    anomaly_model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "isolation_forest",
            IsolationForest(
                n_estimators=200,
                contamination=0.05,
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    print(
        "\nTraining Isolation Forest..."
    )

    anomaly_model.fit(
        X_train
    )

    print(
        "Anomaly detection model trained."
    )

    # --------------------------------------------------
    # Predict test anomalies
    #
    # Isolation Forest:
    #  1  = normal
    # -1  = anomaly
    # --------------------------------------------------

    predictions = anomaly_model.predict(
        X_test
    )

    anomaly_scores = anomaly_model.decision_function(
        X_test
    )

    # --------------------------------------------------
    # Add results
    # --------------------------------------------------

    results = test_df[
        [
            "business_id",
            "month",
            "risk_target"
        ]
    ].copy()

    results["anomaly_prediction"] = predictions

    results["anomaly_score"] = anomaly_scores

    results["anomaly_status"] = results[
        "anomaly_prediction"
    ].map({
        1: "NORMAL",
        -1: "ANOMALY"
    })

    # --------------------------------------------------
    # Anomaly distribution
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("ANOMALY DISTRIBUTION")
    print("=" * 70)

    counts = results[
        "anomaly_status"
    ].value_counts()

    print(counts)

    # --------------------------------------------------
    # Percentage
    # --------------------------------------------------

    percentages = (
        results[
            "anomaly_status"
        ]
        .value_counts(normalize=True)
        * 100
    )

    print("\nPercentage:")

    for status, percentage in percentages.items():

        print(
            f"{status}: "
            f"{percentage:.2f}%"
        )

    # --------------------------------------------------
    # Show detected anomalies
    # --------------------------------------------------

    anomalies = results[
        results["anomaly_status"] == "ANOMALY"
    ].copy()

    print("\n" + "=" * 70)
    print("DETECTED ANOMALIES")
    print("=" * 70)

    print(
        f"Number of anomalies: {len(anomalies)}"
    )

    if len(anomalies) > 0:

        print(
            anomalies[
                [
                    "business_id",
                    "month",
                    "anomaly_score",
                    "anomaly_status"
                ]
            ]
            .sort_values(
                "anomaly_score"
            )
            .head(20)
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    results_path = os.path.join(
        OUTPUT_DIR,
        "anomaly_detection_results.csv"
    )

    results.to_csv(
        results_path,
        index=False
    )

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    model_path = os.path.join(
        MODEL_DIR,
        "anomaly_detection_model.pkl"
    )

    joblib.dump(
        anomaly_model,
        model_path
    )

    # --------------------------------------------------
    # Final output
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print(
        f"Results saved to:\n"
        f"{results_path}"
    )

    print(
        f"\nModel saved to:\n"
        f"{model_path}"
    )

    print("\n" + "=" * 70)
    print("PHASE 2.7 ANOMALY DETECTION COMPLETE")
    print("=" * 70)


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":
    main()
import os
import sys

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


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


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 70)
    print("UDYAMPULSE AI - PHASE 2.3")
    print("FINANCIAL RISK ML MODEL")
    print("=" * 70)

    # --------------------------------------------------
    # Load datasets
    # --------------------------------------------------

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print("\nTraining records:", len(train_df))
    print("Testing records:", len(test_df))

    # --------------------------------------------------
    # Identify features
    # --------------------------------------------------

    excluded_columns = [
        "business_id",
        "month",
        "risk_target"
    ]

    feature_columns = [
        column
        for column in train_df.columns
        if column not in excluded_columns
    ]

    X_train = train_df[feature_columns]
    y_train = train_df["risk_target"]

    X_test = test_df[feature_columns]
    y_test = test_df["risk_target"]

    print("\nNumber of features:", len(feature_columns))

    # --------------------------------------------------
    # Train Random Forest
    # --------------------------------------------------

    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(
        X_train,
        y_train
    )

    print("Model training completed.")

    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------

    y_pred = model.predict(X_test)

    # --------------------------------------------------
    # Accuracy
    # --------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE")
    print("=" * 70)

    print(
        f"\nAccuracy: {accuracy:.4f}"
    )

    # --------------------------------------------------
    # Classification report
    # --------------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "LOW",
                "MEDIUM",
                "HIGH"
            ],
            zero_division=0
        )
    )

    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------

    matrix = confusion_matrix(
        y_test,
        y_pred
    )

    print("Confusion Matrix:")
    print(matrix)

    # --------------------------------------------------
    # Feature importance
    # --------------------------------------------------

    importance = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    print("\n" + "=" * 70)
    print("TOP FEATURE IMPORTANCE")
    print("=" * 70)

    print(
        importance.head(10)
        .to_string(index=False)
    )

    # --------------------------------------------------
    # Save feature importance
    # --------------------------------------------------

    output_path = os.path.join(
        ROOT_DIR,
        "data",
        "processed",
        "feature_importance.csv"
    )

    importance.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nFeature importance saved to:\n"
        f"{output_path}"
    )

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

    import joblib

    model_directory = os.path.join(
        ROOT_DIR,
        "models"
    )

    os.makedirs(
        model_directory,
        exist_ok=True
    )

    model_path = os.path.join(
        model_directory,
        "financial_risk_model.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        f"Model saved to:\n"
        f"{model_path}"
    )

    print("\n" + "=" * 70)
    print("PHASE 2.3 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
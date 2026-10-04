import os
import sys

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score
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
    print("UDYAMPULSE AI - PHASE 2.4")
    print("MODEL EVALUATION & IMPROVEMENT")
    print("=" * 70)

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    train_df["month"] = pd.to_datetime(
        train_df["month"]
    )

    test_df["month"] = pd.to_datetime(
        test_df["month"]
    )

    # --------------------------------------------------
    # Features
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

    # --------------------------------------------------
    # Chronological validation split
    #
    # Last 20% of training period becomes validation.
    # The original test set remains untouched.
    # --------------------------------------------------

    unique_train_months = sorted(
        train_df["month"].unique()
    )

    validation_start = int(
        len(unique_train_months) * 0.80
    )

    model_train_months = unique_train_months[
        :validation_start
    ]

    validation_months = unique_train_months[
        validation_start:
    ]

    model_train = train_df[
        train_df["month"].isin(
            model_train_months
        )
    ].copy()

    validation = train_df[
        train_df["month"].isin(
            validation_months
        )
    ].copy()

    X_train = model_train[feature_columns]
    y_train = model_train["risk_target"]

    X_validation = validation[feature_columns]
    y_validation = validation["risk_target"]

    X_test = test_df[feature_columns]
    y_test = test_df["risk_target"]

    print("\nData split:")
    print(
        f"Model training records: {len(model_train)}"
    )
    print(
        f"Validation records: {len(validation)}"
    )
    print(
        f"Final test records: {len(test_df)}"
    )

    print(
        f"\nModel training period:"
        f" {model_train_months[0]}"
        f" -> {model_train_months[-1]}"
    )

    print(
        f"Validation period:"
        f" {validation_months[0]}"
        f" -> {validation_months[-1]}"
    )

    print(
        f"Final test period:"
        f" {test_df['month'].min()}"
        f" -> {test_df['month'].max()}"
    )

    # --------------------------------------------------
    # Candidate models
    # --------------------------------------------------

    candidates = {

        "Baseline Balanced": {
            "n_estimators": 200,
            "max_depth": 10,
            "class_weight": "balanced"
        },

        "High Risk Emphasis": {
            "n_estimators": 250,
            "max_depth": 12,
            "class_weight": {
                0: 1.0,
                1: 1.2,
                2: 2.5
            }
        },

        "Stronger High Risk": {
            "n_estimators": 250,
            "max_depth": 14,
            "class_weight": {
                0: 1.0,
                1: 1.5,
                2: 3.0
            }
        }
    }

    results = []

    print("\n" + "=" * 70)
    print("VALIDATION MODEL COMPARISON")
    print("=" * 70)

    # --------------------------------------------------
    # Train candidates
    # --------------------------------------------------

    for name, parameters in candidates.items():

        print(
            f"\nTesting: {name}"
        )

        model = RandomForestClassifier(
            n_estimators=parameters[
                "n_estimators"
            ],
            max_depth=parameters[
                "max_depth"
            ],
            class_weight=parameters[
                "class_weight"
            ],
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            X_train,
            y_train
        )

        validation_pred = model.predict(
            X_validation
        )

        accuracy = accuracy_score(
            y_validation,
            validation_pred
        )

        macro_f1 = f1_score(
            y_validation,
            validation_pred,
            average="macro"
        )

        high_recall = recall_score(
            y_validation,
            validation_pred,
            labels=[2],
            average=None,
            zero_division=0
        )[0]

        results.append({
            "model": name,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "high_risk_recall": high_recall
        })

        print(
            f"Accuracy: {accuracy:.4f}"
        )

        print(
            f"Macro F1: {macro_f1:.4f}"
        )

        print(
            f"HIGH-risk recall: {high_recall:.4f}"
        )

    # --------------------------------------------------
    # Results table
    # --------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # Select model
    #
    # Primary objective:
    # HIGH-risk recall
    #
    # Secondary objective:
    # Macro F1
    # --------------------------------------------------

    results_df = results_df.sort_values(
        [
            "high_risk_recall",
            "macro_f1"
        ],
        ascending=False
    )

    best_name = results_df.iloc[0][
        "model"
    ]

    print(
        f"\nSelected model: {best_name}"
    )

    # --------------------------------------------------
    # Get selected parameters
    # --------------------------------------------------

    best_parameters = candidates[
        best_name
    ]

    # --------------------------------------------------
    # Retrain selected model using
    # ALL original training data
    # --------------------------------------------------

    print(
        "\nRetraining selected model "
        "using all training data..."
    )

    final_model = RandomForestClassifier(
        n_estimators=best_parameters[
            "n_estimators"
        ],
        max_depth=best_parameters[
            "max_depth"
        ],
        class_weight=best_parameters[
            "class_weight"
        ],
        random_state=42,
        n_jobs=-1
    )

    final_model.fit(
        train_df[feature_columns],
        train_df["risk_target"]
    )

    # --------------------------------------------------
    # FINAL TEST EVALUATION
    # --------------------------------------------------

    test_pred = final_model.predict(
        X_test
    )

    test_accuracy = accuracy_score(
        y_test,
        test_pred
    )

    test_macro_f1 = f1_score(
        y_test,
        test_pred,
        average="macro"
    )

    test_high_recall = recall_score(
        y_test,
        test_pred,
        labels=[2],
        average=None,
        zero_division=0
    )[0]

    print("\n" + "=" * 70)
    print("FINAL TEST PERFORMANCE")
    print("=" * 70)

    print(
        f"\nAccuracy: {test_accuracy:.4f}"
    )

    print(
        f"Macro F1: {test_macro_f1:.4f}"
    )

    print(
        f"HIGH-risk recall: {test_high_recall:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            test_pred,
            target_names=[
                "LOW",
                "MEDIUM",
                "HIGH"
            ],
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            test_pred
        )
    )

    # --------------------------------------------------
    # Save final model
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
        final_model,
        model_path
    )

    # --------------------------------------------------
    # Save validation results
    # --------------------------------------------------

    results_path = os.path.join(
        ROOT_DIR,
        "data",
        "processed",
        "model_comparison.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print(
        f"Final model:\n{model_path}"
    )

    print(
        f"\nModel comparison:\n{results_path}"
    )

    print("\n" + "=" * 70)
    print("PHASE 2.4 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
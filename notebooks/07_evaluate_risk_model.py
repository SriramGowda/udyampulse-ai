"""UdyamPulse AI | Phase 07: select by validation, then evaluate once on test.

Inputs: chronological data/processed/ml_train.csv and ml_test.csv
Outputs: final Random Forest artifact and validation comparison CSV
Method: purged chronological validation, high-risk recall then Macro F1 selection.
"""

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

TARGET_HORIZON_MONTHS = 3
FEATURE_COLUMNS = [
    "employee_count", "revenue", "expenses", "cash_balance",
    "accounts_receivable", "accounts_payable", "inventory_value",
    "loan_emi", "loan_outstanding", "profit", "profit_margin",
    "working_capital", "cash_ratio", "receivable_ratio",
    "payable_ratio", "debt_service_ratio", "inventory_ratio",
    "revenue_growth", "expense_growth", "cash_growth",
    "receivable_growth",
]


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 70)
    print("UDYAMPULSE AI - PHASE 07")
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

    feature_columns = FEATURE_COLUMNS
    forbidden_features = [
        feature for feature in feature_columns
        if any(term in feature.lower() for term in (
            "future", "stress", "risk_label", "risk_target"
        ))
    ]
    if forbidden_features:
        raise ValueError(f"Future/target leakage in model features: {forbidden_features}")
    missing_columns = sorted(
        set(feature_columns + ["risk_target"]).difference(train_df.columns)
        | set(feature_columns + ["risk_target"]).difference(test_df.columns)
    )
    if missing_columns:
        raise ValueError(f"Train/test data is missing required columns: {missing_columns}")

    # --------------------------------------------------
    # Chronological validation split
    #
    # Last 20% of training period becomes validation.
    # The original test set remains untouched.
    # --------------------------------------------------

    unique_train_months = sorted(
        train_df["month"].unique()
    )

    if len(unique_train_months) < 2 * TARGET_HORIZON_MONTHS + 4:
        raise ValueError(
            "Not enough training months for a chronological validation split "
            "with a target-horizon embargo."
        )

    validation_start = int(
        len(unique_train_months) * 0.80
    )

    if validation_start <= 0 or validation_start >= len(unique_train_months):
        raise ValueError("Unable to create non-empty chronological validation periods.")

    validation_months = unique_train_months[
        validation_start:
    ]
    validation_start_month = pd.Timestamp(validation_months[0])
    model_train_months = [
        month for month in unique_train_months[:validation_start]
        if pd.Timestamp(month)
        < validation_start_month - pd.DateOffset(months=TARGET_HORIZON_MONTHS)
    ]
    purged_months = [
        month for month in unique_train_months[:validation_start]
        if month not in model_train_months
    ]
    if not model_train_months or not validation_months:
        raise ValueError("Validation split has an empty partition after target-horizon embargo.")

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
    if model_train["month"].max() + pd.DateOffset(
        months=TARGET_HORIZON_MONTHS
    ) >= validation["month"].min():
        raise ValueError("Training target windows overlap validation features.")
    if train_df["month"].max() + pd.DateOffset(
        months=TARGET_HORIZON_MONTHS
    ) >= test_df["month"].min():
        raise ValueError("Training target windows overlap final test features.")

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
        f"Purged validation-boundary months: {len(purged_months)} "
        f"(target horizon: {TARGET_HORIZON_MONTHS} months)"
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
        "Model selection used chronological validation only; these final-test "
        "results are reported once and were not used to choose the model."
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
            test_pred,
            labels=[0, 1, 2],
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

    importance = pd.DataFrame({
        "feature": feature_columns,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance_path = os.path.join(
        ROOT_DIR,
        "data",
        "processed",
        "feature_importance.csv"
    )
    importance.to_csv(importance_path, index=False)

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
    print(
        f"\nFinal-model feature importance:\n{importance_path}"
    )

    print("\n" + "=" * 70)
    print("PHASE 07 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
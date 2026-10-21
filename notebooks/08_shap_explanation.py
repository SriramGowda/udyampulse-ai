"""UdyamPulse AI | Phase 08: explain one risk prediction with Tree SHAP.

Inputs: production Random Forest and held-out ml_test.csv
Outputs: per-record SHAP CSV and contribution chart
Method: explain the predicted class in the production feature order.
"""

import os
import sys

import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt


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

MODEL_PATH = os.path.join(
    ROOT_DIR,
    "models",
    "financial_risk_model.pkl"
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
    print("UDYAMPULSE AI - PHASE 08")
    print("SHAP EXPLAINABLE AI TEST")
    print("=" * 70)

    # --------------------------------------------------
    # Load model and test data
    # --------------------------------------------------

    model = joblib.load(
        MODEL_PATH
    )

    test_df = pd.read_csv(
        TEST_PATH
    )

    feature_columns = [
        "employee_count", "revenue", "expenses", "cash_balance",
        "accounts_receivable", "accounts_payable", "inventory_value",
        "loan_emi", "loan_outstanding", "profit", "profit_margin",
        "working_capital", "cash_ratio", "receivable_ratio",
        "payable_ratio", "debt_service_ratio", "inventory_ratio",
        "revenue_growth", "expense_growth", "cash_growth",
        "receivable_growth",
    ]
    if list(getattr(model, "feature_names_in_", feature_columns)) != feature_columns:
        raise ValueError("Loaded model feature order does not match the production feature list.")
    missing_features = sorted(set(feature_columns).difference(test_df.columns))
    if missing_features:
        raise ValueError(f"Test data is missing model features: {missing_features}")

    X_test = test_df[
        feature_columns
    ]

    # --------------------------------------------------
    # Select one business record
    # --------------------------------------------------

    sample = X_test.iloc[[0]]

    actual_risk = test_df[
        "risk_target"
    ].iloc[0]

    # --------------------------------------------------
    # Model prediction
    # --------------------------------------------------

    prediction = model.predict(
        sample
    )[0]

    probabilities = model.predict_proba(
        sample
    )[0]

    risk_labels = {
        0: "LOW",
        1: "MEDIUM",
        2: "HIGH"
    }
    class_ids = list(model.classes_)
    prediction_index = class_ids.index(prediction)

    predicted_label = risk_labels[
        prediction
    ]

    actual_label = risk_labels[
        actual_risk
    ]

    print("\nActual Risk:")
    print(actual_label)

    print("\nPredicted Risk:")
    print(predicted_label)

    print("\nPrediction Probabilities:")

    for class_id, probability in zip(class_ids, probabilities):
        print(
            f"{risk_labels[class_id]}: "
            f"{probability * 100:.2f}%"
        )

    # --------------------------------------------------
    # Create SHAP explainer
    # --------------------------------------------------

    print("\nCreating SHAP explainer...")

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer(sample)

    print("SHAP explanation generated.")

    # --------------------------------------------------
    # Handle SHAP output
    # --------------------------------------------------

    values = getattr(shap_values, "values", shap_values)
    if isinstance(values, list):
        selected_values = values[prediction_index][0]
    else:
        if getattr(values, "ndim", 0) == 3:
            selected_values = values[0, :, prediction_index]
        else:
            selected_values = values[0]

    # --------------------------------------------------
    # Feature contribution table
    # --------------------------------------------------

    explanation_df = pd.DataFrame({
        "feature": feature_columns,
        "shap_value": selected_values
    })

    explanation_df[
        "absolute_importance"
    ] = explanation_df[
        "shap_value"
    ].abs()

    explanation_df = explanation_df.sort_values(
        "absolute_importance",
        ascending=False
    )

    print("\n" + "=" * 70)
    print("TOP SHAP FEATURES")
    print("=" * 70)

    print(
        explanation_df[
            [
                "feature",
                "shap_value"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    # --------------------------------------------------
    # Save explanation CSV
    # --------------------------------------------------

    output_path = os.path.join(
        ROOT_DIR,
        "data",
        "processed",
        "shap_explanation.csv"
    )

    explanation_df.to_csv(
        output_path,
        index=False
    )

    print("\nSHAP explanation saved to:")
    print(output_path)

    # --------------------------------------------------
    # Create SHAP bar plot
    # --------------------------------------------------

    plot_path = os.path.join(
        ROOT_DIR,
        "data",
        "processed",
        "shap_explanation.png"
    )

    plot_df = explanation_df.head(10).copy()

    plot_df = plot_df.sort_values(
        "shap_value"
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.barh(
        plot_df["feature"],
        plot_df["shap_value"]
    )

    plt.xlabel(
        "SHAP Contribution"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        f"Top SHAP Features - "
        f"Predicted Risk: {predicted_label}"
    )

    plt.tight_layout()

    plt.savefig(
        plot_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("\nSHAP plot saved to:")
    print(plot_path)

    # --------------------------------------------------
    # Final status
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 08 SHAP EXPLANATION COMPLETE")
    print("=" * 70)


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":
    main()
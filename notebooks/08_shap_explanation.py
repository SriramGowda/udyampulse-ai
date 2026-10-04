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
    print("UDYAMPULSE AI - PHASE 2.6")
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

    excluded_columns = [
        "business_id",
        "month",
        "risk_target"
    ]

    feature_columns = [
        column
        for column in test_df.columns
        if column not in excluded_columns
    ]

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

    for index, probability in enumerate(
        probabilities
    ):
        print(
            f"{risk_labels[index]}: "
            f"{probability * 100:.2f}%"
        )

    # --------------------------------------------------
    # Create SHAP explainer
    # --------------------------------------------------

    print("\nCreating SHAP explainer...")

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer(
        sample
    )

    print("SHAP explanation generated.")

    # --------------------------------------------------
    # Handle SHAP output
    # --------------------------------------------------

    values = shap_values.values

    if values.ndim == 3:

        selected_values = values[
            0,
            :,
            prediction
        ]

    else:

        selected_values = values[
            0
        ]

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
    print("PHASE 2.6 SHAP TEST COMPLETE")
    print("=" * 70)


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":
    main()
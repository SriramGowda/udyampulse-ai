import os
import sys

import pandas as pd


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

INPUT_PATH = os.path.join(
    ROOT_DIR,
    "data",
    "processed",
    "financial_risk_dataset.csv"
)

OUTPUT_DIR = os.path.join(
    ROOT_DIR,
    "data",
    "processed"
)

TRAIN_PATH = os.path.join(
    OUTPUT_DIR,
    "ml_train.csv"
)

TEST_PATH = os.path.join(
    OUTPUT_DIR,
    "ml_test.csv"
)


# --------------------------------------------------
# Feature engineering
# --------------------------------------------------

def prepare_features(df):
    """
    Create current-month financial features.

    Only information available at the current month
    is used as a model feature.
    """

    df = df.copy()

    # Make sure data is chronological
    df["month"] = pd.to_datetime(df["month"])

    df = df.sort_values(
        ["business_id", "month"]
    ).reset_index(drop=True)

    # --------------------------------------------------
    # Basic financial features
    # --------------------------------------------------

    df["profit"] = (
        df["revenue"]
        - df["expenses"]
    )

    df["profit_margin"] = (
        df["profit"]
        / df["revenue"]
        * 100
    )

    df["working_capital"] = (
        df["accounts_receivable"]
        + df["inventory_value"]
        - df["accounts_payable"]
    )

    df["cash_ratio"] = (
        df["cash_balance"]
        / df["expenses"]
    )

    df["receivable_ratio"] = (
        df["accounts_receivable"]
        / df["revenue"]
    )

    df["payable_ratio"] = (
        df["accounts_payable"]
        / df["expenses"]
    )

    df["debt_service_ratio"] = (
        df["loan_emi"]
        / df["revenue"]
    )

    df["inventory_ratio"] = (
        df["inventory_value"]
        / df["revenue"]
    )

    # --------------------------------------------------
    # Growth features
    # --------------------------------------------------

    grouped = df.groupby("business_id")

    df["revenue_growth"] = (
        grouped["revenue"]
        .pct_change()
        * 100
    )

    df["expense_growth"] = (
        grouped["expenses"]
        .pct_change()
        * 100
    )

    df["cash_growth"] = (
        grouped["cash_balance"]
        .pct_change()
        * 100
    )

    df["receivable_growth"] = (
        grouped["accounts_receivable"]
        .pct_change()
        * 100
    )

    # --------------------------------------------------
    # Replace infinite values
    # --------------------------------------------------

    df = df.replace(
        [float("inf"), float("-inf")],
        pd.NA
    )

    # Remove rows where current features
    # cannot be calculated.
    df = df.dropna().reset_index(drop=True)

    return df


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 70)
    print("UDYAMPULSE AI - PHASE 2.2")
    print("ML TRAINING DATASET PREPARATION")
    print("=" * 70)

    # --------------------------------------------------
    # Load risk dataset
    # --------------------------------------------------

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Risk dataset not found:\n{INPUT_PATH}\n\n"
            "Run Phase 2.1 first."
        )

    df = pd.read_csv(INPUT_PATH)

    print("\nLoaded dataset:")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # --------------------------------------------------
    # Prepare current financial features
    # --------------------------------------------------

    df = prepare_features(df)

    print("\nAfter feature preparation:")
    print(f"Rows: {len(df)}")

    # --------------------------------------------------
    # Define model features
    # --------------------------------------------------

    feature_columns = [
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

    target_column = "risk_target"

    # --------------------------------------------------
    # Check features exist
    # --------------------------------------------------

    missing_features = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing feature columns:\n"
            + "\n".join(missing_features)
        )

    if target_column not in df.columns:
        raise ValueError(
            "risk_target column not found."
        )

    # --------------------------------------------------
    # Leakage check
    # --------------------------------------------------

    forbidden_terms = [
        "future",
        "stress",
        "risk_label",
        "risk_target"
    ]

    leakage_features = [
        column
        for column in feature_columns
        if any(
            term in column.lower()
            for term in forbidden_terms
        )
    ]

    print("\n" + "=" * 70)
    print("LEAKAGE CHECK")
    print("=" * 70)

    if leakage_features:
        raise ValueError(
            "Potential leakage detected:\n"
            + "\n".join(leakage_features)
        )

    print("No future/target columns found in features.")
    print("STATUS: PASS")

    # --------------------------------------------------
    # Create model dataset
    # --------------------------------------------------

    model_df = df[
        [
            "business_id",
            "month"
        ]
        + feature_columns
        + [target_column]
    ].copy()

    # --------------------------------------------------
    # Time-based split
    # --------------------------------------------------

    unique_months = sorted(
        model_df["month"].unique()
    )

    split_index = int(
        len(unique_months) * 0.80
    )

    train_months = unique_months[
        :split_index
    ]

    test_months = unique_months[
        split_index:
    ]

    train_df = model_df[
        model_df["month"].isin(train_months)
    ].copy()

    test_df = model_df[
        model_df["month"].isin(test_months)
    ].copy()

    # --------------------------------------------------
    # Display split information
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("TIME-BASED TRAIN / TEST SPLIT")
    print("=" * 70)

    print(
        f"Total months: {len(unique_months)}"
    )

    print(
        f"Training months: {len(train_months)}"
    )

    print(
        f"Testing months: {len(test_months)}"
    )

    print(
        f"\nTraining period:"
        f" {train_months[0]} → {train_months[-1]}"
    )

    print(
        f"Testing period:"
        f" {test_months[0]} → {test_months[-1]}"
    )

    print(
        f"\nTraining records: {len(train_df)}"
    )

    print(
        f"Testing records: {len(test_df)}"
    )

    # --------------------------------------------------
    # Check chronological separation
    # --------------------------------------------------

    latest_train_month = train_df["month"].max()
    earliest_test_month = test_df["month"].min()

    print("\nChronological leakage check:")

    if latest_train_month < earliest_test_month:
        print("PASS - Test data occurs after training data.")
    else:
        raise ValueError(
            "FAIL - Training and test periods overlap."
        )

    # --------------------------------------------------
    # Class distribution
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING CLASS DISTRIBUTION")
    print("=" * 70)

    print(
        train_df["risk_target"]
        .value_counts()
        .sort_index()
    )

    print("\n" + "=" * 70)
    print("TEST CLASS DISTRIBUTION")
    print("=" * 70)

    print(
        test_df["risk_target"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------
    # Save datasets
    # --------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    train_df.to_csv(
        TRAIN_PATH,
        index=False
    )

    test_df.to_csv(
        TEST_PATH,
        index=False
    )

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print(
        f"Training dataset:\n{TRAIN_PATH}"
    )

    print(
        f"\nTesting dataset:\n{TEST_PATH}"
    )

    print("\nFeature count:")
    print(len(feature_columns))

    print("\nFeatures:")

    for feature in feature_columns:
        print(f"  - {feature}")

    print("\n" + "=" * 70)
    print("PHASE 2.2 DATA PREPARATION COMPLETE")
    print("=" * 70)


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":
    main()
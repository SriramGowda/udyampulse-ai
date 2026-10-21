"""UdyamPulse AI | Phase 04: define a deterministic future-stress target.

Inputs: raw financial history via services.data_service
Outputs: financial_risk_dataset.csv with three-month target labels
Method: current month t features -> stress in t+1..t+3 -> LOW/MEDIUM/HIGH.
"""

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


from services.data_service import (
    load_data,
    create_risk_target
)


# --------------------------------------------------
# Phase 2.1
# Financial Risk Target Design
# --------------------------------------------------

# Current-month features -> next 3 months of stress -> deterministic risk label.
def main():

    print("=" * 70)
    print("UDYAMPULSE AI - PHASE 04")
    print("FINANCIAL RISK TARGET DESIGN")
    print("=" * 70)

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    df = load_data()

    print("\nOriginal dataset:")
    print(f"Rows: {len(df)}")
    print(f"Businesses: {df['business_id'].nunique()}")
    if df.duplicated(["business_id", "month"]).any():
        raise ValueError("Duplicate business-month rows prevent a valid future horizon.")
    month_ordinals = df["month"].dt.to_period("M").astype("int64")
    month_steps = (
        df.assign(_month_ordinal=month_ordinals)
        .sort_values(["business_id", "_month_ordinal"])
        .groupby("business_id")["_month_ordinal"]
        .diff()
        .dropna()
    )
    if not month_steps.eq(1).all():
        raise ValueError(
            "Risk target requires one observation per consecutive calendar month "
            "for every business."
        )

    # --------------------------------------------------
    # Create future-risk target
    # --------------------------------------------------

    target_df = create_risk_target(
        df,
        future_months=3
    )
    if target_df["risk_target"].isna().any():
        raise ValueError("Risk target creation produced unlabeled records.")
    print(
        "Method: the mean financial indicators in months t+1, t+2, and t+3 "
        "form the future stress target; month t is reserved for model features."
    )
    print(
        "Stress rules: profit margin < 10%; cash change < -15%; revenue change "
        "< -10%; receivables/revenue > 0.30; EMI/revenue > 0.12. "
        "0 indicators = LOW, 1 = MEDIUM, 2+ = HIGH."
    )

    print("\nAfter target creation:")
    print(f"Rows available for ML: {len(target_df)}")

    # --------------------------------------------------
    # Risk distribution
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("1. RISK LABEL DISTRIBUTION")
    print("=" * 70)

    distribution = (
        target_df["risk_label"]
        .value_counts()
        .reindex(
            ["LOW", "MEDIUM", "HIGH"],
            fill_value=0
        )
    )

    print(distribution)

    # --------------------------------------------------
    # Percentage distribution
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("2. CLASS PERCENTAGES")
    print("=" * 70)

    percentages = (
        distribution
        / distribution.sum()
        * 100
    )

    print(
        percentages.round(2)
    )

    # --------------------------------------------------
    # Stress score distribution
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("3. FUTURE STRESS SCORE")
    print("=" * 70)

    print(
        target_df[
            "future_stress_score"
        ]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------
    # Stress indicators
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("4. STRESS INDICATORS")
    print("=" * 70)

    stress_columns = [
        "stress_low_profit",
        "stress_cash_decline",
        "stress_revenue_decline",
        "stress_high_receivables",
        "stress_high_debt"
    ]

    for column in stress_columns:

        count = target_df[column].sum()

        percentage = (
            count
            / len(target_df)
            * 100
        )

        print(
            f"{column}: "
            f"{count} records "
            f"({percentage:.2f}%)"
        )

    # --------------------------------------------------
    # Check class balance
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("5. CLASS BALANCE CHECK")
    print("=" * 70)

    highest = percentages.max()
    lowest = percentages.min()

    print(
        f"Largest class: {highest:.2f}%"
    )

    print(
        f"Smallest class: {lowest:.2f}%"
    )

    if lowest >= 5:
        print(
            "\nSTATUS: Classes are acceptable "
            "for initial ML experimentation."
        )
    else:
        print(
            "\nWARNING: Smallest class is below 5%."
        )

    # --------------------------------------------------
    # Sample records
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("6. SAMPLE TARGET RECORDS")
    print("=" * 70)

    display_columns = [
        "business_id",
        "month",
        "revenue",
        "expenses",
        "cash_balance",
        "future_profit_margin",
        "future_cash_change_pct",
        "future_revenue_change_pct",
        "future_stress_score",
        "risk_label",
        "risk_target"
    ]

    print(
        target_df[
            display_columns
        ]
        .head(10)
        .to_string(index=False)
    )

    # --------------------------------------------------
    # Save ML-ready dataset
    # --------------------------------------------------

    output_directory = os.path.join(
        ROOT_DIR,
        "data",
        "processed"
    )

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    output_path = os.path.join(
        output_directory,
        "financial_risk_dataset.csv"
    )

    target_df.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("7. OUTPUT")
    print("=" * 70)

    print(
        f"ML-ready dataset saved to:\n"
        f"{output_path}"
    )

    print("\nPhase 04 completed successfully.")


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":
    main()
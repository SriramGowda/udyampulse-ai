import os
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


# --------------------------------------------------
# Paths
# --------------------------------------------------

RESULTS_PATH = os.path.join(
    ROOT_DIR,
    "data",
    "processed",
    "anomaly_detection_results.csv"
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
    print("UDYAMPULSE AI - PHASE 2.7")
    print("ANOMALY VALIDATION")
    print("=" * 70)

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    anomalies = pd.read_csv(
        RESULTS_PATH
    )

    test_df = pd.read_csv(
        TEST_PATH
    )

    print("\nTotal test records:")
    print(len(anomalies))

    # --------------------------------------------------
    # Overall distribution
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("1. OVERALL ANOMALY DISTRIBUTION")
    print("=" * 70)

    print(
        anomalies[
            "anomaly_status"
        ].value_counts()
    )

    # --------------------------------------------------
    # Business-wise anomaly count
    # --------------------------------------------------

    anomaly_only = anomalies[
        anomalies["anomaly_status"] == "ANOMALY"
    ].copy()

    business_counts = (
        anomaly_only
        .groupby("business_id")
        .size()
        .sort_values(
            ascending=False
        )
    )

    print("\n" + "=" * 70)
    print("2. ANOMALIES BY BUSINESS")
    print("=" * 70)

    print(
        business_counts
        .head(20)
        .to_string()
    )

    # --------------------------------------------------
    # Number of businesses affected
    # --------------------------------------------------

    total_businesses = (
        test_df["business_id"]
        .nunique()
    )

    affected_businesses = (
        anomaly_only["business_id"]
        .nunique()
    )

    print("\nTotal businesses in test set:")
    print(total_businesses)

    print(
        "\nBusinesses with at least one anomaly:"
    )

    print(affected_businesses)

    print(
        f"\nPercentage of businesses affected: "
        f"{affected_businesses / total_businesses * 100:.2f}%"
    )

    # --------------------------------------------------
    # Persistent anomalies
    # --------------------------------------------------

    persistent = business_counts[
        business_counts >= 3
    ]

    print("\n" + "=" * 70)
    print("3. BUSINESSES WITH 3+ ANOMALOUS MONTHS")
    print("=" * 70)

    if len(persistent) > 0:

        print(
            persistent.to_string()
        )

    else:

        print(
            "No business has 3 or more anomalous months."
        )

    # --------------------------------------------------
    # Single-month anomalies
    # --------------------------------------------------

    single_month = business_counts[
        business_counts == 1
    ]

    print("\nBusinesses with exactly one anomaly:")
    print(len(single_month))

    # --------------------------------------------------
    # Merge with financial data
    # --------------------------------------------------

    financial_columns = [
        "business_id",
        "month",
        "revenue",
        "expenses",
        "cash_balance",
        "accounts_receivable",
        "accounts_payable",
        "loan_outstanding",
        "profit",
        "profit_margin",
        "cash_ratio",
        "receivable_ratio",
        "debt_service_ratio"
    ]

    merged = anomaly_only.merge(
        test_df[
            financial_columns
        ],
        on=[
            "business_id",
            "month"
        ],
        how="left"
    )

    # --------------------------------------------------
    # Compare anomalies vs normal records
    # --------------------------------------------------

    normal = anomalies[
        anomalies["anomaly_status"] == "NORMAL"
    ].merge(
        test_df[
            financial_columns
        ],
        on=[
            "business_id",
            "month"
        ],
        how="left"
    )

    comparison_columns = [
        "revenue",
        "expenses",
        "cash_balance",
        "accounts_receivable",
        "loan_outstanding",
        "profit",
        "profit_margin",
        "cash_ratio",
        "receivable_ratio",
        "debt_service_ratio"
    ]

    print("\n" + "=" * 70)
    print("4. ANOMALY VS NORMAL FINANCIAL PROFILE")
    print("=" * 70)

    comparison = pd.DataFrame({
        "ANOMALY": merged[
            comparison_columns
        ].mean(),
        "NORMAL": normal[
            comparison_columns
        ].mean()
    })

    print(
        comparison.round(2)
    )

    # --------------------------------------------------
    # Top strongest anomalies
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("5. STRONGEST ANOMALIES")
    print("=" * 70)

    strongest = merged.sort_values(
        "anomaly_score"
    ).head(15)

    print(
        strongest[
            [
                "business_id",
                "month",
                "anomaly_score",
                "revenue",
                "expenses",
                "cash_balance",
                "profit_margin",
                "cash_ratio",
                "debt_service_ratio"
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # Save validation report
    # --------------------------------------------------

    output_path = os.path.join(
        ROOT_DIR,
        "data",
        "processed",
        "anomaly_validation.csv"
    )

    comparison.to_csv(
        output_path
    )

    print("\n" + "=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print(
        f"Validation summary saved to:\n"
        f"{output_path}"
    )

    print("\n" + "=" * 70)
    print("ANOMALY VALIDATION COMPLETE")
    print("=" * 70)


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":
    main()
    
"""UdyamPulse AI | Phase 03: inspect and smoke-test production KPIs.

Inputs: data/raw/msme_financial_data.csv via services.data_service
Outputs: KPI diagnostics printed to the console
Method: invoke the production KPI implementation and verify required outputs.
"""

import sys
import os

import numpy as np

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)
from services.data_service import (
    load_data,
    calculate_kpis,
    get_latest_business_data
)


print("=" * 70)
print("UDYAMPULSE AI - KPI ENGINE TEST")
print("=" * 70)


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

df = load_data()

print("\n1. RAW DATA")
print("-" * 50)

print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")


# ------------------------------------------------------------
# Calculate KPIs
# ------------------------------------------------------------

df = calculate_kpis(df)


print("\n2. KPI COLUMNS")
print("-" * 50)

kpi_columns = [
    "profit",
    "profit_margin",
    "revenue_growth",
    "expense_growth",
    "working_capital",
    "cash_ratio",
    "receivable_ratio",
    "payable_ratio",
    "debt_service_ratio",
    "inventory_ratio"
]

for column in kpi_columns:
    if column not in df.columns:
        raise AssertionError(f"Production KPI is missing: {column}")
    print(column)

if not df[["business_id", "month"]].equals(
    df.sort_values(["business_id", "month"])[["business_id", "month"]]
):
    raise AssertionError("KPI rows are not sorted chronologically by business.")

expected_profit = df["revenue"] - df["expenses"]
if not (df["profit"].sub(expected_profit).abs() < 1e-9).all():
    raise AssertionError("Production profit calculation is inconsistent.")

expected_margin = (
    expected_profit.div(df["revenue"]).mul(100)
).replace([float("inf"), float("-inf")], np.nan)
if not np.allclose(
    df["profit_margin"].to_numpy(dtype=float, na_value=np.nan),
    expected_margin.to_numpy(dtype=float, na_value=np.nan),
    equal_nan=True,
):
    raise AssertionError("Production profit-margin calculation is inconsistent.")

expected_revenue_growth = (
    df.groupby("business_id")["revenue"].pct_change() * 100
).replace([float("inf"), float("-inf")], np.nan)
if not np.allclose(
    df["revenue_growth"].to_numpy(dtype=float, na_value=np.nan),
    expected_revenue_growth.to_numpy(dtype=float, na_value=np.nan),
    equal_nan=True,
):
    raise AssertionError("Revenue growth is not calculated chronologically per business.")

if df[kpi_columns].isin([float("inf"), float("-inf")]).any().any():
    raise AssertionError("Infinite values found in production KPI output.")
print("\nMissing values in KPI columns (expected for first growth observations):")
print(df[kpi_columns].isna().sum().to_string())


# ------------------------------------------------------------
# Display sample
# ------------------------------------------------------------

print("\n3. SAMPLE KPI DATA")
print("-" * 50)

print(
    df[
        [
            "business_id",
            "month",
            "revenue",
            "expenses",
            "profit",
            "profit_margin",
            "working_capital",
            "cash_ratio",
            "receivable_ratio",
            "debt_service_ratio"
        ]
    ].head(10).to_string(index=False)
)


# ------------------------------------------------------------
# Latest business data
# ------------------------------------------------------------

latest = get_latest_business_data(df)


print("\n4. LATEST BUSINESS SNAPSHOT")
print("-" * 50)

print(
    latest[
        [
            "business_id",
            "month",
            "revenue",
            "expenses",
            "profit",
            "profit_margin",
            "cash_balance",
            "working_capital",
            "cash_ratio",
            "debt_service_ratio"
        ]
    ].head(10).to_string(index=False)
)


# ------------------------------------------------------------
# KPI summary
# ------------------------------------------------------------

print("\n5. AVERAGE KPI VALUES")
print("-" * 50)

print(
    f"Average Profit Margin   : "
    f"{df['profit_margin'].mean():.2f}%"
)

print(
    f"Average Cash Ratio      : "
    f"{df['cash_ratio'].mean():.2f}"
)

print(
    f"Average Receivable Ratio: "
    f"{df['receivable_ratio'].mean():.2f}"
)

print(
    f"Average Payable Ratio   : "
    f"{df['payable_ratio'].mean():.2f}"
)

print(
    f"Average Debt Service    : "
    f"{df['debt_service_ratio'].mean():.2f}"
)

print(
    f"Average Inventory Ratio : "
    f"{df['inventory_ratio'].mean():.2f}"
)


print("\n" + "=" * 70)
print("KPI ENGINE TEST COMPLETED")
print("=" * 70)
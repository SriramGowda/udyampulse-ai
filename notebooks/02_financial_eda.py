"""UdyamPulse AI | Phase 02: explore financial summaries and distributions.

Inputs: data/raw/msme_financial_data.csv
Outputs: console summaries and offline HTML Plotly figures
Method: production KPI calculations, monthly aggregation, and group analysis.
"""

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHART_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "financial_eda"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.data_service import calculate_kpis


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "msme_financial_data.csv"


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

if not DATASET_PATH.is_file():
    raise FileNotFoundError(f"Required dataset not found: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH, parse_dates=["month"])
df = calculate_kpis(df)
CHART_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Basic Financial Calculations
# --------------------------------------------------

# --------------------------------------------------
# Print Header
# --------------------------------------------------

print("=" * 70)
print("UDYAMPULSE AI - FINANCIAL EXPLORATORY DATA ANALYSIS")
print("=" * 70)


# --------------------------------------------------
# 1. Overall Financial Summary
# --------------------------------------------------

print("\n1. OVERALL FINANCIAL SUMMARY")
print("-" * 50)

summary_columns = [
    "revenue",
    "expenses",
    "profit",
    "cash_balance",
    "accounts_receivable",
    "accounts_payable",
    "inventory_value",
    "loan_emi",
    "loan_outstanding"
]

summary = df[summary_columns].describe().T

print(summary)


# --------------------------------------------------
# 2. Overall Average KPIs
# --------------------------------------------------

print("\n2. AVERAGE FINANCIAL KPIs")
print("-" * 50)

print(
    f"Average Revenue       : "
    f"INR {df['revenue'].mean():,.2f}"
)

print(
    f"Average Expenses      : "
    f"INR {df['expenses'].mean():,.2f}"
)

print(
    f"Average Profit        : "
    f"INR {df['profit'].mean():,.2f}"
)

print(
    f"Average Profit Margin : "
    f"{df['profit_margin'].mean():.2f}%"
)

print(
    f"Average Cash Balance  : "
    f"INR {df['cash_balance'].mean():,.2f}"
)

print(
    f"Average Receivables   : "
    f"INR {df['accounts_receivable'].mean():,.2f}"
)

print(
    f"Average Payables      : "
    f"INR {df['accounts_payable'].mean():,.2f}"
)


# --------------------------------------------------
# 3. Monthly Financial Trend
# --------------------------------------------------

monthly = (
    df.groupby("month")
    .agg(
        revenue=("revenue", "sum"),
        expenses=("expenses", "sum"),
        profit=("profit", "sum"),
        cash_balance=("cash_balance", "sum"),
        receivables=("accounts_receivable", "sum"),
        payables=("accounts_payable", "sum")
    )
    .reset_index()
)


print("\n3. MONTHLY FINANCIAL TREND")
print("-" * 50)

print(
    monthly[
        [
            "month",
            "revenue",
            "expenses",
            "profit"
        ]
    ].to_string(index=False)
)


# --------------------------------------------------
# 4. Sector-wise Analysis
# --------------------------------------------------

sector_analysis = (
    df.groupby("sector")
    .agg(
        average_revenue=("revenue", "mean"),
        average_expenses=("expenses", "mean"),
        average_profit=("profit", "mean"),
        average_cash=("cash_balance", "mean"),
        average_receivables=(
            "accounts_receivable",
            "mean"
        )
    )
    .reset_index()
)


print("\n4. SECTOR-WISE ANALYSIS")
print("-" * 50)

print(
    sector_analysis.to_string(
        index=False
    )
)


# --------------------------------------------------
# 5. State-wise Analysis
# --------------------------------------------------

state_analysis = (
    df.groupby("state")
    .agg(
        average_revenue=("revenue", "mean"),
        average_expenses=("expenses", "mean"),
        average_profit=("profit", "mean")
    )
    .reset_index()
)


print("\n5. STATE-WISE ANALYSIS")
print("-" * 50)

print(
    state_analysis.to_string(
        index=False
    )
)


# --------------------------------------------------
# 6. Financial Relationships
# --------------------------------------------------

print("\n6. FINANCIAL CORRELATIONS")
print("-" * 50)

correlation_columns = [
    "revenue",
    "expenses",
    "cash_balance",
    "accounts_receivable",
    "accounts_payable",
    "inventory_value",
    "loan_emi",
    "loan_outstanding",
    "profit"
]

correlation_matrix = (
    df[correlation_columns]
    .corr()
)

print(
    correlation_matrix.round(2)
)
print(
    "Interpretation: profit is calculated from revenue minus expenses, so "
    "its correlations with those two inputs are partly mechanical."
)


# --------------------------------------------------
# 7. Create Charts
# --------------------------------------------------

# Revenue and expense trends in one comparable chart.
trend_long = monthly.melt(
    id_vars="month",
    value_vars=[
        "revenue",
        "expenses"
    ],
    var_name="Metric",
    value_name="Amount"
)

fig_comparison = px.line(
    trend_long,
    x="month",
    y="Amount",
    color="Metric",
    markers=True,
    title="Revenue vs Expenses"
)

fig_comparison.update_layout(
    xaxis_title="Month",
    yaxis_title="Amount (INR)"
)

fig_comparison.write_html(
    CHART_OUTPUT_DIR / "monthly_revenue_expenses.html",
    include_plotlyjs="directory",
    auto_open=False,
)


# Sector-wise profit
fig_sector = px.bar(
    sector_analysis,
    x="sector",
    y="average_profit",
    title="Average Profit by Sector"
)

fig_sector.update_layout(
    xaxis_title="Sector",
    yaxis_title="Average Profit (INR)"
)

fig_sector.write_html(
    CHART_OUTPUT_DIR / "average_profit_by_sector.html",
    include_plotlyjs="directory",
    auto_open=False,
)


# Revenue vs Expenses scatter
fig_scatter = px.scatter(
    df,
    x="revenue",
    y="expenses",
    color="sector",
    title="Revenue vs Expenses",
    hover_data=[
        "business_id",
        "month"
    ]
)

fig_scatter.update_layout(
    xaxis_title="Revenue (INR)",
    yaxis_title="Expenses (INR)"
)

fig_scatter.write_html(
    CHART_OUTPUT_DIR / "revenue_vs_expenses.html",
    include_plotlyjs="directory",
    auto_open=False,
)


# Cash balance trend
fig_cash = px.line(
    monthly,
    x="month",
    y="cash_balance",
    markers=True,
    title="Total Monthly Cash Balance"
)

fig_cash.update_layout(
    xaxis_title="Month",
    yaxis_title="Cash Balance (INR)"
)

fig_cash.write_html(
    CHART_OUTPUT_DIR / "cash_balance_trend.html",
    include_plotlyjs="directory",
    auto_open=False,
)


print("\n" + "=" * 70)
print("FINANCIAL EDA COMPLETED")
print(f"Interactive charts saved to: {CHART_OUTPUT_DIR}")
print("=" * 70)
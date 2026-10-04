import pandas as pd
import plotly.express as px


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET_PATH = "data/raw/msme_financial_data.csv"


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df = pd.read_csv(
    DATASET_PATH,
    parse_dates=["month"]
)


# --------------------------------------------------
# Basic Financial Calculations
# --------------------------------------------------

df["profit"] = (
    df["revenue"]
    - df["expenses"]
)

df["profit_margin"] = (
    df["profit"]
    / df["revenue"]
) * 100


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
    f"₹{df['revenue'].mean():,.2f}"
)

print(
    f"Average Expenses      : "
    f"₹{df['expenses'].mean():,.2f}"
)

print(
    f"Average Profit        : "
    f"₹{df['profit'].mean():,.2f}"
)

print(
    f"Average Profit Margin : "
    f"{df['profit_margin'].mean():.2f}%"
)

print(
    f"Average Cash Balance  : "
    f"₹{df['cash_balance'].mean():,.2f}"
)

print(
    f"Average Receivables   : "
    f"₹{df['accounts_receivable'].mean():,.2f}"
)

print(
    f"Average Payables      : "
    f"₹{df['accounts_payable'].mean():,.2f}"
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


# --------------------------------------------------
# 7. Create Charts
# --------------------------------------------------

# Revenue trend
fig_revenue = px.line(
    monthly,
    x="month",
    y="revenue",
    markers=True,
    title="Total Monthly Revenue"
)

fig_revenue.update_layout(
    xaxis_title="Month",
    yaxis_title="Revenue (₹)"
)

fig_revenue.show()


# Expense trend
fig_expenses = px.line(
    monthly,
    x="month",
    y="expenses",
    markers=True,
    title="Total Monthly Expenses"
)

fig_expenses.update_layout(
    xaxis_title="Month",
    yaxis_title="Expenses (₹)"
)

fig_expenses.show()


# Revenue vs Expenses
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
    yaxis_title="Amount (₹)"
)

fig_comparison.show()


# Sector-wise profit
fig_sector = px.bar(
    sector_analysis,
    x="sector",
    y="average_profit",
    title="Average Profit by Sector"
)

fig_sector.update_layout(
    xaxis_title="Sector",
    yaxis_title="Average Profit (₹)"
)

fig_sector.show()


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
    xaxis_title="Revenue (₹)",
    yaxis_title="Expenses (₹)"
)

fig_scatter.show()


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
    yaxis_title="Cash Balance (₹)"
)

fig_cash.show()


print("\n" + "=" * 70)
print("FINANCIAL EDA COMPLETED")
print("=" * 70)
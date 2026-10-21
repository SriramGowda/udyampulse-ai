"""UdyamPulse AI | Phase 01: validate the raw synthetic financial dataset.

Inputs: data/raw/msme_financial_data.csv
Outputs: validation diagnostics printed to the console
Method: schema, completeness, uniqueness, date, and financial-value checks.
"""

from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "msme_financial_data.csv"


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

if not DATASET_PATH.is_file():
    raise FileNotFoundError(f"Required dataset not found: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH, parse_dates=["month"])
required_columns = {
    "business_id", "month", "sector", "state", "revenue", "expenses",
    "cash_balance", "accounts_receivable", "accounts_payable",
    "inventory_value", "loan_emi", "loan_outstanding",
}
missing_columns = sorted(required_columns.difference(df.columns))
if missing_columns:
    raise ValueError(f"Dataset is missing required columns: {missing_columns}")


print("=" * 60)
print("UDYAMPULSE AI - DATASET VALIDATION")
print("=" * 60)


# --------------------------------------------------
# 1. Dataset Shape
# --------------------------------------------------

print("\n1. DATASET SHAPE")
print("-" * 40)

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


# --------------------------------------------------
# 2. Column Names
# --------------------------------------------------

print("\n2. COLUMNS")
print("-" * 40)

for column in df.columns:
    print(column)


# --------------------------------------------------
# 3. Data Types
# --------------------------------------------------

print("\n3. DATA TYPES")
print("-" * 40)

print(df.dtypes)


# --------------------------------------------------
# 4. Missing Values
# --------------------------------------------------

print("\n4. MISSING VALUES")
print("-" * 40)

missing_values = df.isnull().sum()

print(missing_values)


# --------------------------------------------------
# 5. Duplicate Rows
# --------------------------------------------------

print("\n5. DUPLICATE ROWS")
print("-" * 40)

duplicate_count = df.duplicated().sum()

print(
    f"Duplicate rows: {duplicate_count}"
)
business_month_duplicates = df.duplicated(
    ["business_id", "month"]
).sum()
print(f"Duplicate business-month records: {business_month_duplicates}")


# --------------------------------------------------
# 6. Number of Businesses
# --------------------------------------------------

print("\n6. BUSINESS COUNT")
print("-" * 40)

business_count = (
    df["business_id"]
    .nunique()
)

print(
    f"Unique businesses: {business_count}"
)


# --------------------------------------------------
# 7. Sector Distribution
# --------------------------------------------------

print("\n7. SECTOR DISTRIBUTION")
print("-" * 40)

print(
    df["sector"]
    .value_counts()
)


# --------------------------------------------------
# 8. State Distribution
# --------------------------------------------------

print("\n8. STATE DISTRIBUTION")
print("-" * 40)

print(
    df["state"]
    .value_counts()
)


# --------------------------------------------------
# 9. Date Range
# --------------------------------------------------

print("\n9. DATE RANGE")
print("-" * 40)

print(
    f"Start date: {df['month'].min().date()}"
)

print(
    f"End date  : {df['month'].max().date()}"
)
periods_per_business = df.groupby("business_id")["month"].nunique()
print(
    "Months per business (min/max): "
    f"{periods_per_business.min()} / {periods_per_business.max()}"
)


# --------------------------------------------------
# 10. Financial Statistics
# --------------------------------------------------

print("\n10. FINANCIAL STATISTICS")
print("-" * 40)

financial_columns = [
    "revenue",
    "expenses",
    "cash_balance",
    "accounts_receivable",
    "accounts_payable",
    "inventory_value",
    "loan_emi",
    "loan_outstanding"
]

print(
    df[financial_columns].describe()
)


# --------------------------------------------------
# 11. Check Negative Financial Values
# --------------------------------------------------

print("\n11. NEGATIVE VALUES")
print("-" * 40)

for column in financial_columns:

    negative_count = (
        df[column] < 0
    ).sum()

    print(
        f"{column}: {negative_count}"
    )


# --------------------------------------------------
# 12. First Five Records
# --------------------------------------------------

print("\n12. SAMPLE DATA")
print("-" * 40)

print(
    df.head()
)


# --------------------------------------------------
# Validation Summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)

print(
    f"Rows: {len(df)}"
)

print(
    f"Businesses: {df['business_id'].nunique()}"
)

print(
    f"Missing values: {df.isnull().sum().sum()}"
)

print(
    f"Duplicate rows: {df.duplicated().sum()}"
)
print(
    f"Duplicate business-month records: "
    f"{df.duplicated(['business_id', 'month']).sum()}"
)

print(
    f"Negative financial values: "
    f"{(df[financial_columns] < 0).sum().sum()}"
)

print("\nDataset validation completed.")
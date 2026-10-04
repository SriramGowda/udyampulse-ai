from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# UDYAMPULSE AI
# Financial Data & KPI Service
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "msme_financial_data.csv"


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

def load_data():
    """
    Load the synthetic MSME financial dataset.

    Returns:
        pandas.DataFrame
    """

    df = pd.read_csv(DATA_PATH, parse_dates=["month"])

    return df


# ------------------------------------------------------------
# Calculate basic financial KPIs
# ------------------------------------------------------------

def calculate_kpis(df):
    """
    Calculate financial KPIs for each business-month record.

    The original financial columns are preserved.
    New KPI columns are added to the DataFrame.
    """

    df = df.copy()

    # --------------------------------------------------------
    # 1. Profit
    # --------------------------------------------------------

    df["profit"] = (
        df["revenue"]
        - df["expenses"]
    )

    # --------------------------------------------------------
    # 2. Profit Margin
    # --------------------------------------------------------

    df["profit_margin"] = (
        df["profit"]
        / df["revenue"]
        * 100
    )

    # --------------------------------------------------------
    # 3. Revenue Growth
    # --------------------------------------------------------

    df = df.sort_values(
        ["business_id", "month"]
    )

    df["previous_revenue"] = (
        df.groupby("business_id")["revenue"]
        .shift(1)
    )

    df["revenue_growth"] = (
        (
            df["revenue"]
            - df["previous_revenue"]
        )
        / df["previous_revenue"]
        * 100
    )

    # --------------------------------------------------------
    # 4. Expense Growth
    # --------------------------------------------------------

    df["previous_expenses"] = (
        df.groupby("business_id")["expenses"]
        .shift(1)
    )

    df["expense_growth"] = (
        (
            df["expenses"]
            - df["previous_expenses"]
        )
        / df["previous_expenses"]
        * 100
    )

    # --------------------------------------------------------
    # 5. Working Capital
    #
    # Working Capital =
    # Receivables + Inventory - Payables
    # --------------------------------------------------------

    df["working_capital"] = (
        df["accounts_receivable"]
        + df["inventory_value"]
        - df["accounts_payable"]
    )

    # --------------------------------------------------------
    # 6. Cash Ratio
    #
    # For this project:
    # Cash Ratio = Cash Balance / Expenses
    # --------------------------------------------------------

    df["cash_ratio"] = (
        df["cash_balance"]
        / df["expenses"]
    )

    # --------------------------------------------------------
    # 7. Receivable Ratio
    #
    # Receivables relative to revenue.
    # --------------------------------------------------------

    df["receivable_ratio"] = (
        df["accounts_receivable"]
        / df["revenue"]
    )

    # --------------------------------------------------------
    # 8. Payable Ratio
    #
    # Payables relative to expenses.
    # --------------------------------------------------------

    df["payable_ratio"] = (
        df["accounts_payable"]
        / df["expenses"]
    )

    # --------------------------------------------------------
    # 9. Debt Service Ratio
    #
    # Loan EMI relative to revenue.
    # --------------------------------------------------------

    df["debt_service_ratio"] = (
        df["loan_emi"]
        / df["revenue"]
    )

    # --------------------------------------------------------
    # 10. Inventory Ratio
    #
    # Inventory relative to revenue.
    # --------------------------------------------------------

    df["inventory_ratio"] = (
        df["inventory_value"]
        / df["revenue"]
    )

    # --------------------------------------------------------
    # Replace infinite values
    # --------------------------------------------------------

    df = df.replace(
        [float("inf"), float("-inf")],
        pd.NA
    )

    return df


# ------------------------------------------------------------
# Get latest record for each business
# ------------------------------------------------------------

def get_latest_business_data(df):
    """
    Return the latest available financial record
    for every business.
    """

    latest = (
        df.sort_values(
            ["business_id", "month"]
        )
        .groupby("business_id")
        .tail(1)
        .reset_index(drop=True)
    )

    return latest


# ------------------------------------------------------------
# Get one business's history
# ------------------------------------------------------------

def get_business_history(
    df,
    business_id
):
    """
    Return all monthly records for one business.
    """

    business_df = (
        df[
            df["business_id"]
            == business_id
        ]
        .sort_values("month")
        .reset_index(drop=True)
    )

    return business_df


# ------------------------------------------------------------
# Forecasting: chronological, damped linear trend with holdout evaluation
# ------------------------------------------------------------


def _fit_trend(history, metric, forecast_months):
    """Fit a linear trend using only the supplied chronological history."""
    values = history[metric].astype(float)
    valid = values.notna() & np.isfinite(values.to_numpy(dtype=float))
    values = values.loc[valid].tail(24)
    months = history.loc[values.index, "month"]

    if len(values) < 2:
        raise ValueError(f"At least 2 valid historical {metric} values are required.")

    x = (
        (months.dt.year - months.iloc[0].year) * 12
        + (months.dt.month - months.iloc[0].month)
    ).to_numpy(dtype=float)
    y = values.to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)

    future_x = x[-1] + np.arange(1, forecast_months + 1, dtype=float)
    # Dampen the fitted slope to avoid implausible long-range extrapolation.
    predictions = y[-1] + 0.75 * slope * (future_x - x[-1])
    return np.maximum(predictions, 0.0)


def evaluate_forecast(df, business_id, holdout_months=6):
    """Evaluate each forecast with a chronological holdout; no future rows train the model."""
    business_df = get_business_history(df, business_id)
    metrics = []

    for metric in ("revenue", "cash_balance"):
        series = business_df[["month", metric]].dropna().reset_index(drop=True)
        test_size = min(holdout_months, max(0, len(series) // 4))
        if test_size < 1 or len(series) - test_size < 2:
            metrics.append({
                "metric": metric,
                "mae": np.nan,
                "rmse": np.nan,
                "holdout_months": 0,
            })
            continue

        train = series.iloc[:-test_size]
        actual = series.iloc[-test_size:][metric].to_numpy(dtype=float)
        predicted = _fit_trend(train, metric, test_size)
        errors = actual - predicted
        metrics.append({
            "metric": metric,
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "holdout_months": test_size,
        })

    return pd.DataFrame(metrics)


def forecast_business(df, business_id, months_ahead=6):
    """
    Forecast future revenue and cash balance for one business.

    Fits a damped linear trend on up to 24 latest chronological monthly values.
    No future observations are used. The compatible return shape is unchanged.

    This is Phase 3 forecasting logic, not yet a full ML model.
    """

    business_df = get_business_history(df, business_id).copy()

    if business_df.empty:
        raise ValueError(f"No data found for business_id: {business_id}")

    business_df = business_df.sort_values("month").copy()

    if len(business_df) < 2:
        raise ValueError(
            "At least 2 historical months are required for forecasting."
        )

    if months_ahead < 1:
        raise ValueError("months_ahead must be at least 1.")

    max_month = business_df["month"].max()

    future_months = pd.date_range(
        start=max_month + pd.offsets.MonthBegin(1),
        periods=months_ahead,
        freq="MS"
    )

    forecast_rows = []

    for metric in ["revenue", "cash_balance"]:
        future_values = _fit_trend(business_df, metric, months_ahead)

        for month, value in zip(future_months, future_values):
            forecast_rows.append({
                "business_id": business_id,
                "month": month,
                "metric": metric,
                "forecast_value": float(value)
            })

    forecast_df = pd.DataFrame(forecast_rows)

    forecast_pivot = (
        forecast_df
        .pivot(index="month", columns="metric", values="forecast_value")
        .reset_index()
    )

    forecast_pivot["business_id"] = business_id
    forecast_pivot = forecast_pivot[
        ["business_id", "month", "revenue", "cash_balance"]
    ]

    return forecast_pivot


# ------------------------------------------------------------
# Phase 2.1: Future Financial Risk Target
# ------------------------------------------------------------

def create_risk_target(df, future_months=3):
    """
    Create a future financial-risk target for supervised ML.

    The target is based on financial stress observed during the
    following 3 months.

    Important:
    - Current-month financial information is used later as ML features.
    - Future financial stress is used only to create the target.
    - This helps avoid target leakage.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Sort data chronologically for every business
    # --------------------------------------------------------

    df = df.sort_values(
        ["business_id", "month"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Calculate required financial indicators
    # --------------------------------------------------------

    df["profit"] = (
        df["revenue"]
        - df["expenses"]
    )

    df["profit_margin"] = (
        df["profit"]
        / df["revenue"]
        * 100
    )

    df["cash_ratio"] = (
        df["cash_balance"]
        / df["expenses"]
    )

    df["receivable_ratio"] = (
        df["accounts_receivable"]
        / df["revenue"]
    )

    df["debt_service_ratio"] = (
        df["loan_emi"]
        / df["revenue"]
    )

    # --------------------------------------------------------
    # Create future 3-month averages
    # --------------------------------------------------------

    grouped = df.groupby("business_id")

    df["future_profit_margin"] = sum(
        grouped["profit_margin"].shift(-i)
        for i in range(1, future_months + 1)
    ) / future_months

    df["future_cash_balance"] = sum(
        grouped["cash_balance"].shift(-i)
        for i in range(1, future_months + 1)
    ) / future_months

    df["future_revenue"] = sum(
        grouped["revenue"].shift(-i)
        for i in range(1, future_months + 1)
    ) / future_months

    df["future_receivable_ratio"] = sum(
        grouped["receivable_ratio"].shift(-i)
        for i in range(1, future_months + 1)
    ) / future_months

    df["future_debt_service_ratio"] = sum(
        grouped["debt_service_ratio"].shift(-i)
        for i in range(1, future_months + 1)
    ) / future_months

    # --------------------------------------------------------
    # Calculate future changes
    # --------------------------------------------------------

    df["future_cash_change_pct"] = (
        (
            df["future_cash_balance"]
            - df["cash_balance"]
        )
        / df["cash_balance"]
        * 100
    )

    df["future_revenue_change_pct"] = (
        (
            df["future_revenue"]
            - df["revenue"]
        )
        / df["revenue"]
        * 100
    )

    # --------------------------------------------------------
    # Remove records without enough future history
    # --------------------------------------------------------

    required_columns = [
        "future_profit_margin",
        "future_cash_balance",
        "future_revenue",
        "future_receivable_ratio",
        "future_debt_service_ratio"
    ]

    target_df = df.dropna(
        subset=required_columns
    ).copy()

    # --------------------------------------------------------
    # Financial stress indicators
    # --------------------------------------------------------

    target_df["stress_low_profit"] = (
        target_df["future_profit_margin"] < 10
    ).astype(int)

    target_df["stress_cash_decline"] = (
        target_df["future_cash_change_pct"] < -15
    ).astype(int)

    target_df["stress_revenue_decline"] = (
        target_df["future_revenue_change_pct"] < -10
    ).astype(int)

    target_df["stress_high_receivables"] = (
        target_df["future_receivable_ratio"] > 0.30
    ).astype(int)

    target_df["stress_high_debt"] = (
        target_df["future_debt_service_ratio"] > 0.12
    ).astype(int)

    # --------------------------------------------------------
    # Total future financial stress score
    # --------------------------------------------------------

    stress_columns = [
        "stress_low_profit",
        "stress_cash_decline",
        "stress_revenue_decline",
        "stress_high_receivables",
        "stress_high_debt"
    ]

    target_df["future_stress_score"] = (
        target_df[stress_columns]
        .sum(axis=1)
    )

    # --------------------------------------------------------
    # Risk labels
    #
    # 0 stress indicators  -> LOW
    # 1 stress indicator    -> MEDIUM
    # 2+ indicators         -> HIGH
    # --------------------------------------------------------

    target_df["risk_label"] = "LOW"

    target_df.loc[
        target_df["future_stress_score"] == 1,
        "risk_label"
    ] = "MEDIUM"

    target_df.loc[
        target_df["future_stress_score"] >= 2,
        "risk_label"
    ] = "HIGH"

    # --------------------------------------------------------
    # Numeric target for ML models
    # --------------------------------------------------------

    target_df["risk_target"] = (
        target_df["risk_label"]
        .map({
            "LOW": 0,
            "MEDIUM": 1,
            "HIGH": 2
        })
    )

    return target_df
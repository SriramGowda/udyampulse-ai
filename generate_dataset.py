import os

import numpy as np
import pandas as pd


# ============================================================
# UDYAMPULSE AI
# Synthetic MSME Financial Dataset Generator
# ============================================================

OUTPUT_PATH = "data/raw/msme_financial_data.csv"

NUM_BUSINESSES = 100
NUM_MONTHS = 36
RANDOM_SEED = 42


def generate_dataset(
    num_businesses=NUM_BUSINESSES,
    months=NUM_MONTHS,
    seed=RANDOM_SEED
):
    """
    Generate synthetic monthly financial data
    for manufacturing MSMEs.

    IMPORTANT:
    This dataset is synthetic and intended only
    for academic/demo purposes.

    No ML risk label is created here.
    """

    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # Reference data
    # --------------------------------------------------------

    states = [
        "Karnataka",
        "Tamil Nadu",
        "Maharashtra",
        "Telangana",
        "Andhra Pradesh",
        "Gujarat",
        "Kerala",
        "Rajasthan"
    ]

    # Distribution of business financial conditions.
    # This is used internally only to create realistic
    # financial trajectories.
    conditions = (
        ["Healthy"] * 35
        + ["Stable"] * 30
        + ["Moderate Stress"] * 25
        + ["High Stress"] * 10
    )

    dates = pd.date_range(
        start="2023-10-01",
        periods=months,
        freq="MS"
    )

    records = []

    # ========================================================
    # Business generation
    # ========================================================

    for business_number in range(
        1,
        num_businesses + 1
    ):

        business_id = (
            f"MSME{business_number:03d}"
        )

        business_name = (
            f"Manufacturing MSME {business_number}"
        )

        state = rng.choice(states)

        sector = "Manufacturing"

        employee_count = int(
            rng.integers(5, 100)
        )

        # Internal financial profile.
        # NOT stored in the final CSV.
        condition = conditions[
            business_number - 1
        ]

        # ----------------------------------------------------
        # Initial business size
        # ----------------------------------------------------

        base_revenue = rng.uniform(
            400000,
            1500000
        )

        # Starting cash is deliberately moderate.
        cash_balance = (
            base_revenue
            * rng.uniform(
                0.30,
                0.60
            )
        )

        base_receivables = (
            base_revenue
            * rng.uniform(
                0.15,
                0.25
            )
        )

        base_payables = (
            base_revenue
            * rng.uniform(
                0.10,
                0.20
            )
        )

        base_inventory = (
            base_revenue
            * rng.uniform(
                0.15,
                0.30
            )
        )

        # ----------------------------------------------------
        # Loan
        # ----------------------------------------------------

        initial_loan = rng.uniform(
            800000,
            5000000
        )

        remaining_loan = initial_loan

        loan_term_months = 48

        loan_emi = (
            initial_loan
            / loan_term_months
        )

        previous_revenue = base_revenue

        previous_receivables = (
            base_receivables
        )

        previous_payables = (
            base_payables
        )

        # ====================================================
        # Financial profile configuration
        # ====================================================

        if condition == "Healthy":

            revenue_trend = rng.uniform(
                0.004,
                0.010
            )

            expense_ratio = rng.uniform(
                0.62,
                0.70
            )

            receivable_growth = rng.uniform(
                0.001,
                0.004
            )

            cash_efficiency = rng.uniform(
                0.85,
                0.95
            )

        elif condition == "Stable":

            revenue_trend = rng.uniform(
                -0.001,
                0.005
            )

            expense_ratio = rng.uniform(
                0.66,
                0.75
            )

            receivable_growth = rng.uniform(
                0.003,
                0.007
            )

            cash_efficiency = rng.uniform(
                0.75,
                0.90
            )

        elif condition == "Moderate Stress":

            revenue_trend = rng.uniform(
                -0.010,
                -0.003
            )

            expense_ratio = rng.uniform(
                0.73,
                0.84
            )

            receivable_growth = rng.uniform(
                0.008,
                0.015
            )

            cash_efficiency = rng.uniform(
                0.55,
                0.75
            )

        else:

            revenue_trend = rng.uniform(
                -0.020,
                -0.010
            )

            expense_ratio = rng.uniform(
                0.82,
                0.92
            )

            receivable_growth = rng.uniform(
                0.015,
                0.025
            )

            cash_efficiency = rng.uniform(
                0.35,
                0.60
            )

        # ====================================================
        # Monthly financial records
        # ====================================================

        for month_index, date in enumerate(dates):

            # ------------------------------------------------
            # Seasonal revenue effect
            # ------------------------------------------------

            seasonal_factor = (
                1
                + 0.04
                * np.sin(
                    (
                        month_index
                        / 12
                    )
                    * 2
                    * np.pi
                )
            )

            random_growth = rng.normal(
                0,
                0.015
            )

            # ------------------------------------------------
            # Revenue
            # ------------------------------------------------

            revenue = (
                previous_revenue
                * (
                    1
                    + revenue_trend
                    + random_growth
                )
                * seasonal_factor
            )

            revenue = max(
                revenue,
                100000
            )

            # ------------------------------------------------
            # Expenses
            # ------------------------------------------------

            expense_noise = rng.normal(
                0,
                0.012
            )

            current_expense_ratio = (
                expense_ratio
                + expense_noise
            )

            current_expense_ratio = np.clip(
                current_expense_ratio,
                0.55,
                0.95
            )

            expenses = (
                revenue
                * current_expense_ratio
            )

            # ------------------------------------------------
            # Profit
            # ------------------------------------------------

            profit = (
                revenue
                - expenses
            )

            # ------------------------------------------------
            # Receivables
            # ------------------------------------------------

            accounts_receivable = (
                base_receivables
                * (
                    1
                    + receivable_growth
                ) ** month_index
            )

            # Additional receivable pressure
            # for stressed businesses.
            if condition == "Moderate Stress":

                accounts_receivable *= (
                    1
                    + 0.004
                    * month_index
                )

            elif condition == "High Stress":

                accounts_receivable *= (
                    1
                    + 0.008
                    * month_index
                )

            # ------------------------------------------------
            # Payables
            # ------------------------------------------------

            accounts_payable = (
                base_payables
                * (
                    expenses
                    / base_revenue
                )
            )

            if condition == "High Stress":

                accounts_payable *= (
                    1
                    + 0.004
                    * month_index
                )

            # ------------------------------------------------
            # Inventory
            # ------------------------------------------------

            inventory_value = (
                base_inventory
                * (
                    revenue
                    / base_revenue
                )
            )

            # ------------------------------------------------
            # Loan repayment
            # ------------------------------------------------

            loan_payment = min(
                loan_emi,
                remaining_loan
            )

            remaining_loan = max(
                remaining_loan
                - loan_payment,
                0
            )

            # ------------------------------------------------
            # Working capital changes
            # ------------------------------------------------

            receivable_change = (
                accounts_receivable
                - previous_receivables
            )

            payable_change = (
                accounts_payable
                - previous_payables
            )

            # ------------------------------------------------
            # Operating cash flow
            # ------------------------------------------------

            operating_cash_flow = (
                profit
                - receivable_change
                + payable_change
                - loan_payment
            )

            # Only a portion of accounting profit becomes
            # available cash because of operating needs.
            operating_cash_flow *= cash_efficiency

            # ------------------------------------------------
            # Additional stress effects
            # ------------------------------------------------

            if condition == "Healthy":

                additional_pressure = (
                    base_revenue
                    * rng.uniform(
                        0.005,
                        0.015
                    )
                )

            elif condition == "Stable":

                additional_pressure = (
                    base_revenue
                    * rng.uniform(
                        0.010,
                        0.020
                    )
                )

            elif condition == "Moderate Stress":

                additional_pressure = (
                    base_revenue
                    * rng.uniform(
                        0.020,
                        0.040
                    )
                )

            else:

                additional_pressure = (
                    base_revenue
                    * rng.uniform(
                        0.040,
                        0.070
                    )
                )

            # ------------------------------------------------
            # Final cash movement
            # ------------------------------------------------

            cash_change = (
                operating_cash_flow
                - additional_pressure
            )

            cash_change += rng.normal(
                0,
                base_revenue * 0.008
            )

            cash_balance += cash_change

            # Keep cash positive but allow stressed
            # businesses to approach a low-cash state.
            cash_balance = max(
                cash_balance,
                5000
            )

            # ------------------------------------------------
            # Store record
            # ------------------------------------------------

            records.append(
                {
                    "business_id": business_id,
                    "business_name": business_name,
                    "month": date,
                    "sector": sector,
                    "state": state,
                    "employee_count": employee_count,
                    "revenue": round(
                        revenue,
                        2
                    ),
                    "expenses": round(
                        expenses,
                        2
                    ),
                    "cash_balance": round(
                        cash_balance,
                        2
                    ),
                    "accounts_receivable": round(
                        accounts_receivable,
                        2
                    ),
                    "accounts_payable": round(
                        accounts_payable,
                        2
                    ),
                    "inventory_value": round(
                        inventory_value,
                        2
                    ),
                    "loan_emi": round(
                        loan_payment,
                        2
                    ),
                    "loan_outstanding": round(
                        remaining_loan,
                        2
                    )
                }
            )

            previous_revenue = revenue

            previous_receivables = (
                accounts_receivable
            )

            previous_payables = (
                accounts_payable
            )

    return pd.DataFrame(records)


# ============================================================
# Main
# ============================================================

def main():

    os.makedirs(
        "data/raw",
        exist_ok=True
    )

    df = generate_dataset()

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        "\nSynthetic MSME dataset created successfully!"
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print(
        f"Businesses: "
        f"{df['business_id'].nunique()}"
    )

    print(
        f"Sector: "
        f"{df['sector'].unique().tolist()}"
    )

    print(
        f"Date range: "
        f"{df['month'].min().date()} "
        f"to "
        f"{df['month'].max().date()}"
    )

    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
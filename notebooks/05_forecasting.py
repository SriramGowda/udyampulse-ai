import os
import sys

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
    calculate_kpis,
    forecast_business,
    evaluate_forecast
)


# --------------------------------------------------
# Phase 3: Forecasting
# --------------------------------------------------


def main():
    df = load_data()
    df = calculate_kpis(df)

    sample_business = df["business_id"].unique()[0]
    forecast_df = forecast_business(df, sample_business, months_ahead=6)
    evaluation_df = evaluate_forecast(df, sample_business)

    print("=" * 70)
    print("UDYAMPULSE AI - FORECASTING")
    print("=" * 70)

    print(f"\nBusiness: {sample_business}")
    print("\nForecast for next 6 months")
    print("-" * 40)
    print(forecast_df.to_string(index=False))
    print("\nChronological holdout evaluation")
    print("-" * 40)
    print(evaluation_df.to_string(index=False))

    print("\nInterpretation")
    print("-" * 40)
    print(
        "This forecast is a damped linear trend based only on earlier "
        "monthly observations. Holdout MAE and RMSE provide a basic "
        "chronological evaluation."
    )


if __name__ == "__main__":
    main()

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

from services.data_service import load_data, calculate_kpis
from services.ai_service import generate_financial_summary


# --------------------------------------------------
# Phase 4: Generative AI
# --------------------------------------------------


def main():
    df = load_data()
    df = calculate_kpis(df)

    sample_business = df[df["business_id"] == df["business_id"].unique()[0]].copy()
    sample_business = sample_business.sort_values("month")

    summary = generate_financial_summary(sample_business)

    print("=" * 70)
    print("UDYAMPULSE AI - GENERATIVE AI SUMMARY")
    print("=" * 70)

    print(f"\nBusiness: {sample_business['business_id'].iloc[0]}")
    print("-" * 40)
    print(summary)

    print("\nExplanation")
    print("-" * 40)
    print(
        "This module generates a plain-English business explanation from the latest financial metrics. "
        "If a Gemini API key is configured, it uses Gemini for the final wording; otherwise, it falls back to a local summary."
    )


if __name__ == "__main__":
    main()

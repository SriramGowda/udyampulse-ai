import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from services.ai_service import (
    _generate_gemini,
    generate_financial_summary,
    generate_rag_answer,
    retrieve_documents,
)
from services.data_service import evaluate_forecast, forecast_business


def make_history(months=24):
    month_values = pd.date_range("2024-01-01", periods=months, freq="MS")
    return pd.DataFrame({
        "business_id": ["DEMO"] * months,
        "month": month_values,
        "revenue": [1000 + 20 * index for index in range(months)],
        "expenses": [750 + 12 * index for index in range(months)],
        "cash_balance": [500 + 5 * index for index in range(months)],
        "business_name": ["Demo MSME"] * months,
        "sector": ["Manufacturing"] * months,
        "profit": [250 + 8 * index for index in range(months)],
        "profit_margin": [
            (250 + 8 * index) / (1000 + 20 * index) * 100
            for index in range(months)
        ],
    })


class ForecastTests(unittest.TestCase):
    def test_six_month_forecast_is_chronological_and_uses_compatible_shape(self):
        history = make_history().sample(frac=1, random_state=5)
        forecast = forecast_business(history, "DEMO", months_ahead=6)

        self.assertEqual(list(forecast.columns), [
            "business_id", "month", "revenue", "cash_balance"
        ])
        self.assertEqual(len(forecast), 6)
        self.assertEqual(
            forecast["month"].tolist(),
            list(pd.date_range("2026-01-01", periods=6, freq="MS")),
        )
        self.assertTrue((forecast[["revenue", "cash_balance"]] >= 0).all().all())

    def test_holdout_evaluation_is_reported_for_each_metric(self):
        evaluation = evaluate_forecast(make_history(), "DEMO")

        self.assertEqual(evaluation["metric"].tolist(), ["revenue", "cash_balance"])
        self.assertEqual(evaluation["holdout_months"].tolist(), [6, 6])
        self.assertTrue(evaluation[["mae", "rmse"]].notna().all().all())

    def test_unknown_business_and_invalid_horizon_are_reported(self):
        with self.assertRaisesRegex(ValueError, "No data found"):
            forecast_business(make_history(), "MISSING")
        with self.assertRaisesRegex(ValueError, "at least 1"):
            forecast_business(make_history(), "DEMO", months_ahead=0)


class RetrievalTests(unittest.TestCase):
    def test_retrieval_ranks_chunks_and_local_fallback_cites_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            reference = Path(temp_dir) / "working-capital.txt"
            reference.write_text(
                "Monitor cash flow and working capital. Rising receivables can "
                "create liquidity pressure. Review monthly revenue and expenses.",
                encoding="utf-8",
            )
            results = retrieve_documents(
                "How do receivables affect liquidity and working capital?",
                doc_dir=temp_dir,
            )
            self.assertTrue(results)
            self.assertEqual(results[0]["source"], "working-capital.txt")
            self.assertGreater(results[0]["score"], 0)

            with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
                answer = generate_rag_answer(
                    "How do receivables affect liquidity?",
                    doc_dir=temp_dir,
                )
            self.assertIn("working-capital.txt", answer["sources"])
            self.assertFalse(answer["used_gemini"])
            self.assertIn("retrieved", answer["answer"])

    def test_empty_query_and_no_documents_return_no_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertEqual(retrieve_documents(" ", doc_dir=temp_dir), [])
            self.assertEqual(retrieve_documents("cash flow", doc_dir=temp_dir), [])

    def test_financial_explanation_fallback_covers_model_and_forecast_results(self):
        history = make_history()
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            summary = generate_financial_summary(
                history,
                analysis_context={
                    "risk_label": "HIGH",
                    "anomaly_result": "anomaly",
                    "revenue_forecast_change_pct": 6.2,
                    "cash_forecast_change_pct": -2.1,
                },
            )
        summary = summary.casefold()
        for detail in ("revenue", "expenses", "profit", "cash balance",
                       "high", "anomaly", "+6.2%", "-2.1%"):
            self.assertIn(detail, summary)

    def test_supported_genai_sdk_request_shape(self):
        response = Mock(text="Gemini explanation")
        client = Mock()
        client.models.generate_content.return_value = response
        with patch("google.genai.Client", return_value=client):
            answer = _generate_gemini("Explain these metrics", api_key="demo-key")

        self.assertEqual(answer, "Gemini explanation")
        client.models.generate_content.assert_called_once()
        self.assertEqual(
            client.models.generate_content.call_args.kwargs["model"],
            os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        )


if __name__ == "__main__":
    unittest.main()

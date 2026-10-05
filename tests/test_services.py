import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
from urllib.error import URLError

from services.ai_service import (
    _generate_gemini,
    generate_financial_summary,
    generate_financial_summary_result,
    generate_rag_answer,
    get_ai_status,
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

            with (
                patch.dict(os.environ, {"GEMINI_API_KEY": ""}),
                patch("services.ai_service._generate_gemini", return_value=None),
                patch("services.ai_service.generate_local_response", return_value=None),
            ):
                answer = generate_rag_answer(
                    "How do receivables affect liquidity?",
                    doc_dir=temp_dir,
                )
            self.assertIn("working-capital.txt", answer["sources"])
            self.assertFalse(answer["used_gemini"])
            self.assertEqual(answer["provider"], "retrieval_fallback")
            self.assertIn("retrieved document information", answer["answer"])
            self.assertEqual(
                answer["retrieved_documents"][0]["source"],
                "working-capital.txt",
            )

    def test_gemini_response_is_used_before_local_generation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "finance.txt").write_text(
                "Working capital is current assets less current liabilities.",
                encoding="utf-8",
            )
            with (
                patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
                patch(
                    "services.ai_service._generate_gemini",
                    return_value="Gemini grounded answer",
                ) as gemini,
                patch("services.ai_service.generate_local_response") as local,
            ):
                result = generate_rag_answer(
                    "What is working capital?",
                    doc_dir=temp_dir,
                )

        self.assertEqual(result["provider"], "gemini")
        self.assertEqual(result["answer"], "Gemini grounded answer")
        self.assertEqual(result["sources"], ["finance.txt"])
        gemini.assert_called_once()
        local.assert_not_called()

    def test_failed_gemini_uses_local_ollama_generation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "finance.txt").write_text(
                "Working capital is current assets less current liabilities.",
                encoding="utf-8",
            )
            with (
                patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
                patch("services.ai_service._generate_gemini", return_value=None),
                patch(
                    "services.ai_service.generate_local_response",
                    return_value="Local grounded answer",
                ) as local,
            ):
                result = generate_rag_answer(
                    "What is working capital?",
                    doc_dir=temp_dir,
                )

        self.assertEqual(result["provider"], "local")
        self.assertEqual(result["answer"], "Local grounded answer")
        self.assertEqual(result["sources"], ["finance.txt"])
        local.assert_called_once()

    def test_local_provider_status_and_no_internet_retrieval(self):
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": ""}),
            patch("services.local_llm_service.urlopen", side_effect=URLError("offline")),
            patch("services.local_llm_service.shutil.which", return_value=None),
        ):
            status = get_ai_status()

        self.assertFalse(status["gemini_available"])
        self.assertFalse(status["local_llm_available"])
        self.assertTrue(status["offline_ready"])
        self.assertEqual(status["preferred_provider"], "retrieval_fallback")
        self.assertNotIn("GEMINI_API_KEY", status)

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.dict(os.environ, {"GEMINI_API_KEY": ""}),
            patch("services.ai_service._generate_gemini", return_value=None),
            patch("services.ai_service.generate_local_response", return_value=None),
        ):
            Path(temp_dir, "finance.txt").write_text(
                "Cash flow monitoring helps track receipts, payments, and liquidity.",
                encoding="utf-8",
            )
            result = generate_rag_answer("How do I monitor cash flow?", doc_dir=temp_dir)

        self.assertEqual(result["provider"], "retrieval_fallback")
        self.assertEqual(result["sources"], ["finance.txt"])

    def test_financial_summary_falls_back_to_ollama_then_local_summary(self):
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch("services.ai_service._generate_gemini", return_value=None),
            patch(
                "services.ai_service.generate_local_response",
                return_value="Ollama financial explanation",
            ),
        ):
            local_result = generate_financial_summary_result(make_history())
        self.assertEqual(local_result["provider"], "local")
        self.assertEqual(local_result["answer"], "Ollama financial explanation")

        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": ""}),
            patch("services.ai_service.generate_local_response", return_value=None),
        ):
            fallback_result = generate_financial_summary_result(make_history())
        self.assertEqual(fallback_result["provider"], "local_summary")
        self.assertIn("Revenue", fallback_result["answer"])

    def test_ollama_model_status_and_generation_use_loopback(self):
        from services.local_llm_service import generate_local_response, get_local_llm_status

        tags_response = MagicMock()
        tags_response.__enter__.return_value.read.return_value = (
            b'{"models":[{"name":"custom-finance:latest"}]}'
        )
        generation_response = MagicMock()
        generation_response.__enter__.return_value.read.return_value = (
            b'{"response":"A local response."}'
        )
        with (
            patch.dict(os.environ, {"OLLAMA_MODEL": "custom-finance:latest"}),
            patch("services.local_llm_service.shutil.which", return_value="ollama"),
            patch(
                "services.local_llm_service.urlopen",
                side_effect=[tags_response, tags_response, generation_response],
            ) as local_request,
        ):
            status = get_local_llm_status()
            answer = generate_local_response("Summarize these local records.")

        self.assertTrue(status["ollama_installed"])
        self.assertTrue(status["ollama_running"])
        self.assertTrue(status["local_llm_available"])
        self.assertEqual(answer, "A local response.")
        self.assertTrue(all(
            call.args[0].full_url.startswith("http://127.0.0.1:11434/")
            for call in local_request.call_args_list
        ))

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
                       "high", "anomaly", "6.2%", "2.1%"):
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

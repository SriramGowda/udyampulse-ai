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
from services.data_service import(
     calculate_kpis,
     evaluate_forecast,
    forecast_business,
)


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


class FinancialKpiTests(unittest.TestCase):
    def test_kpis_follow_existing_formulas_and_handle_zero_denominators(self):
        records = pd.DataFrame({
            "business_id": ["DEMO", "DEMO"],
            "month": pd.to_datetime(["2025-01-01", "2025-02-01"]),
            "revenue": [0.0, 100.0],
            "expenses": [0.0, 50.0],
            "cash_balance": [-10.0, 25.0],
            "accounts_receivable": [5.0, 20.0],
            "accounts_payable": [2.0, 10.0],
            "inventory_value": [3.0, 5.0],
            "loan_emi": [1.0, 5.0],
        })

        result = calculate_kpis(records)
        first, second = result.iloc[0], result.iloc[1]

        self.assertEqual(first["profit"], 0)
        self.assertTrue(pd.isna(first["profit_margin"]))
        self.assertTrue(pd.isna(second["revenue_growth"]))
        self.assertTrue(pd.isna(second["expense_growth"]))
        self.assertTrue(pd.isna(first["cash_ratio"]))
        self.assertTrue(pd.isna(first["receivable_ratio"]))
        self.assertEqual(second["profit"], 50)
        self.assertEqual(second["profit_margin"], 50)
        self.assertEqual(second["working_capital"], 15)
        self.assertEqual(second["cash_ratio"], 0.5)
        self.assertEqual(second["receivable_ratio"], 0.2)
        self.assertEqual(second["payable_ratio"], 0.2)
        self.assertEqual(second["debt_service_ratio"], 0.05)
        self.assertEqual(second["inventory_ratio"], 0.05)
        for column in (
            "profit_margin", "revenue_growth", "expense_growth", "cash_ratio",
            "receivable_ratio", "payable_ratio", "debt_service_ratio",
            "inventory_ratio",
        ):
            values = result[column].dropna().to_numpy(dtype=float)
            self.assertTrue(
                pd.Series(values).map(
                    lambda value: abs(value) < float("inf")
                ).all()
            )


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
    def test_official_msme_references_are_retrievable_and_attributed(self):
        expected_sources = {
            "credit guarantee institutional credit": "official_cgtmse_reference.txt",
            "Udyam registration private agency": "official_udyam_registration_reference.txt",
            "PMEGP scheme guidelines": "official_pmegp_reference.txt",
            "public procurement policy MSE": "official_public_procurement_reference.txt",
        }

        for question, source in expected_sources.items():
            with self.subTest(question=question):
                results = retrieve_documents(question)
                self.assertTrue(results)
                self.assertEqual(results[0]["source"], source)
                reference = Path(__file__).resolve().parent.parent / "documents" / source
                content = reference.read_text(encoding="utf-8")
                self.assertIn("Publisher:", content)
                self.assertIn("Official URL: https://", content)
                verification_label = (
                    "Source checked"
                    if source in {
                        "official_pmegp_reference.txt",
                        "official_public_procurement_reference.txt",
                    }
                    else "Last verified"
                )
                self.assertIn(f"{verification_label}: 2026-10-21", content)

    def test_unavailable_scheme_details_are_not_fabricated_or_generated(self):
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch(
                "services.ai_service._generate_gemini",
                return_value="The local reference does not verify these details "
                "[official_pmegp_reference.txt].",
            ) as gemini,
            patch("services.ai_service.generate_local_response") as local,
        ):
            result = generate_rag_answer(
                "What is the PMEGP subsidy amount and application deadline?"
            )

        self.assertEqual(result["provider"], "gemini")
        self.assertIn("official_pmegp_reference.txt", result["sources"])
        self.assertIn("official_pmegp_reference.txt", result["answer"])
        prompt = gemini.call_args.args[0]
        self.assertIn("not available in the excerpts", prompt)
        self.assertIn("official_pmegp_reference.txt", prompt)
        local.assert_not_called()

    def test_unsupported_question_returns_no_local_source_or_generated_answer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "source.txt").write_text(
                "Udyam registration is handled on the official portal.",
                encoding="utf-8",
            )
            with (
                patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
                patch("services.ai_service._generate_gemini") as gemini,
                patch("services.ai_service.generate_local_response") as local,
            ):
                result = generate_rag_answer(
                    "Explain international space launch vehicle propellants.",
                    doc_dir=temp_dir,
                )

        self.assertEqual(result["provider"], "retrieval_fallback")
        self.assertEqual(result["sources"], [])
        self.assertIn("No relevant content was found", result["answer"])
        gemini.assert_not_called()
        local.assert_not_called()

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
        with patch.dict(os.environ):
            os.environ.pop("GEMINI_MODEL", None)
            with patch("google.genai.Client", return_value=client) as genai_client:
                answer = _generate_gemini(
                    "Explain these metrics", api_key="demo-key"
                )

        self.assertEqual(answer, "Gemini explanation")
        self.assertEqual(
            genai_client.call_args.kwargs["http_options"].timeout,
            10_000,
        )
        client.models.generate_content.assert_called_once()
        self.assertEqual(
            client.models.generate_content.call_args.kwargs["model"],
            "gemini-3.5-flash-lite",
        )

    def test_gemini_failure_logs_safe_status_classification(self):
        for status_code, category in (
            (429, "rate_limited"),
            (503, "service_unavailable"),
            (504, "gateway_timeout"),
            (500, "other"),
        ):
            with self.subTest(status_code=status_code):
                error = RuntimeError("sensitive exception message")
                error.code = status_code
                client = Mock()
                client.models.generate_content.side_effect = error

                with (
                    patch("google.genai.Client", return_value=client),
                    patch("services.ai_service.GEMINI_MAX_RETRIES", 0),
                    patch("services.ai_service.LOGGER.warning") as warning,
                ):
                    self.assertIsNone(
                        _generate_gemini("sensitive prompt", api_key="demo-key")
                    )

                message, exception_class, logged_status, logged_category = (
                    warning.call_args.args
                )
                self.assertIn("Gemini request failed; using local fallback", message)
                self.assertEqual(exception_class, "RuntimeError")
                self.assertEqual(logged_status, status_code)
                self.assertEqual(logged_category, category)
                self.assertNotIn("sensitive", " ".join(map(str, warning.call_args.args)))

    def test_gemini_failure_logs_unknown_for_missing_or_invalid_status(self):
        for status_code in (None, "503", True, 99, 600):
            with self.subTest(status_code=status_code):
                error = RuntimeError("sensitive exception message")
                if status_code is not None:
                    error.code = status_code
                client = Mock()
                client.models.generate_content.side_effect = error

                with (
                    patch("google.genai.Client", return_value=client),
                    patch("services.ai_service.GEMINI_MAX_RETRIES", 0),
                    patch("services.ai_service.LOGGER.warning") as warning,
                ):
                    self.assertIsNone(
                        _generate_gemini("sensitive prompt", api_key="demo-key")
                    )

                _, exception_class, logged_status, logged_category = (
                    warning.call_args.args
                )
                self.assertEqual(exception_class, "RuntimeError")
                self.assertEqual(logged_status, "unknown")
                self.assertEqual(logged_category, "unknown")

    def test_transient_gemini_error_retries_once_after_server_delay(self):
        error = RuntimeError("test error")
        error.code = 503
        error.details = {
            "error": {
                "details": [{
                    "@type": "type.googleapis.com/google.rpc.RetryInfo",
                    "retryDelay": "1.25s",
                }],
            },
        }
        client = Mock()
        client.models.generate_content.side_effect = [
            error,
            Mock(text="Gemini explanation"),
        ]

        with (
            patch("google.genai.Client", return_value=client),
            patch("services.ai_service.time.sleep") as sleep,
        ):
            answer = _generate_gemini("Explain these metrics", api_key="demo-key")

        self.assertEqual(answer, "Gemini explanation")
        self.assertEqual(client.models.generate_content.call_count, 2)
        sleep.assert_called_once_with(1.25)

    def test_transient_gemini_error_respects_retry_after_header(self):
        error = RuntimeError("test error")
        error.code = 429
        error.response = Mock(headers={"Retry-After": "0.75"})
        client = Mock()
        client.models.generate_content.side_effect = [
            error,
            Mock(text="Gemini explanation"),
        ]

        with (
            patch("google.genai.Client", return_value=client),
            patch("services.ai_service.time.sleep") as sleep,
        ):
            answer = _generate_gemini("Explain these metrics", api_key="demo-key")

        self.assertEqual(answer, "Gemini explanation")
        self.assertEqual(client.models.generate_content.call_count, 2)
        sleep.assert_called_once_with(0.75)

    def test_transient_gemini_error_with_long_delay_falls_back_without_retry(self):
        error = RuntimeError("test error")
        error.code = 429
        error.details = {
            "error": {
                "details": [{
                    "@type": "type.googleapis.com/google.rpc.RetryInfo",
                    "retryDelay": "30s",
                }],
            },
        }
        client = Mock()
        client.models.generate_content.side_effect = error

        with (
            patch("google.genai.Client", return_value=client),
            patch("services.ai_service.time.sleep") as sleep,
        ):
            answer = _generate_gemini("Explain these metrics", api_key="demo-key")

        self.assertIsNone(answer)
        client.models.generate_content.assert_called_once()
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()

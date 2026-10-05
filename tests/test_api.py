import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.business_id = "MSME001"

    def test_health_and_business_list_are_json(self):
        health = self.client.get("/api/health")
        businesses = self.client.get("/api/businesses")

        self.assertEqual(health.status_code, 200)
        self.assertIn(health.json()["status"], ("ok", "degraded"))
        self.assertIsInstance(health.json()["gemini_configured"], bool)
        self.assertEqual(businesses.status_code, 200)
        self.assertTrue(any(
            item["business_id"] == self.business_id for item in businesses.json()
        ))

    def test_business_and_kpis_return_selected_record_and_history(self):
        business = self.client.get(f"/api/business/{self.business_id}")
        kpis = self.client.get(f"/api/business/{self.business_id}/kpis")

        self.assertEqual(business.status_code, 200)
        self.assertGreater(len(business.json()["history"]), 1)
        self.assertEqual(kpis.status_code, 200)
        self.assertIn("working_capital", kpis.json())
        self.assertIn("profit_margin", kpis.json())

    def test_risk_anomaly_shap_and_forecast_endpoints(self):
        risk = self.client.get(f"/api/business/{self.business_id}/risk")
        anomaly = self.client.get(f"/api/business/{self.business_id}/anomaly")
        shap = self.client.get(f"/api/business/{self.business_id}/shap")
        forecast = self.client.get(f"/api/business/{self.business_id}/forecast")

        self.assertEqual(risk.status_code, 200, risk.text)
        self.assertIn(risk.json()["label"], ("LOW", "MEDIUM", "HIGH"))
        self.assertEqual(anomaly.status_code, 200, anomaly.text)
        self.assertIn(anomaly.json()["status"], ("NORMAL", "ANOMALY"))
        self.assertEqual(shap.status_code, 200, shap.text)
        self.assertEqual(len(shap.json()["features"]), 10)
        self.assertEqual(forecast.status_code, 200, forecast.text)
        for metric in ("revenue", "cash_balance"):
            self.assertEqual(len(forecast.json()["forecasts"][metric]["forecast"]), 6)

    def test_unknown_business_returns_not_found(self):
        response = self.client.get("/api/business/UNKNOWN/kpis")

        self.assertEqual(response.status_code, 404)
        self.assertIn("not found", response.json()["detail"])

    def test_ai_summary_uses_existing_service_and_returns_status(self):
        with patch(
            "backend.routes.api.generate_financial_summary_result",
            return_value={
                "answer": "Generated summary",
                "used_gemini": True,
                "provider": "gemini",
            },
        ) as generate:
            response = self.client.post(
                "/api/ai/summary",
                json={"business_id": self.business_id},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["answer"], "Generated summary")
        self.assertTrue(response.json()["used_gemini"])
        self.assertEqual(response.json()["provider"], "gemini")
        generate.assert_called_once()

    def test_compliance_query_uses_existing_rag_service(self):
        expected = {
            "answer": "Grounded answer",
            "sources": ["msme_finance_reference_1.txt"],
            "retrieved_documents": [],
            "used_gemini": False,
            "provider": "retrieval_fallback",
        }
        with patch("backend.routes.api.generate_rag_answer", return_value=expected) as query:
            response = self.client.post(
                "/api/compliance/query",
                json={"question": "How should cash flow be monitored?"},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), expected)
        query.assert_called_once()

    def test_ai_status_reports_safe_provider_readiness(self):
        with patch(
            "backend.routes.api.get_ai_status",
            return_value={
                "gemini_available": False,
                "gemini_configured": False,
                "ollama_installed": False,
                "ollama_running": False,
                "local_llm_available": False,
                "model": "llama3.2:3b",
                "retrieval_available": True,
                "offline_ready": True,
                "preferred_provider": "retrieval_fallback",
            },
        ):
            response = self.client.get("/api/ai/status")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(response.json()["gemini_available"])
        self.assertTrue(response.json()["offline_ready"])
        self.assertNotIn("GEMINI_API_KEY", response.json())

    def test_compliance_endpoint_returns_sources_and_selected_provider(self):
        expected = {
            "answer": "Local grounded answer",
            "sources": ["msme_finance_reference_1.txt"],
            "retrieved_documents": [{
                "source": "msme_finance_reference_1.txt",
                "chunk": 1,
                "snippet": "Working capital should be monitored.",
            }],
            "used_gemini": False,
            "provider": "local",
        }
        with patch("backend.routes.api.generate_rag_answer", return_value=expected):
            response = self.client.post(
                "/api/compliance/query",
                json={"question": "How should working capital be monitored?"},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["provider"], "local")
        self.assertEqual(response.json()["sources"], expected["sources"])
        self.assertEqual(
            response.json()["retrieved_documents"],
            expected["retrieved_documents"],
        )


if __name__ == "__main__":
    unittest.main()

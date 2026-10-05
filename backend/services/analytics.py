from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RISK_MODEL_PATH = PROJECT_ROOT / "models" / "financial_risk_model.pkl"
ANOMALY_MODEL_PATH = PROJECT_ROOT / "models" / "anomaly_detection_model.pkl"
RISK_LABELS = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
MODEL_FEATURES = [
    "employee_count",
    "revenue",
    "expenses",
    "cash_balance",
    "accounts_receivable",
    "accounts_payable",
    "inventory_value",
    "loan_emi",
    "loan_outstanding",
    "profit",
    "profit_margin",
    "working_capital",
    "cash_ratio",
    "receivable_ratio",
    "payable_ratio",
    "debt_service_ratio",
    "inventory_ratio",
    "revenue_growth",
    "expense_growth",
    "cash_growth",
    "receivable_growth",
]


@lru_cache(maxsize=1)
def _load_models():
    return (
        joblib.load(RISK_MODEL_PATH),
        joblib.load(ANOMALY_MODEL_PATH),
    )


def _ratio(numerator, denominator):
    if not denominator or pd.isna(denominator):
        return 0.0
    return numerator / denominator


def build_model_input(history):
    current = history.sort_values("month").iloc[-1]
    ordered_history = history.sort_values("month")

    values = {
        "employee_count": current["employee_count"],
        "revenue": current["revenue"],
        "expenses": current["expenses"],
        "cash_balance": current["cash_balance"],
        "accounts_receivable": current["accounts_receivable"],
        "accounts_payable": current["accounts_payable"],
        "inventory_value": current["inventory_value"],
        "loan_emi": current["loan_emi"],
        "loan_outstanding": current["loan_outstanding"],
        "profit": current["revenue"] - current["expenses"],
    }
    values["profit_margin"] = _ratio(values["profit"], current["revenue"]) * 100
    values["working_capital"] = (
        current["accounts_receivable"]
        + current["inventory_value"]
        - current["accounts_payable"]
    )
    values["cash_ratio"] = _ratio(current["cash_balance"], current["expenses"])
    values["receivable_ratio"] = _ratio(
        current["accounts_receivable"], current["revenue"]
    )
    values["payable_ratio"] = _ratio(
        current["accounts_payable"], current["expenses"]
    )
    values["debt_service_ratio"] = _ratio(current["loan_emi"], current["revenue"])
    values["inventory_ratio"] = _ratio(current["inventory_value"], current["revenue"])

    for column, feature in (
        ("revenue", "revenue_growth"),
        ("expenses", "expense_growth"),
        ("cash_balance", "cash_growth"),
        ("accounts_receivable", "receivable_growth"),
    ):
        growth = ordered_history[column].pct_change().iloc[-1] * 100
        values[feature] = 0.0 if pd.isna(growth) else growth

    return pd.DataFrame([[values[name] for name in MODEL_FEATURES]], columns=MODEL_FEATURES)


class AnalyticsService:
    def __init__(self):
        self.risk_model, self.anomaly_model = _load_models()

    def analyze(self, history):
        model_input = build_model_input(history)
        prediction = int(self.risk_model.predict(model_input)[0])
        probabilities = self.risk_model.predict_proba(model_input)[0]
        probability_map = {
            RISK_LABELS.get(int(class_id), str(class_id)): float(probability)
            for class_id, probability in zip(self.risk_model.classes_, probabilities)
        }
        anomaly_prediction = int(self.anomaly_model.predict(model_input)[0])
        anomaly_score = float(self.anomaly_model.decision_function(model_input)[0])
        shap_values = shap.TreeExplainer(self.risk_model).shap_values(model_input)

        if isinstance(shap_values, list):
            class_values = shap_values[prediction][0]
        else:
            shap_array = np.asarray(shap_values)
            if shap_array.ndim == 3:
                class_values = shap_array[0, :, prediction]
            else:
                class_values = shap_array[0]

        features = sorted(
            (
                {
                    "feature": name,
                    "value": float(value),
                    "absolute_impact": abs(float(value)),
                }
                for name, value in zip(MODEL_FEATURES, class_values)
            ),
            key=lambda item: item["absolute_impact"],
            reverse=True,
        )[:10]
        label = RISK_LABELS.get(prediction, str(prediction))
        current = history.sort_values("month").iloc[-1]
        anomaly_status = "ANOMALY" if anomaly_prediction == -1 else "NORMAL"
        return {
            "risk": {
                "business_id": str(current["business_id"]),
                "label": label,
                "probabilities": probability_map,
                "explanation": (
                    f"The existing Random Forest model classifies the latest record as {label}. "
                    "Probabilities and SHAP feature contributions are model indicators, not a "
                    "guarantee of future business performance."
                ),
            },
            "anomaly": {
                "business_id": str(current["business_id"]),
                "month": current["month"].isoformat(),
                "status": anomaly_status,
                "prediction": anomaly_prediction,
                "score": anomaly_score,
                "explanation": (
                    "The existing Isolation Forest model identified an unusual financial pattern."
                    if anomaly_prediction == -1
                    else "The existing Isolation Forest model did not flag an unusual pattern."
                ),
                "indicators": {
                    name: float(model_input.iloc[0][name])
                    for name in MODEL_FEATURES
                },
            },
            "shap": features,
        }

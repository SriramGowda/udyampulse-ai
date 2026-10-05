import logging
import os
import json

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.services.analytics import AnalyticsService
from services.ai_service import (
    get_ai_status,
    generate_financial_summary_result,
    generate_rag_answer,
    is_gemini_configured,
)
from services.data_service import (
    calculate_kpis,
    evaluate_forecast,
    forecast_business,
    get_business_history,
    get_latest_business_data,
    load_data,
)


LOGGER = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


class ComplianceQuery(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class SummaryRequest(BaseModel):
    business_id: str = Field(min_length=1, max_length=100)


def _json_records(frame):
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _json_record(record):
    return json.loads(
        pd.DataFrame([record]).to_json(orient="records", date_format="iso")
    )[0]


def _data():
    return calculate_kpis(load_data())


def _business_data(business_id):
    frame = _data()
    if business_id not in set(frame["business_id"].astype(str)):
        raise HTTPException(status_code=404, detail=f"Business '{business_id}' was not found.")
    return frame, get_business_history(frame, business_id)


def _analysis(history):
    return AnalyticsService().analyze(history)


@router.get("/health")
def health():
    try:
        data = load_data()
        data_status = "ready" if not data.empty else "empty"
    except Exception:
        LOGGER.exception("Unable to load the financial dataset for health check.")
        data_status = "error"
    return {
        "status": "ok" if data_status == "ready" else "degraded",
        "data": data_status,
        "gemini_configured": is_gemini_configured(),
    }


@router.get("/ai/status")
def ai_status():
    return get_ai_status()


@router.get("/businesses")
def businesses():
    latest = get_latest_business_data(_data())
    return _json_records(
        latest[["business_id", "business_name", "sector", "state", "month"]]
        .sort_values("business_id")
    )


@router.get("/business/{business_id}")
def business(business_id: str):
    _, history = _business_data(business_id)
    return {
        "business": _json_record(history.iloc[-1]),
        "history": _json_records(history.sort_values("month")),
    }


@router.get("/business/{business_id}/kpis")
def business_kpis(business_id: str):
    _, history = _business_data(business_id)
    return _json_record(history.sort_values("month").iloc[-1])


@router.get("/business/{business_id}/risk")
def business_risk(business_id: str):
    _, history = _business_data(business_id)
    result = _analysis(history)
    return result["risk"]


@router.get("/business/{business_id}/anomaly")
def business_anomaly(business_id: str):
    _, history = _business_data(business_id)
    result = _analysis(history)
    return result["anomaly"]


@router.get("/business/{business_id}/shap")
def business_shap(business_id: str):
    _, history = _business_data(business_id)
    result = _analysis(history)
    return {"business_id": business_id, "features": result["shap"]}


@router.get("/business/{business_id}/forecast")
def business_forecast(business_id: str):
    frame, history = _business_data(business_id)
    forecasts = forecast_business(frame, business_id, months_ahead=6)
    evaluations = evaluate_forecast(frame, business_id)
    result = {}
    for metric in ("revenue", "cash_balance"):
        evaluation = evaluations.loc[evaluations["metric"] == metric].iloc[0]
        result[metric] = {
            "history": _json_records(history[["month", metric]]),
            "forecast": _json_records(forecasts[["month", metric]]),
            "evaluation": _json_record(evaluation),
        }
    return {"business_id": business_id, "forecasts": result}


@router.post("/ai/summary")
def ai_summary(request: SummaryRequest):
    _, history = _business_data(request.business_id)
    history = history.sort_values("month")
    result = _analysis(history)
    forecast = forecast_business(_data(), request.business_id, months_ahead=6)
    latest = history.iloc[-1]
    revenue_change = (
        float((forecast["revenue"].iloc[-1] / latest["revenue"] - 1) * 100)
        if latest["revenue"] else None
    )
    cash_change = (
        float((forecast["cash_balance"].iloc[-1] / latest["cash_balance"] - 1) * 100)
        if latest["cash_balance"] else None
    )
    summary = generate_financial_summary_result(
        history,
        business_history=history,
        analysis_context={
            "risk_label": result["risk"]["label"],
            "risk_probabilities": result["risk"]["probabilities"],
            "anomaly_result": result["anomaly"]["status"].lower(),
            "revenue_forecast_change_pct": revenue_change,
            "cash_forecast_change_pct": cash_change,
            "forecast": {
                "revenue_next_six_months": forecast["revenue"].tolist(),
                "cash_next_six_months": forecast["cash_balance"].tolist(),
            },
        },
    )
    return summary


@router.post("/compliance/query")
def compliance_query(request: ComplianceQuery):
    try:
        return generate_rag_answer(
            request.question,
            doc_dir=os.path.join(os.path.dirname(__file__), "..", "..", "documents"),
        )
    except Exception as error:
        LOGGER.exception("Compliance assistant request failed.")
        raise HTTPException(
            status_code=503,
            detail="The compliance assistant could not process this request.",
        ) from error

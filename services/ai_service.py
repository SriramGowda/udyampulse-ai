import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCUMENT_DIR = PROJECT_ROOT / "documents"

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    LOGGER.warning("python-dotenv is not installed; environment variables will be used.")


def is_gemini_configured():
    return bool(os.getenv("GEMINI_API_KEY"))


def _generate_gemini(prompt, api_key=None):
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=prompt,
        )
        answer = response.text
        return answer.strip() if answer else None
    except Exception as error:
        LOGGER.warning("Gemini request failed; using local fallback (%s).", error.__class__.__name__)
        return None


def _load_document_files(doc_dir):
    folder = Path(doc_dir)
    if not folder.is_absolute():
        folder = PROJECT_ROOT / folder
    if not folder.exists():
        return []

    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}
    )


def _chunk_text(text, chunk_size=160, overlap=30):
    words = re.findall(r"\S+", text)
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def _document_chunks(doc_dir):
    chunks = []
    for path in _load_document_files(doc_dir):
        text = path.read_text(encoding="utf-8", errors="replace")
        for index, content in enumerate(_chunk_text(text)):
            chunks.append({
                "source": path.name,
                "chunk": index + 1,
                "content": content,
            })
    return chunks


def retrieve_documents(question, doc_dir="documents", max_results=3):
    """Retrieve the most relevant local document chunks using TF-IDF cosine similarity."""
    if not question or not question.strip() or max_results < 1:
        return []

    chunks = _document_chunks(doc_dir)
    if not chunks:
        return []

    texts = [chunk["content"] for chunk in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform(texts + [question.strip()])
    except ValueError:
        return []

    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    ranked_indices = np.argsort(scores)[::-1]
    results = []
    for index in ranked_indices:
        if scores[index] <= 0:
            break
        chunk = chunks[index]
        results.append({
            "source": chunk["source"],
            "chunk": chunk["chunk"],
            "score": float(scores[index]),
            "snippet": chunk["content"],
        })
        if len(results) == max_results:
            break
    return results


def _format_metric(context, key, formatter):
    value = context.get(key)
    if value is None or pd.isna(value):
        return "not available"
    return formatter(value)


def _local_financial_summary(business_data, analysis_context):
    latest = business_data.iloc[-1]
    history = business_data.sort_values("month")
    recent = history.tail(min(6, len(history)))

    def trend(column):
        first, last = recent[column].iloc[0], recent[column].iloc[-1]
        if pd.isna(first) or pd.isna(last) or first == 0:
            return "unavailable"
        return f"{((last - first) / abs(first)) * 100:+.1f}% over the latest {len(recent)} months"

    context = analysis_context or {}
    business_name = latest.get("business_name", "The selected business")
    month = pd.to_datetime(latest["month"]).strftime("%B %Y")
    risk = context.get("risk_label", "not available")
    anomaly = context.get("anomaly_result", "not available")
    revenue_forecast = _format_metric(
        context, "revenue_forecast_change_pct", lambda value: f"{value:+.1f}%"
    )
    cash_forecast = _format_metric(
        context, "cash_forecast_change_pct", lambda value: f"{value:+.1f}%"
    )
    return (
        f"{business_name} ({latest.get('sector', 'sector not available')}) — latest data: {month}. "
        f"Revenue is ₹{latest['revenue']:,.0f} and expenses are ₹{latest['expenses']:,.0f}; "
        f"profit is ₹{latest['profit']:,.0f} with a {latest['profit_margin']:.1f}% margin. "
        f"Revenue changed {trend('revenue')} and expenses changed {trend('expenses')}. "
        f"Cash balance is ₹{latest['cash_balance']:,.0f}; the model's current risk classification is {risk}, "
        f"and anomaly detection reports {anomaly}. "
        f"The six-month forecast indicates revenue change of {revenue_forecast} and cash change of "
        f"{cash_forecast} versus the latest month. These are synthetic-data indicators, not financial advice."
    )


def generate_financial_summary(
    business_data, business_history=None, api_key=None, analysis_context=None
):
    """Explain the latest financial position and supplied model outputs."""
    if business_data.empty:
        return "No financial data is available for this business."

    history = business_history if business_history is not None else business_data
    context = analysis_context or {}
    local_summary = _local_financial_summary(business_data, context)

    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return local_summary

    latest = business_data.sort_values("month").iloc[-1]
    recent = history.sort_values("month").tail(6)
    metrics = {
        "business": latest.get("business_name", "Selected business"),
        "sector": latest.get("sector"),
        "month": str(pd.to_datetime(latest["month"]).date()),
        "revenue": float(latest["revenue"]),
        "expenses": float(latest["expenses"]),
        "profit": float(latest["profit"]),
        "profit_margin_pct": float(latest["profit_margin"]),
        "cash_balance": float(latest["cash_balance"]),
        "revenue_trend_recent": recent["revenue"].pct_change().dropna().tolist(),
        "expense_trend_recent": recent["expenses"].pct_change().dropna().tolist(),
        "risk_prediction": context.get("risk_label", "not available"),
        "risk_probabilities": context.get("risk_probabilities", {}),
        "anomaly_result": context.get("anomaly_result", "not available"),
        "six_month_forecast": context.get("forecast", {}),
    }
    answer = _generate_gemini(
        "Explain the supplied synthetic MSME financial metrics in plain language. "
        "Cover financial condition, revenue and expense trends, profit, cash, risk prediction, "
        "anomaly result, and six-month revenue/cash forecast. Explain uncertainty and do not invent "
        "facts, give generic recommendations, or claim access to private records. "
        "Treat all values as synthetic demo data.\n\n"
        f"Metrics: {metrics}\n"
        f"Local analytical summary: {local_summary}",
        api_key=api_key,
    )
    return answer or local_summary


def generate_rag_answer(question, doc_dir="documents", api_key=None):
    """Answer from retrieved local references and return their source filenames."""
    results = retrieve_documents(question, doc_dir=doc_dir, max_results=3)
    if not results:
        return {
            "answer": (
                "No relevant content was found in the local reference documents. "
                "This assistant does not access private GST, banking, or government databases."
            ),
            "sources": [],
            "used_gemini": False,
        }

    context = "\n\n".join(
        f"[{item['source']}, chunk {item['chunk']}]\n{item['snippet']}"
        for item in results
    )
    sources = list(dict.fromkeys(item["source"] for item in results))
    api_key = api_key or os.getenv("GEMINI_API_KEY")

    if api_key:
        answer = _generate_gemini(
            "Answer the question using only the reference excerpts. Cite supporting statements "
            "with their bracketed source filename. If evidence is insufficient, say so. "
            "Do not claim access to private GST, bank, or government databases. "
            "Do not invent legal/compliance requirements.\n\n"
            f"Question: {question}\n\nReference excerpts:\n{context}",
            api_key=api_key,
        )
        if answer:
            return {"answer": answer, "sources": sources, "used_gemini": True}

    excerpts = "\n\n".join(
        f"**{item['source']}**: {item['snippet']}" for item in results
    )
    return {
        "answer": (
            "Gemini is unavailable, so this response shows the most relevant passages "
            "retrieved from the local reference library rather than generating new advice.\n\n"
            f"{excerpts}\n\n"
            "These demo references are not a substitute for official compliance guidance."
        ),
        "sources": sources,
        "used_gemini": False,
    }
